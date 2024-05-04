unit GaussianYoloLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$PointerMath on}

interface

uses
  SysUtils, Darknet, blas, Activations;

type
  TGaussianYoloLayer = TLayer;

function make_gaussian_yolo_layer(const batch, w, h, n, total: longint; const mask: TArray<longint>; const classes, max_boxes: longint):TGaussianYoloLayer;
procedure resize_gaussian_yolo_layer(var l: TGaussianYoloLayer; const w, h: longint);
function get_gaussian_yolo_box(const x, biases: PSingle; const n, index, i, j, lw, lh, w, h, stride: longint; const yolo_point: TYOLOPoint):TBox;
function fix_nan_inf(const val: single):single;inline;
function clip_value(const val: single; const max_val: single):single;inline;
function delta_gaussian_yolo_box(const truth: TBox; const x, biases: PSingle; const n, index, i, j, lw, lh, w, h: longint; const delta: PSingle; const scale: single; const stride: longint; const iou_normalizer: single; const iou_loss: TIOULoss; const uc_normalizer: single; const accumulate: longint; const yolo_point: TYOLOPoint; const max_delta: single):single;
procedure averages_gaussian_yolo_deltas(const class_index, box_index, stride, classes: longint; const delta: PSingle);
procedure delta_gaussian_yolo_class(const output, delta: PSingle; const index, class_id, classes, stride: longint; const avg_cat: PSingle; const label_smooth_eps: single; const classes_multipliers: PSingle; const cls_normalizer: single);
function compare_gaussian_yolo_class(const output: PSingle; classes, class_index, stride: longint; const objectness: single; const class_id: longint; const conf_thresh: single):longint;
procedure forward_gaussian_yolo_layer(var l: TGaussianYoloLayer; const state: PNetworkState);
procedure backward_gaussian_yolo_layer(var l: TGaussianYoloLayer; const state: PNetworkState);
procedure correct_gaussian_yolo_boxes(const dets: Pdetection; const n, w, h, netw, neth: longint; const relative, letter: boolean);
function gaussian_yolo_num_detections(const l: TGaussianYoloLayer; const thresh: single):longint;
function get_gaussian_yolo_detections(const l: TGaussianYoloLayer; const w, h, netw, neth: longint; const thresh: single; const map: Plongint; const relative: boolean; const dets: Pdetection; const letter: boolean):longint;

{$ifdef GPU}
procedure forward_gaussian_yolo_layer_gpu(var l: TGaussianYoloLayer; const state: PNetworkState);
procedure backward_gaussian_yolo_layer_gpu(const l: layer; state: network_state);
{$endif}

implementation
uses math, box;

function make_gaussian_yolo_layer(const batch, w, h, n, total: longint; const mask: TArray<longint>; const classes, max_boxes: longint):TGaussianYoloLayer;
var
    i: longint;
begin
    result := Default(TGaussianYoloLayer);
    result.&type := ltGaussianYOLO;
    result.n := n;
    result.total := total;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := n * (classes+8+1);
    result.out_w := result.w;
    result.out_h := result.h;
    result.out_c := result.c;
    result.classes := classes;
    result.cost := TSingles.Create(1);
    result.biases := TSingles.Create(total * 2);
    if assigned(mask) then
        result.mask := mask
    else
        begin
            setLength(result.mask, n);
            for i := 0 to n -1 do
                result.mask[i] := i
        end;
    result.bias_updates := TSingles.Create(n * 2);
    result.outputs := h * w * n * (classes+8+1);
    result.inputs := result.outputs;
    result.max_boxes := max_boxes;
    result.truth_size := 4+2;
    result.truths := result.max_boxes * result.truth_size;
    result.delta := TSingles.Create(batch * result.outputs);
    result.output := TSingles.Create(batch * result.outputs);
    for i := 0 to total * 2 -1 do
        result.biases[i] := 0.5;
    result.forward := forward_gaussian_yolo_layer;
    result.backward := backward_gaussian_yolo_layer;
{$ifdef GPU}
    result.forward_gpu := forward_gaussian_yolo_layer_gpu;
    result.backward_gpu := backward_gaussian_yolo_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.delta_gpu := cuda_make_array(result.delta, batch * result.outputs);
    free(result.output);
    if cudaSuccess = cudaHostAlloc( and result.output, batch * result.outputs * sizeof(float), cudaHostRegisterMapped) then
        result.output_pinned := 1
    else
        begin
            cudaGetLastError();
            result.output := single(calloc(batch * result.outputs, sizeof(float)))
        end;
    free(result.delta);
    if cudaSuccess = cudaHostAlloc( and result.delta, batch * result.outputs * sizeof(float), cudaHostRegisterMapped) then
        result.delta_pinned := 1
    else
        begin
            cudaGetLastError();
            result.delta := single(calloc(batch * result.outputs, sizeof(float)))
        end;
{$endif}
    Randomize;

end;

procedure resize_gaussian_yolo_layer(var l: TGaussianYoloLayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.outputs := h * w * l.n * (l.classes+8+1);
    l.inputs := l.outputs;
    if l.output_pinned=0 then
        l.output.reAllocate( l.batch * l.outputs);
    if l.delta_pinned=0 then
        l.delta.reAllocate(l.batch * l.outputs);
{$ifdef GPU}
    if l.output_pinned then
        begin
            CHECK_CUDA(cudaFreeHost(l.output));
            if cudaSuccess <> cudaHostAlloc( and l.output, l.batch * l.outputs * sizeof(float), cudaHostRegisterMapped) then
                begin
                    cudaGetLastError();
                    l.output := single(calloc(l.batch * l.outputs, sizeof(float)));
                    l.output_pinned := 0
                end
        end;
    if l.delta_pinned then
        begin
            CHECK_CUDA(cudaFreeHost(l.delta));
            if cudaSuccess <> cudaHostAlloc( and l.delta, l.batch * l.outputs * sizeof(float), cudaHostRegisterMapped) then
                begin
                    cudaGetLastError();
                    l.delta := single(calloc(l.batch * l.outputs, sizeof(float)));
                    l.delta_pinned := 0
                end
        end;
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);
    l.delta_gpu := cuda_make_array(l.delta, l.batch * l.outputs);
    l.output_gpu := cuda_make_array(l.output, l.batch * l.outputs)
{$endif}
end;

function get_gaussian_yolo_box(const x, biases: PSingle; const n, index, i, j, lw, lh, w, h, stride: longint; const yolo_point: TYOLOPoint):TBox;
begin
    result.w := exp(x[index+4 * stride]) * biases[2 * n] / w;
    result.h := exp(x[index+6 * stride]) * biases[2 * n+1] / h;
    result.x := (i+x[index+0 * stride]) / lw;
    result.y := (j+x[index+2 * stride]) / lh;
    if yolo_point = ypYOLO_CENTER then

    else
        if yolo_point = ypYOLO_LEFT_TOP then
            begin
                result.x := (i+x[index+0 * stride]) / lw+result.w / 2;
                result.y := (j+x[index+2 * stride]) / lh+result.h / 2
            end
    else
        if yolo_point = ypYOLO_RIGHT_BOTTOM then
            begin
                result.x := (i+x[index+0 * stride]) / lw-result.w / 2;
                result.y := (j+x[index+2 * stride]) / lh-result.h / 2
            end;
end;

function fix_nan_inf(const val: single):single;inline;
begin
    if IsNan(val) or IsInfinite(val) then
        result := 0;
    result:=val
end;

function clip_value(const val: single; const max_val: single):single;inline;
begin
    if val > max_val then
        exit( max_val)
    else  if result < -max_val then
        exit( -max_val);
    result := val
end;

function delta_gaussian_yolo_box(const truth: TBox; const x, biases: PSingle; const n, index, i, j, lw, lh, w, h: longint; const delta: PSingle; const scale: single; const stride: longint; const iou_normalizer: single; const iou_loss: TIOULoss; const uc_normalizer: single; const accumulate: longint; const yolo_point: TYOLOPoint; const max_delta: single):single;
var
    pred: TBox;
    iou: single;
    all_ious: TIOUs;
    sigma_const, epsi, dx, dy, dw, dh, tx, ty, tw, th,
      in_exp_x, in_exp_x_2, normal_dist_x, in_exp_y, in_exp_y_2, normal_dist_y,
      in_exp_w, in_exp_w_2, normal_dist_w, in_exp_h, in_exp_h_2, normal_dist_h,
      temp_x, temp_y, temp_w, temp_h,
      delta_x, delta_y, delta_w, delta_h,
      delta_ux, delta_uy, delta_uw, delta_uh : single;
begin
    pred := get_gaussian_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, yolo_point);
    all_ious := default(TIOUs);
    all_ious.iou := box_iou(pred, truth);
    all_ious.giou := box_giou(pred, truth);
    all_ious.diou := box_diou(pred, truth);
    all_ious.ciou := box_ciou(pred, truth);
    if pred.w = 0 then
        pred.w := 1.0;
    if pred.h = 0 then
        pred.h := 1.0;
    sigma_const := 0.3;
    epsi := power(10, -9);
    iou := all_ious.iou;
    tx := (truth.x * lw-i);
    ty := (truth.y * lh-j);
    tw := ln(truth.w * w / biases[2 * n]);
    th := ln(truth.h * h / biases[2 * n+1]);
    if yolo_point = ypYOLO_CENTER then

    else
        if yolo_point = ypYOLO_LEFT_TOP then
            begin
                tx := ((truth.x-truth.w / 2) * lw-i);
                ty := ((truth.y-truth.h / 2) * lh-j)
            end
    else
        if yolo_point = ypYOLO_RIGHT_BOTTOM then
            begin
                tx := ((truth.x+truth.w / 2) * lw-i);
                ty := ((truth.y+truth.h / 2) * lh-j)
            end;
    dx := (tx-x[index+0 * stride]);
    dy := (ty-x[index+2 * stride]);
    dw := (tw-x[index+4 * stride]);
    dh := (th-x[index+6 * stride]);
    in_exp_x := dx / x[index+1 * stride];
    in_exp_x_2 := sqr(in_exp_x{, 2});
    normal_dist_x := exp(in_exp_x_2 * (-1.0 / 2.0)) / (sqrt(PI() * 2.0) * (x[index+1 * stride]+sigma_const));
    in_exp_y := dy / x[index+3 * stride];
    in_exp_y_2 := sqr(in_exp_y{, 2});
    normal_dist_y := exp(in_exp_y_2 * (-1.0 / 2.0)) / (sqrt(PI() * 2.0) * (x[index+3 * stride]+sigma_const));
    in_exp_w := dw / x[index+5 * stride];
    in_exp_w_2 := sqr(in_exp_w{, 2});
    normal_dist_w := exp(in_exp_w_2 * (-1.0 / 2.0)) / (sqrt(PI() * 2.0) * (x[index+5 * stride]+sigma_const));
    in_exp_h := dh / x[index+7 * stride];
    in_exp_h_2 := sqr(in_exp_h{, 2});
    normal_dist_h := exp(in_exp_h_2 * (-1.0 / 2.0)) / (sqrt(PI() * 2.0) * (x[index+7 * stride]+sigma_const));
    temp_x := (1.0 / 2.0) * 1.0 / (normal_dist_x+epsi) * normal_dist_x * scale;
    temp_y := (1.0 / 2.0) * 1.0 / (normal_dist_y+epsi) * normal_dist_y * scale;
    temp_w := (1.0 / 2.0) * 1.0 / (normal_dist_w+epsi) * normal_dist_w * scale;
    temp_h := (1.0 / 2.0) * 1.0 / (normal_dist_h+epsi) * normal_dist_h * scale;
    if accumulate=0 then
        begin
            delta[index+0 * stride] := 0;
            delta[index+1 * stride] := 0;
            delta[index+2 * stride] := 0;
            delta[index+3 * stride] := 0;
            delta[index+4 * stride] := 0;
            delta[index+5 * stride] := 0;
            delta[index+6 * stride] := 0;
            delta[index+7 * stride] := 0
        end;
    delta_x := temp_x * in_exp_x * (1.0 / x[index+1 * stride]);
    delta_y := temp_y * in_exp_y * (1.0 / x[index+3 * stride]);
    delta_w := temp_w * in_exp_w * (1.0 / x[index+5 * stride]);
    delta_h := temp_h * in_exp_h * (1.0 / x[index+7 * stride]);
    delta_ux := temp_x * (in_exp_x_2 / x[index+1 * stride]-1.0 / (x[index+1 * stride]+sigma_const));
    delta_uy := temp_y * (in_exp_y_2 / x[index+3 * stride]-1.0 / (x[index+3 * stride]+sigma_const));
    delta_uw := temp_w * (in_exp_w_2 / x[index+5 * stride]-1.0 / (x[index+5 * stride]+sigma_const));
    delta_uh := temp_h * (in_exp_h_2 / x[index+7 * stride]-1.0 / (x[index+7 * stride]+sigma_const));
    if iou_loss <> ilMSE then
        begin
            iou := all_ious.giou;
            all_ious.dx_iou := dx_box_iou(pred, truth, iou_loss);
            dx := all_ious.dx_iou.dt;
            dy := all_ious.dx_iou.db;
            dw := all_ious.dx_iou.dl;
            dh := all_ious.dx_iou.dr;
            if yolo_point = ypYOLO_CENTER then

            else
                if yolo_point = ypYOLO_LEFT_TOP then
                    begin
                        dx := dx-dw / 2;
                        dy := dy-dh / 2
                    end
            else
                if yolo_point = ypYOLO_RIGHT_BOTTOM then
                    begin
                        dx := dx+dw / 2;
                        dy := dy+dh / 2
                    end;
            dw := dw * exp(x[index+4 * stride]);
            dh := dh * exp(x[index+6 * stride]);
            delta_x := dx;
            delta_y := dy;
            delta_w := dw;
            delta_h := dh
        end;
    delta_x := delta_x * iou_normalizer;
    delta_y := delta_y * iou_normalizer;
    delta_w := delta_w * iou_normalizer;
    delta_h := delta_h * iou_normalizer;
    delta_ux := delta_ux * uc_normalizer;
    delta_uy := delta_uy * uc_normalizer;
    delta_uw := delta_uw * uc_normalizer;
    delta_uh := delta_uh * uc_normalizer;
    delta_x := fix_nan_inf(delta_x);
    delta_y := fix_nan_inf(delta_y);
    delta_w := fix_nan_inf(delta_w);
    delta_h := fix_nan_inf(delta_h);
    delta_ux := fix_nan_inf(delta_ux);
    delta_uy := fix_nan_inf(delta_uy);
    delta_uw := fix_nan_inf(delta_uw);
    delta_uh := fix_nan_inf(delta_uh);
    if max_delta <> MaxSingle then
        begin
            delta_x := clip_value(delta_x, max_delta);
            delta_y := clip_value(delta_y, max_delta);
            delta_w := clip_value(delta_w, max_delta);
            delta_h := clip_value(delta_h, max_delta);
            delta_ux := clip_value(delta_ux, max_delta);
            delta_uy := clip_value(delta_uy, max_delta);
            delta_uw := clip_value(delta_uw, max_delta);
            delta_uh := clip_value(delta_uh, max_delta)
        end;
    delta[index+0 * stride] := delta[index+0 * stride] + delta_x;
    delta[index+2 * stride] := delta[index+2 * stride] + delta_y;
    delta[index+4 * stride] := delta[index+4 * stride] + delta_w;
    delta[index+6 * stride] := delta[index+6 * stride] + delta_h;
    delta[index+1 * stride] := delta[index+1 * stride] + delta_ux;
    delta[index+3 * stride] := delta[index+3 * stride] + delta_uy;
    delta[index+5 * stride] := delta[index+5 * stride] + delta_uw;
    delta[index+7 * stride] := delta[index+7 * stride] + delta_uh;
    exit(iou)
end;

procedure averages_gaussian_yolo_deltas(const class_index, box_index, stride, classes: longint; const delta: PSingle);
var
    classes_in_one_box: longint;
    c: longint;
begin
    classes_in_one_box := 0;
    for c := 0 to classes -1 do
        if delta[class_index+stride * c] > 0 then
            inc(classes_in_one_box);
    if classes_in_one_box > 0 then
        begin
            delta[box_index+0 * stride] := delta[box_index+0 * stride] / classes_in_one_box;
            delta[box_index+1 * stride] := delta[box_index+1 * stride] / classes_in_one_box;
            delta[box_index+2 * stride] := delta[box_index+2 * stride] / classes_in_one_box;
            delta[box_index+3 * stride] := delta[box_index+3 * stride] / classes_in_one_box;
            delta[box_index+4 * stride] := delta[box_index+4 * stride] / classes_in_one_box;
            delta[box_index+5 * stride] := delta[box_index+5 * stride] / classes_in_one_box;
            delta[box_index+6 * stride] := delta[box_index+6 * stride] / classes_in_one_box;
            delta[box_index+7 * stride] := delta[box_index+7 * stride] / classes_in_one_box
        end
end;

procedure delta_gaussian_yolo_class(const output, delta: PSingle; const index, class_id, classes, stride: longint; const avg_cat: PSingle; const label_smooth_eps: single; const classes_multipliers: PSingle; const cls_normalizer: single);
var
    n: longint;
    y_true: single;
begin
    if delta[index]<>0 then
        begin
            y_true := 1;
            if label_smooth_eps<>0 then
                y_true := y_true * (1-label_smooth_eps)+0.5 * label_smooth_eps;
            delta[index+stride * class_id] := y_true-output[index+stride * class_id];
            if assigned(classes_multipliers) then
                delta[index+stride * class_id] := delta[index+stride * class_id] * classes_multipliers[class_id];
            if assigned(avg_cat) then
                 avg_cat[0] :=  avg_cat[0] + output[index+stride * class_id];
            exit()
        end;
    for n := 0 to classes -1 do
        begin
            y_true := longint(n = class_id);
            if label_smooth_eps<>0 then
                y_true := y_true * (1-label_smooth_eps)+0.5 * label_smooth_eps;
            delta[index+stride * n] := y_true-output[index+stride * n];
            if assigned(classes_multipliers) and (n = class_id) then
                delta[index+stride * class_id] := delta[index+stride * class_id] * (classes_multipliers[class_id] * cls_normalizer);
            if (n = class_id) and assigned(avg_cat) then
                 avg_cat[0] :=  avg_cat[0] + output[index+stride * n]
        end
end;

function compare_gaussian_yolo_class(const output: PSingle; classes, class_index, stride: longint; const objectness: single; const class_id: longint; const conf_thresh: single):longint;
var
    j: longint;
    prob: single;
begin
    for j := 0 to classes -1 do
        begin
            prob := output[class_index+stride * j];
            if prob > conf_thresh then
                exit(1)
        end;
    exit(0)
end;

function entry_gaussian_index(const l: TGaussianYoloLayer; const batch, location, entry: longint):longint;
var
    n: longint;
    loc: longint;
begin
    n := location div (l.w * l.h);
    loc := location mod (l.w * l.h);
    exit(batch * l.outputs+n * l.w * l.h * (8+l.classes+1)+entry * l.w * l.h+loc)
end;

procedure forward_gaussian_yolo_layer(var l: TGaussianYoloLayer; const state: PNetworkState);
var
    i, j, b, t, n, index, count, class_count, class_index, obj_index, box_index, stride,
      best_match_t, best_t, class_id, class_id_match, cl_id, best_n, mask_n: longint;
    avg_iou, recall, recall75, avg_cat, avg_obj, avg_anyobj, best_match_iou,
      best_iou, objectness, iou, iou_multiplier, scale, class_multiplier,
      class_loss, except_uc_loss, loss, uc_loss, iou_loss : single;
    pred, truth, truth_shift: TBox;
    classification_lost, except_uncertainty_lost:TArray<single>;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    move(state.input[0], l.output[0], l.outputs * l.batch * sizeof(single));
{$ifndef GPU}
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            begin
                index := entry_gaussian_index(l, b, n * l.w * l.h, 0);
                activate_array(l.output+index, 2 * l.w * l.h, acLOGISTIC);
                scal_add_cpu(l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output+index, 1);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 2);
                activate_array(l.output+index, 2 * l.w * l.h, acLOGISTIC);
                scal_add_cpu(l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output+index, 1);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 5);
                activate_array(l.output+index, l.w * l.h, acLOGISTIC);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 7);
                activate_array(l.output+index, l.w * l.h, acLOGISTIC);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 8);
                activate_array(l.output+index, (1+l.classes) * l.w * l.h, acLOGISTIC)
            end;
{$endif}
    filldword(l.delta[0], l.outputs * l.batch ,0);
    if not state.train then
        exit();
    avg_iou := 0;
    recall := 0;
    recall75 := 0;
    avg_cat := 0;
    avg_obj := 0;
    avg_anyobj := 0;
    count := 0;
    class_count := 0;
    l.cost[0] := 0;
    for b := 0 to l.batch -1 do
        begin
            for j := 0 to l.h -1 do
                for i := 0 to l.w -1 do
                    for n := 0 to l.n -1 do
                        begin
                            class_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 9);
                            obj_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 8);
                            box_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                            stride := l.w * l.h;
                            pred := get_gaussian_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.yolo_point);
                            best_match_iou := 0;
                            best_match_t := 0;
                            best_iou := 0;
                            best_t := 0;
                            for t := 0 to l.max_boxes -1 do
                                begin
                                    truth := float_to_box_stride(state.truth+t * l.truth_size+b * l.truths, 1);
                                    class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                                    if class_id >= l.classes then
                                        begin
                                            write(format(#10' Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] ',[ class_id, l.classes, l.classes-1]));
                                            writeln(format(' truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d ', [truth.x, truth.y, truth.w, truth.h, class_id]));
                                            continue
                                        end;
                                    if truth.x=0 then
                                        break;
                                    objectness := l.output[obj_index];
                                    class_id_match := compare_gaussian_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25);
                                    iou := box_iou(pred, truth);
                                    if (iou > best_match_iou) and (class_id_match = 1) then
                                        begin
                                            best_match_iou := iou;
                                            best_match_t := t
                                        end;
                                    if iou > best_iou then
                                        begin
                                            best_iou := iou;
                                            best_t := t
                                        end
                                end;
                            avg_anyobj := avg_anyobj + l.output[obj_index];
                            l.delta[obj_index] := l.obj_normalizer * (-l.output[obj_index]);
                            if best_match_iou > l.ignore_thresh then
                                begin
                                    iou_multiplier := best_match_iou * best_match_iou;
                                    if l.objectness_smooth then
                                        begin
                                            l.delta[obj_index] := l.obj_normalizer * (iou_multiplier-l.output[obj_index]);
                                            class_id := trunc(state.truth[best_match_t * l.truth_size+b * l.truths+4]);
                                            if assigned(l.map) then
                                                class_id := l.map[class_id];
                                            delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer)
                                        end
                                    else
                                        l.delta[obj_index] := 0
                                end
                            else
                                if state.net.adversarial then
                                    begin
                                        scale := pred.w * pred.h;
                                        if scale > 0 then
                                            scale := sqrt(scale);
                                        l.delta[obj_index] := scale * l.obj_normalizer * (-l.output[obj_index]);
                                        for cl_id := 0 to l.classes -1 do
                                            if l.output[class_index+stride * cl_id] * l.output[obj_index] > 0.25 then
                                                l.delta[class_index+stride * cl_id] := scale * (-l.output[class_index+stride * cl_id])
                                    end;
                            if best_iou > l.truth_thresh then
                                begin
                                    iou_multiplier := best_iou * best_iou;
                                    if l.objectness_smooth then
                                        l.delta[obj_index] := l.obj_normalizer * (iou_multiplier-l.output[obj_index])
                                    else
                                        l.delta[obj_index] := l.obj_normalizer * (1-l.output[obj_index]);
                                    class_id := trunc(state.truth[best_t * l.truth_size+b * l.truths+4]);
                                    if assigned(l.map) then
                                        class_id := l.map[class_id];
                                    delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, 0, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                                    if assigned(l.classes_multipliers) then
                                        class_multiplier := l.classes_multipliers[class_id]
                                    else
                                        class_multiplier := 1.0;
                                    if l.objectness_smooth then
                                        l.delta[class_index+stride * class_id] := class_multiplier * (iou_multiplier-l.output[class_index+stride * class_id]);
                                    truth := float_to_box_stride(state.truth+best_t * l.truth_size+b * l.truths, 1);
                                    delta_gaussian_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, l.uc_normalizer, 1, l.yolo_point, l.max_delta)
                                end
                        end;
            for t := 0 to l.max_boxes -1 do
                begin
                    truth := float_to_box_stride(state.truth+t * l.truth_size+b * l.truths, 1);
                    if truth.x=0 then
                        break;
                    best_iou := 0;
                    best_n := 0;
                    i := trunc(truth.x * l.w);
                    j := trunc(truth.y * l.h);
                    if l.yolo_point = ypYOLO_CENTER then

                    else
                        if l.yolo_point = ypYOLO_LEFT_TOP then
                            begin
                                i := trunc(min(l.w-1, max(0, ((truth.x-truth.w / 2) * l.w))));
                                j := trunc(min(l.h-1, max(0, ((truth.y-truth.h / 2) * l.h))))
                            end
                    else
                        if l.yolo_point = ypYOLO_RIGHT_BOTTOM then
                            begin
                                i := trunc(min(l.w-1, max(0, ((truth.x+truth.w / 2) * l.w))));
                                j := trunc(min(l.h-1, max(0, ((truth.y+truth.h / 2) * l.h))))
                            end;
                    truth_shift := truth;
                    truth_shift.x := 0;truth_shift.y := 0;
                    for n := 0 to l.total -1 do
                        begin
                            pred := default(TBox);
                            pred.w := l.biases[2 * n] / state.net.w;
                            pred.h := l.biases[2 * n+1] / state.net.h;
                            iou := box_iou(pred, truth_shift);
                            if iou > best_iou then
                                begin
                                    best_iou := iou;
                                    best_n := n
                                end
                        end;
                    mask_n := int_index(l.mask, best_n, l.n);
                    if mask_n >= 0 then
                        begin
                            class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                            if assigned(l.map) then
                                class_id := l.map[class_id];
                            box_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 0);
                            if assigned(l.classes_multipliers) then
                                class_multiplier := l.classes_multipliers[class_id]
                            else
                                class_multiplier := 1.0;
                            iou := delta_gaussian_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, l.uc_normalizer, 1, l.yolo_point, l.max_delta);
                            obj_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 8);
                            avg_obj := avg_obj + l.output[obj_index];
                            l.delta[obj_index] := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                            class_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 9);
                            delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h,  @avg_cat, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                            inc(count);
                            inc(class_count);
                            if iou > 0.5 then
                                recall := recall + 1;
                            if iou > 0.75 then
                                recall75 := recall75 + 1;
                            avg_iou := avg_iou + iou
                        end;
                    for n := 0 to l.total -1 do
                        begin
                            mask_n := int_index(l.mask, n, l.n);
                            if (mask_n >= 0) and (n <> best_n) and (l.iou_thresh < 1.0) then
                                begin
                                    pred := default(TBox);
                                    pred.w := l.biases[2 * n] / state.net.w;
                                    pred.h := l.biases[2 * n+1] / state.net.h;
                                    iou := box_iou_kind(pred, truth_shift, l.iou_thresh_kind);
                                    if iou > l.iou_thresh then
                                        begin
                                            class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                                            if assigned(l.map) then
                                                class_id := l.map[class_id];
                                            box_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 0);
                                            if assigned(l.classes_multipliers) then
                                                class_multiplier := l.classes_multipliers[class_id]
                                            else
                                                class_multiplier := 1.0;
                                            iou := delta_gaussian_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, l.uc_normalizer, 1, l.yolo_point, l.max_delta);
                                            obj_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 8);
                                            avg_obj := avg_obj + l.output[obj_index];
                                            l.delta[obj_index] := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                                            class_index := entry_gaussian_index(l, b, mask_n * l.w * l.h+j * l.w+i, 9);
                                            delta_gaussian_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h,  @avg_cat, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                                            inc(count);
                                            inc(class_count);
                                            if iou > 0.5 then
                                                recall := recall + 1;
                                            if iou > 0.75 then
                                                recall75 := recall75 + 1;
                                            avg_iou := avg_iou + iou
                                        end
                                end
                        end
                end;
            for j := 0 to l.h -1 do
                for i := 0 to l.w -1 do
                    for n := 0 to l.n -1 do
                        begin
                            box_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                            class_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 9);
                            stride := l.w * l.h;
                            averages_gaussian_yolo_deltas(class_index, box_index, stride, l.classes, l.delta)
                        end
        end;
    stride := l.w * l.h;
    setLength(classification_lost,l.batch * l.outputs);
    move(l.delta[0], classification_lost[0], l.batch * l.outputs * sizeof(single));
    for b := 0 to l.batch -1 do
        for j := 0 to l.h -1 do
            for i := 0 to l.w -1 do
                for n := 0 to l.n -1 do
                    begin
                        box_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                        classification_lost[box_index+0 * stride] := 0;
                        classification_lost[box_index+1 * stride] := 0;
                        classification_lost[box_index+2 * stride] := 0;
                        classification_lost[box_index+3 * stride] := 0;
                        classification_lost[box_index+4 * stride] := 0;
                        classification_lost[box_index+5 * stride] := 0;
                        classification_lost[box_index+6 * stride] := 0;
                        classification_lost[box_index+7 * stride] := 0
                    end;
    class_loss := sqr(mag_array(@classification_lost[0], l.outputs * l.batch){, 2});
    //free(classification_lost);
    setLength(except_uncertainty_lost ,l.batch * l.outputs);
    move(l.delta[0], except_uncertainty_lost[0], l.batch * l.outputs * sizeof(single));
    for b := 0 to l.batch -1 do
        for j := 0 to l.h -1 do
            for i := 0 to l.w -1 do
                for n := 0 to l.n -1 do
                    begin
                        box_index := entry_gaussian_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                        except_uncertainty_lost[box_index+4 * stride] := 0;
                        except_uncertainty_lost[box_index+5 * stride] := 0;
                        except_uncertainty_lost[box_index+6 * stride] := 0;
                        except_uncertainty_lost[box_index+7 * stride] := 0
                    end;
    except_uc_loss := sqr(mag_array(@except_uncertainty_lost[0], l.outputs * l.batch){, 2});
    //free(except_uncertainty_lost);
    l.cost[0] := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
    loss := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
    uc_loss := loss-except_uc_loss;
    iou_loss := except_uc_loss-class_loss;
    loss := loss / l.batch;
    class_loss := class_loss / l.batch;
    uc_loss := uc_loss / l.batch;
    iou_loss := iou_loss / l.batch;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

    writeln(ErrOutput, format('Region %d Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, .5R: %f, .75R: %f,  count: %d, class_loss = %.2f, iou_loss = %.2f, uc_loss = %.2f, total_loss = %.2f ', [state.index, avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w * l.h * l.n * l.batch), recall / count, recall75 / count, count, class_loss, iou_loss, uc_loss, loss]))
end;

procedure backward_gaussian_yolo_layer(var l: TGaussianYoloLayer; const state: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, state.delta, 1)
end;

procedure correct_gaussian_yolo_boxes(const dets: Pdetection; const n, w, h, netw, neth: longint; const relative, letter: boolean);
var
    i, new_w, new_h: longint;
    b: TBox;
begin
    new_w := 0;
    new_h := 0;
    if letter then
        begin
            if (single(netw) / w) < (single(neth) / h) then
                begin
                    new_w := netw;
                    new_h := (h * netw) div w
                end
            else
                begin
                    new_h := neth;
                    new_w := (w * neth) div h
                end
        end
    else
        begin
            new_w := netw;
            new_h := neth
        end;
    for i := 0 to n -1 do
        begin
            b := dets[i].bbox;
            b.x := (b.x-(netw-new_w) / 2.0 / netw) / (single(new_w) / netw);
            b.y := (b.y-(neth-new_h) / 2.0 / neth) / (single(new_h) / neth);
            b.w := b.w * (single(netw) / new_w);
            b.h := b.h * (single(neth) / new_h);
            if not relative then
                begin
                    b.x := b.x * w;
                    b.w := b.w * w;
                    b.y := b.y * h;
                    b.h := b.h * h
                end;
            dets[i].bbox := b
        end
end;

function gaussian_yolo_num_detections(const l: TGaussianYoloLayer; const thresh: single):longint;
var
    i, n, count, obj_index:longint;
begin
    count := 0;
    for i := 0 to l.w * l.h -1 do
        for n := 0 to l.n -1 do
            begin
                obj_index := entry_gaussian_index(l, 0, n * l.w * l.h+i, 8);
                if l.output[obj_index] > thresh then
                    inc(count)
            end;
    exit(count)
end;

function get_gaussian_yolo_detections(const l: TGaussianYoloLayer; const w, h,
  netw, neth: longint; const thresh: single; const map: Plongint;
  const relative: boolean; const dets: Pdetection; const letter: boolean
  ): longint;
var
    i, j, n, count, row, col, obj_index, box_index, class_index: longint;
    predictions: PSingle;
    objectness, uc_aver, prob: single;
begin
    predictions := l.output;
    count := 0;
    for i := 0 to l.w * l.h -1 do
        begin
            row := i div l.w;
            col := i mod l.w;
            for n := 0 to l.n -1 do
                begin
                    obj_index := entry_gaussian_index(l, 0, n * l.w * l.h+i, 8);
                    objectness := predictions[obj_index];
                    if objectness <= thresh then
                        continue;
                    if objectness > thresh then
                        begin
                            box_index := entry_gaussian_index(l, 0, n * l.w * l.h+i, 0);
                            dets[count].bbox := get_gaussian_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h, l.yolo_point);
                            dets[count].objectness := objectness;
                            dets[count].classes := l.classes;
                            dets[count].uc[0] := predictions[entry_gaussian_index(l, 0, n * l.w * l.h+i, 1)];
                            dets[count].uc[1] := predictions[entry_gaussian_index(l, 0, n * l.w * l.h+i, 3)];
                            dets[count].uc[2] := predictions[entry_gaussian_index(l, 0, n * l.w * l.h+i, 5)];
                            dets[count].uc[3] := predictions[entry_gaussian_index(l, 0, n * l.w * l.h+i, 7)];
                            dets[count].points := longint(l.yolo_point);
                            for j := 0 to l.classes -1 do
                                begin
                                    class_index := entry_gaussian_index(l, 0, n * l.w * l.h+i, 9+j);
                                    uc_aver := (dets[count].uc[0]+dets[count].uc[1]+dets[count].uc[2]+dets[count].uc[3]) / 4.0;
                                    prob := objectness * predictions[class_index] * (1.0-uc_aver);
                                    if (prob > thresh) then
                                        dets[count].prob[j] := prob
                                    else
                                        dets[count].prob[j] := 0
                                end;
                            inc(count)
                        end
                end
        end;
    correct_gaussian_yolo_boxes(dets, count, w, h, netw, neth, relative, letter);
    exit(count)
end;

{$ifdef GPU}
procedure forward_gaussian_yolo_layer_gpu(var l: TGaussianYoloLayer; const state: PNetworkState);
var
    b: longint;
    n: longint;
    index: longint;
    in_cpu: PSingle;
    truth_cpu: PSingle;
    num_truth: longint;
    cpu_state: TNetworkState;
begin
    copy_ongpu(l.batch * l.inputs, state.input, 1, l.output_gpu, 1);
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            begin
                index := entry_gaussian_index(l, b, n * l.w * l.h, 0);
                activate_array_ongpu(l.output_gpu+index, 2 * l.w * l.h, LOGISTIC);
                scal_add_ongpu(l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output_gpu+index, 1);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 2);
                activate_array_ongpu(l.output_gpu+index, 2 * l.w * l.h, LOGISTIC);
                scal_add_ongpu(l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output_gpu+index, 1);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 5);
                activate_array_ongpu(l.output_gpu+index, l.w * l.h, LOGISTIC);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 7);
                activate_array_ongpu(l.output_gpu+index, l.w * l.h, LOGISTIC);
                index := entry_gaussian_index(l, b, n * l.w * l.h, 8);
                activate_array_ongpu(l.output_gpu+index, (1+l.classes) * l.w * l.h, LOGISTIC)
            end;
    if not state.train or l.onlyforward then
        begin
            cuda_pull_array_async(l.output_gpu, l.output, l.batch * l.outputs);
            CHECK_CUDA(cudaPeekAtLastError());
            exit()
        end;
    in_cpu := PSingle(calloc(l.batch * l.inputs, sizeof(float)));
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    memcpy(in_cpu, l.output, l.batch * l.outputs * sizeof(float));
    truth_cpu := 0;
    if state.truth then
        begin
            num_truth := l.batch * l.truths;
            truth_cpu := PSingle(calloc(num_truth, sizeof(float)));
            cuda_pull_array(state.truth, truth_cpu, num_truth)
        end;
    cpu_state := state;
    cpu_state.net := state.net;
    cpu_state.index := state.index;
    cpu_state.train := state.train;
    cpu_state.truth := truth_cpu;
    cpu_state.input := in_cpu;
    forward_gaussian_yolo_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    free(in_cpu);
    if cpu_state.truth then
        free(cpu_state.truth)
end;

procedure backward_gaussian_yolo_layer_gpu(const l: layer; state: network_state);
begin
    axpy_ongpu(l.batch * l.inputs, l.delta_normalizer, l.delta_gpu, 1, state.delta, 1)
end;
{$endif}
end.

