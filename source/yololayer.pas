unit YoloLayer;

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
  Classes, SysUtils, lightnet, box, blas, Activations;

type
  PYoloLayer = ^TYoloLayer;
  TYoloLayer = TLayer;

  PTrainYoloArgs = ^TTrainYoloArgs;
  TTrainYoloArgs = record
    l : TLayer;
    state : PNetworkState ;
    b : longint;
    tot_iou : single;
    tot_giou_loss : single;
    tot_iou_loss : single;
    count : longint;
    class_count : longint;
  end;


function make_yolo_layer(const batch, w, h, n, total: longint; const mask: TArray<longint>; const classes, max_boxes: longint):TYoloLayer;
procedure resize_yolo_layer(var l: TYoloLayer; const w, h: longint);
function get_yolo_box(const x, biases: PSingle; const n, index, i, j, lw, lh, w, h, stride:longint;const new_coords: boolean):TBox;
function delta_yolo_box(const truth: TBox; const x, biases: Psingle; const n, index, i, j, lw, lh, w, h: longint; const delta: Psingle; const scale: single; const stride: longint; const iou_normalizer: single; const iou_loss: TIOULoss; const accumulate: boolean; const max_delta: single; const rewritten_bbox: TArray<longint>; const new_coords: boolean):TIOUs;
procedure averages_yolo_deltas(const class_index, box_index, stride, classes: longint; const delta: PSingle);
procedure delta_yolo_class(const output: PSingle; const delta: PSingle; const index, class_id, classes, stride: longint; const avg_cat: PSingle; const focal_loss: boolean; const label_smooth_eps: single; const classes_multipliers: PSingle; const cls_normalizer: single);
function compare_yolo_class(const output: Psingle; const classes, class_index, stride: longint; const objectness: single; const class_id: longint; const conf_thresh: single):longint;
function entry_index(const l: TYoloLayer; const batch, location, entry: longint):longint;
//procedure process_batch(ptr: Pointer);
procedure forward_yolo_layer(var l: TYoloLayer; const state: PNetworkState);
procedure backward_yolo_layer(var l: TYoloLayer; const state: PNetworkState);
procedure correct_yolo_boxes(const dets: PDetection; const n, w, h, netw, neth:longint; const relative, letter: boolean);
function yolo_num_detections(const l: PYoloLayer; const thresh: single):longint;
function yolo_num_detections_batch(l: PYoloLayer; const thresh: single; const batch: longint):longint;
procedure avg_flipped_yolo(const l: TYoloLayer);
function get_yolo_detections(const l: PYoloLayer; const w, h, netw, neth: longint; const thresh: single; const map: TIntegers; const relative: boolean; const dets: PDetection; const letter: boolean):longint;
function get_yolo_detections_batch(const l: PYoloLayer; const w, h, netw, neth: longint; const thresh: single; const map: Plongint; const relative: boolean; dets: PDetection; letter: boolean; batch: longint):longint;

{$ifdef GPU}
procedure forward_yolo_layer_gpu(const l: layer; state: network_state);
procedure backward_yolo_layer_gpu(const l: layer; state: network_state);
{$endif}

implementation
uses Math, Steroids;

function ifthen(const cond:boolean; const this, that:string):string;overload;
begin
    if cond then
        result:=this
    else
        result:=that
end;

function make_yolo_layer(const batch, w, h, n, total: longint; const mask: TArray<longint>; const classes, max_boxes: longint):TYoloLayer;
var
    i: longint;
begin
    result := Default(TYoloLayer);
    result.&type := ltYOLO;
    result.n := n;
    result.total := total;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := n * (classes+4+1);
    result.out_w := result.w;
    result.out_h := result.h;
    result.out_c := result.c;
    result.classes := classes;
//    result.cost := TSingles.Create(1);
    result.biases := TSingles.Create(total * 2);
    result.nbiases := total * 2;
    if assigned(mask) then
        result.mask := mask
    else
        begin
            setLength(result.mask, n);
            for i := 0 to n -1 do
                result.mask[i] := i
        end;
    result.bias_updates := TSingles.Create(n * 2);
    result.outputs := h * w * n * (classes+4+1);
    result.inputs := result.outputs;
    result.max_boxes := max_boxes;
    result.truth_size := 4+2;
    result.truths := result.max_boxes * result.truth_size;
    setLength(result.labels, batch * result.w * result.h * result.n);// := longint(xcalloc(batch * result.w * result.h * result.n, sizeof(int)));
    for i := 0 to batch * result.w * result.h * result.n -1 do
        result.labels[i] := -1;
    setLength(result.class_ids, batch * result.w * result.h * result.n) ;//:= longint(xcalloc(batch * result.w * result.h * result.n, sizeof(int)));
    for i := 0 to batch * result.w * result.h * result.n -1 do
        result.class_ids[i] := -1;
    result.delta := TSingles.Create(batch * result.outputs);
    result.output := TSingles.Create(batch * result.outputs);
    for i := 0 to total * 2 -1 do
        result.biases[i] := 0.5;
    result.forward := forward_yolo_layer;
    result.backward := backward_yolo_layer;
{$ifdef GPU}
    result.forward_gpu := forward_yolo_layer_gpu;
    result.backward_gpu := backward_yolo_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.output_avg_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.delta_gpu := cuda_make_array(result.delta, batch * result.outputs);
    free(result.output);
    if cudaSuccess = cudaHostAlloc( and result.output, batch * result.outputs * sizeof(float), cudaHostRegisterMapped) then
        result.output_pinned := 1
    else
        begin
            cudaGetLastError();
            result.output := single(xcalloc(batch * result.outputs, sizeof(float)))
        end;
    free(result.delta);
    if cudaSuccess = cudaHostAlloc( and result.delta, batch * result.outputs * sizeof(float), cudaHostRegisterMapped) then
        result.delta_pinned := 1
    else
        begin
            cudaGetLastError();
            result.delta := single(xcalloc(batch * result.outputs, sizeof(float)))
        end;
{$endif}
    writeln(ErrOutput, 'yolo');
    randomize;
end;

procedure resize_yolo_layer(var l: TYoloLayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.outputs := h * w * l.n * (l.classes+4+1);
    l.inputs := l.outputs;
    if assigned(l.embedding_output) then
        l.embedding_output.reAllocate(l.batch * l.embedding_size * l.n * l.h * l.w);
    if assigned(l.labels) then
        setLength(l.labels, l.batch * l.n * l.h * l.w);
    if assigned(l.class_ids) then
        setLength(l.class_ids, l.batch * l.n * l.h * l.w);
    if not (l.output_pinned<>0) then
        l.output.reAllocate(l.batch * l.outputs);
    if not (l.delta_pinned<>0) then
        l.delta.reAllocate(l.batch * l.outputs);
{$ifdef GPU}
    if assigned(l.output_pinned) then
        begin
            CHECK_CUDA(cudaFreeHost(l.output));
            if cudaSuccess <> cudaHostAlloc( and l.output, l.batch * l.outputs * sizeof(float), cudaHostRegisterMapped) then
                begin
                    cudaGetLastError();
                    l.output := single(xcalloc(l.batch * l.outputs, sizeof(float)));
                    l.output_pinned := 0
                end
        end;
    if l.delta_pinned then
        begin
            CHECK_CUDA(cudaFreeHost(l.delta));
            if cudaSuccess <> cudaHostAlloc( and l.delta, l.batch * l.outputs * sizeof(float), cudaHostRegisterMapped) then
                begin
                    cudaGetLastError();
                    l.delta := single(xcalloc(l.batch * l.outputs, sizeof(float)));
                    l.delta_pinned := 0
                end
        end;
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);
    cuda_free(l.output_avg_gpu);
    l.delta_gpu := cuda_make_array(l.delta, l.batch * l.outputs);
    l.output_gpu := cuda_make_array(l.output, l.batch * l.outputs);
    l.output_avg_gpu := cuda_make_array(l.output, l.batch * l.outputs)
{$endif}
end;

function get_yolo_box(const x, biases: PSingle; const n, index, i, j, lw, lh, w, h, stride:longint;const new_coords: boolean):TBox;
begin
    if new_coords then
        begin
            result.x := (i+x[index+0 * stride]) / lw;
            result.y := (j+x[index+1 * stride]) / lh;
            result.w := x[index+2 * stride] * x[index+2 * stride] * 4 * biases[2 * n] / w;
            result.h := x[index+3 * stride] * x[index+3 * stride] * 4 * biases[2 * n+1] / h
        end
    else
        begin
            result.x := (i+x[index+0 * stride]) / lw;
            result.y := (j+x[index+1 * stride]) / lh;
            result.w := exp(x[index+2 * stride]) * biases[2 * n] / w;
            result.h := exp(x[index+3 * stride]) * biases[2 * n+1] / h
        end;
end;

function fix_nan_inf(const val: single):single;
begin
    if isnan(val) or IsInfinite(val) then
        exit(0);
    exit(val)
end;

function clip_value(const val: single; const max_val: single):single;
begin
    if val > max_val then
        exit( max_val);
    if val < -max_val then
        exit( -max_val);
    exit(val)
end;

function delta_yolo_box(const truth: TBox; const x, biases: Psingle; const n, index, i, j, lw, lh, w, h: longint; const delta: Psingle; const scale: single; const stride: longint; const iou_normalizer: single; const iou_loss: TIOULoss; const accumulate: boolean; const max_delta: single; const rewritten_bbox: TArray<longint>; const new_coords: boolean):TIOUs;
var
    all_ious: TIOUs;
    pred: TBox;
    tx, ty, tw, th, dx, dy, dw, dh: single;
begin
    if (delta[index+0 * stride]<>0) or (delta[index+1 * stride]<>0) or (delta[index+2 * stride]<>0) or (delta[index+3 * stride]<>0) then
        inc(rewritten_bbox[0]);
    all_ious := default(TIOUs);
    pred := get_yolo_box(x, biases, n, index, i, j, lw, lh, w, h, stride, new_coords);
    all_ious.iou := box_iou(pred, truth);
    all_ious.giou := box_giou(pred, truth);
    all_ious.diou := box_diou(pred, truth);
    all_ious.ciou := box_ciou(pred, truth);
    if pred.w = 0 then
        pred.w := 1.0;
    if pred.h = 0 then
        pred.h := 1.0;
    if iou_loss = ilMSE then
        begin
            tx := (truth.x * lw-i);
            ty := (truth.y * lh-j);
            tw := ln(truth.w * w / biases[2 * n]);
            th := ln(truth.h * h / biases[2 * n+1]);
            if new_coords then
                begin
                    tw := sqrt(truth.w * w / (4 * biases[2 * n]));
                    th := sqrt(truth.h * h / (4 * biases[2 * n+1]))
                end;
            delta[index+0 * stride] := delta[index+0 * stride] + (scale * (tx-x[index+0 * stride]) * iou_normalizer);
            delta[index+1 * stride] := delta[index+1 * stride] + (scale * (ty-x[index+1 * stride]) * iou_normalizer);
            delta[index+2 * stride] := delta[index+2 * stride] + (scale * (tw-x[index+2 * stride]) * iou_normalizer);
            delta[index+3 * stride] := delta[index+3 * stride] + (scale * (th-x[index+3 * stride]) * iou_normalizer)
        end
    else
        begin
            all_ious.dx_iou := dx_box_iou(pred, truth, iou_loss);
            dx := all_ious.dx_iou.dt;
            dy := all_ious.dx_iou.db;
            dw := all_ious.dx_iou.dl;
            dh := all_ious.dx_iou.dr;
            if new_coords then

            else
                begin
                    dw := dw * exp(x[index+2 * stride]);
                    dh := dh * exp(x[index+3 * stride])
                end;
            dx := dx * iou_normalizer;
            dy := dy * iou_normalizer;
            dw := dw * iou_normalizer;
            dh := dh * iou_normalizer;
            dx := fix_nan_inf(dx);
            dy := fix_nan_inf(dy);
            dw := fix_nan_inf(dw);
            dh := fix_nan_inf(dh);
            if max_delta <> MaxSingle then
                begin
                    dx := clip_value(dx, max_delta);
                    dy := clip_value(dy, max_delta);
                    dw := clip_value(dw, max_delta);
                    dh := clip_value(dh, max_delta)
                end;
            if not accumulate then
                begin
                    delta[index+0 * stride] := 0;
                    delta[index+1 * stride] := 0;
                    delta[index+2 * stride] := 0;
                    delta[index+3 * stride] := 0
                end;
            delta[index+0 * stride] := delta[index+0 * stride] + dx;
            delta[index+1 * stride] := delta[index+1 * stride] + dy;
            delta[index+2 * stride] := delta[index+2 * stride] + dw;
            delta[index+3 * stride] := delta[index+3 * stride] + dh
        end;
    exit(all_ious)
end;

procedure averages_yolo_deltas(const class_index, box_index, stride, classes: longint; const delta: PSingle);
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
            delta[box_index+3 * stride] := delta[box_index+3 * stride] / classes_in_one_box
        end
end;

procedure delta_yolo_class(const output: PSingle; const delta: PSingle; const index, class_id, classes, stride: longint; const avg_cat: PSingle; const focal_loss: boolean; const label_smooth_eps: single; const classes_multipliers: PSingle; const cls_normalizer: single);
var
    n: longint;
    y_true: single;
    result_delta: single;
    alpha: single;
    ti: longint;
    pt: single;
    grad: single;
begin
    if delta[index+stride * class_id]<>0 then
        begin
            y_true := 1;
            if label_smooth_eps<>0 then
                y_true := y_true * (1-label_smooth_eps)+0.5 * label_smooth_eps;
            result_delta := y_true-output[index+stride * class_id];
            if not isnan(result_delta) and not IsInfinite(result_delta) then
                delta[index+stride * class_id] := result_delta;
            if assigned(classes_multipliers) then
                delta[index+stride * class_id] := delta[index+stride * class_id] * classes_multipliers[class_id];
            if assigned(avg_cat) then
                 avg_cat[0] := avg_cat[0] + output[index+stride * class_id];
            exit()
        end;
    if focal_loss then
        begin
            alpha := 0.5;
            ti := index+stride * class_id;
            pt := output[ti]+0.000000000000001;
            grad := -(1-pt) * (2 * pt * ln(pt)+pt-1);
            for n := 0 to classes -1 do
                begin
                    delta[index+stride * n] := ((ifthen((n = class_id), 1, 0))-output[index+stride * n]);
                    delta[index+stride * n] := delta[index+stride * n] * (alpha * grad);
                    if (n = class_id) and assigned(avg_cat) then
                        avg_cat[0] := avg_cat[0] + output[index+stride * n]
                end
        end
    else
        for n := 0 to classes -1 do
            begin
                y_true := (ifthen((n = class_id), 1, 0));
                if label_smooth_eps<>0 then
                    y_true := y_true * (1-label_smooth_eps)+0.5 * label_smooth_eps;
                result_delta := y_true-output[index+stride * n];
                if not isnan(result_delta) and not IsInfinite(result_delta) then
                    delta[index+stride * n] := result_delta;
                if assigned(classes_multipliers) and (n = class_id) then
                    delta[index+stride * class_id] := delta[index+stride * class_id] * (classes_multipliers[class_id] * cls_normalizer);
                if (n = class_id) and assigned(avg_cat) then
                    avg_cat[0] := avg_cat[0] + output[index+stride * n]
            end
end;

function compare_yolo_class(const output: Psingle; const classes, class_index, stride: longint; const objectness: single; const class_id: longint; const conf_thresh: single):longint;
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

function entry_index(const l: TYoloLayer; const batch, location, entry: longint):longint;
var
    n, loc: longint;
begin
    n := location div (l.w * l.h);
    loc := location mod (l.w * l.h);
    exit(batch * l.outputs+n * l.w * l.h * (4+l.classes+1)+entry * l.w * l.h+loc)
end;

procedure process_batch(ptr: Pointer);
var
    l: TYoloLayer;
    state: PNetworkState;
    found_object:boolean;
    b, i, j, t, n, class_index, obj_index, box_index, stride, best_match_t, best_t, class_id: longint;
    class_id_match, cl_id, best_n, mask_n, truth_in_index, track_id, truth_out_index: longint;
    tot_giou,tot_diou,tot_ciou,tot_diou_loss,tot_ciou_loss,recall,recall75,avg_cat,avg_obj,
      avg_anyobj: single;
    best_match_iou, best_iou, objectness, iou, delta_obj, scale, iou_multiplier, class_multiplier: single;
    buff: string;
    truth, pred, truth_shift: TBox;
    all_ious: TIOUs;
    args : PTrainYoloArgs absolute ptr;
begin
    l := args.l;
    state := args.state;
    b := args.b;
    tot_giou := 0;
    tot_diou := 0;
    tot_ciou := 0;
    tot_diou_loss := 0;
    tot_ciou_loss := 0;
    recall := 0;
    recall75 := 0;
    avg_cat := 0;
    avg_obj := 0;
    avg_anyobj := 0;
    for j := 0 to l.h -1 do
        for i := 0 to l.w -1 do
            for n := 0 to l.n -1 do
                begin
                    class_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 4+1);
                    obj_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 4);
                    box_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                    stride := l.w * l.h;
                    pred := get_yolo_box(l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.w * l.h, l.new_coords);
                    best_match_iou := 0;
                    best_match_t := 0;
                    best_iou := 0;
                    best_t := 0;
                    for t := 0 to l.max_boxes -1 do
                        begin
                            truth := float_to_box_stride(state.truth+t * l.truth_size+b * l.truths, 1);
                            if truth.x=0 then
                                break;
                            class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                            if (class_id >= l.classes) or (class_id < 0) then
                                begin
                                    writeln(format(#10' Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] ', [class_id, l.classes, l.classes-1]));
                                    writeln(format(#10' truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f, class_id = %d ', [truth.x, truth.y, truth.w, truth.h, class_id]));
                                    continue
                                end;
                            objectness := l.output[obj_index];
                            if isnan(objectness) or IsInfinite(objectness) then
                                l.output[obj_index] := 0;
                            class_id_match := compare_yolo_class(l.output, l.classes, class_index, l.w * l.h, objectness, class_id, 0.25);
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
                            if l.objectness_smooth then
                                begin
                                    delta_obj := l.obj_normalizer * (best_match_iou-l.output[obj_index]);
                                    if delta_obj > l.delta[obj_index] then
                                        l.delta[obj_index] := delta_obj
                                end
                            else
                                l.delta[obj_index] := 0
                        end
                    else
                        if state.net.adversarial then
                            begin
                                stride := l.w * l.h;
                                scale := pred.w * pred.h;
                                if scale > 0 then
                                    scale := sqrt(scale);
                                l.delta[obj_index] := scale * l.obj_normalizer * (-l.output[obj_index]);
                                found_object := false;
                                for cl_id := 0 to l.classes -1 do
                                    if l.output[class_index+stride * cl_id] * l.output[obj_index] > 0.25 then
                                        begin
                                            l.delta[class_index+stride * cl_id] := scale * (-l.output[class_index+stride * cl_id]);
                                            found_object := true
                                        end;
                                if found_object then
                                    begin
                                        for cl_id := 0 to l.classes -1 do
                                            if l.output[class_index+stride * cl_id] * l.output[obj_index] < 0.25 then
                                                l.delta[class_index+stride * cl_id] := scale * (1-l.output[class_index+stride * cl_id]);
                                        l.delta[box_index+0 * stride] := l.delta[box_index+0 * stride] + (scale * (-l.output[box_index+0 * stride]));
                                        l.delta[box_index+1 * stride] := l.delta[box_index+1 * stride] + (scale * (-l.output[box_index+1 * stride]));
                                        l.delta[box_index+2 * stride] := l.delta[box_index+2 * stride] + (scale * (-l.output[box_index+2 * stride]));
                                        l.delta[box_index+3 * stride] := l.delta[box_index+3 * stride] + (scale * (-l.output[box_index+3 * stride]))
                                    end
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
                            delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h, nil, l.focal_loss, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                            class_multiplier := ifthen(assigned(l.classes_multipliers), l.classes_multipliers[class_id], 1.0);
                            if l.objectness_smooth then
                                l.delta[class_index+stride * class_id] := class_multiplier * (iou_multiplier-l.output[class_index+stride * class_id]);
                            truth := float_to_box_stride(state.truth+best_t * l.truth_size+b * l.truths, 1);
                            delta_yolo_box(truth, l.output, l.biases, l.mask[n], box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, true, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                            inc(state.net.total_bbox[0])
                        end
                end;
    for t := 0 to l.max_boxes -1 do
        begin
            truth := float_to_box_stride(state.truth+t * l.truth_size+b * l.truths, 1);
            if truth.x=0 then
                break;
            if (truth.x < 0) or (truth.y < 0) or (truth.x > 1) or (truth.y > 1) or (truth.w < 0) or (truth.h < 0) then
                begin
                    writeln(format(' Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f ',[ truth.x, truth.y, truth.w, truth.h]));
                    //sprintf(buff, 'echo "Wrong label: truth.x = %f, truth.y = %f, truth.w = %f, truth.h = %f" >> bad_label.list', truth.x, truth.y, truth.w, truth.h);
                    //system(buff)
                end;
            class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
            if (class_id >= l.classes) or (class_id < 0) then
                continue;
            best_iou := 0;
            best_n := 0;
            i := trunc(truth.x * l.w);
            j := trunc(truth.y * l.h);
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
                    box_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 0);
                    if assigned(l.classes_multipliers) then
                        class_multiplier := l.classes_multipliers[class_id]
                    else
                        class_multiplier := 1.0;
                    all_ious := delta_yolo_box(truth, l.output, l.biases, best_n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, true, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                    inc(state.net.total_bbox[0]);
                    truth_in_index := t * l.truth_size+b * l.truths+5;
                    track_id := trunc(state.truth[truth_in_index]);
                    truth_out_index := b * l.n * l.w * l.h+mask_n * l.w * l.h+j * l.w+i;
                    l.labels[truth_out_index] := track_id;
                    l.class_ids[truth_out_index] := class_id;
                    args.tot_iou := args.tot_iou + all_ious.iou;
                    args.tot_iou_loss := args.tot_iou_loss + (1-all_ious.iou);
                    tot_giou := tot_giou + all_ious.giou;
                    args.tot_giou_loss := args.tot_giou_loss + (1-all_ious.giou);
                    tot_diou := tot_diou + all_ious.diou;
                    tot_diou_loss := tot_diou_loss + (1-all_ious.diou);
                    tot_ciou := tot_ciou + all_ious.ciou;
                    tot_ciou_loss := tot_ciou_loss + (1-all_ious.ciou);
                    obj_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 4);
                    avg_obj := avg_obj + l.output[obj_index];
                    if l.objectness_smooth then
                        begin
                            delta_obj := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                            if l.delta[obj_index] = 0 then
                                l.delta[obj_index] := delta_obj
                        end
                    else
                        l.delta[obj_index] := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                    class_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 4+1);
                    delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h,  @avg_cat, l.focal_loss, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                    inc(args.count);
                    inc(args.class_count);
                    if all_ious.iou > 0.5 then
                        recall := recall + 1;
                    if all_ious.iou > 0.75 then
                        recall75 := recall75 + 1
                end;
            for n := 0 to l.total -1 do
                begin
                    mask_n := int_index(l.mask, n, l.n);
                    if (mask_n >= 0) and (n <> best_n) and (l.iou_thresh < 1.0) then
                        begin
                            pred := Default(TBox);
                            pred.w := l.biases[2 * n] / state.net.w;
                            pred.h := l.biases[2 * n+1] / state.net.h;
                            iou := box_iou_kind(pred, truth_shift, l.iou_thresh_kind);
                            if iou > l.iou_thresh then
                                begin
                                    class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                                    if assigned(l.map) then
                                        class_id := l.map[class_id];
                                    box_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 0);
                                    if assigned(l.classes_multipliers) then
                                        class_multiplier := l.classes_multipliers[class_id]
                                    else
                                        class_multiplier := 1.0;
                                    all_ious := delta_yolo_box(truth, l.output, l.biases, n, box_index, i, j, l.w, l.h, state.net.w, state.net.h, l.delta, (2-truth.w * truth.h), l.w * l.h, l.iou_normalizer * class_multiplier, l.iou_loss, true, l.max_delta, state.net.rewritten_bbox, l.new_coords);
                                    inc(state.net.total_bbox[0]);
                                    args.tot_iou := args.tot_iou + all_ious.iou;
                                    args.tot_iou_loss := args.tot_iou_loss + (1-all_ious.iou);
                                    tot_giou := tot_giou + all_ious.giou;
                                    args.tot_giou_loss := args.tot_giou_loss + (1-all_ious.giou);
                                    tot_diou := tot_diou + all_ious.diou;
                                    tot_diou_loss := tot_diou_loss + (1-all_ious.diou);
                                    tot_ciou := tot_ciou + all_ious.ciou;
                                    tot_ciou_loss := tot_ciou_loss + (1-all_ious.ciou);
                                    obj_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 4);
                                    avg_obj := avg_obj + l.output[obj_index];
                                    if l.objectness_smooth then
                                        begin
                                            delta_obj := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                                            if l.delta[obj_index] = 0 then
                                                l.delta[obj_index] := delta_obj
                                        end
                                    else
                                        l.delta[obj_index] := class_multiplier * l.obj_normalizer * (1-l.output[obj_index]);
                                    class_index := entry_index(l, b, mask_n * l.w * l.h+j * l.w+i, 4+1);
                                    delta_yolo_class(l.output, l.delta, class_index, class_id, l.classes, l.w * l.h,  @avg_cat, l.focal_loss, l.label_smooth_eps, @l.classes_multipliers[0], l.cls_normalizer);
                                    inc(args.count);
                                    inc(args.class_count);
                                    if all_ious.iou > 0.5 then
                                        recall := recall + 1;
                                    if all_ious.iou > 0.75 then
                                        recall75 := recall75 + 1
                                end
                        end
                end
        end;
    if l.iou_thresh < 1.0 then
        for j := 0 to l.h -1 do
            for i := 0 to l.w -1 do
                for n := 0 to l.n -1 do
                    begin
                        obj_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 4);
                        box_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 0);
                        class_index := entry_index(l, b, n * l.w * l.h+j * l.w+i, 4+1);
                        stride := l.w * l.h;
                        if l.delta[obj_index] <> 0 then
                            averages_yolo_deltas(class_index, box_index, stride, l.classes, l.delta)
                    end;
end;

procedure forward_yolo_layer(var l: TYoloLayer; const state: PNetworkState);
var
    b, n, bbox_index, obj_index, i: longint;
    tot_iou, tot_giou, tot_diou, tot_ciou, tot_iou_loss, tot_giou_loss, tot_diou_loss, tot_ciou_loss: single;
    recall, recall75, avg_cat, avg_obj, avg_anyobj, iteration_num, start_point, count, class_count, counter, counter_all, counter_reject: longint;
    yolo_args: TArray<TTrainYoloArgs>;
    threads :TArray<TThread>;
    progress_it, progress, ep_loss_threshold, cur_max, cur_avg, rolling_std, rolling_max, rolling_avg: single;
    progress_badlabels, cur_std, final_badlebels_threshold, badlabels_threshold, num_deltas_per_anchor: single;
    cur_percent, loss: single;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    move(state.input[0], l.output[0], l.outputs * l.batch * sizeof(single));
{$ifndef GPU}
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            begin
                bbox_index := entry_index(l, b, n * l.w * l.h, 0);
                if l.new_coords then

                else
                    begin
                        activate_array(l.output+bbox_index, 2 * l.w * l.h, acLOGISTIC);
                        obj_index := entry_index(l, b, n * l.w * l.h, 4);
                        activate_array(l.output+obj_index, (1+l.classes) * l.w * l.h, acLOGISTIC)
                    end;
                scal_add_cpu(2 * l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output+bbox_index, 1)
            end;
{$endif}
    filldword(l.delta[0], l.outputs * l.batch ,0);
    if not state.train then begin
        {$ifdef USE_TELEMETRY}
        if benchmark then metrics.forward.finish(l.&type);
        {$endif}

        exit();
    end;
    for i := 0 to l.batch * l.w * l.h * l.n -1 do
        l.labels[i] := -1;
    for i := 0 to l.batch * l.w * l.h * l.n -1 do
        l.class_ids[i] := -1;
    tot_iou := 0;
    tot_giou := 0;
    tot_diou := 0;
    tot_ciou := 0;
    tot_iou_loss := 0;
    tot_giou_loss := 0;
    tot_diou_loss := 0;
    tot_ciou_loss := 0;
    recall := 0;
    recall75 := 0;
    avg_cat := 0;
    avg_obj := 0;
    avg_anyobj := 0;
    count := 0;
    class_count := 0;
    l.cost := 0;
    setLength(threads, l.batch);
    setLength(yolo_args, l.batch);
    for b := 0 to l.batch -1 do
        begin
            yolo_args[b].l := l;
            yolo_args[b].state := state;
            yolo_args[b].b := b;
            yolo_args[b].tot_iou := 0;
            yolo_args[b].tot_iou_loss := 0;
            yolo_args[b].tot_giou_loss := 0;
            yolo_args[b].count := 0;
            yolo_args[b].class_count := 0;
            threads[b] := ExecuteInThread(process_batch, @yolo_args[b]);
            //if pthread_create( and threads[b], 0, process_batch,  and (yolo_args[b])) then
                //error('Thread creation failed', DARKNET_LOC)
        end;
    for b := 0 to l.batch -1 do
        begin
            //pthread_join(threads[b], 0);
            threads[b].WaitFor;
            tot_iou := tot_iou + yolo_args[b].tot_iou;
            tot_iou_loss := tot_iou_loss + yolo_args[b].tot_iou_loss;
            tot_giou_loss := tot_giou_loss + yolo_args[b].tot_giou_loss;
            count := count + yolo_args[b].count;
            class_count := class_count + yolo_args[b].class_count
        end;
    //free(yolo_args);
    //free(threads);
    iteration_num := state.net.cur_iteration[0];//get_current_iteration(state.net);
    start_point := state.net.max_batches * 3 div 4;
    if ((state.net.badlabels_rejection_percentage<>0) and (start_point < iteration_num)) or ((state.net.num_sigmas_reject_badlabels<>0) and (start_point < iteration_num)) or ((state.net.equidistant_point<>0) and (state.net.equidistant_point < iteration_num)) then
        begin
            progress_it := iteration_num-state.net.equidistant_point;
            progress := progress_it / (state.net.max_batches-state.net.equidistant_point);
            ep_loss_threshold := (state.net.delta_rolling_avg[0]) * progress * 1.4;
            cur_max := 0;
            cur_avg := 0;
            counter := 0;
            for i := 0 to l.batch * l.outputs -1 do
                if l.delta[i] <> 0 then
                    begin
                        inc(counter);
                        cur_avg := cur_avg + abs(l.delta[i]);
                        if cur_max < abs(l.delta[i]) then
                            cur_max := abs(l.delta[i])
                    end;
            cur_avg := cur_avg / counter;
            if state.net.delta_rolling_max[0] = 0 then
                state.net.delta_rolling_max[0] := cur_max;
            state.net.delta_rolling_max[0] := state.net.delta_rolling_max[0] * 0.99 + cur_max * 0.01;
            state.net.delta_rolling_avg[0] := state.net.delta_rolling_avg[0] * 0.99 + cur_avg * 0.01;
            if (state.net.num_sigmas_reject_badlabels<>0) and (start_point < iteration_num) then
                begin
                    rolling_std := state.net.delta_rolling_std[0];
                    rolling_max := state.net.delta_rolling_max[0];
                    rolling_avg := state.net.delta_rolling_avg[0];
                    progress_badlabels := (iteration_num-start_point) / (start_point);
                    cur_std := 0;
                    counter := 0;
                    for i := 0 to l.batch * l.outputs -1 do
                        if l.delta[i] <> 0 then
                            begin
                                inc(counter);
                                cur_std := cur_std + sqr(l.delta[i]-rolling_avg{, 2})
                            end;
                    cur_std := sqrt(cur_std / counter);
                    state.net.delta_rolling_std[0] := state.net.delta_rolling_std[0] * 0.99+cur_std * 0.01;
                    final_badlebels_threshold := rolling_avg+rolling_std * state.net.num_sigmas_reject_badlabels;
                    badlabels_threshold := rolling_max-progress_badlabels * abs(rolling_max-final_badlebels_threshold);
                    badlabels_threshold := lightnet.max(final_badlebels_threshold, badlabels_threshold);
                    for i := 0 to l.batch * l.outputs -1 do
                        if abs(l.delta[i]) > badlabels_threshold then
                            l.delta[i] := 0;
                    writeln(format(' rolling_std = %f, rolling_max = %f, rolling_avg = %f ', [rolling_std, rolling_max, rolling_avg]));
                    writeln(format(' badlabels loss_threshold = %f, start_it = %d, progress = %f ', [badlabels_threshold, start_point, progress_badlabels * 100]));
                    ep_loss_threshold := lightnet.min(final_badlebels_threshold, rolling_avg) * progress
                end;
            if (state.net.badlabels_rejection_percentage<>0) and (start_point < iteration_num) then
                begin
                    if state.net.badlabels_reject_threshold[0] = 0 then
                        state.net.badlabels_reject_threshold[0] := state.net.delta_rolling_max[0];
                    writeln(' badlabels_reject_threshold = %f ', state.net.badlabels_reject_threshold[0]);
                    num_deltas_per_anchor := (l.classes+4+1);
                    counter_reject := 0;
                    counter_all := 0;
                    for i := 0 to l.batch * l.outputs -1 do
                        if l.delta[i] <> 0 then
                            begin
                                inc(counter_all);
                                if abs(l.delta[i]) > state.net.badlabels_reject_threshold[0] then
                                    begin
                                        inc(counter_reject);
                                        l.delta[i] := 0
                                    end
                            end;
                    cur_percent := 100 * (counter_reject * num_deltas_per_anchor / counter_all);
                    if cur_percent > state.net.badlabels_rejection_percentage then
                        begin
                            state.net.badlabels_reject_threshold[0] := state.net.badlabels_reject_threshold[0] + 0.01;
                            writeln(' increase!!! ')
                        end
                    else
                        if  state.net.badlabels_reject_threshold[0] > 0.01 then
                            begin
                                state.net.badlabels_reject_threshold[0] := state.net.badlabels_reject_threshold[0] - 0.01;
                                writeln(' decrease!!! ')
                            end;
                    writeln(format(' badlabels_reject_threshold = %f, cur_percent = %f, badlabels_rejection_percentage = %f, delta_rolling_max = %f ', [state.net.badlabels_reject_threshold[0], cur_percent, state.net.badlabels_rejection_percentage, state.net.delta_rolling_max[0]]))
                end;
            if (state.net.equidistant_point<>0) and (state.net.equidistant_point < iteration_num) then
                begin
                    writeln(format(' equidistant_point loss_threshold = %f, start_it = %d, progress = %3.1f %% ', [ep_loss_threshold, state.net.equidistant_point, progress * 100]));
                    for i := 0 to l.batch * l.outputs -1 do
                        if abs(l.delta[i]) < ep_loss_threshold then
                            l.delta[i] := 0
                end
        end;
    if count = 0 then
        count := 1;
    if class_count = 0 then
        class_count := 1;
    if l.show_details = 0 then
        begin
            loss := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
            l.cost := loss;
            loss := loss / l.batch;
            writeln(ErrOutput, format('v3 (%s loss, Normalizer: (iou: %.2f, obj: %.2f, cls: %.2f) Region %d Avg (IOU: %f), count: %d, total_loss = %f ', [(ifthen(l.iou_loss = ilMSE, 'mse', (ifthen(l.iou_loss = ilGIOU, 'giou', 'iou')))), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, state.index, tot_iou / count, count, loss]))
        end;

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_yolo_layer(var l: TYoloLayer; const state: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, state.delta, 1)
end;

procedure correct_yolo_boxes(const dets: PDetection; const n, w, h, netw,
  neth: longint; const relative, letter: boolean);
var
    i, new_w, new_h: longint;
    deltaw, deltah, ratiow, ratioh: single;
    b: TBox;
begin
    new_w := 0;
    new_h := 0;
    if letter then
        begin
            if (netw / w) < (neth / h) then
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
    deltaw := netw-new_w;
    deltah := neth-new_h;
    ratiow := new_w / netw;
    ratioh := new_h / neth;
    for i := 0 to n -1 do
        begin
            b := dets[i].bbox;
            b.x := (b.x-deltaw / 2.0 / netw) / ratiow;
            b.y := (b.y-deltah / 2.0 / neth) / ratioh;
            b.w := b.w * (1 / ratiow);
            b.h := b.h * (1 / ratioh);
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

function yolo_num_detections(const l: PYoloLayer; const thresh: single
  ): longint;
var
    i, n, count, obj_index: longint;
begin
    count := 0;
    for n := 0 to l.n -1 do
        for i := 0 to l.w * l.h -1 do
            begin
                obj_index := entry_index(l^, 0, n * l.w * l.h+i, 4);
                if //not isnan(l.output[obj_index]) and
                   (l.output[obj_index] > thresh) then
                    inc(count)
            end;
    exit(count)
end;

function yolo_num_detections_batch(l: PYoloLayer; const thresh: single;
  const batch: longint): longint;
var
    i, n, count, obj_index: longint;
begin
    count := 0;
    for i := 0 to l.w * l.h -1 do
        for n := 0 to l.n -1 do
            begin
                obj_index := entry_index(l^, batch, n * l.w * l.h+i, 4);
                if //not isnan(l.output[obj_index]) and
                   (l.output[obj_index] > thresh) then
                    inc(count)
            end;
    exit(count)
end;

procedure avg_flipped_yolo(const l: TYoloLayer);
var
    i, j, n, z, i1, i2: longint;
    flip: PSingle;
    swap: single;
begin
    flip := l.output + l.outputs;
    for j := 0 to l.h -1 do
        for i := 0 to l.w div 2 -1 do
            for n := 0 to l.n -1 do
                for z := 0 to l.classes+4+1 -1 do
                    begin
                        i1 := z * l.w * l.h * l.n+n * l.w * l.h+j * l.w+i;
                        i2 := z * l.w * l.h * l.n+n * l.w * l.h+j * l.w+(l.w-i-1);
                        swap := flip[i1];
                        flip[i1] := flip[i2];
                        flip[i2] := swap;
                        if z = 0 then
                            begin
                                flip[i1] := -flip[i1];
                                flip[i2] := -flip[i2]
                            end
                    end;
    for i := 0 to l.outputs -1 do
        l.output[i] := (l.output[i]+flip[i]) / 2.0
end;

function get_yolo_detections(const l: PYoloLayer; const w, h, netw,
  neth: longint; const thresh: single; const map: TIntegers;
  const relative: boolean; const dets: PDetection; const letter: boolean
  ): longint;
var
    predictions: PSingle;
    i, j, n, count, row,  col, obj_index: longint;
    box_index, class_index: longint;
    objectness, prob: single;
begin
    predictions := l.output;
    count := 0;
    for i := 0 to l.w * l.h -1 do
        begin
            row := i div l.w;
            col := i mod l.w;
            for n := 0 to l.n -1 do
                begin
                    obj_index := entry_index(l^, 0, n * l.w * l.h+i, 4);
                    objectness := predictions[obj_index];
                    if //not isnan(objectness) and
                       (objectness > thresh) then
                        begin
                            box_index := entry_index(l^, 0, n * l.w * l.h+i, 0);
                            dets[count].bbox := get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h, l.new_coords);
                            dets[count].objectness := objectness;
                            dets[count].classes := l.classes;
                            if assigned(l.embedding_output) then
                                get_embedding(l.embedding_output, l.w, l.h, l.n * l.embedding_size, l.embedding_size, col, row, n, 0, @dets[count].embeddings[0]);
                            for j := 0 to l.classes -1 do
                                begin
                                    class_index := entry_index(l^, 0, n * l.w * l.h+i, 4+1+j);
                                    prob := objectness * predictions[class_index];
                                    if (prob > thresh) then
                                        dets[count].prob[j] := prob
                                    else
                                        dets[count].prob[j] := 0
                                end;
                            inc(count)
                        end
                end
        end;
    correct_yolo_boxes(@dets[0], count, w, h, netw, neth, relative, letter);
    exit(count)
end;

function get_yolo_detections_batch(const l: PYoloLayer; const w, h, netw,
  neth: longint; const thresh: single; const map: Plongint;
  const relative: boolean; dets: PDetection; letter: boolean; batch: longint
  ): longint;
var
    i: longint;
    j: longint;
    n: longint;
    predictions: PSingle;
    count: longint;
    row: longint;
    col: longint;
    obj_index: longint;
    objectness: single;
    box_index: longint;
    class_index: longint;
    prob: single;
begin
    predictions := l.output;
    count := 0;
    for i := 0 to l.w * l.h -1 do
        begin
            row := i div l.w;
            col := i mod l.w;
            for n := 0 to l.n -1 do
                begin
                    obj_index := entry_index(l^, batch, n * l.w * l.h+i, 4);
                    objectness := predictions[obj_index];
                    if //not isnan(objectness) and
                       (objectness > thresh) then
                        begin
                            box_index := entry_index(l^, batch, n * l.w * l.h+i, 0);
                            dets[count].bbox := get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row, l.w, l.h, netw, neth, l.w * l.h, l.new_coords);
                            dets[count].objectness := objectness;
                            dets[count].classes := l.classes;
                            if assigned(l.embedding_output) then
                                get_embedding(l.embedding_output, l.w, l.h, l.n * l.embedding_size, l.embedding_size, col, row, n, batch, @dets[count].embeddings[0]);
                            for j := 0 to l.classes -1 do
                                begin
                                    class_index := entry_index(l^, batch, n * l.w * l.h+i, 4+1+j);
                                    prob := objectness * predictions[class_index];
                                    if (prob > thresh) then
                                        dets[count].prob[j] := prob
                                    else
                                        dets[count].prob[j] := 0
                                end;
                            &inc(count)
                        end
                end
        end;
    correct_yolo_boxes(@dets[0], count, w, h, netw, neth, relative, letter);
    exit(count)
end;

{$ifdef GPU}
procedure forward_yolo_layer_gpu(const l: layer; state: network_state);
var
    le: layer;
    b: longint;
    n: longint;
    bbox_index: longint;
    obj_index: longint;
    in_cpu: PSingle;
    truth_cpu: PSingle;
    num_truth: longint;
    cpu_state: network_state;
begin
    if l.embedding_output then
        begin
            le := state.net.layers[l.embedding_layer_id];
            cuda_pull_array_async(le.output_gpu, l.embedding_output, le.batch * le.outputs)
        end;
    simple_copy_ongpu(l.batch * l.inputs, state.input, l.output_gpu);
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            begin
                bbox_index := entry_index(l, b, n * l.w * l.h, 0);
                if l.new_coords then

                else
                    begin
                        activate_array_ongpu(l.output_gpu+bbox_index, 2 * l.w * l.h, LOGISTIC);
                        obj_index := entry_index(l, b, n * l.w * l.h, 4);
                        activate_array_ongpu(l.output_gpu+obj_index, (1+l.classes) * l.w * l.h, LOGISTIC)
                    end;
                if l.scale_x_y <> 1 then
                    scal_add_ongpu(2 * l.w * l.h, l.scale_x_y, -0.5 * (l.scale_x_y-1), l.output_gpu+bbox_index, 1)
            end;
    if not state.train or l.onlyforward then
        begin
            if l.mean_alpha and l.output_avg_gpu then
                mean_array_gpu(l.output_gpu, l.batch * l.outputs, l.mean_alpha, l.output_avg_gpu);
            cuda_pull_array_async(l.output_gpu, l.output, l.batch * l.outputs);
            CHECK_CUDA(cudaPeekAtLastError());
            exit()
        end;
    in_cpu := single(xcalloc(l.batch * l.inputs, sizeof(float)));
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    memcpy(in_cpu, l.output, l.batch * l.outputs * sizeof(float));
    truth_cpu := 0;
    if state.truth then
        begin
            num_truth := l.batch * l.truths;
            truth_cpu := single(xcalloc(num_truth, sizeof(float)));
            cuda_pull_array(state.truth, truth_cpu, num_truth)
        end;
    cpu_state := state;
    cpu_state.net := state.net;
    cpu_state.index := state.index;
    cpu_state.train := state.train;
    cpu_state.truth := truth_cpu;
    cpu_state.input := in_cpu;
    forward_yolo_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    free(in_cpu);
    if cpu_state.truth then
        free(cpu_state.truth)
end;

procedure backward_yolo_layer_gpu(const l: layer; state: network_state);
begin
    axpy_ongpu(l.batch * l.inputs, state.net.loss_scale * l.delta_normalizer, l.delta_gpu, 1, state.delta, 1)
end;
{$endif}
end.

