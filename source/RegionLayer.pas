unit RegionLayer;

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
  SysUtils, lightnet, activations, SoftmaxLayer, box, blas, tree;

type
  PRegionLayer = ^TRegionLayer;
  TRegionLayer = TLayer;
(*function make_region_layer(const batch, w, h, n, classes, coords: longint):TRegionLayer;
procedure resize_region_layer(const l: PRegionlayer; const w, h: longint);
function get_region_box(const x, biases: TSingles; const n, index, i, j, w, h, stride: longint):TBox;
function delta_region_box(const truth: TBox; const x, biases: TSingles; const n, index, i, j, w, h: longint; const delta: TSingles; const scale: single; const stride: longint):single;
procedure delta_region_mask(const truth, x: TSingles; const n, index: longint; const delta: TSingles; const stride, scale: longint);
procedure delta_region_class(const output, delta: TSingles; const index, &class, classes: longint; const hier: PTree; const scale: single; const stride: longint; const avg_cat: TSingles; const tag: boolean);
function logit(const x: single):single;
function tisnan(const x: single):single;
function entry_index(l: TRegionLayer; batch: longint; location: longint; entry: longint):longint;
procedure forward_region_layer(const l: TRegionLayer; net: TNetwork);
procedure backward_region_layer(const l: TRegionLayer; net: TNetwork);
procedure correct_region_boxes(dets: Pdetection; n: longint; w: longint; h: longint; netw: longint; neth: longint; relative: longint);
procedure get_region_detections(l: TRegionLayer; w: longint; h: longint; netw: longint; neth: longint; thresh: single; map: TIntegers; tree_thresh: single; relative: longint; dets: Pdetection);
{$ifdef GPU}
procedure forward_region_layer_gpu(const l: TRegionLayer; net: TNetwork);
procedure backward_region_layer_gpu(const l: TRegionLayer; net: TNetwork);
{$endif}
procedure zero_objectness(l: TRegionLayer);
*)

function make_region_layer(const batch, w, h, n, classes, coords, max_boxes: longint):TRegionLayer;
procedure resize_region_layer(var l: TRegionLayer; const w, h: longint);
function get_region_box(const x, biases: Psingle; const n, index, i, j, w, h: longint):TBox;
function delta_region_box(const truth: TBox; const x, biases: Psingle; const n, index, i, j, w, h: longint; const delta: Psingle; const scale: single):single;
procedure delta_region_class(const output, delta: Psingle; const index:longint; class_id:longint; const classes: longint; const hier: PTree; const scale: single; const avg_cat: Psingle; const focal_loss: boolean);
function logit(const x: single):single;
function tisnan(const x: single):boolean;
procedure forward_region_layer(var l: TRegionLayer; const state: PNetworkState);
procedure backward_region_layer(var l: TRegionLayer; const state: PNetworkState);
procedure get_region_boxes(const l: PRegionLayer; const w, h: longint; const thresh: single; const probs: TArray<TArray<Single>>; const boxes: TArray<TBox>; const only_objectness: boolean; const map: Plongint);
{$ifdef GPU}

procedure forward_region_layer_gpu(const l: region_layer; state: network_state);
procedure backward_region_layer_gpu(l: region_layer; state: network_state);
{$endif}
procedure correct_region_boxes(const dets: PDetection; const n, w, h, netw, neth: longint; const relative: boolean);
procedure get_region_detections(const l: TRegionLayer; const w, h, netw, neth: longint; const thresh: single; const map: Plongint; tree_thresh: single; const relative: boolean; const dets: Pdetection);
procedure zero_objectness(const l: TRegionLayer);

implementation
uses math;
const DOABS = true;// todo check why

function make_region_layer(const batch, w, h, n, classes, coords, max_boxes: longint):TRegionLayer;
var
    i: longint;
begin
    result := default(TRegionLayer);
    result.&type := ltREGION;
    result.n := n;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := n * (classes+coords+1);
    result.out_w := result.w;
    result.out_h := result.h;
    result.out_c := result.c;
    result.classes := classes;
    result.coords := coords;
    result.cost := [0];//TSingles.Create(1);
    result.biases := TSingles.Create(n * 2);
    result.bias_updates := TSingles.Create(n * 2);
    result.outputs := h * w * n * (classes+coords+1);
    result.inputs := result.outputs;
    result.max_boxes := max_boxes;
    result.truth_size := 4+2;
    result.truths := max_boxes * result.truth_size;
    result.delta := TSingles.Create(batch * result.outputs);
    result.output := TSingles.Create(batch * result.outputs);
    for i := 0 to n * 2 -1 do
        result.biases[i] := 0.5;
    result.forward := forward_region_layer;
    result.backward := backward_region_layer;
{$ifdef GPU}
    result.forward_gpu := forward_region_layer_gpu;
    result.backward_gpu := backward_region_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.delta_gpu := cuda_make_array(result.delta, batch * result.outputs);
{$endif}
    writeln(ErrOutput, 'detection');
    randomize
end;

procedure resize_region_layer(var l: TRegionLayer; const w, h: longint);
var
    old_w, old_h: longint;
begin
{$ifdef GPU}
    old_w := l.w;
    old_h := l.h;
{$endif}
    l.w := w;
    l.h := h;
    l.outputs := h * w * l.n * (l.classes+l.coords+1);
    l.inputs := l.outputs;
    l.output.reAllocate(l.batch * l.outputs);
    l.delta.reAllocate(l.batch * l.outputs);
{$ifdef GPU}
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);

    l.delta_gpu = cuda_make_array(l.delta, l.batch*l.outputs);
    l.output_gpu = cuda_make_array(l.output, l.batch*l.outputs);
{$endif}
end;

function get_region_box(const x, biases: Psingle; const n, index, i, j, w, h: longint):TBox;
begin
    result.x := (i+logistic_activate(x[index+0])) / w;
    result.y := (j+logistic_activate(x[index+1])) / h;
    result.w := exp(x[index+2]) * biases[2 * n];
    result.h := exp(x[index+3]) * biases[2 * n+1];
    if DOABS then
        begin
            result.w := exp(x[index+2]) * biases[2 * n] / w;
            result.h := exp(x[index+3]) * biases[2 * n+1] / h
        end;
end;

function delta_region_box(const truth: TBox; const x, biases: Psingle; const n, index, i, j, w, h: longint; const delta: Psingle; const scale: single):single;
var
    pred: TBox;
    iou, tx, ty, tw, th: single;
begin
    pred := get_region_box(x, biases, n, index, i, j, w, h);
    iou := box_iou(pred, truth);
    tx := (truth.x * w-i);
    ty := (truth.y * h-j);
    tw := ln(truth.w / biases[2 * n]);
    th := ln(truth.h / biases[2 * n+1]);
    if DOABS then
        begin
            tw := ln(truth.w * w / biases[2 * n]);
            th := ln(truth.h * h / biases[2 * n+1])
        end;
    delta[index+0] := scale * (tx-logistic_activate(x[index+0])) * logistic_gradient(logistic_activate(x[index+0]));
    delta[index+1] := scale * (ty-logistic_activate(x[index+1])) * logistic_gradient(logistic_activate(x[index+1]));
    delta[index+2] := scale * (tw-x[index+2]);
    delta[index+3] := scale * (th-x[index+3]);
    exit(iou)
end;

procedure delta_region_class(const output, delta: Psingle; const index:longint; class_id:longint; const classes: longint; const hier: PTree; const scale: single; const avg_cat: Psingle; const focal_loss: boolean);
var
    i, n, g, offset, ti: longint;
    pred, alpha, pt, grad: single;
begin
    if assigned(hier) then
        begin
            pred := 1;
            while (class_id >= 0) do
                begin
                    pred := pred * output[index+class_id];
                    g := hier.group[class_id];
                    offset := hier.group_offset[g];
                    for i := 0 to hier.group_size[g] -1 do
                        delta[index+offset+i] := scale * (-output[index+offset+i]);
                    delta[index+class_id] := scale * (1-output[index+class_id]);
                    class_id := hier.parent[class_id]
                end;
            avg_cat[0] := avg_cat[0] + pred
        end
    else
        begin
            if focal_loss then
                begin
                    alpha := 0.5;
                    ti := index+class_id;
                    pt := output[ti]+0.000000000000001;
                    grad := -(1-pt) * (2 * pt * ln(pt)+pt-1);
                    for n := 0 to classes -1 do
                        begin
                            delta[index+n] := scale * ((ifthen((n = class_id), 1, 0))-output[index+n]);
                            delta[index+n] := delta[index+n] * (alpha * grad);
                            if n = class_id then
                                avg_cat[0] := avg_cat[0] + output[index+n]
                        end
                end
            else
                for n := 0 to classes -1 do
                    begin
                        delta[index+n] := scale * ((ifthen((n = class_id), 1, 0))-output[index+n]);
                        if n = class_id then
                            avg_cat[0] := avg_cat[0] + output[index+n]
                    end
        end
end;

function logit(const x: single):single;
begin
    exit(ln(x / (1.0-x)))
end;

function tisnan(const x: single):boolean;
begin
    exit(IsNan(x))
end;

function entry_index(const l: TRegionLayer; const batch, location, entry: longint):longint;
var
    n, loc: longint;
begin
    n := location div (l.w * l.h);
    loc := location mod (l.w * l.h);
    exit(batch * l.outputs+n * l.w * l.h * (l.coords+l.classes+1)+entry * l.w * l.h+loc)
end;

procedure forward_region_layer(var l: TRegionLayer; const state: PNetworkState);
var
    i, j, b, t, n, size, index, count, class_count, class_id, onlyclass_id, maxi, best_class_id, best_index, best_n: longint;
    avg_iou, recall, avg_cat, avg_obj, avg_anyobj, maxp, scale, p, best_iou, iou: single;
    truth, pred, truth_shift: TBox;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    size := l.coords+l.classes+1;
    move(state.input[0], l.output[0], l.outputs * l.batch * sizeof(Single));
  {$ifndef GPU}
    flatten(l.output, l.w * l.h, size * l.n, l.batch, true);
  {$endif}
    for b := 0 to l.batch -1 do
        for i := 0 to l.h * l.w * l.n -1 do
            begin
                index := size * i+b * l.outputs;
                l.output[index+4] := logistic_activate(l.output[index+4])
            end;
  {$ifndef GPU}
    if assigned(l.softmax_tree) then
        for b := 0 to l.batch -1 do
            for i := 0 to l.h * l.w * l.n -1 do
                begin
                    index := size * i+b * l.outputs;
                    softmax_tree(l.output+index+5, 1, 0, 1, @l.softmax_tree[0], l.output+index+5)
                end
    else
        if l.softmax then
            for b := 0 to l.batch -1 do
                for i := 0 to l.h * l.w * l.n -1 do
                    begin
                        index := size * i+b * l.outputs;
                        softmax(l.output+index+5, l.classes, 1, l.output+index+5, 1)
                    end;
  {$endif}
    if not state.train then
        exit();
    filldword(l.delta[0], l.outputs * l.batch ,0);//;
    avg_iou := 0;
    recall := 0;
    avg_cat := 0;
    avg_obj := 0;
    avg_anyobj := 0;
    count := 0;
    class_count := 0;
    l.cost[0] := 0;
    for b := 0 to l.batch -1 do  begin
            if assigned(l.softmax_tree) then begin
                onlyclass_id := 0;
                for t := 0 to l.max_boxes -1 do begin
                    truth := float_to_box(state.truth+t * l.truth_size+b * l.truths);
                    if truth.x=0 then
                        break;
                    class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                    maxp := 0;
                    maxi := 0;
                    if (truth.x > 100000) and (truth.y > 100000) then begin
                        for n := 0 to l.n * l.w * l.h -1 do begin
                            index := size * n+b * l.outputs+5;
                            scale := l.output[index-1];
                            p := scale * get_hierarchy_probability(l.output+index, l.softmax_tree[0], class_id);
                            if p > maxp then begin
                                maxp := p;
                                maxi := n
                            end
                        end;
                        index := size * maxi+b * l.outputs+5;
                        delta_region_class(l.output, l.delta, index, class_id, l.classes, @l.softmax_tree[0], l.class_scale, @avg_cat, l.focal_loss);
                        inc(class_count);
                        onlyclass_id := 1;
                        break
                    end
                end;
                if onlyclass_id<>0 then
                    continue
            end;
            for j := 0 to l.h -1 do
                for i := 0 to l.w -1 do
                    for n := 0 to l.n -1 do
                        begin
                            index := size * (j * l.w * l.n+i * l.n+n)+b * l.outputs;
                            pred := get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                            best_iou := 0;
                            best_class_id := -1;
                            for t := 0 to l.max_boxes -1 do
                                begin
                                    truth := float_to_box(state.truth+t * l.truth_size+b * l.truths);
                                    class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                                    if class_id >= l.classes then
                                        continue;
                                    if truth.x=0 then
                                        break;
                                    iou := box_iou(pred, truth);
                                    if iou > best_iou then
                                        begin
                                            best_class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                                            best_iou := iou
                                        end
                                end;
                            avg_anyobj := avg_anyobj + l.output[index+4];
                            l.delta[index+4] := l.noobject_scale * ((-l.output[index+4]) * logistic_gradient(l.output[index+4]));
                            if l.classfix = -1 then
                                l.delta[index+4] := l.noobject_scale * ((best_iou-l.output[index+4]) * logistic_gradient(l.output[index+4]))
                            else
                                if best_iou > l.thresh then
                                    begin
                                        l.delta[index+4] := 0;
                                        if l.classfix > 0 then
                                            begin
                                                delta_region_class(l.output, l.delta, index+5, best_class_id, l.classes, @l.softmax_tree[0], l.class_scale * (ifthen(l.classfix = 2, l.output[index+4], 1)),  @avg_cat, l.focal_loss);
                                                inc(class_count)
                                            end
                                    end;
                            if  state.net.seen[0] < 12800 then
                                begin
                                    truth := default(TBox);
                                    truth.x := (i+0.5) / l.w;
                                    truth.y := (j+0.5) / l.h;
                                    truth.w := l.biases[2 * n];
                                    truth.h := l.biases[2 * n+1];
                                    if DOABS then
                                        begin
                                            truth.w := l.biases[2 * n] / l.w;
                                            truth.h := l.biases[2 * n+1] / l.h
                                        end;
                                    delta_region_box(truth, l.output, l.biases, n, index, i, j, l.w, l.h, l.delta, 0.01)
                                end
                        end;
            for t := 0 to l.max_boxes -1 do
                begin
                    truth := float_to_box(state.truth+t * l.truth_size+b * l.truths);
                    class_id := trunc(state.truth[t * l.truth_size+b * l.truths+4]);
                    if class_id >= l.classes then
                        begin
                            writeln(format(#10' Warning: in txt-labels class_id=%d >= classes=%d in cfg-file. In txt-labels class_id should be [from 0 to %d] ',[ class_id, l.classes, l.classes-1]));
                            continue
                        end;
                    if truth.x=0 then
                        break;
                    best_iou := 0;
                    best_index := 0;
                    best_n := 0;
                    i := trunc(truth.x * l.w);
                    j := trunc(truth.y * l.h);
                    truth_shift := truth;
                    truth_shift.x := 0;
                    truth_shift.y := 0;
                    for n := 0 to l.n -1 do
                        begin
                            index := size * (j * l.w * l.n+i * l.n+n)+b * l.outputs;
                            pred := get_region_box(l.output, l.biases, n, index, i, j, l.w, l.h);
                            if l.bias_match then
                                begin
                                    pred.w := l.biases[2 * n];
                                    pred.h := l.biases[2 * n+1];
                                    if DOABS then
                                        begin
                                            pred.w := l.biases[2 * n] / l.w;
                                            pred.h := l.biases[2 * n+1] / l.h
                                        end
                                end;
                            pred.x := 0;
                            pred.y := 0;
                            iou := box_iou(pred, truth_shift);
                            if iou > best_iou then
                                begin
                                    best_index := index;
                                    best_iou := iou;
                                    best_n := n
                                end
                        end;
                    iou := delta_region_box(truth, l.output, l.biases, best_n, best_index, i, j, l.w, l.h, l.delta, l.coord_scale);
                    if iou > 0.5 then
                        recall := recall + 1;
                    avg_iou := avg_iou + iou;
                    avg_obj := avg_obj + l.output[best_index+4];
                    l.delta[best_index+4] := l.object_scale * (1-l.output[best_index+4]) * logistic_gradient(l.output[best_index+4]);
                    if l.rescore then
                        l.delta[best_index+4] := l.object_scale * (iou-l.output[best_index+4]) * logistic_gradient(l.output[best_index+4]);
                    if assigned(l.map) then
                        class_id := l.map[class_id];
                    delta_region_class(l.output, l.delta, best_index+5, class_id, l.classes, @l.softmax_tree[0], l.class_scale,  @avg_cat, l.focal_loss);
                    inc(count);
                    inc(class_count)
                end
        end;
    {$ifndef GPU}
    flatten(l.delta, l.w * l.h, size * l.n, l.batch, false);
    {$endif}
    l.cost[0] := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

    writeln(format('Region Avg IOU: %f, Class: %f, Obj: %f, No Obj: %f, Avg Recall: %f,  count: %d', [avg_iou / count, avg_cat / class_count, avg_obj / count, avg_anyobj / (l.w * l.h * l.n * l.batch), recall / count, count]))
end;

procedure backward_region_layer(var l: TRegionLayer; const state: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, state.delta, 1)
end;

procedure get_region_boxes(const l: PRegionLayer; const w, h: longint;
  const thresh: single; const probs: TArray<TArray<Single>>; const boxes: TArray
  <TBox>; const only_objectness: boolean; const map: Plongint);
var
    i, j, n, row, col, index, p_index, box_index, class_index: longint;
    found :boolean;
    predictions: Psingle;
    scale, prob: single;
begin
    predictions := l.output;
    // todo [get_region_boxes] Parallelize
    for i := 0 to l.w * l.h -1 do
        begin
            row := i div l.w;
            col := i mod l.w;
            for n := 0 to l.n -1 do
                begin
                    index := i * l.n+n;
                    p_index := index * (l.classes+5)+4;
                    scale := predictions[p_index];
                    if (l.classfix = -1) and (scale < 0.5) then
                        scale := 0;
                    box_index := index * (l.classes+5);
                    boxes[index] := get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
                    boxes[index].x := boxes[index].x * w;
                    boxes[index].y := boxes[index].y * h;
                    boxes[index].w := boxes[index].w * w;
                    boxes[index].h := boxes[index].h * h;
                    class_index := index * (l.classes+5)+5;
                    if assigned(l.softmax_tree) then
                        begin
                            hierarchy_predictions(predictions+class_index, l.classes, l.softmax_tree[0], false);
                            found := false;
                            if assigned(map) then
                                for j := 0 to 200 -1 do
                                    begin
                                        prob := scale * predictions[class_index+map[j]];
                                        if (prob > thresh) then
                                            probs[index][j] := prob
                                        else
                                            probs[index][j] := 0
                                    end
                            else
                                j := l.classes-1;
                                while j >= 0 do begin
                                    if not found and (predictions[class_index+j] > 0.5) then
                                        found := true
                                    else
                                        predictions[class_index+j] := 0;
                                    prob := predictions[class_index+j];
                                    if (scale > thresh) then
                                        probs[index][j] := prob
                                    else
                                        probs[index][j] := 0;
                                    dec(j)
                                end
                        end
                    else
                        for j := 0 to l.classes -1 do
                            begin
                                prob := scale * predictions[class_index+j];
                                if (prob > thresh) then
                                    probs[index][j] := prob
                                else
                                    probs[index][j] := 0
                            end;
                    if only_objectness then
                        probs[index][0] := scale
                end
        end
end;

{$ifdef GPU}

procedure forward_region_layer_gpu(const l: region_layer; state: network_state);
var
    i: longint;
    count: longint;
    group_size: longint;
    truth_cpu: PSingle;
    num_truth: longint;
    cpu_state: network_state;
begin
    flatten_ongpu(state.input, l.h * l.w, l.n * (l.coords+l.classes+1), l.batch, 1, l.output_gpu);
    if l.softmax_tree then
        begin
            count := 5;
            for i := 0 to l.softmax_tree.groups -1 do
                begin
                    group_size := l.softmax_tree.group_size[i];
                    softmax_gpu(l.output_gpu+count, group_size, l.classes+5, l.w * l.h * l.n * l.batch, 1, l.output_gpu+count);
                    count := count + group_size
                end
        end
    else
        if l.softmax then
            softmax_gpu(l.output_gpu+5, l.classes, l.classes+5, l.w * l.h * l.n * l.batch, 1, l.output_gpu+5);
    float * in_cpu := single(xcalloc(l.batch * l.inputs, sizeof(float)));
    truth_cpu := 0;
    if state.truth then
        begin
            num_truth := l.batch * l.truths;
            truth_cpu := single(xcalloc(num_truth, sizeof(float)));
            cuda_pull_array(state.truth, truth_cpu, num_truth)
        end;
    cuda_pull_array(l.output_gpu, in_cpu, l.batch * l.inputs);
    cpu_state := state;
    cpu_state.train := state.train;
    cpu_state.truth := truth_cpu;
    cpu_state.input := in_cpu;
    forward_region_layer(l, cpu_state);
    free(cpu_state.input);
    if not state.train then
        exit();
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    if cpu_state.truth then
        free(cpu_state.truth)
end;

procedure backward_region_layer_gpu(l: region_layer; state: network_state);
begin
    flatten_ongpu(l.delta_gpu, l.h * l.w, l.n * (l.coords+l.classes+1), l.batch, 0, state.delta)
end;

{$endif}

procedure correct_region_boxes(const dets: PDetection; const n, w, h, netw, neth: longint; const relative: boolean);
var
    i, new_w, new_h: longint;
    b: TBox;
begin
    new_w := 0;
    new_h := 0;
    if (netw / w) < (neth / h) then
        begin
            new_w := netw;
            new_h := (h * netw) div w
        end
    else
        begin
            new_h := neth;
            new_w := (w * neth) div h
        end;
    for i := 0 to n -1 do
        begin
            b := dets[i].bbox;
            b.x := (b.x-(netw-new_w) / 2.0 / netw) / (new_w / netw);
            b.y := (b.y-(neth-new_h) / 2.0 / neth) / (new_h / neth);
            b.w := b.w * (netw / new_w);
            b.h := b.h * (neth / new_h);
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

procedure get_region_detections(const l: TRegionLayer; const w, h, netw, neth: longint; const thresh: single; const map: Plongint; tree_thresh: single; const relative: boolean; const dets: Pdetection);
var
    predictions, flip: Psingle;
    i, j, n, z, i1, i2, row, col, index, obj_index, box_index, mask_index, class_index: longint;
    swap, scale, prob: single;
begin
    predictions := l.output;
    if (l.batch = 2) then
        begin
            flip := l.output+l.outputs;
            for j := 0 to l.h -1 do
                for i := 0 to l.w div 2 -1 do
                    for n := 0 to l.n -1 do
                        for z := 0 to l.classes+l.coords+1 -1 do
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
    for i := 0 to l.w * l.h -1 do
        begin
            row := i div l.w;
            col := i mod l.w;
            for n := 0 to l.n -1 do
                begin
                    index := n * l.w * l.h+i;
                    for j := 0 to l.classes -1 do
                        dets[index].prob[j] := 0;
                    obj_index := entry_index(l, 0, n * l.w * l.h+i, l.coords);
                    box_index := entry_index(l, 0, n * l.w * l.h+i, 0);
                    mask_index := entry_index(l, 0, n * l.w * l.h+i, 4);
                    if l.background<>0 then
                        scale := 1
                    else
                        scale := predictions[obj_index];
                    dets[index].bbox := get_region_box(predictions, l.biases, n, box_index, col, row, l.w, l.h);
                    if scale > thresh then
                        dets[index].objectness := scale
                    else
                        dets[index].objectness := 0;
                    if assigned(dets[index].mask) then
                        for j := 0 to l.coords-4 -1 do
                            dets[index].mask[j] := l.output[mask_index+j * l.w * l.h];
                    class_index := entry_index(l, 0, n * l.w * l.h+i, not l.background);
                    if assigned(l.softmax_tree) then
                        begin
                            hierarchy_predictions(predictions+class_index, l.classes, l.softmax_tree[0], false);
                            if assigned(map) then
                                for j := 0 to 200 -1 do
                                    begin
                                        class_index := entry_index(l, 0, n * l.w * l.h+i, l.coords+1+map[j]);
                                        prob := scale * predictions[class_index];
                                        if (prob > thresh) then
                                            dets[index].prob[j] := prob
                                        else
                                            dets[index].prob[j] := 0
                                    end
                            else
                                begin
                                    j := hierarchy_top_prediction(predictions+class_index, l.softmax_tree[0], tree_thresh, l.w * l.h);
                                    if (scale > thresh) then
                                        dets[index].prob[j] := scale
                                    else
                                        dets[index].prob[j] := 0
                                end
                        end
                    else
                        if dets[index].objectness<>0 then
                            for j := 0 to l.classes -1 do
                                begin
                                    class_index := entry_index(l, 0, n * l.w * l.h+i, l.coords+1+j);
                                    prob := scale * predictions[class_index];
                                    if (prob > thresh) then
                                        dets[index].prob[j] := prob
                                    else
                                        dets[index].prob[j] := 0
                                end
                end
        end;
    correct_region_boxes(dets, l.w * l.h * l.n, w, h, netw, neth, relative)
end;

procedure zero_objectness(const l: TRegionLayer);
var
    i, n, obj_index: longint;
begin
    for i := 0 to l.w * l.h -1 do
        for n := 0 to l.n -1 do
            begin
                obj_index := entry_index(l, 0, n * l.w * l.h+i, l.coords);
                l.output[obj_index] := 0
            end
end;

end.
