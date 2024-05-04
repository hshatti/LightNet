unit DetectionLayer;

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
  SysUtils, math, darknet, box, blas;

type TDetectionLayer = TLayer;

function make_detection_layer(const batch, inputs, n, side, classes, coords:longint; const rescore: boolean):TDetectionLayer;
procedure forward_detection_layer(var l: TDetectionLayer; const state: PNetworkState);
procedure backward_detection_layer(var l: TDetectionLayer; const state: PNetworkState);
procedure get_detection_detections(const l: TDetectionLayer; const w, h: longint; const thresh: single; const dets: PDetection);
procedure get_detection_boxes(const l: TDetectionLayer; const w, h: longint; const thresh: single; const probs: PPsingle; const boxes: PBox; const only_objectness: boolean);
{$ifdef GPU}
procedure forward_detection_layer_gpu(const l: TDetectionLayer; net: TNetwork);
procedure backward_detection_layer_gpu(l: TDetectionLayer; net: TNetwork);
{$endif}

implementation

function make_detection_layer(const batch, inputs, n, side, classes, coords:longint; const rescore: boolean):TDetectionLayer;
begin
    result := default(TDetectionLayer);
    result.&type := ltDETECTION;
    result.n := n;
    result.batch := batch;
    result.inputs := inputs;
    result.classes := classes;
    result.coords := coords;
    result.rescore := rescore;
    result.side := side;
    result.w := side;
    result.h := side;
    assert(side * side * ((1+result.coords) * result.n+result.classes) = inputs);
    result.cost := TSingles.Create(1);
    result.outputs := result.inputs;
    result.truths := result.side * result.side * (1+result.coords+result.classes);
    result.output := TSingles.Create(batch * result.outputs);
    result.delta := TSingles.Create(batch * result.outputs);
    result.forward := forward_detection_layer;
    result.backward := backward_detection_layer;
{$ifdef GPU}
    result.forward_gpu := forward_detection_layer_gpu;
    result.backward_gpu := backward_detection_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.delta_gpu := cuda_make_array(result.delta, batch * result.outputs);
{$endif}
    writeln(ErrOutput, 'Detection Layer');
    //RandSeed:=0; // note why? RandSeed is set to  0
    Randomize;
    //srand(0);
    //exit(l)
end;

procedure forward_detection_layer(var l: TDetectionLayer;
  const state: PNetworkState);

var
    locations, i, j, b, index, offset, count, size, truth_index,
    p_index, best_index, class_index, box_index, tbox_index: longint;
    is_obj: boolean;
    indexes : array[0..99] of longint;

    avg_iou, avg_cat, avg_allcat, avg_obj, avg_anyobj,
    best_iou, best_rmse, iou, rmse, cutoff: single;

    costs : TSingles;

    truth, _out: TBox;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    locations := l.side * l.side;
    move(state.input[0], l.output[0], l.outputs * l.batch * sizeof(single));
    //memcpy(l.output, net.input, l.outputs * l.batch * sizeof(float));
    if l.softmax then
        for b := 0 to l.batch -1 do
            begin
                index := b * l.inputs;
                for i := 0 to locations -1 do
                    begin
                        offset := i * l.classes;
                        softmax(@l.output[index+offset], l.classes, 1, 1, @l.output[index+offset])
                    end
            end;
    if state.train then
        begin
            avg_iou := 0;
            avg_cat := 0;
            avg_allcat := 0;
            avg_obj := 0;
            avg_anyobj := 0;
            count := 0;
            l.cost[0] := 0;
            size := l.inputs * l.batch;
            FillDWord(l.delta[0], size, 0);
            for b := 0 to l.batch -1 do
                begin
                    index := b * l.inputs;
                    for i := 0 to locations -1 do
                        begin
                            truth_index := (b * locations+i) * (1+l.coords+l.classes);
                            // todo [forward_detection_layer] maybe truncate float then cast to boolean?
                            is_obj := state.truth[truth_index]<>0;
                            for j := 0 to l.n -1 do
                                begin
                                    p_index := index+locations * l.classes+i * l.n+j;
                                    l.delta[p_index] := l.noobject_scale * (-l.output[p_index]);
                                    l.cost[0] :=  l.cost[0] + (l.noobject_scale * sqr(l.output[p_index]{, 2}));
                                    avg_anyobj := avg_anyobj + l.output[p_index]
                                end;
                            best_index := -1;
                            best_iou := 0;
                            best_rmse := 20;
                            if not is_obj then
                                continue;
                            class_index := index+i * l.classes;
                            for j := 0 to l.classes -1 do
                                begin
                                    l.delta[class_index+j] := l.class_scale * (state.truth[truth_index+1+j]-l.output[class_index+j]);
                                    l.cost[0] := l.cost[0] + (l.class_scale * sqr(state.truth[truth_index+1+j]-l.output[class_index+j]{, 2}));
                                    if state.truth[truth_index+1+j]<>0 then
                                        avg_cat := avg_cat + l.output[class_index+j];
                                    avg_allcat := avg_allcat + l.output[class_index+j]
                                end;
                            truth := float_to_box(state.truth+truth_index+1+l.classes);
                            truth.x := truth.x / l.side;
                            truth.y := truth.y / l.side;
                            for j := 0 to l.n -1 do
                                begin
                                    box_index := index+locations * (l.classes+l.n)+(i * l.n+j) * l.coords;
                                    _out := float_to_box(l.output+box_index);
                                    _out.x := _out.x / l.side;
                                    _out.y := _out.y / l.side;
                                    if l.sqrt then
                                        begin
                                            _out.w := _out.w * _out.w;
                                            _out.h := _out.h * _out.h
                                        end;
                                    iou := box_iou(_out, truth);
                                    rmse := box_rmse(_out, truth);
                                    if (best_iou > 0) or (iou > 0) then
                                        if iou > best_iou then
                                            begin
                                                best_iou := iou;
                                                best_index := j
                                            end
                                    else
                                        if rmse < best_rmse then
                                            begin
                                                best_rmse := rmse;
                                                best_index := j
                                            end
                                end;
                            if l.forced then
                                begin
                                    if truth.w * truth.h < 0.1 then
                                        best_index := 1
                                    else
                                        best_index := 0
                                end;
                            if l.random and ( state.net.seen[0] < 64000) then
                                best_index := random(l.n);
                            box_index := index+locations * (l.classes+l.n)+(i * l.n+best_index) * l.coords;
                            tbox_index := truth_index+1+l.classes;
                            _out := float_to_box(@l.output[box_index]);
                            _out.x := _out.x / l.side;
                            _out.y := _out.y / l.side;
                            if l.sqrt then
                                begin
                                    _out.w := _out.w * _out.w;
                                    _out.h := _out.h * _out.h
                                end;
                            iou := box_iou(_out, truth);
                            p_index := index+locations * l.classes+i * l.n+best_index;
                            l.cost[0] :=  l.cost[0] - (l.noobject_scale * sqr(l.output[p_index]{, 2}));
                            l.cost[0] :=  l.cost[0] + (l.object_scale * sqr(1-l.output[p_index]{, 2}));
                            avg_obj := avg_obj + l.output[p_index];
                            l.delta[p_index] := l.object_scale * (1-l.output[p_index]);
                            if l.rescore then
                                l.delta[p_index] := l.object_scale * (iou-l.output[p_index]);
                            l.delta[box_index+0] := l.coord_scale * (state.truth[tbox_index+0]-l.output[box_index+0]);
                            l.delta[box_index+1] := l.coord_scale * (state.truth[tbox_index+1]-l.output[box_index+1]);
                            l.delta[box_index+2] := l.coord_scale * (state.truth[tbox_index+2]-l.output[box_index+2]);
                            l.delta[box_index+3] := l.coord_scale * (state.truth[tbox_index+3]-l.output[box_index+3]);
                            if l.sqrt then
                                begin
                                    l.delta[box_index+2] := l.coord_scale * (sqrt(state.truth[tbox_index+2])-l.output[box_index+2]);
                                    l.delta[box_index+3] := l.coord_scale * (sqrt(state.truth[tbox_index+3])-l.output[box_index+3])
                                end;
                            l.cost[0] :=  l.cost[0] + sqr(1-iou{, 2});
                            avg_iou := avg_iou + iou;
                            inc(count)
                        end
                end;
            if false then
                begin
                    costs := TSingles.Create(l.batch * locations * l.n);
                    for b := 0 to l.batch -1 do
                        begin
                            index := b * l.inputs;
                            for i := 0 to locations -1 do
                                for j := 0 to l.n -1 do
                                    begin
                                        p_index := index+locations * l.classes+i * l.n+j;
                                        costs[b * locations * l.n+i * l.n+j] := l.delta[p_index] * l.delta[p_index]
                                    end
                        end;
                    top_k(costs, l.batch * locations * l.n, 100, @indexes[0]);
                    cutoff := costs[indexes[99]];
                    for b := 0 to l.batch -1 do
                        begin
                            index := b * l.inputs;
                            for i := 0 to locations -1 do
                                for j := 0 to l.n -1 do
                                    begin
                                        p_index := index+locations * l.classes+i * l.n+j;
                                        if l.delta[p_index] * l.delta[p_index] < cutoff then
                                            l.delta[p_index] := 0
                                    end
                        end;
                    costs.free
                end;
            l.cost[0] := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
            writeln(format('Detection Avg IOU: %f, Pos Cat: %f, All Cat: %f, Pos Obj: %f, Any Obj: %f, count: %d', [avg_iou / count, avg_cat / count, avg_allcat / (count * l.classes), avg_obj / count, avg_anyobj / (l.batch * locations * l.n), count]))
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_detection_layer(var l: TDetectionLayer;
  const state: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, state.delta, 1)
end;

procedure get_detection_detections(const l: TDetectionLayer; const w,
  h: longint; const thresh: single; const dets: PDetection);
var
    i, j, n, row, col, index, p_index, box_index, class_index: longint;
    scale, prob: single;
    b: TBox;
    predictions : TSingles;
begin
    predictions := l.output;
    for i := 0 to l.side * l.side -1 do
        begin
            row := i div l.side;
            col := i mod l.side;
            for n := 0 to l.n -1 do
                begin
                    index := i * l.n+n;
                    p_index := l.side * l.side * l.classes+i * l.n+n;
                    scale := predictions[p_index];
                    box_index := l.side * l.side * (l.classes+l.n)+(i * l.n+n) * 4;
                    b.x := (predictions[box_index+0]+col) / l.side * w;
                    b.y := (predictions[box_index+1]+row) / l.side * h;
                    b.w := power(predictions[box_index+2], ifthen(l.sqrt, 2.0, 1)) * w;
                    b.h := power(predictions[box_index+3], ifthen(l.sqrt, 2.0, 1)) * h;
                    dets[index].bbox := b;
                    dets[index].objectness := scale;
                    for j := 0 to l.classes -1 do
                        begin
                            class_index := i * l.classes;
                            prob := scale * predictions[class_index+j];
                            if (prob > thresh) then
                                dets[index].prob[j] := prob
                            else
                                dets[index].prob[j] := 0
                        end
                end
        end
end;

procedure get_detection_boxes(const l: TDetectionLayer; const w, h: longint; const thresh: single; const probs: PPsingle; const boxes: PBox; const only_objectness: boolean);
var
    predictions: PSingle;
    i, j, n, row, col, index, p_index, box_index, class_index: longint;
    scale, prob: single;
begin
    predictions := l.output;
    for i := 0 to l.side * l.side -1 do
        begin
            row := i div l.side;
            col := i mod l.side;
            for n := 0 to l.n -1 do
                begin
                    index := i * l.n+n;
                    p_index := l.side * l.side * l.classes+i * l.n+n;
                    scale := predictions[p_index];
                    box_index := l.side * l.side * (l.classes+l.n)+(i * l.n+n) * 4;
                    boxes[index].x := (predictions[box_index+0]+col) / l.side * w;
                    boxes[index].y := (predictions[box_index+1]+row) / l.side * h;
                    boxes[index].w := power(predictions[box_index+2], (ifthen(l.sqrt, 2, 1))) * w;
                    boxes[index].h := power(predictions[box_index+3], (ifthen(l.sqrt, 2, 1))) * h;
                    for j := 0 to l.classes -1 do
                        begin
                            class_index := i * l.classes;
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
procedure forward_detection_layer_gpu(const l: TDetectionLayer; net: TNetwork);
begin
    if not net.train then
        begin
            copy_gpu(l.batch * l.inputs, net.input_gpu, 1, l.output_gpu, 1);
            exit()
        end;
    cuda_pull_array(net.input_gpu, net.input, l.batch * l.inputs);
    forward_detection_layer(l, net);
    cuda_push_array(l.output_gpu, l.output, l.batch * l.outputs);
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.inputs)
end;

procedure backward_detection_layer_gpu(l: TDetectionLayer; net: TNetwork);
begin
    axpy_gpu(l.batch * l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1)
end;
{$endif}

end.

