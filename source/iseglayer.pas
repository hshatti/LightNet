unit iSegLayer;

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
  SysUtils, math, lightnet, activations, blas;

type
    TISegLayer = TLayer;

function make_iseg_layer(const batch, w, h, classes, ids: longint):TISegLayer;
procedure resize_iseg_layer(var l: TISeglayer; const w, h: longint);
procedure forward_iseg_layer(var l: TISegLayer; const net: PNetworkState);
procedure backward_iseg_layer(var l: TISegLayer; const net: PNetworkState);

implementation

function make_iseg_layer(const batch, w, h, classes, ids: longint):TISegLayer;
var
    i: longint;
begin
    result := default(TISegLayer);
    result.&type := ltISEG;
    result.h := h;
    result.w := w;
    result.c := classes+ids;
    result.out_w := result.w;
    result.out_h := result.h;
    result.out_c := result.c;
    result.classes := classes;
    result.batch := batch;
    result.extra := ids;
//    result.cost := TSingles.Create(1);
    result.outputs := h * w * result.c;
    result.inputs := result.outputs;
    result.truths := 90 * (result.w * result.h+1);
    result.delta := TSingles.Create(batch * result.outputs);
    result.output := TSingles.Create(batch * result.outputs);
    setLength(result.counts, 90);// := TIntegers.Create(90);
    result.sums := AllocMem(90* sizeof(TSingles));
    if boolean(ids) then
        begin
            for i := 0 to 90 -1 do
                result.sums[i] := TSingles.Create(ids)
        end;
    result.forward := forward_iseg_layer;
    result.backward := backward_iseg_layer;
{$ifdef GPU}
    result.forward_gpu := forward_iseg_layer_gpu;
    result.backward_gpu := backward_iseg_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, batch * result.outputs);
    result.delta_gpu := cuda_make_array(result.delta, batch * result.outputs);
{$endif}
    writeln(ErrOutput, 'iseg');
    //srand(0);
    randomize;
    //RandSeed:=0;// note [make_iseg_layer] again RANDSEED =0 why?
end;

procedure resize_iseg_layer(var l: TISeglayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.outputs := h * w * l.c;
    l.inputs := l.outputs;
    l.output.ReAllocate( l.batch * l.outputs );
    l.delta.reAllocate( l.batch * l.outputs);
{$ifdef GPU}
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);
    l.delta_gpu := cuda_make_array(l.delta, l.batch * l.outputs);
    l.output_gpu := cuda_make_array(l.output, l.batch * l.outputs)
{$endif}
end;

procedure forward_iseg_layer(var l: TISegLayer; const net: PNetworkState);
var
    time: double;
    i, b, j, k, ids, index, c, z: longint;
    v, sum, diff: single;
    mse :TSingles;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    ids := l.extra;
    move(net.input[0], l.output[0], l.outputs * l.batch * sizeof(single));
    FillDWord(l.delta[0], l.outputs * l.batch, 0);
{$ifndef GPU}
    for b := 0 to l.batch -1 do
        begin
            index := b * l.outputs;
            activate_array(l.output+index, l.classes * l.w * l.h, acLOGISTIC)
        end;
{$endif}
    for b := 0 to l.batch -1 do
        begin
            for i := 0 to l.classes -1 do
                for k := 0 to l.w * l.h -1 do
                    begin
                        index := b * l.outputs+i * l.w * l.h+k;
                        l.delta[index] := -l.output[index]
                    end;
            for i := 0 to ids -1 do
                for k := 0 to l.w * l.h -1 do
                    begin
                        index := b * l.outputs+(i+l.classes) * l.w * l.h+k;
                        l.delta[index] := 0.1 * (-l.output[index])
                    end;
            FillChar(l.counts[0], 90 * sizeof(longint),0);
            for i := 0 to 90 -1 do
                begin
                    fill_cpu(ids, 0, l.sums[i], 1);
                    c := trunc(net.truth[b * l.truths+i * (l.w * l.h+1)]);
                    if c < 0 then
                        break;
                    for k := 0 to l.w * l.h -1 do
                        begin
                            index := b * l.outputs+c * l.w * l.h+k;
                            v := net.truth[b * l.truths+i * (l.w * l.h+1)+1+k];
                            if v<>0 then
                                begin
                                    l.delta[index] := v-l.output[index];
                                    axpy_cpu(ids, 1, l.output+b * l.outputs+l.classes * l.w * l.h+k, l.w * l.h, l.sums[i], 1);
                                    inc(l.counts[i])
                                end
                        end
                end;
            // todo replace with stack [ mse: array[0..89] of single ]
            mse := TSingles.Create(90);
            for i := 0 to 90 -1 do
                begin
                    c := trunc(net.truth[b * l.truths+i * (l.w * l.h+1)]);
                    if c < 0 then
                        break;
                    for k := 0 to l.w * l.h -1 do
                        begin
                            v := net.truth[b * l.truths+i * (l.w * l.h+1)+1+k];
                            if v<>0 then
                                begin
                                    sum := 0;
                                    for z := 0 to ids -1 do
                                        begin
                                            index := b * l.outputs+(l.classes+z) * l.w * l.h+k;
                                            sum := sum + sqr(l.sums[i][z] / l.counts[i]-l.output[index]{, 2})
                                        end;
                                    mse[i] := mse[i] + sum
                                end
                        end;
                    mse[i] := mse[i] / l.counts[i]
                end;
            for i := 0 to 90 -1 do
                begin
                    if not boolean(l.counts[i]) then
                        continue;
                    scal_cpu(ids, 1 / l.counts[i], l.sums[i], 1);
                    if (b = 0)
{$ifdef GPU}
                    and (net.gpu_index = 0)
{$endif}
                    then
                        begin
                            write(format('%4d, %6.3f, ', [l.counts[i], mse[i]]));
                            for j := 0 to ids -1 do
                                write(format('%6.3f,', [l.sums[i][j]]));
                            writeln('')
                        end
                end;
            mse.free;
            for i := 0 to 90 -1 do
                begin
                    if not boolean(l.counts[i]) then
                        continue;
                    for k := 0 to l.w * l.h -1 do
                        begin
                            v := net.truth[b * l.truths+i * (l.w * l.h+1)+1+k];
                            if v<>0 then
                                for j := 0 to 90 -1 do
                                    begin
                                        if not boolean(l.counts[j]) then
                                            continue;
                                        for z := 0 to ids -1 do
                                            begin
                                                index := b * l.outputs+(l.classes+z) * l.w * l.h+k;
                                                diff := l.sums[j][z]-l.output[index];
                                                if j = i then
                                                    l.delta[index] := l.delta[index] + ifthen(diff < 0, -0.1, 0.1)
                                                else
                                                    l.delta[index] := l.delta[index] + (-(ifthen(diff < 0, -0.1, 0.1)))
                                            end
                                    end
                        end
                end;
            for i := 0 to ids -1 do
                for k := 0 to l.w * l.h -1 do
                    begin
                        index := b * l.outputs+(i+l.classes) * l.w * l.h+k;
                        l.delta[index] := l.delta[index] * 0.01
                    end
        end;
    l.cost := sqr(mag_array(l.delta, l.outputs * l.batch){, 2});
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_iseg_layer(var l: TISegLayer; const net: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, 1, l.delta, 1, net.delta, 1)
end;

{$ifdef GPU}
procedure forward_iseg_layer_gpu(var l: TISegLayer; const net: PNetworkState);
var
    b: longint;
begin
    copy_gpu(l.batch * l.inputs, net.input_gpu, 1, l.output_gpu, 1);
    for b := 0 to l.batch -1 do
        activate_array_gpu(l.output_gpu+b * l.outputs, l.classes * l.w * l.h, LOGISTIC);
    cuda_pull_array(l.output_gpu, net.input, l.batch * l.inputs);
    forward_iseg_layer(l, net);
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs)
end;

procedure backward_iseg_layer_gpu(var l: TISegLayer; const net: PNetworkState);
var
    b: longint;
begin
    for b := 0 to l.batch -1 do
      //if l.extra<>0 then
      //  gradient_array_gpu(l.output_gpu + b*l.outputs + l.classes*l.w*l.h, l.extra*l.w*l.h, acLOGISTIC, l.delta_gpu + b*l.outputs + l.classes*l.w*l.h)
    ;
    axpy_gpu(l.batch * l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1)
end;
{$endif}

end.

