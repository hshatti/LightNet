unit MaxpoolLayer;

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
  SysUtils, math, lightnet, gemm, ConvolutionalLayer, imagedata;

type
  PMaxPoolLayer = ^TMaxPoolLayer;
  TMaxpoolLayer = TLayer;

function get_maxpool_delta(const l: TMaxPoolLayer):TImageData;

function get_maxpool_image(const l: TMaxPoolLayer):TImageData;
function make_maxpool_layer(const batch, h, w, c, size:longint; stride_x, stride_y:longint; const padding, maxpool_depth, out_channels, antialiasing:longint; const avgpool, train: boolean):TMaxpoolLayer;
procedure resize_maxpool_layer(var l: TMaxPoolLayer; const w, h: longint);
procedure forward_maxpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
procedure backward_maxpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);

procedure forward_local_avgpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
procedure backward_local_avgpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);

procedure cudnn_maxpool_setup(const l: PMaxPoolLayer);


implementation

function get_maxpool_image(const l: TMaxPoolLayer):TImageData;
var
    h, w, c: longint;
begin
    h := l.out_h;
    w := l.out_w;
    c := l.c;
    exit(float_to_image(w, h, c, l.output))
end;

function get_maxpool_delta(const l: TMaxPoolLayer):TImageData;
var
    h, w, c: longint;
begin
    h := l.out_h;
    w := l.out_w;
    c := l.c;
    exit(float_to_image(w, h, c, l.delta))
end;

procedure create_maxpool_cudnn_tensors(const l: PMaxPoolLayer);
begin
{$ifdef CUDNN}
    CHECK_CUDNN(cudnnCreatePoolingDescriptor( @l.poolingDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.srcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.dstTensorDesc))
{$endif}
end;

procedure cudnn_maxpool_setup(const l: PMaxPoolLayer);
begin
{$ifdef CUDNN}
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(l.poolingDesc, CUDNN_POOLING_MAX, CUDNN_NOT_PROPAGATE_NAN, l.size, l.size, l.pad div 2, l.pad div 2, l.stride_x, l.stride_y));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w))
{$endif}
end;

procedure cudnn_local_avgpool_setup(const l: PMaxPoolLayer);
begin
{$ifdef CUDNN}
    CHECK_CUDNN(cudnnSetPooling2dDescriptor(l.poolingDesc, CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING, CUDNN_NOT_PROPAGATE_NAN, l.size, l.size, l.pad div 2, l.pad div 2, l.stride_x, l.stride_y));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w))
{$endif}
end;

function make_maxpool_layer(const batch, h, w, c, size:longint; stride_x, stride_y:longint; const padding, maxpool_depth, out_channels, antialiasing:longint; const avgpool, train: boolean):TMaxpoolLayer;
var
    blur_stride_x, blur_stride_y, output_size, blur_size, blur_pad, blur_nweights, i: longint;
begin
    result := Default(TMaxPoolLayer);
    result.avgpool := avgpool;
    if avgpool then
        result.&type := ltLOCAL_AVGPOOL
    else
        result.&type := ltMAXPOOL;
    result.train := train;
    blur_stride_x := stride_x;
    blur_stride_y := stride_y;
    result.antialiasing := antialiasing;
    if antialiasing<>0 then begin
        stride_x := 1;
        stride_y := 1;
        result.stride := 1;
        result.stride_x := 1;
        result.stride_y := 1
    end;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := c;
    result.pad := padding;
    result.maxpool_depth := maxpool_depth;
    result.out_channels := out_channels;
    if maxpool_depth<>0 then
        begin
            result.out_c := out_channels;
            result.out_w := result.w;
            result.out_h := result.h
        end
    else
        begin
            result.out_w := (w+padding-size) div stride_x+1;
            result.out_h := (h+padding-size) div stride_y+1;
            result.out_c := c
        end;
    result.outputs := result.out_h * result.out_w * result.out_c;
    result.inputs := h * w * c;
    result.size := size;
    result.stride := stride_x;
    result.stride_x := stride_x;
    result.stride_y := stride_y;
    output_size := result.out_h * result.out_w * result.out_c * batch;
    if train then
        begin
            if not avgpool then
                setLength(result.indexes, output_size);
            result.delta := TSingles.Create(output_size);
        end;
    result.output := TSingles.Create(output_size);
    if avgpool then
        begin
            result.forward := forward_local_avgpool_layer;
            result.backward := backward_local_avgpool_layer
        end
    else
        begin
            result.forward := forward_maxpool_layer;
            result.backward := backward_maxpool_layer
        end;
{$ifdef GPU}
    if avgpool then
        begin
            result.forward_gpu := forward_local_avgpool_layer_gpu;
            result.backward_gpu := backward_local_avgpool_layer_gpu
        end
    else
        begin
            result.forward_gpu := forward_maxpool_layer_gpu;
            result.backward_gpu := backward_maxpool_layer_gpu
        end;
    if train then
        begin
            if not avgpool then
                result.indexes_gpu := cuda_make_int_array(output_size);
            result.delta_gpu := cuda_make_array(result.delta, output_size)
        end;
    result.output_gpu := cuda_make_array(result.output, output_size);
    create_maxpool_cudnn_tensors( and result);
    if avgpool then
        cudnn_local_avgpool_setup( and result)
    else
        cudnn_maxpool_setup( and result);
{$endif}
    result.bflops := (result.size * result.size * result.c * result.out_h * result.out_w) / 1000000000.0;
    if avgpool then
        begin
            if stride_x = stride_y then
                writeln(ErrOutput, format('avg               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
            else
                writeln(ErrOutput, format('avg              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, stride_y, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
        end
    else
        begin
            if maxpool_depth<>0 then
                writeln(ErrOutput, format('max-depth         %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
            else
                if stride_x = stride_y then
                    writeln(ErrOutput, format('max               %2dx%2d/%2d   %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
            else
                writeln(ErrOutput, format('max              %2dx%2d/%2dx%2d %4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [size, size, stride_x, stride_y, w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]))
        end;
    if result.antialiasing<>0 then
        begin
            writeln('AA:  ');
            setLength(result.input_layer, 1);
            blur_size := 3;
            blur_pad := blur_size div 2;
            if result.antialiasing = 2 then
                begin
                    blur_size := 2;
                    blur_pad := 0
                end;
            result.input_layer[0] := make_convolutional_layer(batch, 1, result.out_h, result.out_w, result.out_c, result.out_c, result.out_c, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, acLINEAR, false, false, false, false, false, 1, 0, nil, 0, false, train);
            blur_nweights := result.out_c * blur_size * blur_size;
            if blur_size = 2 then begin
                i := 0;
                while i < blur_nweights do begin
                    result.input_layer[0].weights[i+0] := 1 / 4.0;
                    result.input_layer[0].weights[i+1] := 1 / 4.0;
                    result.input_layer[0].weights[i+2] := 1 / 4.0;
                    result.input_layer[0].weights[i+3] := 1 / 4.0;
                    i := i + (blur_size * blur_size)
                end
            end else begin
                i := 0;
                while i < blur_nweights do begin
                    result.input_layer[0].weights[i+0] := 1 / 16.0;
                    result.input_layer[0].weights[i+1] := 2 / 16.0;
                    result.input_layer[0].weights[i+2] := 1 / 16.0;
                    result.input_layer[0].weights[i+3] := 2 / 16.0;
                    result.input_layer[0].weights[i+4] := 4 / 16.0;
                    result.input_layer[0].weights[i+5] := 2 / 16.0;
                    result.input_layer[0].weights[i+6] := 1 / 16.0;
                    result.input_layer[0].weights[i+7] := 2 / 16.0;
                    result.input_layer[0].weights[i+8] := 1 / 16.0;
                    i := i + (blur_size * blur_size)
                end;
            end;
            for i := 0 to result.out_c -1 do
                result.input_layer[0].biases[i] := 0;
{$ifdef GPU}
            if gpu_index >= 0 then
                begin
                    if result.antialiasing then
                        result.input_antialiasing_gpu := cuda_make_array(NULL, result.batch * result.outputs);
                    push_convolutional_layer( * (result.input_layer))
                end
{$endif}
        end;
end;

procedure resize_maxpool_layer(var l: TMaxPoolLayer; const w, h: longint);
var
    output_size: longint;
begin
    l.h := h;
    l.w := w;
    l.inputs := h * w * l.c;
    l.out_w := (w+l.pad-l.size) div l.stride_x+1;
    l.out_h := (h+l.pad-l.size) div l.stride_y+1;
    l.outputs := l.out_w * l.out_h * l.out_c;
    output_size := l.outputs * l.batch;
    if l.train then
        begin
            if not l.avgpool then
                setLength(l.indexes, output_size);
            l.delta.reAllocate(output_size)
        end;
    l.output.reAllocate(output_size);
{$ifdef GPU}
    CHECK_CUDA(cudaFree(l.output_gpu));
    l.output_gpu := cuda_make_array(l.output, output_size);
    if l.train then
        begin
            if not l.avgpool then
                begin
                    CHECK_CUDA(cudaFree(PSingle(l.indexes_gpu)));
                    l.indexes_gpu := cuda_make_int_array(output_size)
                end;
            CHECK_CUDA(cudaFree(l.delta_gpu));
            l.delta_gpu := cuda_make_array(l.delta, output_size)
        end;
    if l.avgpool then
        cudnn_local_avgpool_setup(l)
    else
        cudnn_maxpool_setup(l)
{$endif}
end;

procedure forward_maxpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
var
    b, i, j, k, g, out_index, max_i, in_index: longint;
    max, val: single;
    m, n, w_offset, h_offset, h, w, c, cur_h, cur_w, index: longint;
    s: TNetworkState;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    if l.maxpool_depth<>0 then
        begin
            for b := 0 to l.batch -1 do
                for i := 0 to l.h -1 do
                    for j := 0 to l.w -1 do
                        for g := 0 to l.out_c -1 do
                            begin
                                out_index := j+l.w * (i+l.h * (g+l.out_c * b));
                                max := -MaxSingle;
                                max_i := -1;
                                k := g;
                                while k < l.c do begin
                                    in_index := j+l.w * (i+l.h * (k+l.c * b));
                                    val := state.input[in_index];
                                    if (val > max) then
                                        max_i := in_index
                                    else
                                        max_i := max_i;
                                    if (val > max) then
                                        max := val
                                    else
                                        max := max;
                                    k := k + l.out_c
                                end;
                                l.output[out_index] := max;
                                if assigned(l.indexes) then
                                    l.indexes[out_index] := max_i
                            end;
            {$ifdef USE_TELEMETRY}
            if benchmark then metrics.forward.start(l.&type);
            {$endif}
            exit()
        end;
    if not state.train and (l.stride_x = l.stride_y) then
        forward_maxpool_layer_avx(state.input, l.output, PLongint(l.indexes), l.size, l.w, l.h, l.out_w, l.out_h, l.c, l.pad, l.stride, l.batch)
    else
        begin
            w_offset := -l.pad div 2;
            h_offset := -l.pad div 2;
            h := l.out_h;
            w := l.out_w;
            c := l.c;
            for b := 0 to l.batch -1 do
                for k := 0 to c -1 do
                    for i := 0 to h -1 do
                        for j := 0 to w -1 do
                            begin
                                out_index := j+w * (i+h * (k+c * b));
                                max := -MaxSingle;
                                max_i := -1;
                                for n := 0 to l.size -1 do
                                    for m := 0 to l.size -1 do
                                        begin
                                            cur_h := h_offset+i * l.stride_y+n;
                                            cur_w := w_offset+j * l.stride_x+m;
                                            index := cur_w+l.w * (cur_h+l.h * (k+b * l.c));
                                            if (cur_h >= 0) and (cur_h < l.h) and (cur_w >= 0) and (cur_w < l.w) then
                                                val := state.input[index]
                                            else
                                                val := -MaxSingle;
                                            if (val > max) then
                                                max_i := index
                                            else
                                                max_i := max_i;
                                            if (val > max) then
                                                max := val
                                            else
                                                max := max
                                        end;
                                l.output[out_index] := max;
                                if assigned(l.indexes) then
                                    l.indexes[out_index] := max_i
                            end
        end;
    if l.antialiasing<>0 then
        begin
            s := default(TNetworkState);
            s.train := state.train;
            s.workspace := state.workspace;
            s.net := state.net;
            s.input := l.output;
            forward_convolutional_layer( l.input_layer[0], @s);
            move(l.input_layer[0].output[0], l.output[0], l.input_layer[0].outputs * l.input_layer[0].batch * sizeof(single))
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_maxpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
var
    i, h, w, c, index: longint;
begin
    h := l.out_h;
    w := l.out_w;
    c := l.out_c;
    for i := 0 to h * w * c * l.batch -1 do
        begin
            index := l.indexes[i];
            state.delta[index] := state.delta[index] + l.delta[i]
        end
end;

procedure forward_local_avgpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
var
    b, i, j, k, m, n, w_offset, h_offset, h, w, c, out_index, counter, cur_h, cur_w, index: longint;
    avg: single;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    w_offset := -l.pad div 2;
    h_offset := -l.pad div 2;
    h := l.out_h;
    w := l.out_w;
    c := l.c;
    for b := 0 to l.batch -1 do
        for k := 0 to c -1 do
            for i := 0 to h -1 do
                for j := 0 to w -1 do
                    begin
                        out_index := j+w * (i+h * (k+c * b));
                        avg := 0;
                        counter := 0;
                        for n := 0 to l.size -1 do
                            for m := 0 to l.size -1 do
                                begin
                                    cur_h := h_offset+i * l.stride_y+n;
                                    cur_w := w_offset+j * l.stride_x+m;
                                    index := cur_w+l.w * (cur_h+l.h * (k+b * l.c));
                                    if (cur_h >= 0) and (cur_h < l.h) and (cur_w >= 0) and (cur_w < l.w) then
                                        begin
                                            inc(counter);
                                            avg := avg + state.input[index]
                                        end
                                end;
                        l.output[out_index] := avg / counter
                    end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}
end;

procedure backward_local_avgpool_layer(var l: TMaxpoolLayer; const state: PNetworkState);
var
    b, i, j, k, m, n, w_offset, h_offset, h, w, c, out_index, cur_h, cur_w, index: longint;
begin
    w_offset := -l.pad div 2;
    h_offset := -l.pad div 2;
    h := l.out_h;
    w := l.out_w;
    c := l.c;
    for b := 0 to l.batch -1 do
        for k := 0 to c -1 do
            for i := 0 to h -1 do
                for j := 0 to w -1 do
                    begin
                        out_index := j+w * (i+h * (k+c * b));
                        for n := 0 to l.size -1 do
                            for m := 0 to l.size -1 do
                                begin
                                    cur_h := h_offset+i * l.stride_y+n;
                                    cur_w := w_offset+j * l.stride_x+m;
                                    index := cur_w+l.w * (cur_h+l.h * (k+b * l.c));
                                    if (cur_h >= 0) and (cur_h < l.h) and (cur_w >= 0) and (cur_w < l.w) then
                                        state.delta[index] := state.delta[index] + (l.delta[out_index] / (l.size * l.size))
                                end
                    end
end;

end.

