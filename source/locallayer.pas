unit LocalLayer;

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
  SysUtils, darknet, blas, col2im, Activations, gemm;

type
  TLocalLayer = TLayer;


function local_out_height(const l: TLocalLayer):longint;
function local_out_width(const l: TLocalLayer):longint;
function make_local_layer(const batch, h, w, c, n, size, stride, pad: longint; const activation: TActivation):TLocalLayer;
procedure forward_local_layer(var l: TLocalLayer; const state: PNetworkState);
procedure backward_local_layer(var l: TLocalLayer; const state: PNetworkState);
procedure update_local_layer(const l: TLocalLayer; const a: TUpdateArgs);

{$ifdef GPU}
procedure forward_local_layer_gpu(const l: TLocalLayer; net: TNetwork);
procedure backward_local_layer_gpu(l: TLocalLayer; net: TNetwork);
procedure update_local_layer_gpu(l: TLocalLayer; a: TUpdateArgs);
procedure pull_local_layer(l: TLocalLayer);
procedure push_local_layer(l: TLocalLayer);
{$endif}

implementation

function local_out_height(const l: TLocalLayer):longint;
var
    h: longint;
begin
    h := l.h;
    if not boolean(l.pad )then
        h := h - l.size
    else
        h := h - 1;
    exit(h div l.stride+1)
end;

function local_out_width(const l: TLocalLayer):longint;
var
    w: longint;
begin
    w := l.w;
    if not boolean(l.pad) then
        w := w - l.size
    else
        w := w - 1;
    exit(w div l.stride+1)
end;

function make_local_layer(const batch, h, w, c, n, size, stride, pad: longint;
  const activation: TActivation): TLocalLayer;
var
    i, out_h, out_w, locations: longint;
    scale: single;
begin
    result := default(TLocalLayer);
    result.&type := ltLOCAL;

    result.h := h;
    result.w := w;
    result.c := c;
    result.n := n;
    result.batch := batch;
    result.stride := stride;
    result.size := size;
    result.pad := pad;

    out_h := local_out_height(result);
    out_w := local_out_width(result);
    locations := out_h * out_w;
    result.out_h := out_h;
    result.out_w := out_w;
    result.out_c := n;
    result.outputs := result.out_h * result.out_w * result.out_c;
    result.inputs := result.w * result.h * result.c;

    result.weights := TSingles.Create(c * n * size * size * locations);
    result.weight_updates := TSingles.Create(c * n * size * size * locations);

    result.biases := TSingles.Create(result.outputs);
    result.bias_updates := TSingles.Create(result.outputs);

    scale := sqrt(2 / (size * size * c));
    for i := 0 to c * n * size * size -1 do
        result.weights[i] := scale * rand_uniform(-1, 1);

    result.col_image := TSingles.Create(out_h * out_w * size * size * c);
    result.output := TSingles.Create(result.batch * out_h * out_w * n);
    result.delta := TSingles.Create(result.batch * out_h * out_w * n);

    //result.workspace_size := out_h * out_w * size * size * c;

    result.forward := forward_local_layer;
    result.backward := backward_local_layer;
    result.update := update_local_layer;
{$ifdef GPU}
    result.forward_gpu := forward_local_layer_gpu;
    result.backward_gpu := backward_local_layer_gpu;
    result.update_gpu := update_local_layer_gpu;

    result.weights_gpu := cuda_make_array(result.weights, c * n * size * size * locations);
    result.weight_updates_gpu := cuda_make_array(result.weight_updates, c * n * size * size * locations);

    result.biases_gpu := cuda_make_array(result.biases, result.outputs);
    result.bias_updates_gpu := cuda_make_array(result.bias_updates, result.outputs);

    result.delta_gpu := cuda_make_array(result.delta, result.batch * out_h * out_w * n);
    result.output_gpu := cuda_make_array(result.output, result.batch * out_h * out_w * n);
{$endif}
    result.activation := activation;
    writeln(ErrOutput, format('Local Layer: %d x %d x %d image, %d filters -> %d x %d x %d image', [h, w, c, n, out_h, out_w, n]));
end;

procedure forward_local_layer(var l: TLocalLayer; const state: PNetworkState);
var
    out_h, out_w, i, j, locations, m, n, k: longint;
    input, output, a, b, c: TSingles;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    out_h := local_out_height(l);
    out_w := local_out_width(l);
    locations := out_h * out_w;

    for i := 0 to l.batch -1 do
        copy_cpu(l.outputs, l.biases, 1, l.output+i * l.outputs, 1);

    for i := 0 to l.batch -1 do
        begin
            input := state.input+i * l.w * l.h * l.c;
            im2col_cpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad, @l.col_image[0]);
            output := l.output+i * l.outputs;
            for j := 0 to locations -1 do
                begin
                    a := l.weights+j * l.size * l.size * l.c * l.n;
                    b := @l.col_image[j];
                    c := output+j;

                    m := l.n;
                    n := 1;
                    k := l.size * l.size * l.c;
                    sgemm(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations)
                end
        end;
    activate_array(l.output, l.outputs * l.batch, l.activation);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_local_layer(var l: TLocalLayer; const state: PNetworkState);
var
    i, j, locations, m, n, k: longint;
    input, a, b, c: TSingles;
begin
    locations := l.out_w * l.out_h;
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    for i := 0 to l.batch -1 do
        axpy_cpu(l.outputs, 1, l.delta+i * l.outputs, 1, l.bias_updates, 1);
    for i := 0 to l.batch -1 do
        begin
            input := state.input+i * l.w * l.h * l.c;
            im2col_cpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad, @l.col_image[0]);
            for j := 0 to locations -1 do
                begin
                    a := l.delta+i * l.outputs+j;
                    b := @l.col_image[j];
                    c := l.weight_updates+j * l.size * l.size * l.c * l.n;
                    m := l.n;
                    n := l.size * l.size * l.c;
                    k := 1;

                    sgemm(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n)
                end;
            if assigned(state.delta) then
                begin
                    for j := 0 to locations -1 do
                        begin
                            a := l.weights+j * l.size * l.size * l.c * l.n;
                            b := l.delta+i * l.outputs+j;
                            c := @l.col_image[j];

                            m := l.size * l.size * l.c;
                            n := 1;
                            k := l.n;

                            sgemm(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations)
                        end;
                    col2im_cpu(@l.col_image[0], l.c, l.h, l.w, l.size, l.stride, l.pad, state.delta+i * l.c * l.h * l.w)
                end
        end
end;

procedure update_local_layer(const l: TLocalLayer; const a: TUpdateArgs);
var
    //learning_rate, momentum, decay: single;
    batch, locations, size: longint;
begin
    //learning_rate := a.learning_rate * l.learning_rate_scale;
    //momentum := a.momentum;
    //decay := a.decay;
    //batch := a.batch;
    locations := l.out_w * l.out_h;
    size := l.size * l.size * l.c * l.n * locations;
    axpy_cpu(l.outputs, a.learning_rate / a.batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, a.momentum, l.bias_updates, 1);

    axpy_cpu(size, -a.decay * a.batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, a.learning_rate / a.batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, a.momentum, l.weight_updates, 1)
end;

{$ifdef GPU}
procedure forward_local_layer_gpu(const l: TLocalLayer; net: TNetwork);
var
    out_h: longint;
    out_w: longint;
    i: longint;
    j: longint;
    locations: longint;
    input: TSingles;
    output: TSingles;
    a: TSingles;
    b: TSingles;
    c: TSingles;
    m: longint;
    n: longint;
    k: longint;
begin
    out_h := local_out_height(l);
    out_w := local_out_width(l);
    locations := out_h * out_w;
    for i := 0 to l.batch -1 do
        copy_gpu(l.outputs, l.biases_gpu, 1, l.output_gpu+i * l.outputs, 1);
    for i := 0 to l.batch -1 do
        begin
            input := net.input_gpu+i * l.w * l.h * l.c;
            im2col_gpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad, net.workspace);
            output := l.output_gpu+i * l.outputs;
            for j := 0 to locations -1 do
                begin
                    a := l.weights_gpu+j * l.size * l.size * l.c * l.n;
                    b := net.workspace+j;
                    c := output+j;
                    m := l.n;
                    n := 1;
                    k := l.size * l.size * l.c;
                    gemm_gpu(0, 0, m, n, k, 1, a, k, b, locations, 1, c, locations)
                end
        end;
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation)
end;

procedure backward_local_layer_gpu(l: TLocalLayer; net: TNetwork);
var
    i: longint;
    j: longint;
    locations: longint;
    input: TSingles;
    a: TSingles;
    b: TSingles;
    c: TSingles;
    m: longint;
    n: longint;
    k: longint;
begin
    locations := l.out_w * l.out_h;
    gradient_array_gpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    for i := 0 to l.batch -1 do
        axpy_gpu(l.outputs, 1, l.delta_gpu+i * l.outputs, 1, l.bias_updates_gpu, 1);
    for i := 0 to l.batch -1 do
        begin
            input := net.input_gpu+i * l.w * l.h * l.c;
            im2col_gpu(input, l.c, l.h, l.w, l.size, l.stride, l.pad, net.workspace);
            for j := 0 to locations -1 do
                begin
                    a := l.delta_gpu+i * l.outputs+j;
                    b := net.workspace+j;
                    c := l.weight_updates_gpu+j * l.size * l.size * l.c * l.n;
                    m := l.n;
                    n := l.size * l.size * l.c;
                    k := 1;
                    gemm_gpu(0, 1, m, n, k, 1, a, locations, b, locations, 1, c, n)
                end;
            if net.delta_gpu then
                begin
                    for j := 0 to locations -1 do
                        begin
                            a := l.weights_gpu+j * l.size * l.size * l.c * l.n;
                            b := l.delta_gpu+i * l.outputs+j;
                            c := net.workspace+j;
                            m := l.size * l.size * l.c;
                            n := 1;
                            k := l.n;
                            gemm_gpu(1, 0, m, n, k, 1, a, m, b, locations, 0, c, locations)
                        end;
                    col2im_gpu(net.workspace, l.c, l.h, l.w, l.size, l.stride, l.pad, net.delta_gpu+i * l.c * l.h * l.w)
                end
        end
end;

procedure update_local_layer_gpu(l: TLocalLayer; a: TUpdateArgs);
var
    learning_rate: single;
    momentum: single;
    decay: single;
    batch: longint;
    locations: longint;
    size: longint;
begin
    learning_rate := a.learning_rate * l.learning_rate_scale;
    momentum := a.momentum;
    decay := a.decay;
    batch := a.batch;
    locations := l.out_w * l.out_h;
    size := l.size * l.size * l.c * l.n * locations;
    axpy_gpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_gpu(l.outputs, momentum, l.bias_updates_gpu, 1);
    axpy_gpu(size, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_gpu(size, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_gpu(size, momentum, l.weight_updates_gpu, 1)
end;

procedure pull_local_layer(l: TLocalLayer);
var
    locations: longint;
    size: longint;
begin
    locations := l.out_w * l.out_h;
    size := l.size * l.size * l.c * l.n * locations;
    cuda_pull_array(l.weights_gpu, l.weights, size);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs)
end;

procedure push_local_layer(l: TLocalLayer);
var
    locations: longint;
    size: longint;
begin
    locations := l.out_w * l.out_h;
    size := l.size * l.size * l.c * l.n * locations;
    cuda_push_array(l.weights_gpu, l.weights, size);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs)
end;
{$endif}

end.

