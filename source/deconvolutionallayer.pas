unit DeConvolutionalLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$PointerMath On}

interface

uses
  SysUtils, lightnet, blas, col2im, Activations, BatchNormLayer, gemm;

type
  TDeConvolutionalLayer = TLayer;

function deconvolutional_out_height(const l: TDeConvolutionalLayer):longint;
function deconvolutional_out_width(const l: TDeConvolutionalLayer):longint;
function deconvolutional_out_size(const l: TDeConvolutionalLayer):longint;
function get_deconvolutional_image(const l: TDeConvolutionalLayer):TImageData;
function get_deconvolutional_delta(const l: TDeConvolutionalLayer):TImageData;
function make_deconvolutional_layer(const batch, h, w, c, n, size, stride: longint; const activation: TActivation):TDeConvolutionalLayer;
procedure resize_deconvolutional_layer(var l: TDeConvolutionalLayer; const h, w: longint);
procedure forward_deconvolutional_layer(var l: TDeConvolutionalLayer; const state: PNetworkState);
procedure backward_deconvolutional_layer(var l: TDeConvolutionalLayer; const state: PNetworkState);
procedure update_deconvolutional_layer(const l: TDeConvolutionalLayer; const arg :TUpdateArgs);


implementation
uses imagedata;
function deconvolutional_out_height(const l: TDeConvolutionalLayer):longint;
begin
    result := l.stride * (l.h-1)+l.size;
end;

function deconvolutional_out_width(const l: TDeConvolutionalLayer):longint;
begin
    result := l.stride * (l.w-1)+l.size;
end;

function deconvolutional_out_size(const l: TDeConvolutionalLayer):longint;
begin
    exit(deconvolutional_out_height(l) * deconvolutional_out_width(l))
end;

function get_deconvolutional_image(const l: TDeConvolutionalLayer):TImageData;
var
    h, w, c: longint;
begin
    h := deconvolutional_out_height(l);
    w := deconvolutional_out_width(l);
    c := l.n;
    exit(float_to_image(w, h, c, l.output))
end;

function get_deconvolutional_delta(const l: TDeConvolutionalLayer):TImageData;
var
    h, w, c: longint;
begin
    h := deconvolutional_out_height(l);
    w := deconvolutional_out_width(l);
    c := l.n;
    exit(float_to_image(w, h, c, l.delta))
end;

function make_deconvolutional_layer(const batch, h, w, c, n, size, stride: longint; const activation: TActivation):TDeConvolutionalLayer;
var
    i, out_h, out_w: longint;
    scale: single;
begin
    result := default(TDeConvolutionalLayer);
    result.&type := ltDECONVOLUTIONAL;
    result.h := h;
    result.w := w;
    result.c := c;
    result.n := n;
    result.batch := batch;
    result.stride := stride;
    result.size := size;
    result.weights := TSingles.create(c * n * size * size);
    result.weight_updates := TSingles.create(c * n * size * size);
    result.biases := TSingles.create(n);
    result.bias_updates := TSingles.create(n);
    scale := 1. / sqrt(size * size * c);
    for i := 0 to c * n * size * size -1 do
        result.weights[i] := scale * rand_normal();
    for i := 0 to n -1 do
        result.biases[i] := scale;
    out_h := deconvolutional_out_height(result);
    out_w := deconvolutional_out_width(result);
    result.out_h := out_h;
    result.out_w := out_w;
    result.out_c := n;
    result.outputs := result.out_w * result.out_h * result.out_c;
    result.inputs := result.w * result.h * result.c;
    result.col_image := TSingles.create(h * w * size * size * n);
    result.output := TSingles.create(result.batch * out_h * out_w * n);
    result.delta := TSingles.create(result.batch * out_h * out_w * n);
    result.forward := forward_deconvolutional_layer;
    result.backward := backward_deconvolutional_layer;
    result.update := update_deconvolutional_layer;
  {$ifdef GPU}
    result.weights_gpu := cuda_make_array(result.weights, c * n * size * size);
    result.weight_updates_gpu := cuda_make_array(result.weight_updates, c * n * size * size);
    result.biases_gpu := cuda_make_array(result.biases, n);
    result.bias_updates_gpu := cuda_make_array(result.bias_updates, n);
    result.col_image_gpu := cuda_make_array(result.col_image, h * w * size * size * n);
    result.delta_gpu := cuda_make_array(result.delta, result.batch * out_h * out_w * n);
    result.output_gpu := cuda_make_array(result.output, result.batch * out_h * out_w * n);
  {$endif}
    result.activation := activation;
    writeln(ErrOutput, format('Deconvolutional Layer: %d x %d x %d image, %d filters -> %d x %d x %d image', [h, w, c, n, out_h, out_w, n]));
end;

procedure resize_deconvolutional_layer(var l: TDeConvolutionalLayer; const h, w: longint);
var
    out_h, out_w: longint;
begin
    l.h := h;
    l.w := w;
    out_h := deconvolutional_out_height( l);
    out_w := deconvolutional_out_width( l);
    l.col_image.reAllocate(out_h * out_w * l.size * l.size * l.c);
    l.output.reAllocate(l.batch * out_h * out_w);
    l.delta.reAllocate(l.batch * out_h * out_w);
  {$ifdef GPU}
    cuda_free(l.col_image_gpu);
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);
    l.col_image_gpu := cuda_make_array(l.col_image, out_h * out_w * l.size * l.size * l.c);
    l.delta_gpu := cuda_make_array(l.delta, l.batch * out_h * out_w * l.n);
    l.output_gpu := cuda_make_array(l.output, l.batch * out_h * out_w * l.n)
  {$endif}
end;

procedure forward_deconvolutional_layer(var l: TDeConvolutionalLayer;
  const state: PNetworkState);
var
    i, out_h, out_w, size, m, n, k: longint;
    a, b, c: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    out_h := deconvolutional_out_height(l);
    out_w := deconvolutional_out_width(l);
    size := out_h * out_w;
    m := l.size * l.size * l.n;
    n := l.h * l.w;
    k := l.c;
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    for i := 0 to l.batch -1 do
        begin
            a := l.weights;
            b := state.input+i * l.c * l.h * l.w;
            c := l.col_image;
            sgemm(1, 0, m, n, k, 1, a, m, b, n, 0, c, n);
            col2im_cpu(c, l.n, out_h, out_w, l.size, l.stride, 0, l.output+i * l.n * size)
        end;
    add_bias(l.output, l.biases, l.batch, l.n, size);
    activate_array(l.output, l.batch * l.n * size, l.activation) ;

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_deconvolutional_layer(var l: TDeConvolutionalLayer;
  const state: PNetworkState);
var
    alpha: single;
    out_h, out_w, size, i, m, n, k: longint;
    a, b, c: PSingle;
begin
    alpha := 1. / l.batch;
    out_h := deconvolutional_out_height(l);
    out_w := deconvolutional_out_width(l);
    size := out_h * out_w;
    gradient_array(l.output, size * l.n * l.batch, l.activation, l.delta);
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, size);
    for i := 0 to l.batch -1 do
        begin
            m := l.c;
            n := l.size * l.size * l.n;
            k := l.h * l.w;
            a := state.input+i * m * n;
            b := l.col_image;
            c := l.weight_updates;
            im2col_cpu(l.delta+i * l.n * size, l.n, out_h, out_w, l.size, l.stride, 0, b);
            sgemm(0, 1, m, n, k, alpha, a, k, b, k, 1, c, n);
            if assigned(state.delta) then
                begin
                    m := l.c;
                    n := l.h * l.w;
                    k := l.size * l.size * l.n;
                    a := l.weights;
                    b := l.col_image;
                    c := state.delta+i * n * m;
                    sgemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n)
                end
        end
end;

procedure update_deconvolutional_layer(const l: TDeConvolutionalLayer; const arg :TUpdateArgs);
var
    size: longint;
begin
    size := l.size * l.size * l.c * l.n;
    axpy_cpu(l.n, arg.learning_rate, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, arg.momentum, l.bias_updates, 1);
    axpy_cpu(size, -arg.decay, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(size, arg.learning_rate, l.weight_updates, 1, l.weights, 1);
    scal_cpu(size, arg.momentum, l.weight_updates, 1)
end;

end.

