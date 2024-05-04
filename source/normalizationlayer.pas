unit NormalizationLayer;

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
  SysUtils, lightnet, blas;

type

  PNormalizationLayer = ^TNormalizationLayer;
  TNormalizationLayer = TLayer;

function make_normalization_layer(const batch, w, h, c, size: longint; const alpha, beta, kappa: single):TNormalizationLayer;
procedure resize_normalization_layer(var layer: TNormalizationLayer; const w, h: longint);
procedure forward_normalization_layer(var layer: TNormalizationLayer; const net: PNetworkState);
procedure backward_normalization_layer(var layer: TNormalizationLayer; const net: PNetworkState);

{$ifdef GPU}
procedure forward_normalization_layer_gpu(var layer: TNormalizationLayer; var net: network);
procedure backward_normalization_layer_gpu(var layer: TNormalizationLayer; var net: network);
{$endif}

implementation

function make_normalization_layer(const batch, w, h, c, size: longint; const alpha, beta, kappa: single):TNormalizationLayer;
begin
    writeln(ErrOutput, format('Local Response Normalization result: %d x %d x %d image, %d size', [w, h, c, size]));
    result := default(TNormalizationLayer);
    result.&type := ltNORMALIZATION;
    result.batch := batch;
    result.h := h; result.out_h := h;
    result.w := w; result.out_w := w;
    result.c := c; result.out_c := c;
    result.kappa := kappa;
    result.size := size;
    result.alpha := alpha;
    result.beta := beta;
    result.output := TSingles.Create(h * w * c * batch);
    result.delta := TSingles.Create(h * w * c * batch);
    result.squared := TSingles.Create(h * w * c * batch);
    result.norms := TSingles.Create(h * w * c * batch);
    result.inputs := w * h * c;
    result.outputs := result.inputs;
    result.forward := forward_normalization_layer;
    result.backward := backward_normalization_layer;
  {$ifdef GPU}
    result.forward_gpu := forward_normalization_layer_gpu;
    result.backward_gpu := backward_normalization_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, h * w * c * batch);
    result.delta_gpu := cuda_make_array(result.delta, h * w * c * batch);
    result.squared_gpu := cuda_make_array(result.squared, h * w * c * batch);
    result.norms_gpu := cuda_make_array(result.norms, h * w * c * batch);
  {$endif}
    //exit(result)
end;

procedure resize_normalization_layer(var layer: TNormalizationLayer; const w,
  h: longint);
var
    c: longint;
    batch: longint;
begin
    c := layer.c;
    batch := layer.batch;
    layer.h := h;
    layer.w := w;
    layer.out_h := h;
    layer.out_w := w;
    layer.inputs := w * h * c;
    layer.outputs := layer.inputs;
    layer.output.reAllocate( h * w * c * batch);
    layer.delta.reAllocate( h * w * c * batch);
    layer.squared.reAllocate( h * w * c * batch);
    layer.norms.reAllocate( h * w * c * batch);
  {$ifdef GPU}
    cuda_free(layer.output_gpu);
    cuda_free(layer.delta_gpu);
    cuda_free(layer.squared_gpu);
    cuda_free(layer.norms_gpu);
    layer.output_gpu := cuda_make_array(layer.output, h * w * c * batch);
    layer.delta_gpu := cuda_make_array(layer.delta, h * w * c * batch);
    layer.squared_gpu := cuda_make_array(layer.squared, h * w * c * batch);
    layer.norms_gpu := cuda_make_array(layer.norms, h * w * c * batch)
  {$endif}
end;

procedure forward_normalization_layer(var layer: TNormalizationLayer; const net: PNetworkState);
var
    k, b, w, h, c, prev, next: longint;
    squared, norms, input: TSingles;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(layer.&type);
    {$endif}

    w := layer.w;
    h := layer.h;
    c := layer.c;
    scal_cpu(w * h * c * layer.batch, 0, layer.squared, 1);
    for b := 0 to layer.batch -1 do
        begin
            squared := layer.squared+w * h * c * b;
            norms := layer.norms+w * h * c * b;
            input := net.input+w * h * c * b;
            pow_cpu(w * h * c, 2, input, 1, squared, 1);
            const_cpu(w * h, layer.kappa, norms, 1);
            for k := 0 to layer.size div 2 -1 do
                axpy_cpu(w * h, layer.alpha, squared+w * h * k, 1, norms, 1);
            for k := 1 to layer.c -1 do
                begin
                    copy_cpu(w * h, norms+w * h * (k-1), 1, norms+w * h * k, 1);
                    prev := k-((layer.size-1) div 2)-1;
                    next := k+(layer.size div 2);
                    if prev >= 0 then
                        axpy_cpu(w * h, -layer.alpha, squared+w * h * prev, 1, norms+w * h * k, 1);
                    if next < layer.c then
                        axpy_cpu(w * h, layer.alpha, squared+w * h * next, 1, norms+w * h * k, 1)
                end
        end;
    pow_cpu(w * h * c * layer.batch, -layer.beta, layer.norms, 1, layer.output, 1);
    mul_cpu(w * h * c * layer.batch, net.input, 1, layer.output, 1);
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(layer.&type);
    {$endif}

end;

procedure backward_normalization_layer(var layer: TNormalizationLayer; const net: PNetworkState);
var
    w, h, c: longint;
begin
    w := layer.w;
    h := layer.h;
    c := layer.c;
    pow_cpu(w * h * c * layer.batch, -layer.beta, layer.norms, 1, net.delta, 1);
    mul_cpu(w * h * c * layer.batch, layer.delta, 1, net.delta, 1)
end;

{$ifdef GPU}
procedure forward_normalization_layer_gpu(var layer: TNormalizationLayer; var net: network);
var
    k, b, w, h, c, prev, next: longint;
    squared, norms, input: TSingles;
begin
    w := layer.w;
    h := layer.h;
    c := layer.c;
    scal_gpu(w * h * c * layer.batch, 0, layer.squared_gpu, 1);
    for b := 0 to layer.batch -1 do
        begin
            squared := layer.squared_gpu+w * h * c * b;
            norms := layer.norms_gpu+w * h * c * b;
            input := net.input_gpu+w * h * c * b;
            pow_gpu(w * h * c, 2, input, 1, squared, 1);
            const_gpu(w * h, layer.kappa, norms, 1);
            for k := 0 to layer.size div 2 -1 do
                axpy_gpu(w * h, layer.alpha, squared+w * h * k, 1, norms, 1);
            for k := 1 to layer.c -1 do
                begin
                    copy_gpu(w * h, norms+w * h * (k-1), 1, norms+w * h * k, 1);
                    prev := k-((layer.size-1) div 2)-1;
                    next := k+(layer.size div 2);
                    if prev >= 0 then
                        axpy_gpu(w * h, -layer.alpha, squared+w * h * prev, 1, norms+w * h * k, 1);
                    if next < layer.c then
                        axpy_gpu(w * h, layer.alpha, squared+w * h * next, 1, norms+w * h * k, 1)
                end
        end;
    pow_gpu(w * h * c * layer.batch, -layer.beta, layer.norms_gpu, 1, layer.output_gpu, 1);
    mul_gpu(w * h * c * layer.batch, net.input_gpu, 1, layer.output_gpu, 1)
end;

procedure backward_normalization_layer_gpu(var layer: TNormalizationLayer; var net: network);
var
    w, h, c: longint;
begin
    w := layer.w;
    h := layer.h;
    c := layer.c;
    pow_gpu(w * h * c * layer.batch, -layer.beta, layer.norms_gpu, 1, net.delta_gpu, 1);
    mul_gpu(w * h * c * layer.batch, layer.delta_gpu, 1, net.delta_gpu, 1)
end;
{$endif}

end.

