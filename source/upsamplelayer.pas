unit UpSampleLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface

uses
  SysUtils, darknet, blas;

type
  PUpSampleLayer = ^TUpSampleLayer;
  TUpSampleLayer = TLayer;


// note UpSampling is just a [Scaler * Tensor] multiply with a stride
function make_upsample_layer(const batch, w, h, c:longint; stride: longint):TUpSampleLayer;
procedure resize_upsample_layer(var l: TUpSampleLayer; const w, h: longint);
procedure forward_upsample_layer(var l: TUpSampleLayer; const net: PNetworkState);
procedure backward_upsample_layer(var l: TUpSampleLayer; const net: PNetworkState);

{$ifdef GPU}
procedure forward_upsample_layer_gpu(const l: TUpSampleLayer; net: TNetwork);
procedure backward_upsample_layer_gpu(const l: TUpSampleLayer; net: TNetwork);
{$endif}

implementation

function make_upsample_layer(const batch, w, h, c:longint; stride: longint):TUpSampleLayer;
begin
    result := Default(TUpSampleLayer);
    result.&type := ltUPSAMPLE;
    result.batch := batch;
    result.w := w;
    result.h := h;
    result.c := c;
    result.out_w := w * stride;
    result.out_h := h * stride;
    result.out_c := c;
    if stride < 0 then
        begin
            stride := -stride;
            result.reverse := true;
            result.out_w := w div stride;
            result.out_h := h div stride
        end;
    result.stride := stride;
    result.outputs := result.out_w * result.out_h * result.out_c;
    result.inputs := result.w * result.h * result.c;
    result.delta := TSingles.Create(result.outputs * batch);
    result.output := TSingles.Create(result.outputs * batch);

    result.forward := forward_upsample_layer;
    result.backward := backward_upsample_layer;
  {$ifdef GPU}
    result.forward_gpu := forward_upsample_layer_gpu;
    result.backward_gpu := backward_upsample_layer_gpu;
    result.delta_gpu := cuda_make_array(result.delta, result.outputs * batch);
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
  {$endif}
    if result.reverse then
        writeln(ErrOutput, format('downsample         %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d', [stride, w, h, c, result.out_w, result.out_h, result.out_c]))
    else
        writeln(ErrOutput, format('upsample           %2dx  %4d x%4d x%4d   ->  %4d x%4d x%4d', [stride, w, h, c, result.out_w, result.out_h, result.out_c]));
    //exit(l)
end;

procedure resize_upsample_layer(var l: TUpSampleLayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.out_w := w * l.stride;
    l.out_h := h * l.stride;
    if l.reverse then
        begin
            l.out_w := w div l.stride;
            l.out_h := h div l.stride
        end;
    l.outputs := l.out_w * l.out_h * l.out_c;
    l.inputs := l.h * l.w * l.c;
    l.delta.reAllocate(l.outputs * l.batch);
    l.output.reAllocate(l.outputs * l.batch);
  {$ifdef GPU}
    cuda_free(l.output_gpu);
    cuda_free(l.delta_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch);
    l.delta_gpu := cuda_make_array(l.delta, l.outputs * l.batch)
  {$endif}
end;

procedure forward_upsample_layer(var l: TUpSampleLayer; const net: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    if l.reverse then         // todo [forward_upsample_layer] why not using rverse as a parameter instead of [if else then]
        upsample_cpu(l.output, l.out_w, l.out_h, l.c, l.batch, l.stride, false, l.scale, net.input)
    else
        upsample_cpu(net.input, l.w, l.h, l.c, l.batch, l.stride, true, l.scale, l.output);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_upsample_layer(var l: TUpSampleLayer; const net: PNetworkState);
begin
    if l.reverse then  // todo [backward_upsample] why not passing l.reverse to the function instad of [if then else]
        upsample_cpu(l.delta, l.out_w, l.out_h, l.c, l.batch, l.stride, true, l.scale, net.delta)
    else
        upsample_cpu(net.delta, l.w, l.h, l.c, l.batch, l.stride, false, l.scale, l.delta)
end;

{$ifdef GPU}
procedure forward_upsample_layer_gpu(const l: TUpSampleLayer; net: TNetwork);
begin
    fill_gpu(l.outputs * l.batch, 0, l.output_gpu, 1);
    if l.reverse then
        upsample_gpu(l.output_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 0, l.scale, net.input_gpu)
    else
        upsample_gpu(net.input_gpu, l.w, l.h, l.c, l.batch, l.stride, 1, l.scale, l.output_gpu)
end;

procedure backward_upsample_layer_gpu(const l: TUpSampleLayer; net: TNetwork);
begin
    if l.reverse then
        upsample_gpu(l.delta_gpu, l.out_w, l.out_h, l.c, l.batch, l.stride, 1, l.scale, net.delta_gpu)
    else
        upsample_gpu(net.delta_gpu, l.w, l.h, l.c, l.batch, l.stride, 0, l.scale, l.delta_gpu)
end;
{$endif}


end.

