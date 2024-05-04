unit ReOrgLayer;

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
  PReOrgLayer = ^TReOrgLayer;
  TReOrgLayer = TLayer;

function make_reorg_layer(const batch, w, h, c, stride: longint; const reverse: boolean):TReOrgLayer;
procedure resize_reorg_layer(var l: TReOrgLayer; const w, h: longint);
procedure forward_reorg_layer(var l: TReOrgLayer; const state: PNetworkState);
procedure backward_reorg_layer(var l: TReOrgLayer; const state: PNetworkState);
{$ifdef GPU}
procedure forward_reorg_layer_gpu(l: TReOrgLayer; net: TNetwork);
procedure backward_reorg_layer_gpu(l: TReOrgLayer; net: TNetwork);
{$endif}
implementation

function make_reorg_layer(const batch, w, h, c, stride: longint; const reverse: boolean):TReOrgLayer;
var
    output_size: longint;
begin
    result := Default(TReOrgLayer);
    result.&type := ltREORG;
    result.batch := batch;
    result.stride := stride;
    result.h := h;
    result.w := w;
    result.c := c;
    if reverse then
        begin
            result.out_w := w * stride;
            result.out_h := h * stride;
            result.out_c := c div (stride * stride)
        end
    else
        begin
            result.out_w := w div stride;
            result.out_h := h div stride;
            result.out_c := c * (stride * stride)
        end;
    result.reverse := reverse;
    writeln(ErrOutput, format('reorg                    /%2d %4d x%4d x%4d -> %4d x%4d x%4d', [stride, w, h, c, result.out_w, result.out_h, result.out_c]));
    result.outputs := result.out_h * result.out_w * result.out_c;
    result.inputs := h * w * c;
    output_size := result.out_h * result.out_w * result.out_c * batch;
    result.output := TSingles.Create(output_size);
    result.delta := TSingles.Create(output_size);
    result.forward := forward_reorg_layer;
    result.backward := backward_reorg_layer;
{$ifdef GPU}
    result.forward_gpu := forward_reorg_layer_gpu;
    result.backward_gpu := backward_reorg_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, output_size);
    result.delta_gpu := cuda_make_array(result.delta, output_size);
{$endif}
end;

procedure resize_reorg_layer(var l: TReOrgLayer; const w, h: longint);
var
    stride: longint;
    c: longint;
    output_size: longint;
begin
    stride := l.stride;
    c := l.c;
    l.h := h;
    l.w := w;
    if l.reverse then
        begin
            l.out_w := w * stride;
            l.out_h := h * stride;
            l.out_c := c div (stride * stride)
        end
    else
        begin
            l.out_w := w div stride;
            l.out_h := h div stride;
            l.out_c := c * (stride * stride)
        end;
    l.outputs := l.out_h * l.out_w * l.out_c;
    l.inputs := l.outputs;
    output_size := l.outputs * l.batch;
    l.output.reAllocate(output_size);
    l.delta.reAllocate(output_size);
{$ifdef GPU}
    cuda_free(l.output_gpu);
    cuda_free(l.delta_gpu);
    l.output_gpu := cuda_make_array(l.output, output_size);
    l.delta_gpu := cuda_make_array(l.delta, output_size)
{$endif}
end;

procedure forward_reorg_layer(var l: TReOrgLayer; const state: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    //if l.reverse then
        reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, l.reverse, l.output)
    //else
    //    reorg_cpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output)
    ;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_reorg_layer(var l: TReOrgLayer; const state: PNetworkState);
begin
    //if l.reverse then
        reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, l.reverse, state.delta)
    //else
    //    reorg_cpu(l.delta, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta)
end;

{$ifdef GPU}
procedure forward_reorg_layer_gpu(l: layer; state: network_state);
begin
    if l.reverse then
        reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, l.output_gpu)
    else
        reorg_ongpu(state.input, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, l.output_gpu)
end;

procedure backward_reorg_layer_gpu(l: layer; state: network_state);
begin
    if l.reverse then
        reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 0, state.delta)
    else
        reorg_ongpu(l.delta_gpu, l.out_w, l.out_h, l.out_c, l.batch, l.stride, 1, state.delta)
end;
{$endif}

end.

