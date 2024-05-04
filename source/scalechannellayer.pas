unit ScaleChannelLayer;

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
  SysUtils, Darknet, Activations;

type
  TScaleChannelLayer = TLayer;

function make_scale_channels_layer(const batch, index, w, h, c, w2, h2, c2, scale_wh: longint):TScaleChannelLayer;
procedure resize_scale_channels_layer(var l: TScaleChannelLayer; const net: PNetwork);
procedure forward_scale_channels_layer(var l: TScaleChannelLayer; const state: PNetworkState);
procedure backward_scale_channels_layer(var l: TScaleChannelLayer; const state: PNetworkState);

{$ifdef GPU}
procedure forward_scale_channels_layer_gpu(const l: layer; state: network_state);
procedure backward_scale_channels_layer_gpu(const l: layer; state: network_state);
{$endif}

implementation

function make_scale_channels_layer(const batch, index, w, h, c, w2, h2, c2, scale_wh: longint):TScaleChannelLayer;
begin
    writeln(ErrOutput, 'scale Layer: ', index);
    result := Default(TScaleChannelLayer);
    result.&type := ltScaleChannels;
    result.batch := batch;
    result.scale_wh := scale_wh;
    result.w := w;
    result.h := h;
    result.c := c;
    if result.scale_wh=0 then
        assert((w = 1) and (h = 1))
    else
        assert(c = 1);
    result.out_w := w2;
    result.out_h := h2;
    result.out_c := c2;
    if result.scale_wh=0 then
        assert(result.out_c = result.c)
    else
        assert((result.out_w = result.w) and (result.out_h = result.h));
    result.outputs := result.out_w * result.out_h * result.out_c;
    result.inputs := result.outputs;
    result.index := index;
    result.delta := TSingles.Create(result.outputs * batch);//single(xcalloc(result.outputs * batch, sizeof(float)));
    result.output := TSingles.Create(result.outputs * batch);//single(xcalloc(result.outputs * batch, sizeof(float)));
    result.forward := forward_scale_channels_layer;
    result.backward := backward_scale_channels_layer;
{$ifdef GPU}
    result.forward_gpu := forward_scale_channels_layer_gpu;
    result.backward_gpu := backward_scale_channels_layer_gpu;
    result.delta_gpu := cuda_make_array(result.delta, result.outputs * batch);
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
{$endif}

end;

procedure resize_scale_channels_layer(var l: TScaleChannelLayer; const net: PNetwork);
var
    first: TLayer;
begin
    first := net.layers[l.index];
    l.out_w := first.out_w;
    l.out_h := first.out_h;
    l.outputs := l.out_w * l.out_h * l.out_c;
    l.inputs := l.outputs;
    l.delta.reAllocate(l.outputs * l.batch);
    l.output.reAllocate(l.outputs * l.batch);
{$ifdef GPU}
    cuda_free(l.output_gpu);
    cuda_free(l.delta_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch);
    l.delta_gpu := cuda_make_array(l.delta, l.outputs * l.batch)
{$endif}
end;

procedure forward_scale_channels_layer(var l: TScaleChannelLayer; const state: PNetworkState);
var
    size: longint;
    channel_size: longint;
    batch_size: longint;
    from_output: PSingle;
    i: longint;
    input_index: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := l.out_w * l.out_h;
    batch_size := l.out_c * l.out_w * l.out_h;
    from_output := state.net.layers[l.index].output;
    if l.scale_wh<>0 then
        begin
            // todo Parallelize [forward_scale_channels_layer]
            for i := 0 to size -1 do
                begin
                    input_index := i mod channel_size+(i div batch_size) * channel_size;
                    l.output[i] := state.input[input_index] * from_output[i]
                end
        end
    else
        begin
            //todo parallelize [forward_scale_channels_layer2]
            for i := 0 to size -1 do
                l.output[i] := state.input[i div channel_size] * from_output[i]
        end;
    activate_array(l.output, l.outputs * l.batch, l.activation) ;

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_scale_channels_layer(var l: TScaleChannelLayer; const state: PNetworkState);
var
    size: longint;
    channel_size: longint;
    batch_size: longint;
    from_output: PSingle;
    from_delta: PSingle;
    i: longint;
    input_index: longint;
begin
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := l.out_w * l.out_h;
    batch_size := l.out_c * l.out_w * l.out_h;
    from_output := state.net.layers[l.index].output;
    from_delta := state.net.layers[l.index].delta;
    if l.scale_wh<>0 then
        begin
            // todo Parallelize [backward_scale_channels_layer 1]
            for i := 0 to size -1 do
                begin
                    input_index := i mod channel_size+(i div batch_size) * channel_size;
                    state.delta[input_index] := state.delta[input_index] + (l.delta[i] * from_output[i]);
                    from_delta[i] := from_delta[i] + (state.input[input_index] * l.delta[i])
                end
        end
    else
        begin
            // todo Parallelize [backward_scale_channels_layer 2]
            for i := 0 to size -1 do
                begin
                    state.delta[i div channel_size] := state.delta[i div channel_size] + (l.delta[i] * from_output[i]);
                    from_delta[i] := from_delta[i] + (state.input[i div channel_size] * l.delta[i])
                end
        end
end;

{$ifdef GPU}
procedure forward_scale_channels_layer_gpu(const l: layer; state: network_state);
var
    size: longint;
    channel_size: longint;
    batch_size: longint;
begin
    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := l.out_w * l.out_h;
    batch_size := l.out_c * l.out_w * l.out_h;
    scale_channels_gpu(state.net.layers[l.index].output_gpu, size, channel_size, batch_size, l.scale_wh, state.input, l.output_gpu);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation)
end;

procedure backward_scale_channels_layer_gpu(const l: layer; state: network_state);
var
    size: longint;
    channel_size: longint;
    batch_size: longint;
    from_output: PSingle;
    from_delta: PSingle;
begin
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := l.out_w * l.out_h;
    batch_size := l.out_c * l.out_w * l.out_h;
    from_output := state.net.layers[l.index].output_gpu;
    from_delta := state.net.layers[l.index].delta_gpu;
    backward_scale_channels_gpu(l.delta_gpu, size, channel_size, batch_size, l.scale_wh, state.input, from_delta, from_output, state.delta)
end;
{$endif}


end.

