unit SAMLayer;

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
  SysUtils, lightnet, Activations;

type

  TSAMLayer = TLayer;

function make_sam_layer(const batch, index, w, h, c, w2, h2, c2: longint):TSAMLayer;
procedure resize_sam_layer(var l: TSAMLayer; const w, h: longint);
procedure forward_sam_layer(var l: TSamLayer; const state: PNetworkState);
procedure backward_sam_layer(var l: TSamlayer; const state: PNetworkState);

{$ifdef GPU}
procedure forward_sam_layer_gpu(const l: layer; state: network_state);
procedure backward_sam_layer_gpu(const l: layer; state: network_state);
{$endif}

implementation

function make_sam_layer(const batch, index, w, h, c, w2, h2, c2: longint):TSAMLayer;
begin
    writeln(ErrOutput, 'scale Layer: ', index);
    result := default(TSAMLayer);
    result.&type := ltSAM;
    result.batch := batch;
    result.w := w;
    result.h := h;
    result.c := c;
    result.out_w := w2;
    result.out_h := h2;
    result.out_c := c2;
    assert(result.out_c = result.c);
    assert((result.w = result.out_w) and (result.h = result.out_h));
    result.outputs := result.out_w * result.out_h * result.out_c;
    result.inputs := result.outputs;
    result.index := index;
    result.delta := TSingles.Create(result.outputs * batch);
    result.output := TSingles.Create(result.outputs * batch);
    result.forward := forward_sam_layer;
    result.backward := backward_sam_layer;
{$ifdef GPU}
    result.forward_gpu := forward_sam_layer_gpu;
    result.backward_gpu := backward_sam_layer_gpu;
    result.delta_gpu := cuda_make_array(result.delta, result.outputs * batch);
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
{$endif}
end;

procedure resize_sam_layer(var l: TSAMLayer; const w, h: longint);
begin
    l.out_w := w;
    l.out_h := h;
    l.outputs := l.out_w * l.out_h * l.out_c;
    l.inputs := l.outputs;
    l.delta.ReAllocate( l.outputs * l.batch );
    l.output.ReAllocate( l.outputs * l.batch );
{$ifdef GPU}
    cuda_free(l.output_gpu);
    cuda_free(l.delta_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch);
    l.delta_gpu := cuda_make_array(l.delta, l.outputs * l.batch)
{$endif}
end;

procedure forward_sam_layer(var l: TSamLayer; const state: PNetworkState);
var
    size: longint;
    from_output: PSingle;
    i: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    size := l.batch * l.out_c * l.out_w * l.out_h;
    from_output := state.net.layers[l.index].output;
    // todo Parallelize [forward_sam_layer]
    for i := 0 to size -1 do
        l.output[i] := state.input[i] * from_output[i];
    activate_array(l.output, l.outputs * l.batch, l.activation);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_sam_layer(var l: TSamlayer; const state: PNetworkState);
var
    size: longint;
    from_output: PSingle;
    from_delta: PSingle;
    i: longint;
begin
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    size := l.batch * l.out_c * l.out_w * l.out_h;
    from_output := state.net.layers[l.index].output;
    from_delta := state.net.layers[l.index].delta;
    // todo Parallelize [backward_sam_layer]
    for i := 0 to size -1 do
        begin
            state.delta[i] := state.delta[i] + (l.delta[i] * from_output[i]);
            from_delta[i] := state.input[i] * l.delta[i]
        end
end;

{$ifdef GPU}
procedure forward_sam_layer_gpu(const l: layer; state: network_state);
var
    size: longint;
    channel_size: longint;
begin
    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := 1;
    sam_gpu(state.net.layers[l.index].output_gpu, size, channel_size, state.input, l.output_gpu);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation)
end;

procedure backward_sam_layer_gpu(const l: layer; state: network_state);
var
    size: longint;
    channel_size: longint;
begin
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    size := l.batch * l.out_c * l.out_w * l.out_h;
    channel_size := 1;
    float * from_output := state.net.layers[l.index].output_gpu;
    float * from_delta := state.net.layers[l.index].delta_gpu;
    backward_sam_gpu(l.delta_gpu, size, channel_size, state.input, from_delta, from_output, state.delta)
end;
{$endif}

end.

