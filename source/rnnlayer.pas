unit RNNLayer;

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
  SysUtils, lightnet, ConnectedLayer, blas;

type
  PRNNLayer = ^TRNNLayer;
  TRNNLayer = TLayer;

function make_rnn_layer( batch:longint; const inputs, hidden, outputs, steps: longint; const activation: TActivation; const batch_normalize:boolean; const log: longint):TRNNLayer;
procedure update_rnn_layer(const l: TRNNlayer; const arg: TUpdateArgs);
procedure forward_rnn_layer(var l: TRNNLayer; const state: PNetworkState);
procedure backward_rnn_layer(var l: TRNNLayer; const state: PNetworkState);

{$ifdef GPU}
procedure pull_rnn_layer(l: layer);
procedure push_rnn_layer(l: layer);
procedure update_rnn_layer_gpu(l: layer; batch: longint; learning_rate: single; momentum: single; decay: single; loss_scale: single);
procedure forward_rnn_layer_gpu(l: layer; state: network_state);
procedure backward_rnn_layer_gpu(l: layer; state: network_state);
{$endif}

implementation
uses math;

procedure increment_layer(const l: PRNNLayer; const steps: longint);
var
    num: longint;
begin
    num := l.outputs * l.batch * steps;
    l.output := l.output + num;
    l.delta := l.delta + num;
    l.x := l.x + num;
    l.x_norm := l.x_norm + num;
{$ifdef GPU}
    l.output_gpu := l.output_gpu + num;
    l.delta_gpu := l.delta_gpu + num;
    l.x_gpu := l.x_gpu + num;
    l.x_norm_gpu := l.x_norm_gpu + num
{$endif}
end;

function make_rnn_layer( batch:longint; const inputs, hidden, outputs, steps: longint; const activation: TActivation; const batch_normalize:boolean; const log: longint):TRNNLayer;
begin
    writeln(ErrOutput, format('RNN Layer: %d inputs, %d outputs', [inputs, outputs]));
    batch := batch div steps;
    result := default(TRNNLayer);
    result.batch := batch;
    result.&type := ltRNN;
    result.steps := steps;
    result.hidden := hidden;
    result.inputs := inputs;
    result.out_w := 1;
    result.out_h := 1;
    result.out_c := outputs;
    result.state := TSingles.Create(batch * hidden * (steps+1));

    setLength(result.input_layer, 1);
    write(ErrOutput, #9#9);
    result.input_layer[0] := make_connected_layer(batch, steps, inputs, hidden, activation, batch_normalize);
    result.input_layer[0].batch := batch;
    if result.workspace_size < result.input_layer[0].workspace_size then
        result.workspace_size := result.input_layer[0].workspace_size;

    setLength(result.self_layer, 1);
    write(ErrOutput, #9#9);
    result.self_layer[0] := make_connected_layer(batch, steps, hidden, hidden, TActivation(ifthen((log = 2), ord(acLOGGY), (ifthen(log = 1, ord(acLOGISTIC), ord(activation) ) ) ) ), batch_normalize);
    result.self_layer[0].batch := batch;
    if result.workspace_size < result.self_layer[0].workspace_size then
        result.workspace_size := result.self_layer[0].workspace_size;

    setLength(result.output_layer ,1);
    write(ErrOutput, #9#9);
    result.output_layer[0] := make_connected_layer(batch, steps, hidden, outputs, activation, batch_normalize);
    result.output_layer[0].batch := batch;
    if result.workspace_size < result.output_layer[0].workspace_size then
        result.workspace_size := result.output_layer[0].workspace_size;

    result.outputs := outputs;
    result.output := result.output_layer[0].output;
    result.delta := result.output_layer[0].delta;
    result.forward := forward_rnn_layer;
    result.backward := backward_rnn_layer;
    result.update := update_rnn_layer;
{$ifdef GPU}
    l.forward_gpu := forward_rnn_layer_gpu;
    l.backward_gpu := backward_rnn_layer_gpu;
    l.update_gpu := update_rnn_layer_gpu;
    l.state_gpu := cuda_make_array(l.state, batch * hidden * (steps+1));
    l.output_gpu := l.output_layer.output_gpu;
    l.delta_gpu := l.output_layer.delta_gpu;
{$endif}
end;

procedure update_rnn_layer(const l: TRNNlayer; const arg: TUpdateArgs);
begin
    update_connected_layer( l.input_layer[0], arg);
    update_connected_layer( l.self_layer[0], arg);
    update_connected_layer( l.output_layer[0], arg)
end;

procedure forward_rnn_layer(var l: TRNNLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    input_layer: TConnectedLayer;
    self_layer: TConnectedLayer;
    output_layer: TConnectedLayer;
    old_state: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    s := default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    input_layer :=  l.input_layer[0];
    self_layer :=  l.self_layer[0];
    output_layer :=  l.output_layer[0];
    fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
    fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
    if state.train then
        fill_cpu(l.hidden * l.batch, 0, l.state, 1);
    for i := 0 to l.steps -1 do
        begin
            s.input := state.input;
            forward_connected_layer(input_layer, @s);
            s.input := l.state;
            forward_connected_layer(self_layer, @s);
            old_state := l.state;
            if state.train then
                l.state := l.state + (l.hidden * l.batch);
            if l.shortcut then
                copy_cpu(l.hidden * l.batch, old_state, 1, l.state, 1)
            else
                fill_cpu(l.hidden * l.batch, 0, l.state, 1);
            axpy_cpu(l.hidden * l.batch, 1, input_layer.output, 1, l.state, 1);
            axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);
            s.input := l.state;
            forward_connected_layer(output_layer, @s);
            state.input := state.input + (l.inputs * l.batch);
            increment_layer( @input_layer, 1);
            increment_layer( @self_layer, 1);
            increment_layer( @output_layer, 1)
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_rnn_layer(var l: TRNNLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    input_layer: TConnectedLayer;
    self_layer: TConnectedLayer;
    output_layer: TConnectedLayer;
begin
    s := default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    input_layer :=  l.input_layer[0];
    self_layer :=  l.self_layer[0];
    output_layer :=  l.output_layer[0];
    increment_layer( @input_layer, l.steps-1);
    increment_layer( @self_layer, l.steps-1);
    increment_layer( @output_layer, l.steps-1);
    l.state := l.state + (l.hidden * l.batch * l.steps);
    for i := l.steps-1 downto 0 do begin
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);
        s.input := l.state;
        s.delta := self_layer.delta;
        backward_connected_layer(output_layer, @s);
        l.state := l.state - (l.hidden * l.batch);
        s.input := l.state;
        s.delta := self_layer.delta-l.hidden * l.batch;
        if i = 0 then
            s.delta := nil;
        backward_connected_layer(self_layer, @s);
        copy_cpu(l.hidden * l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0) and l.shortcut then
            axpy_cpu(l.hidden * l.batch, 1, self_layer.delta, 1, self_layer.delta-l.hidden * l.batch, 1);
        s.input := state.input+i * l.inputs * l.batch;
        if assigned(state.delta) then
            s.delta := state.delta+i * l.inputs * l.batch
        else
            s.delta := nil;
        backward_connected_layer(input_layer, @s);
        increment_layer( @input_layer, -1);
        increment_layer( @self_layer, -1);
        increment_layer( @output_layer, -1);
    end
end;

{$ifdef GPU}
procedure pull_rnn_layer(l: layer);
begin
    pull_connected_layer( * (l.input_layer));
    pull_connected_layer( * (l.self_layer));
    pull_connected_layer( * (l.output_layer))
end;

procedure push_rnn_layer(l: layer);
begin
    push_connected_layer( * (l.input_layer));
    push_connected_layer( * (l.self_layer));
    push_connected_layer( * (l.output_layer))
end;

procedure update_rnn_layer_gpu(l: layer; batch: longint; learning_rate: single; momentum: single; decay: single; loss_scale: single);
begin
    update_connected_layer_gpu( * (l.input_layer), batch, learning_rate, momentum, decay, loss_scale);
    update_connected_layer_gpu( * (l.self_layer), batch, learning_rate, momentum, decay, loss_scale);
    update_connected_layer_gpu( * (l.output_layer), batch, learning_rate, momentum, decay, loss_scale)
end;

procedure forward_rnn_layer_gpu(l: layer; state: network_state);
var
    s: network_state;
    i: longint;
    input_layer: layer;
    self_layer: layer;
    output_layer: layer;
    old_state: PSingle;
begin
    s := [0];
    s.train := state.train;
    s.workspace := state.workspace;
    input_layer :=  * (l.input_layer);
    self_layer :=  * (l.self_layer);
    output_layer :=  * (l.output_layer);
    fill_ongpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
    fill_ongpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
    fill_ongpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
    if state.train then
        fill_ongpu(l.hidden * l.batch, 0, l.state_gpu, 1);
    for i := 0 to l.steps -1 do
        begin
            s.input := state.input;
            forward_connected_layer_gpu(input_layer, s);
            s.input := l.state_gpu;
            forward_connected_layer_gpu(self_layer, s);
            old_state := l.state_gpu;
            if state.train then
                l.state_gpu := l.state_gpu + (l.hidden * l.batch);
            if l.shortcut then
                copy_ongpu(l.hidden * l.batch, old_state, 1, l.state_gpu, 1)
            else
                fill_ongpu(l.hidden * l.batch, 0, l.state_gpu, 1);
            axpy_ongpu(l.hidden * l.batch, 1, input_layer.output_gpu, 1, l.state_gpu, 1);
            axpy_ongpu(l.hidden * l.batch, 1, self_layer.output_gpu, 1, l.state_gpu, 1);
            s.input := l.state_gpu;
            forward_connected_layer_gpu(output_layer, s);
            state.input := state.input + (l.inputs * l.batch);
            increment_layer( and input_layer, 1);
            increment_layer( and self_layer, 1);
            increment_layer( and output_layer, 1)
        end
end;

procedure backward_rnn_layer_gpu(l: layer; state: network_state);
var
    s: network_state;
    i: longint;
    input_layer: layer;
    self_layer: layer;
    output_layer: layer;
begin
    s := [0];
    s.train := state.train;
    s.workspace := state.workspace;
    input_layer :=  * (l.input_layer);
    self_layer :=  * (l.self_layer);
    output_layer :=  * (l.output_layer);
    increment_layer( and input_layer, l.steps-1);
    increment_layer( and self_layer, l.steps-1);
    increment_layer( and output_layer, l.steps-1);
    l.state_gpu := l.state_gpu + (l.hidden * l.batch * l.steps);
    i := l.steps-1;
    while i >= 0 do begin
        s.input := l.state_gpu;
        s.delta := self_layer.delta_gpu;
        backward_connected_layer_gpu(output_layer, s);
        l.state_gpu := l.state_gpu - (l.hidden * l.batch);
        copy_ongpu(l.hidden * l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
        s.input := l.state_gpu;
        s.delta := self_layer.delta_gpu-l.hidden * l.batch;
        if i = 0 then
            s.delta := 0;
        backward_connected_layer_gpu(self_layer, s);
        if (i > 0) and l.shortcut then
            axpy_ongpu(l.hidden * l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu-l.hidden * l.batch, 1);
        s.input := state.input+i * l.inputs * l.batch;
        if state.delta then
            s.delta := state.delta+i * l.inputs * l.batch
        else
            s.delta := 0;
        backward_connected_layer_gpu(input_layer, s);
        increment_layer( and input_layer, -1);
        increment_layer( and self_layer, -1);
        increment_layer( and output_layer, -1);
        &ced(i)
    end
end;
{$endif}

end.

