unit CRNNLayer;

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
  SysUtils, lightnet, blas, ConvolutionalLayer;

type TCRNNLayer = TLayer;

function make_crnn_layer(batch: longint; const h, w, c, hidden_filters, output_filters, groups, steps, size, stride, dilation, pad: longint; const activation: TActivation; const batch_normalize, xnor, train: boolean): TCRNNLayer;
procedure resize_crnn_layer(var l: TCRNNLayer; const w, h: longint);
procedure free_state_crnn(const l: TCRNNLayer);
procedure update_crnn_layer(const l: TCRNNLayer; const arg: TUpdateArgs);
procedure forward_crnn_layer(var l: TCRNNLayer; const state: PNetworkState);
procedure backward_crnn_layer(var l: TCRNNLayer; const state: PNetworkState);
{$ifdef GPU}
procedure pull_crnn_layer(l: TCRNNLayer);
procedure push_crnn_layer(l: TCRNNLayer);
procedure update_crnn_layer_gpu(l: TCRNNLayer; a: TUpdateArgs);
procedure forward_crnn_layer_gpu(l: TCRNNLayer; net: TNetwork);
procedure backward_crnn_layer_gpu(l: TCRNNLayer; net: TNetwork);
{$endif}

implementation

procedure increment_layer(var l: TCRNNLayer; const steps: longint);
var
    num: longint;
begin
    // todo find an alternative for {$define NO_POINTERS} case
    num := l.outputs * l.batch * steps;
    //l.output := l.output + num;
    inc(l.output, num);
    //l.delta := l.delta + num;
    inc(l.delta, num);
    //l.x := l.x + num;
    inc(l.x,num);
    //l.x_norm := l.x_norm + num;
    inc(l.x_norm, num);
  {$ifdef GPU}
    l.output_gpu := l.output_gpu + num;
    l.delta_gpu := l.delta_gpu + num;
    l.x_gpu := l.x_gpu + num ;
    l.x_norm_gpu := l.x_norm_gpu + num
  {$endif}
end;

function make_crnn_layer(batch: longint; const h, w, c, hidden_filters,
  output_filters, groups, steps, size, stride, dilation, pad: longint;
  const activation: TActivation; const batch_normalize, xnor, train: boolean
  ): TCRNNLayer;
begin
    writeln(format('CRNN Layer: %d x %d x %d image, %d filters', [h, w, c, output_filters]));
    batch := batch div steps;
    result := Default(TCRNNLayer);
    result.train := train;
    result.batch := batch;
    result.&type := ltCRNN;
    result.steps := steps;
    result.size := size;
    result.stride := stride;
    result.dilation := dilation;
    result.pad := pad;
    result.h := h;
    result.w := w;
    result.c := c;
    result.groups := groups;
    result.out_c := output_filters;
    result.inputs := h * w * c;
    result.hidden := h * w * hidden_filters;
    result.xnor := xnor;
    result.state:=TSingles.Create(result.hidden * batch * (steps+1));

    setLength(result.input_layer, 1);
    result.input_layer[0] := make_convolutional_layer(batch, steps, h, w, hidden_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.input_layer[0].batch := batch;
    if result.workspace_size < result.input_layer[0].workspace_size then result.workspace_size := result.input_layer[0].workspace_size;

    setLength(result.self_layer,1);
    result.self_layer[0] := make_convolutional_layer(batch, steps, h, w, hidden_filters, hidden_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.self_layer[0].batch := batch;
    if result.workspace_size < result.self_layer[0].workspace_size then result.workspace_size := result.self_layer[0].workspace_size;

    setLength(result.output_layer,1);
    result.output_layer[0] := make_convolutional_layer(batch, steps, h, w, hidden_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.output_layer[0].batch := batch;
    if result.workspace_size < result.output_layer[0].workspace_size then result.workspace_size := result.output_layer[0].workspace_size;

    result.out_h := result.output_layer[0].out_h;
    result.out_w := result.output_layer[0].out_w;
    result.outputs := result.output_layer[0].outputs;

    assert(result.input_layer[0].outputs = result.self_layer[0].outputs);
    assert(result.input_layer[0].outputs = result.output_layer[0].inputs);

    result.output := result.output_layer[0].output;
    result.delta := result.output_layer[0].delta;

    result.forward := forward_crnn_layer;
    result.backward := backward_crnn_layer;
    result.update := update_crnn_layer;
  {$ifdef GPU}
    result.forward_gpu := forward_crnn_layer_gpu;
    result.backward_gpu := backward_crnn_layer_gpu;
    result.update_gpu := update_crnn_layer_gpu;
    result.state_gpu := cuda_make_array(result.state, result.hidden * batch * (steps+1));
    result.output_gpu := result.output_layer.output_gpu;
    result.delta_gpu := result.output_layer.delta_gpu;
  {$endif}

end;

procedure resize_crnn_layer(var l: TCRNNLayer; const w, h: longint);
var
    hidden_filters: longint;
begin
    resize_convolutional_layer(l.input_layer[0], w, h);
    if l.workspace_size < l.input_layer[0].workspace_size then
        l.workspace_size := l.input_layer[0].workspace_size;
    resize_convolutional_layer(l.self_layer[0], w, h);
    if l.workspace_size < l.self_layer[0].workspace_size then
        l.workspace_size := l.self_layer[0].workspace_size;
    resize_convolutional_layer(l.output_layer[0], w, h);
    if l.workspace_size < l.output_layer[0].workspace_size then
        l.workspace_size := l.output_layer[0].workspace_size;
    l.output := l.output_layer[0].output;
    l.delta := l.output_layer[0].delta;
    hidden_filters := l.self_layer[0].c;
    l.w := w;
    l.h := h;
    l.inputs := h * w * l.c;
    l.hidden := h * w * hidden_filters;
    l.out_h := l.output_layer[0].out_h;
    l.out_w := l.output_layer[0].out_w;
    l.outputs := l.output_layer[0].outputs;
    assert(l.input_layer[0].inputs = l.inputs);
    assert(l.self_layer[0].inputs = l.hidden);
    assert(l.input_layer[0].outputs = l.self_layer[0].outputs);
    assert(l.input_layer[0].outputs = l.output_layer[0].inputs);
    l.state.reAllocate(l.batch * l.hidden * (l.steps+1));
{$ifdef GPU}
    if l.state_gpu then
        cudaFree(l.state_gpu);
    l.state_gpu := cuda_make_array(l.state, l.batch * l.hidden * (l.steps+1));
    l.output_gpu := l.output_layer.output_gpu;
    l.delta_gpu := l.output_layer.delta_gpu
{$endif}
end;

procedure free_state_crnn(const l: TCRNNLayer);
var
    i: longint;
begin
    for i := 0 to l.outputs * l.batch -1 do
        l.self_layer[0].output[i] := rand_uniform(-1, 1);
{$ifdef GPU}
    cuda_push_array(l.self_layer.output_gpu, l.self_layer.output, l.outputs * l.batch)
{$endif}
end;

procedure update_crnn_layer(const l: TCRNNLayer; const arg :TUpdateArgs);
begin
    update_convolutional_layer(l.input_layer[0], arg);
    update_convolutional_layer(l.self_layer[0], arg);
    update_convolutional_layer(l.output_layer[0], arg)
end;

procedure forward_crnn_layer(var l: TCRNNLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    input_layer, self_layer, output_layer: TConvolutionalLayer;
    old_state: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    s := Default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    input_layer :=  l.input_layer[0];
    self_layer :=  l.self_layer[0];
    output_layer :=  l.output_layer[0];
    if state.train then
        begin
            fill_cpu(l.outputs * l.batch * l.steps, 0, output_layer.delta, 1);
            fill_cpu(l.hidden * l.batch * l.steps, 0, self_layer.delta, 1);
            fill_cpu(l.hidden * l.batch * l.steps, 0, input_layer.delta, 1);
            fill_cpu(l.hidden * l.batch, 0, l.state, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            s.input := state.input;
            forward_convolutional_layer(input_layer, @s);
            s.input := l.state;
            forward_convolutional_layer(self_layer, @s);
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
            forward_convolutional_layer(output_layer, @s);
            state.input := state.input + (l.inputs * l.batch);
            increment_layer(input_layer, 1);
            increment_layer(self_layer, 1);
            increment_layer(output_layer, 1)
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_crnn_layer(var l: TCRNNLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    input_layer, self_layer, output_layer: TConvolutionalLayer;
begin
    s := Default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    input_layer :=  l.input_layer[0];
    self_layer :=  l.self_layer[0];
    output_layer :=  l.output_layer[0];
    increment_layer(input_layer, l.steps-1);
    increment_layer(self_layer, l.steps-1);
    increment_layer(output_layer, l.steps-1);
    l.state := l.state + (l.hidden * l.batch * l.steps);
    i := l.steps-1;
    while i >= 0 do begin
        copy_cpu(l.hidden * l.batch, input_layer.output, 1, l.state, 1);
        axpy_cpu(l.hidden * l.batch, 1, self_layer.output, 1, l.state, 1);
        s.input := l.state;
        s.delta := self_layer.delta;
        backward_convolutional_layer(output_layer, @s);
        l.state := l.state - (l.hidden * l.batch);
        s.input := l.state;
        s.delta := self_layer.delta-l.hidden * l.batch;
        if i = 0 then
            s.delta := 0;
        backward_convolutional_layer(self_layer, @s);
        copy_cpu(l.hidden * l.batch, self_layer.delta, 1, input_layer.delta, 1);
        if (i > 0) and l.shortcut then
            axpy_cpu(l.hidden * l.batch, 1, self_layer.delta, 1, self_layer.delta-l.hidden * l.batch, 1);
        s.input := state.input+i * l.inputs * l.batch;
        if assigned(state.delta) then
            s.delta := state.delta+i * l.inputs * l.batch
        else
            s.delta := 0;
        backward_convolutional_layer(input_layer, @s);
        increment_layer(input_layer, -1);
        increment_layer(self_layer, -1);
        increment_layer(output_layer, -1);
        dec(i)
    end
end;
{$ifdef GPU}
procedure pull_crnn_layer(l: layer);
begin
    pull_convolutional_layer( * (l.input_layer));
    pull_convolutional_layer( * (l.self_layer));
    pull_convolutional_layer( * (l.output_layer))
end;

procedure push_crnn_layer(l: layer);
begin
    push_convolutional_layer( * (l.input_layer));
    push_convolutional_layer( * (l.self_layer));
    push_convolutional_layer( * (l.output_layer))
end;

procedure update_crnn_layer_gpu(l: layer; batch: longint; learning_rate: single; momentum: single; decay: single; loss_scale: single);
begin
    update_convolutional_layer_gpu( * (l.input_layer), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu( * (l.self_layer), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu( * (l.output_layer), batch, learning_rate, momentum, decay, loss_scale)
end;

procedure forward_crnn_layer_gpu(l: layer; state: network_state);
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
    s.net := state.net;
    if not state.train then
        s.index := state.index;
    input_layer :=  * (l.input_layer);
    self_layer :=  * (l.self_layer);
    output_layer :=  * (l.output_layer);
    if state.train then
        begin
            fill_ongpu(l.outputs * l.batch * l.steps, 0, output_layer.delta_gpu, 1);
            fill_ongpu(l.hidden * l.batch * l.steps, 0, self_layer.delta_gpu, 1);
            fill_ongpu(l.hidden * l.batch * l.steps, 0, input_layer.delta_gpu, 1);
            fill_ongpu(l.hidden * l.batch, 0, l.state_gpu, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            s.input := state.input;
            forward_convolutional_layer_gpu(input_layer, s);
            s.input := l.state_gpu;
            forward_convolutional_layer_gpu(self_layer, s);
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
            forward_convolutional_layer_gpu(output_layer, s);
            state.input := state.input + (l.inputs * l.batch);
            increment_layer( and input_layer, 1);
            increment_layer( and self_layer, 1);
            increment_layer( and output_layer, 1)
        end
end;

procedure backward_crnn_layer_gpu(l: layer; state: network_state);
var
    s: network_state;
    i: longint;
    input_layer: layer;
    self_layer: layer;
    output_layer: layer;
    init_state_gpu: PSingle;
begin
    s := [0];
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    input_layer :=  * (l.input_layer);
    self_layer :=  * (l.self_layer);
    output_layer :=  * (l.output_layer);
    increment_layer( and input_layer, l.steps-1);
    increment_layer( and self_layer, l.steps-1);
    increment_layer( and output_layer, l.steps-1);
    init_state_gpu := l.state_gpu;
    l.state_gpu := l.state_gpu + (l.hidden * l.batch * l.steps);
    i := l.steps-1;
    while i >= 0 do begin
        s.input := l.state_gpu;
        s.delta := self_layer.delta_gpu;
        backward_convolutional_layer_gpu(output_layer, s);
        l.state_gpu := l.state_gpu - (l.hidden * l.batch);
        copy_ongpu(l.hidden * l.batch, self_layer.delta_gpu, 1, input_layer.delta_gpu, 1);
        s.input := l.state_gpu;
        s.delta := self_layer.delta_gpu-l.hidden * l.batch;
        if i = 0 then
            s.delta := 0;
        backward_convolutional_layer_gpu(self_layer, s);
        if (i > 0) and l.shortcut then
            axpy_ongpu(l.hidden * l.batch, 1, self_layer.delta_gpu, 1, self_layer.delta_gpu-l.hidden * l.batch, 1);
        s.input := state.input+i * l.inputs * l.batch;
        if state.delta then
            s.delta := state.delta+i * l.inputs * l.batch
        else
            s.delta := 0;
        backward_convolutional_layer_gpu(input_layer, s);
        if state.net.try_fix_nan then
            begin
                fix_nan_and_inf(output_layer.delta_gpu, output_layer.inputs * output_layer.batch);
                fix_nan_and_inf(self_layer.delta_gpu, self_layer.inputs * self_layer.batch);
                fix_nan_and_inf(input_layer.delta_gpu, input_layer.inputs * input_layer.batch)
            end;
        increment_layer( and input_layer, -1);
        increment_layer( and self_layer, -1);
        increment_layer( and output_layer, -1);
        &ced(i)
    end;
    fill_ongpu(l.hidden * l.batch, 0, init_state_gpu, 1)
end;
{$endif}

end.

