unit ShortcutLayer;

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
  SysUtils, darknet, Activations, gemm, blas;

type
  PShortcutLayer = ^TShortcutLayer;
  TShortcutLayer = TLayer;

// note Shortcut is just a tensor [add] layer

function make_shortcut_layer(const batch, n: longint; const input_layers, input_sizes: TArray<longint>; const w, h, c: longint;
      const layers_output, layers_delta:TArray<PSingle>; const layers_output_gpu, layers_delta_gpu: PPsingle; const weights_type: TWeightsType;
      const weights_normalization: TWeightsNormalization; const activation: TActivation; const train: boolean):TShortcutLayer;
procedure resize_shortcut_layer(var l: TShorTcutLayer; const w, h: longint; const net: PNetwork);
procedure forward_shortcut_layer(var l: TShortcutLayer; const state: PNetworkState);
procedure backward_shortcut_layer(var l: TShortcutLayer; const state: PNetworkState);
procedure update_shortcut_layer(const l: TShortcutLayer; const arg :TUpdateArgs);

{$ifdef GPU}
procedure forward_shortcut_layer_gpu(const l: layer; state: network_state);
procedure backward_shortcut_layer_gpu(const l: layer; state: network_state);
procedure update_shortcut_layer_gpu(l: layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
procedure pull_shortcut_layer(l: layer);
procedure push_shortcut_layer(l: layer);

{$endif}

implementation

function make_shortcut_layer(const batch, n: longint; const input_layers,
  input_sizes: TArray<longint>; const w, h, c: longint; const layers_output,
  layers_delta: TArray<PSingle>; const layers_output_gpu, layers_delta_gpu: PPsingle;
  const weights_type: TWeightsType;
  const weights_normalization: TWeightsNormalization;
  const activation: TActivation; const train: boolean): TShortcutLayer;
var
    i: longint;
    scale: single;
begin
    write(ErrOutput, 'Shortcut Layer: ');
    for i := 0 to n -1 do
        write(ErrOutput, format('%d, ', [input_layers[i]]));
    result := Default(TShortcutLayer);
    result.train := train;
    result.&type := ltSHORTCUT;
    result.batch := batch;
    result.activation := activation;
    result.n := n;
    result.input_layers := input_layers;
    result.input_sizes := input_sizes;
    result.layers_output := layers_output;
    result.layers_delta := layers_delta;
    result.weights_type := weights_type;
    result.weights_normalization := weights_normalization;
    result.learning_rate_scale := 1;
    result.w := w; result.out_w := w;
    result.h := h; result.out_h := h;
    result.c := c; result.out_c := c;
    result.outputs := w * h * c;
    result.inputs := result.outputs;
    result.index := result.input_layers[0];
    if train then
        result.delta := TSingles.Create(result.outputs * batch);
    result.output := TSingles.Create(result.outputs * batch);
    result.nweights := 0;
    if result.weights_type = wtPER_FEATURE then
        result.nweights := (result.n+1)
    else
        if result.weights_type = wtPER_CHANNEL then
            result.nweights := (result.n+1) * result.c;
    if result.nweights > 0 then
        begin
            result.weights := TSingles.Create(result.nweights);
            scale := sqrt(2 / result.nweights);
            for i := 0 to result.nweights -1 do
                result.weights[i] := 1;
            if train then
                result.weight_updates := TSingles.Create(result.nweights);
            result.update := update_shortcut_layer
        end;
    result.forward := forward_shortcut_layer;
    result.backward := backward_shortcut_layer;
{$ifndef GPU}
    if (result.activation = acSWISH) or (result.activation = acMISH) then
        result.activation_input := TSingles.Create(result.batch * result.outputs);
{$endif}
{$ifdef GPU}
    if result.activation = acSWISH or result.activation = acMISH then
        result.activation_input_gpu := cuda_make_array(result.activation_input, result.batch * result.outputs);
    result.forward_gpu := forward_shortcut_layer_gpu;
    result.backward_gpu := backward_shortcut_layer_gpu;
    if result.nweights > 0 then
        begin
            result.update_gpu := update_shortcut_layer_gpu;
            result.weights_gpu := cuda_make_array(result.weights, result.nweights);
            if train then
                result.weight_updates_gpu := cuda_make_array(result.weight_updates, result.nweights)
        end;
    if train then
        result.delta_gpu := cuda_make_array(result.delta, result.outputs * batch);
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
    result.input_sizes_gpu := cuda_make_int_array_new_api(input_sizes, result.n);
    result.layers_output_gpu := single(cuda_make_array_pointers((layers_output_gpu), result.n));
    result.layers_delta_gpu := single(cuda_make_array_pointers((layers_delta_gpu), result.n));
{$endif}
    result.bflops := result.out_w * result.out_h * result.out_c * result.n div 1000000000;
    if boolean(result.weights_type) then
        result.bflops := result.bflops * 2;
    writeln(ErrOutput, format(' wt = %d, wn = %d, outputs:%4d x%4d x%4d %5.3f BF', [Ord(result.weights_type), ord(result.weights_normalization), result.out_w, result.out_h, result.out_c, result.bflops]));
end;

procedure resize_shortcut_layer(var l: TShorTcutLayer; const w, h: longint; const net: PNetwork);
var
    i: longint;
    index: longint;
    layers_output_gpu: PPSingle;
    layers_delta_gpu: PPSingle;
begin
    l.w := w; l.out_w := w;
    l.h := h; l.out_h := h;
    l.outputs := w * h * l.out_c;
    l.inputs := l.outputs;
    if l.train then
        l.delta.reAllocate(l.outputs * l.batch);
    l.output.reAllocate(l.outputs * l.batch);
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            l.input_sizes[i] := net.layers[index].outputs;
            l.layers_output[i] := net.layers[index].output;
            l.layers_delta[i] := net.layers[index].delta;
            assert((l.w = net.layers[index].out_w) and (l.h = net.layers[index].out_h))
        end;
    if (l.activation = acSWISH) or (l.activation = acMISH) then
        l.activation_input.reAllocate(l.outputs * l.batch);
{$ifdef GPU}
    cuda_free(l.output_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch);
    if l.train then
        begin
            cuda_free(l.delta_gpu);
            l.delta_gpu := cuda_make_array(l.delta, l.outputs * l.batch)
        end;
    layers_output_gpu := PPSingle(calloc(l.n, sizeof(float * )));
    layers_delta_gpu := PPSingle(calloc(l.n, sizeof(float * )));
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            layers_output_gpu[i] := net.layers[index].output_gpu;
            layers_delta_gpu[i] := net.layers[index].delta_gpu
        end;
    memcpy_ongpu(l.input_sizes_gpu, l.input_sizes, l.n * sizeof(int));
    memcpy_ongpu(l.layers_output_gpu, layers_output_gpu, l.n * sizeof(float * ));
    memcpy_ongpu(l.layers_delta_gpu, layers_delta_gpu, l.n * sizeof(float * ));
    free(layers_output_gpu);
    free(layers_delta_gpu);
    if (l.activation = acSWISH) or (l.activation = acMISH) then
        begin
            cuda_free(l.activation_input_gpu);
            l.activation_input_gpu := cuda_make_array(l.activation_input, l.batch * l.outputs)
        end
{$endif}
end;

procedure forward_shortcut_layer(var l: TShortcutLayer; const state: PNetworkState);
var
    from_w: longint;
    from_h: longint;
    from_c: longint;
    a,b:PSingle;
    i: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    from_w := state.net.layers[l.index].w;
    from_h := state.net.layers[l.index].h;
    from_c := state.net.layers[l.index].c;
    if (l.nweights = 0) and (l.n = 1) and (from_w = l.w) and (from_h = l.h) and (from_c = l.c) then
        begin
            a:=state.input;
            b:=state.net.layers[l.index].output;
            for i := 0 to l.batch * l.w * l.h * l.c -1 do
                l.output[i] := a[i] + b[i]
                //l.output[i] := state.input[i]+state.net.layers[l.index].output[i]
        end
    else
        shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes, PPSingle(l.layers_output), l.output, state.input, l.weights, l.nweights, l.weights_normalization);
    if l.activation = acSWISH then
        activate_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.output)
    else
        if l.activation = acMISH then
            activate_array_mish(l.output, l.outputs * l.batch, l.activation_input, l.output)
    else
        //activate_array_cpu_custom(l.output, l.outputs * l.batch, l.activation)
        activate_array(l.output, l.outputs * l.batch, l.activation);
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_shortcut_layer(var l: TShortcutLayer; const state: PNetworkState);
begin
    if l.activation = acSWISH then
        gradient_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.delta)
    else
        if l.activation = acMISH then
            gradient_array_mish(l.outputs * l.batch, l.activation_input, l.delta)
    else
        gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    backward_shortcut_multilayer_cpu(l.outputs * l.batch, l.outputs, l.batch, l.n, l.input_sizes, PPSingle(l.layers_delta), state.delta, l.delta, l.weights, l.weight_updates, l.nweights, state.input, PPSingle(l.layers_output), l.weights_normalization)
end;

procedure update_shortcut_layer(const l: TShortcutLayer; const arg :TUpdateArgs);
var
    learning_rate: single;
begin
    if l.nweights > 0 then
        begin
            learning_rate := arg.learning_rate * l.learning_rate_scale;
            axpy_cpu(l.nweights, -arg.decay * arg.batch, l.weights, 1, l.weight_updates, 1);
            axpy_cpu(l.nweights, arg.learning_rate / arg.batch, l.weight_updates, 1, l.weights, 1);
            scal_cpu(l.nweights, arg.momentum, l.weight_updates, 1)
        end
end;

{$ifdef GPU}
procedure forward_shortcut_layer_gpu(const l: layer; state: network_state);
begin
    shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_output_gpu, l.output_gpu, state.input, l.weights_gpu, l.nweights, l.weights_normalization);
    if (l.activation = SWISH) then
        activate_array_swish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu)
    else
        if l.activation = MISH then
            activate_array_mish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.output_gpu)
    else
        activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation)
end;

procedure backward_shortcut_layer_gpu(const l: layer; state: network_state);
begin
    if l.activation = SWISH then
        gradient_array_swish_ongpu(l.output_gpu, l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu)
    else
        if l.activation = MISH then
            gradient_array_mish_ongpu(l.outputs * l.batch, l.activation_input_gpu, l.delta_gpu)
    else
        gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    backward_shortcut_multilayer_gpu(l.outputs, l.batch, l.n, l.input_sizes_gpu, l.layers_delta_gpu, state.delta, l.delta_gpu, l.weights_gpu, l.weight_updates_gpu, l.nweights, state.input, l.layers_output_gpu, l.weights_normalization)
end;

procedure update_shortcut_layer_gpu(l: layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
var
    learning_rate: single;
begin
    if l.nweights > 0 then
        begin
            learning_rate := learning_rate_init * l.learning_rate_scale / loss_scale;
            reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
            fix_nan_and_inf(l.weights_gpu, l.nweights);
            constrain_ongpu(l.nweights, 1, l.weight_updates_gpu, 1);
            axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
            scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1)
        end
end;

procedure pull_shortcut_layer(l: layer);
begin
    constrain_ongpu(l.nweights, 1, l.weight_updates_gpu, 1);
    cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);
    cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
    CHECK_CUDA(cudaPeekAtLastError());
    CHECK_CUDA(cudaStreamSynchronize(get_cuda_stream()))
end;

procedure push_shortcut_layer(l: layer);
begin
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    CHECK_CUDA(cudaPeekAtLastError())
end;
{$endif}

end.

