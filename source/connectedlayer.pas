unit ConnectedLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}

interface
uses SysUtils, lightnet, utils, blas, BatchNormLayer, Activations, gemm;

type
  TConnectedLayer = TLayer;

function get_connected_workspace_size(const l:TConnectedLayer):size_t;
function make_connected_layer(const batch, steps, inputs, outputs: longint; const activation: TActivation; const batch_normalize: boolean):TConnectedLayer;
procedure update_connected_layer(const l: TConnectedLayer; const arg:TUpdateArgs);
procedure forward_connected_layer(var l: TConnectedLayer; const state: PNetworkState);
procedure backward_connected_layer(var l: TConnectedLayer; const state: PNetworkState);
procedure denormalize_connected_layer(const l: TConnectedLayer);
procedure statistics_connected_layer(l: TConnectedLayer);
{$ifdef GPU}
procedure pull_connected_layer(l: connected_layer);
procedure push_connected_layer(l: connected_layer);
procedure update_connected_layer_gpu(l: connected_layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
procedure forward_connected_layer_gpu(l: connected_layer; state: network_state);
procedure backward_connected_layer_gpu(l: connected_layer; state: network_state);
{$endif}

implementation

function get_connected_workspace_size(const l:TConnectedLayer):size_t;
begin
{$ifdef CUDNN}
    result := get_convolutional_workspace_size(l);
{$else}
    result:=0
{$endif}
end;

function make_connected_layer(const batch, steps, inputs, outputs: longint;
  const activation: TActivation; const batch_normalize: boolean
  ): TConnectedLayer;
var
    total_batch, i: longint;
    scale: single;
begin
    total_batch := batch * steps;
    result := default(TConnectedLayer);
    result.&type := ltCONNECTED;
    result.inputs := inputs;
    result.outputs := outputs;
    result.batch := batch;
    result.batch_normalize := batch_normalize;
    result.h := 1;
    result.w := 1;
    result.c := inputs;
    result.out_h := 1;
    result.out_w := 1;
    result.out_c := outputs;
    result.n := result.out_c;
    result.size := 1;
    result.stride :=1; result.stride_x :=1; result.stride_y := 1;
    result.pad := 0;
    result.activation := activation;
    result.learning_rate_scale := 1;
    result.groups := 1;
    result.dilation := 1;
    result.output := TSingles.Create(total_batch * outputs);
    result.delta := TSingles.Create(total_batch * outputs);
    result.weight_updates := TSingles.Create(inputs * outputs);
    result.bias_updates := TSingles.Create(outputs);
    result.weights := TSingles.Create(outputs * inputs);
    result.biases := TSingles.Create(outputs);
    result.forward := forward_connected_layer;
    result.backward := backward_connected_layer;
    result.update := update_connected_layer;
    scale := sqrt(2 / inputs);
    for i := 0 to outputs * inputs -1 do
        result.weights[i] := scale * rand_uniform(-1, 1);
    for i := 0 to outputs -1 do
        result.biases[i] := 0;
    if batch_normalize then
        begin
            result.scales := TSingles.Create(outputs);
            result.scale_updates := TSingles.Create(outputs);
            for i := 0 to outputs -1 do
                result.scales[i] := 1;
            result.mean := TSingles.Create(outputs);
            result.mean_delta := TSingles.Create(outputs);
            result.variance := TSingles.Create(outputs);
            result.variance_delta := TSingles.Create(outputs);
            result.rolling_mean := TSingles.Create(outputs);
            result.rolling_variance := TSingles.Create(outputs);
            result.x := TSingles.Create(total_batch * outputs);
            result.x_norm := TSingles.Create(total_batch * outputs)
        end;
{$ifdef GPU}
    result.forward_gpu := forward_connected_layer_gpu;
    result.backward_gpu := backward_connected_layer_gpu;
    result.update_gpu := update_connected_layer_gpu;
    result.weights_gpu := cuda_make_array(result.weights, outputs * inputs);
    result.biases_gpu := cuda_make_array(result.biases, outputs);
    result.weight_updates_gpu := cuda_make_array(result.weight_updates, outputs * inputs);
    result.bias_updates_gpu := cuda_make_array(result.bias_updates, outputs);
    result.output_gpu := cuda_make_array(result.output, outputs * total_batch);
    result.delta_gpu := cuda_make_array(result.delta, outputs * total_batch);
    if batch_normalize then
        begin
            result.scales_gpu := cuda_make_array(result.scales, outputs);
            result.scale_updates_gpu := cuda_make_array(result.scale_updates, outputs);
            result.mean_gpu := cuda_make_array(result.mean, outputs);
            result.variance_gpu := cuda_make_array(result.variance, outputs);
            result.rolling_mean_gpu := cuda_make_array(result.mean, outputs);
            result.rolling_variance_gpu := cuda_make_array(result.variance, outputs);
            result.mean_delta_gpu := cuda_make_array(result.mean, outputs);
            result.variance_delta_gpu := cuda_make_array(result.variance, outputs);
            result.x_gpu := cuda_make_array(result.output, total_batch * outputs);
            result.x_norm_gpu := cuda_make_array(result.output, total_batch * outputs)
        end;
  {$ifdef CUDNN}
    create_convolutional_cudnn_tensors( and result);
    cudnn_convolutional_setup( and result, cudnn_fastest, 0);
    result.workspace_size := get_connected_workspace_size(result);
  {$endif}
{$endif}
    writeln(ErrOutput, format('connected                            %4d  ->  %4d', [inputs, outputs]));
end;

procedure update_connected_layer(const l: TConnectedLayer; const arg:TUpdateArgs);
begin
    axpy_cpu(l.outputs, arg.learning_rate / arg.batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.outputs, arg.momentum, l.bias_updates, 1);
    if l.batch_normalize then
        begin
            axpy_cpu(l.outputs, arg.learning_rate / arg.batch, l.scale_updates, 1, l.scales, 1);
            scal_cpu(l.outputs, arg.momentum, l.scale_updates, 1)
        end;
    axpy_cpu(l.inputs * l.outputs, -arg.decay * arg.batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.inputs * l.outputs, arg.learning_rate / arg.batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.inputs * l.outputs, arg.momentum, l.weight_updates, 1)
end;

procedure forward_connected_layer(var l: TConnectedLayer; const state: PNetworkState);
var
    i, m,k, n: longint;
    a, b, c: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    fill_cpu(l.outputs * l.batch, 0, l.output, 1);
    m := l.batch;
    k := l.inputs;
    n := l.outputs;
    a := state.input;
    b := l.weights;
    c := l.output;
    sgemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);

    if l.batch_normalize then
        begin
            if state.train then
                begin
                    mean_cpu(l.output, l.batch, l.outputs, 1, l.mean);
                    variance_cpu(l.output, l.mean, l.batch, l.outputs, 1, l.variance);
                    scal_cpu(l.outputs, 0.95, l.rolling_mean, 1);
                    axpy_cpu(l.outputs, 0.05, l.mean, 1, l.rolling_mean, 1);
                    scal_cpu(l.outputs, 0.95, l.rolling_variance, 1);
                    axpy_cpu(l.outputs, 0.05, l.variance, 1, l.rolling_variance, 1);
                    copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
                    normalize_cpu(l.output, l.mean, l.variance, l.batch, l.outputs, 1);
                    copy_cpu(l.outputs * l.batch, l.output, 1, l.x_norm, 1)
                end
            else
                normalize_cpu(l.output, l.rolling_mean, l.rolling_variance, l.batch, l.outputs, 1);

            scale_bias(l.output, l.scales, l.batch, l.outputs, 1);

        end;

    for i := 0 to l.batch -1 do
        axpy_cpu(l.outputs, 1, l.biases, 1, l.output+i * l.outputs, 1);

    activate_array(l.output, l.outputs * l.batch, l.activation);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_connected_layer(var l: TConnectedLayer; const state: PNetworkState);
var
    i, m, k, n: longint;
    a, b, c: PSingle;
begin
    gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    for i := 0 to l.batch -1 do
        axpy_cpu(l.outputs, 1, l.delta+i * l.outputs, 1, l.bias_updates, 1);
    if l.batch_normalize then
        begin
            backward_scale_cpu(l.x_norm, l.delta, l.batch, l.outputs, 1, l.scale_updates);
            scale_bias(l.delta, l.scales, l.batch, l.outputs, 1);
            mean_delta_cpu(l.delta, l.variance, l.batch, l.outputs, 1, l.mean_delta);
            variance_delta_cpu(l.x, l.delta, l.mean, l.variance, l.batch, l.outputs, 1, l.variance_delta);
            normalize_delta_cpu(l.x, l.mean, l.variance, l.mean_delta, l.variance_delta, l.batch, l.outputs, 1, l.delta)
        end;
    m := l.outputs;
    k := l.batch;
    n := l.inputs;
    a := l.delta;
    b := state.input;
    c := l.weight_updates;
    sgemm(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);
    m := l.batch;
    k := l.outputs;
    n := l.inputs;
    a := l.delta;
    b := l.weights;
    c := state.delta;
    if assigned(c) then
        sgemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n)
end;

procedure denormalize_connected_layer(const l: TConnectedLayer);
var
    i, j: longint;
    scale: single;
begin
    for i := 0 to l.outputs -1 do
        begin
            scale := l.scales[i] / sqrt(l.rolling_variance[i]+0.000001);
            for j := 0 to l.inputs -1 do
                l.weights[i * l.inputs+j] := l.weights[i * l.inputs+j] * scale;
            l.biases[i] := l.biases[i] - (l.rolling_mean[i] * scale);
            l.scales[i] := 1;
            l.rolling_mean[i] := 0;
            l.rolling_variance[i] := 1
        end
end;

procedure statistics_connected_layer(l: TConnectedLayer);
begin
    if l.batch_normalize then
        begin
            write('Scales ');
            print_statistics(l.scales, l.outputs)
        end;
    write('Biases ');
    print_statistics(l.biases, l.outputs);
    write('Weights ');
    print_statistics(l.weights, l.outputs)
end;
{$ifdef GPU}
procedure pull_connected_layer(l: connected_layer);
begin
    cuda_pull_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_pull_array(l.biases_gpu, l.biases, l.outputs);
    cuda_pull_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_pull_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if l.batch_normalize then
        begin
            cuda_pull_array(l.scales_gpu, l.scales, l.outputs);
            cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
            cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs)
        end;
    CHECK_CUDA(cudaPeekAtLastError())
end;

procedure push_connected_layer(l: connected_layer);
begin
    cuda_push_array(l.weights_gpu, l.weights, l.inputs * l.outputs);
    cuda_push_array(l.biases_gpu, l.biases, l.outputs);
    cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.inputs * l.outputs);
    cuda_push_array(l.bias_updates_gpu, l.bias_updates, l.outputs);
    if l.batch_normalize then
        begin
            cuda_push_array(l.scales_gpu, l.scales, l.outputs);
            cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.outputs);
            cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.outputs)
        end;
    CHECK_CUDA(cudaPeekAtLastError())
end;

procedure update_connected_layer_gpu(l: connected_layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
var
    learning_rate: single;
begin
    learning_rate := learning_rate_init * l.learning_rate_scale;
    if loss_scale <> 1.0 then
        begin
            scal_ongpu(l.inputs * l.outputs, 1.0 / loss_scale, l.weight_updates_gpu, 1);
            scal_ongpu(l.outputs, 1.0 / loss_scale, l.bias_updates_gpu, 1);
            scal_ongpu(l.outputs, 1.0 / loss_scale, l.scale_updates_gpu, 1)
        end;
    axpy_ongpu(l.outputs, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.outputs, momentum, l.bias_updates_gpu, 1);
    if l.batch_normalize then
        begin
            axpy_ongpu(l.outputs, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
            scal_ongpu(l.outputs, momentum, l.scale_updates_gpu, 1)
        end;
    axpy_ongpu(l.inputs * l.outputs, -decay * batch, l.weights_gpu, 1, l.weight_updates_gpu, 1);
    axpy_ongpu(l.inputs * l.outputs, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
    scal_ongpu(l.inputs * l.outputs, momentum, l.weight_updates_gpu, 1)
end;

procedure forward_connected_layer_gpu(l: connected_layer; state: network_state);
var
    m: longint;
    k: longint;
    n: longint;
    a: PSingle;
    b: PSingle;
    c: PSingle;
    one: single;
    alpha: single;
begin
    fill_ongpu(l.outputs * l.batch, 0, l.output_gpu, 1);
    m := l.batch;
    k := l.inputs;
    n := l.outputs;
    a := state.input;
    b := l.weights_gpu;
    c := l.output_gpu;
  {$ifdef CUDNN}
    one := 1;
    alpha := 1; beta := 0;
    CHECK_CUDNN(cudnnConvolutionForward(cudnn_handle(),  @alpha, l.srcTensorDesc, state.input, l.weightDesc, l.weights_gpu, l.convDesc, l.fw_algo, state.workspace, l.workspace_size,  @beta, l.dstTensorDesc, l.output_gpu));
  {$else}
    gemm_ongpu(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
  {$endif}
    if l.batch_normalize then
        forward_batchnorm_layer_gpu(l, state)
    else
        add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.outputs, 1);
    activate_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation)
end;

procedure backward_connected_layer_gpu(l: connected_layer; state: network_state);
var
    i: longint;
    one: single;
    m: longint;
    k: longint;
    n: longint;
    a: PSingle;
    b: PSingle;
    c: PSingle;
begin
    constrain_ongpu(l.outputs * l.batch, 1, l.delta_gpu, 1);
    gradient_array_ongpu(l.output_gpu, l.outputs * l.batch, l.activation, l.delta_gpu);
    for i := 0 to l.batch -1 do
        axpy_ongpu(l.outputs, 1, l.delta_gpu+i * l.outputs, 1, l.bias_updates_gpu, 1);
    if l.batch_normalize then
        backward_batchnorm_layer_gpu(l, state);
  {$ifdef CUDNN_DISABLED}
    one := 1;
    CHECK_CUDNN(cudnnConvolutionBackwardFilter(cudnn_handle(),  and one, l.srcTensorDesc, state.input, l.ddstTensorDesc, l.delta_gpu, l.convDesc, l.bf_algo, state.workspace, l.workspace_size,  and one, l.dweightDesc, l.weight_updates_gpu));
    if state.delta then
        CHECK_CUDNN(cudnnConvolutionBackwardData(cudnn_handle(),  and one, l.weightDesc, l.weights_gpu, l.ddstTensorDesc, l.delta_gpu, l.convDesc, l.bd_algo, state.workspace, l.workspace_size,  and one, l.dsrcTensorDesc, state.delta));
  {$else}
    m := l.outputs;
    k := l.batch;
    n := l.inputs;
    a := l.delta_gpu;
    b := state.input;
    c := l.weight_updates_gpu;
    gemm_ongpu(1, 0, m, n, k, 1, a, m, b, n, 1, c, n);
    m := l.batch;
    k := l.outputs;
    n := l.inputs;
    a := l.delta_gpu;
    b := l.weights_gpu;
    c := state.delta;
    if c then
        gemm_ongpu(0, 0, m, n, k, 1, a, k, b, n, 1, c, n)
  {$endif}
end;
{$endif}
end.

