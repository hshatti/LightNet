unit RepresentationLayer;

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
  SysUtils, darknet, blas;

type
  TImplicitLayer = TLayer;

function make_implicit_layer(const batch, index: longint; const mean_init, std_init: single; const filters, atoms: longint):TImplicitLayer;
procedure resize_implicit_layer(l: Player; w: longint; h: longint);
procedure forward_implicit_layer(var l: TImplicitLayer; const state: PNetworkState);
procedure backward_implicit_layer(var l: TImplicitLayer; const state: PNetworkState);
procedure update_implicit_layer(const l: TImplicitLayer; const arg :TUpdateArgs);

{$ifdef PU}
procedure forward_implicit_layer_gpu(const l: layer; state: network_state);
procedure Conv.backward_implicit_layer_gpu(const l: layer; state: network_state);
procedure update_implicit_layer_gpu(l: layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
procedure pull_implicit_layer(l: layer);
procedure push_implicit_layer(l: layer);
{$endif}

implementation

function make_implicit_layer(const batch, index: longint; const mean_init, std_init: single; const filters, atoms: longint):TImplicitLayer;
var
    i: longint;
begin
    writeln(ErrOutput, format('implicit Layer: %d x %d '#9' mean=%.2f, std=%.2f ', [filters, atoms, mean_init, std_init]));
    result := Default(TImplicitLayer);
    result.&type := ltIMPLICIT;
    result.batch := batch;
    result.w := 1;
    result.h := 1;
    result.c := 1;
    result.out_w := 1;
    result.out_h := atoms;
    result.out_c := filters;
    result.outputs := result.out_w * result.out_h * result.out_c;
    result.inputs := 1;
    result.index := index;
    result.nweights := result.out_w * result.out_h * result.out_c;
    result.weight_updates := TSingles.Create(result.nweights);
    result.weights := TSingles.Create(result.nweights);
    for i := 0 to result.nweights -1 do
        result.weights[i] := mean_init + rand_uniform(-std_init, std_init);
    result.delta := TSingles.Create(result.outputs * batch);
    result.output := TSingles.Create(result.outputs * batch);
    result.forward := forward_implicit_layer;
    result.backward := backward_implicit_layer;
    result.update := update_implicit_layer;
{$ifdef GPU}
    result.forward_gpu := forward_implicit_layer_gpu;
    result.backward_gpu := backward_implicit_layer_gpu;
    result.update_gpu := update_implicit_layer_gpu;
    result.delta_gpu := cuda_make_array(result.delta, result.outputs * batch);
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
    result.weight_updates_gpu := cuda_make_array(result.weight_updates, result.nweights);
    result.weights_gpu := cuda_make_array(result.weights, result.nweights);
{$endif}
end;

procedure resize_implicit_layer(l: Player; w: longint; h: longint);
begin

end;

procedure forward_implicit_layer(var l: TImplicitLayer; const state: PNetworkState);
var
    i: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    for i := 0 to l.nweights * l.batch -1 do
        l.output[i] := l.weights[i mod l.nweights];

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_implicit_layer(var l: TImplicitLayer; const state: PNetworkState);
var
    i: longint;
begin
    for i := 0 to l.nweights * l.batch -1 do
        l.weight_updates[i mod l.nweights] := l.weight_updates[i mod l.nweights] + l.delta[i]
end;

procedure update_implicit_layer(const l: TImplicitLayer; const arg: TUpdateArgs);
var
    learning_rate: single;
begin
    learning_rate := arg.learning_rate * l.learning_rate_scale;
    axpy_cpu(l.nweights, -arg.decay * arg.batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, arg.learning_rate / arg.batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, arg.momentum, l.weight_updates, 1)
end;

{$ifdef PU}

procedure forward_implicit_layer_gpu(const l: layer; state: network_state);
begin
    forward_implicit_gpu(l.batch, l.nweights, l.weights_gpu, l.output_gpu)
end;

procedure Conv.backward_implicit_layer_gpu(const l: layer; state: network_state);
begin
    backward_implicit_gpu(l.batch, l.nweights, l.weight_updates_gpu, l.delta_gpu)
end;

procedure update_implicit_layer_gpu(l: layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
var
    learning_rate: single;
begin
    learning_rate := learning_rate_init * l.learning_rate_scale / loss_scale;
    reset_nan_and_inf(l.weight_updates_gpu, l.nweights);
    fix_nan_and_inf(l.weights_gpu, l.nweights);
    if l.adam then
        adam_update_gpu(l.weights_gpu, l.weight_updates_gpu, l.m_gpu, l.v_gpu, l.B1, l.B2, l.eps, decay, learning_rate, l.nweights, batch, l.t)
    else
        begin
            axpy_ongpu(l.nweights, -decay * batch * loss_scale, l.weights_gpu, 1, l.weight_updates_gpu, 1);
            axpy_ongpu(l.nweights, learning_rate / batch, l.weight_updates_gpu, 1, l.weights_gpu, 1);
            scal_ongpu(l.nweights, momentum, l.weight_updates_gpu, 1)
        end;
    if l.clip then
        constrain_ongpu(l.nweights, l.clip, l.weights_gpu, 1)
end;

procedure pull_implicit_layer(l: layer);
begin
    cuda_pull_array_async(l.weights_gpu, l.weights, l.nweights);
    cuda_pull_array_async(l.weight_updates_gpu, l.weight_updates, l.nweights);
    if l.adam then
        begin
            cuda_pull_array_async(l.m_gpu, l.m, l.nweights);
            cuda_pull_array_async(l.v_gpu, l.v, l.nweights)
        end;
    CHECK_CUDA(cudaPeekAtLastError());
    cudaStreamSynchronize(get_cuda_stream())
end;

procedure push_implicit_layer(l: layer);
begin
    cuda_push_array(l.weights_gpu, l.weights, l.nweights);
    if l.train then
        cuda_push_array(l.weight_updates_gpu, l.weight_updates, l.nweights);
    if l.adam then
        begin
            cuda_push_array(l.m_gpu, l.m, l.nweights);
            cuda_push_array(l.v_gpu, l.v, l.nweights)
        end;
    CHECK_CUDA(cudaPeekAtLastError())
end;
{$endif}

end.

