unit LogisticLayer;

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
  SysUtils, lightnet, blas, Activations;

type
  TLogisticLayer = TLayer;

function make_logistic_layer(const batch, inputs: longint):TLogisticLayer;
procedure forward_logistic_layer(var l: TLogisticLayer; const net: PNetworkState);
procedure backward_logistic_layer(var l: TLogisticLayer; const net: PNetworkState);

{$ifdef GPU}
procedure forward_logistic_layer_gpu(const l: TLogisticLayer; net: TNetwork);
procedure backward_logistic_layer_gpu(const l: TLogisticLayer; net: TNetwork);
{$endif}

implementation

function make_logistic_layer(const batch, inputs: longint):TLogisticLayer;
begin
    writeln(ErrOutput, format('logistic x entropy                             %4d', [inputs]));
    result := default(TLogisticLayer);
    result.&type := ltLOGXENT;
    result.batch := batch;
    result.inputs := inputs;
    result.outputs := inputs;
    result.loss := TSingles.Create(inputs * batch);
    result.output := TSingles.Create(inputs * batch);
    result.delta := TSingles.Create(inputs * batch);
    result.cost := [0];//TSingles.Create(1);
    result.forward := forward_logistic_layer;
    result.backward := backward_logistic_layer;
{$ifdef GPU}
    result.forward_gpu := forward_logistic_layer_gpu;
    result.backward_gpu := backward_logistic_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, inputs * batch);
    result.loss_gpu := cuda_make_array(result.loss, inputs * batch);
    result.delta_gpu := cuda_make_array(result.delta, inputs * batch);
{$endif}
end;

procedure forward_logistic_layer(var l: TLogisticLayer; const net: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs * l.batch, acLOGISTIC);
    if assigned(net.truth) then
        begin
            logistic_x_ent_cpu(l.batch * l.inputs, l.output, net.truth, l.delta, l.loss);
            l.cost[0] := sum_array(l.loss, l.batch * l.inputs)
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_logistic_layer(var l: TLogisticLayer; const net: PNetworkState);
begin
    axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1)
end;

{$ifdef GPU}
procedure forward_logistic_layer_gpu(const l: TLogisticLayer; net: TNetwork);
begin
    copy_gpu(l.outputs * l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs * l.batch, LOGISTIC);
    if assigned(net.truth) then
        begin
            logistic_x_ent_gpu(l.batch * l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
            cuda_pull_array(l.loss_gpu, l.loss, l.batch * l.inputs);
            l.cost[0] := sum_array(l.loss, l.batch * l.inputs)
        end
end;

procedure backward_logistic_layer_gpu(const l: TLogisticLayer; net: TNetwork);
begin
    axpy_gpu(l.batch * l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1)
end;

{$endif}

end.

