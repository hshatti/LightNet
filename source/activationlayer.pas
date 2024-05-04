unit ActivationLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}


interface

uses
   lightnet, Activations, Blas;

type
  TActivationLayer=TLayer;

function make_activation_layer(const batch,inputs: longint ; const activation: TActivation):TLayer;

procedure forward_activation_layer(var l:TLayer; const net: PNetworkState);
procedure backward_activation_layer(var l:TLayer; const net: PNetworkState);

{$ifdef GPU}
procedure forward_activation_layer_gpu(const l:TLayer; const net:TNetwork);
procedure backward_activation_layer_gpu(const l:TLayer; const net:TNetwork);
{$endif}
implementation


// todo make TActivationLayer (forward, backward) as helper
function make_activation_layer(const batch,inputs: longint ; const activation: TActivation):TLayer;
begin
    result:=default(TActivationLayer);
    result.&type        := ltACTIVE;

    result.inputs       := inputs;
    result.outputs      := inputs;
    result.batch        := batch;

    result.output       := TSingles.create(batch*inputs);
    result.delta        := TSingles.create(batch*inputs);

    result.forward      := forward_activation_layer;
    result.backward     := backward_activation_layer;
{$ifdef GPU}
    result.forward_gpu  := forward_activation_layer_gpu;
    result.backward_gpu := backward_activation_layer_gpu;

    result.output_gpu   := cuda_make_array(result.output, inputs*batch);
    result.delta_gpu    := cuda_make_array(result.delta, inputs*batch);
{$endif}
    result.activation := activation;
    writeln('Activation Layer: ',inputs,' inputs');

end;

procedure forward_activation_layer(var l: TLayer; const net: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    copy_cpu(l.outputs*l.batch, net.input, 1, l.output, 1);
    activate_array(l.output, l.outputs*l.batch, l.activation);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}
end;

procedure backward_activation_layer(var l: TLayer; const net: PNetworkState);
begin
    gradient_array(l.output, l.outputs*l.batch, l.activation, l.delta);
    copy_cpu(l.outputs*l.batch, l.delta, 1, net.delta, 1);
end;

{$ifdef GPU}

procedure forward_activation_layer_gpu(const l:TLayer; const net:TNetwork);
begin
    copy_gpu(l.outputs*l.batch, net.input_gpu, 1, l.output_gpu, 1);
    activate_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation);
end;

procedure backward_activation_layer_gpu(const l:TLayer; const net:TNetwork);
begin
    gradient_array_gpu(l.output_gpu, l.outputs*l.batch, l.activation, l.delta_gpu);
    copy_gpu(l.outputs*l.batch, l.delta_gpu, 1, net.delta_gpu, 1);
end;
{$endif}

end.

