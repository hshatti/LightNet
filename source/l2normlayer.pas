unit L2NormLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface

uses
  SysUtils, lightnet, blas;

type
  TL2Norm = TLayer;

function make_l2norm_layer(const batch, inputs: longint):TL2Norm;
procedure forward_l2norm_layer(var l: TL2Norm; const net: PNetworkState);
procedure backward_l2norm_layer(var l: TL2Norm; const net: PNetworkState);

implementation

function make_l2norm_layer(const batch, inputs: longint):TL2Norm;
begin
    writeln(ErrOutput, format('l2norm                                         %4d', [inputs]));
    result := default(TL2Norm);
    result.&type := ltL2NORM;
    result.batch := batch;
    result.inputs := inputs;
    result.outputs := inputs;
    result.output := TSingles.Create(inputs * batch);
    result.scales := TSingles.Create(inputs * batch);
    result.delta := TSingles.Create(inputs * batch);
    result.forward := forward_l2norm_layer;
    result.backward := backward_l2norm_layer;
{$ifdef GPU}
    result.forward_gpu := forward_l2norm_layer_gpu;
    result.backward_gpu := backward_l2norm_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, inputs * batch);
    result.scales_gpu := cuda_make_array(result.output, inputs * batch);
    result.delta_gpu := cuda_make_array(result.delta, inputs * batch);
{$endif}
end;

procedure forward_l2norm_layer(var l: TL2Norm; const net: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    copy_cpu(l.outputs * l.batch, net.input, 1, l.output, 1);
    l2normalize_cpu(l.output, l.scales, l.batch, l.out_c, l.out_w * l.out_h);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_l2norm_layer(var l: TL2Norm; const net: PNetworkState);
begin
    axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1)
end;

{$ifdef GPU}
procedure forward_l2norm_layer_gpu(const l: TL2Norm; net: TNetwork);
begin
    copy_gpu(l.outputs * l.batch, net.input_gpu, 1, l.output_gpu, 1);
    l2normalize_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_w * l.out_h)
end;

procedure backward_l2norm_layer_gpu(const l: TL2Norm; net: TNetwork);
begin
    axpy_gpu(l.batch * l.inputs, 1, l.scales_gpu, 1, l.delta_gpu, 1);
    axpy_gpu(l.batch * l.inputs, 1, l.delta_gpu, 1, net.delta_gpu, 1)
end;

{$endif}

end.

