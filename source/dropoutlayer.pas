unit DropoutLayer;

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
  PDropoutLayer = ^TDropoutLayer;
  TDropoutLayer = TLayer;

function make_dropout_layer(const batch, inputs: longint; const probability: single; const dropblock: boolean; const dropblock_size_rel: single; const dropblock_size_abs, w, h, c: longint):TDropoutLayer;
procedure resize_dropout_layer(var l: TDropoutLayer; const inputs: longint);
procedure forward_dropout_layer(var l: TDropoutLayer; const state: PNetworkState);
procedure backward_dropout_layer(var l: TDropoutLayer; const state: PNetworkState);

implementation

function make_dropout_layer(const batch, inputs: longint; const probability: single; const dropblock: boolean; const dropblock_size_rel: single; const dropblock_size_abs, w, h, c: longint):TDropoutLayer;
begin
    result := Default(TDropoutLayer);
    result.&type := ltDROPOUT;
    result.probability := probability;
    result.dropblock := dropblock;
    result.dropblock_size_rel := dropblock_size_rel;
    result.dropblock_size_abs := dropblock_size_abs;
    if result.dropblock then
        begin
            result.out_w :=w;
            result.w := w;
            result.out_h :=h;
            result.h := h;
            result.out_c := c;
            result.c := c;
            if (result.w <= 0) or (result.h <= 0) or (result.c <= 0) then
                begin
                    writeln(format(' Error: DropBlock - there must be positive values for: result.w=%d, result.h=%d, result.c=%d ',[ result.w, result.h, result.c]));
                    raise Exception.Create('Error!')
                end
        end;
    result.inputs := inputs;
    result.outputs := inputs;
    result.batch := batch;
    result.rand := TSingles.Create(inputs * batch);
    result.scale := 1 / (1.0-probability);
    result.forward := forward_dropout_layer;
    result.backward := backward_dropout_layer;
{$ifdef GPU}
    result.forward_gpu := forward_dropout_layer_gpu;
    result.backward_gpu := backward_dropout_layer_gpu;
    result.rand_gpu := cuda_make_array(result.rand, inputs * batch);
    if result.dropblock then
        begin
            result.drop_blocks_scale := cuda_make_array_pinned(result.rand, result.batch);
            result.drop_blocks_scale_gpu := cuda_make_array(result.rand, result.batch)
        end;
{$endif}
    if result.dropblock then
        begin
            if result.dropblock_size_abs<>0 then
                writeln(ErrOutput, format('dropblock    p = %.3f   result.dropblock_size_abs = %d    %4d  ->   %4d', [probability, result.dropblock_size_abs, inputs, inputs]))
            else
                writeln(ErrOutput, format('dropblock    p = %.3f   result.dropblock_size_rel = %.2f    %4d  ->   %4d', [probability, result.dropblock_size_rel, inputs, inputs]))
        end
    else
        writeln(ErrOutput, format('dropout    p = %.3f        %4d  ->   %4d',[ probability, inputs, inputs]));
end;

procedure resize_dropout_layer(var l: TDropoutLayer; const inputs: longint);
begin
    l.inputs := inputs;
    l.outputs := inputs;
    l.rand.reAllocate(l.inputs * l.batch );
{$ifdef GPU}
    cuda_free(l.rand_gpu);
    l.rand_gpu := cuda_make_array(l.rand, l.inputs * l.batch);
    if l.dropblock then
        begin
            cudaFreeHost(l.drop_blocks_scale);
            l.drop_blocks_scale := cuda_make_array_pinned(l.rand, l.batch);
            cuda_free(l.drop_blocks_scale_gpu);
            l.drop_blocks_scale_gpu := cuda_make_array(l.rand, l.batch)
        end
{$endif}
end;

procedure forward_dropout_layer(var l: TDropoutLayer; const state: PNetworkState);
var
    i: longint;
    r: single;
begin
  {$ifdef USE_TELEMETRY}
  if benchmark then metrics.forward.start(l.&type);
  {$endif}

    if not state.train then
        exit();
    for i := 0 to l.batch * l.inputs -1 do
        begin
            r := random();//rand_uniform(0, 1);
            l.rand[i] := r;
            if r < l.probability then
                state.input[i] := 0
            else
                state.input[i] := state.input[i] * l.scale
        end ;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_dropout_layer(var l: TDropoutLayer;
  const state: PNetworkState);
var
    i: longint;
    r: single;
begin
    if not assigned(state.delta) then
        exit();
    for i := 0 to l.batch * l.inputs -1 do
        begin
            r := l.rand[i];
            if r < l.probability then
                state.delta[i] := 0
            else
                state.delta[i] := state.delta[i] * l.scale
        end
end;


end.

