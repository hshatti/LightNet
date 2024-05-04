unit AvgPoolLayer;
{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}

interface
uses SysUtils, darknet;

type
    PAvgPoolLayer = ^TAvgPoolLayer;
    TAvgPoolLayer = TLayer;


function make_avgpool_layer(const batch, w, h, c: longint):TAvgPoolLayer;
procedure resize_avgpool_layer(var l: TAvgPoolLayer; const w, h: longint);
procedure forward_avgpool_layer(var l: TAvgPoolLayer;const net: PNetworkState);
procedure backward_avgpool_layer(var l: TAvgPoolLayer;const net: PNetworkState);


implementation

function make_avgpool_layer(const batch, w, h, c: longint): TAvgPoolLayer;
var
    output_size: longint;
begin
    writeln(format('avg                     %4d x%4d x%4d   ->  %4d',[ w, h, c, c]));
    result:=default(TAvgPoolLayer);
    result.&type := ltAVGPOOL;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := c;
    result.out_w := 1;
    result.out_h := 1;
    result.out_c := c;
    result.outputs := result.out_c;
    result.inputs := h * w * c;
    output_size := result.outputs * batch;
    result.output:=TSingles.Create(output_size);
    //setLength(result.output, output_size);
    result.delta:=TSingles.Create(output_size);
    //setLength(result.delta ,output_size);
    result.forward := forward_avgpool_layer;
    result.backward := backward_avgpool_layer;
{$ifdef GPU}
    result.forward_gpu := forward_avgpool_layer_gpu;
    result.backward_gpu := backward_avgpool_layer_gpu;
    result.output_gpu := cuda_make_array(l.output, output_size);
    result.delta_gpu := cuda_make_array(l.delta, output_size);
{$endif}

end;

procedure resize_avgpool_layer(var l: TAvgPoolLayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.inputs := h * w * l.c
end;

procedure forward_avgpool_layer(var l: TAvgPoolLayer; const net: PNetworkState);
var
    b, i, k, out_index, in_index: longint;
begin
    {$ifdef USE_TELEMeTRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    for b := 0 to l.batch -1 do
        for k := 0 to l.c -1 do
            begin
                out_index := k+b * l.c;
                l.output[out_index] := 0;
                for i := 0 to l.h * l.w -1 do
                    begin
                        in_index := i+l.h * l.w * (k+b * l.c);
                        l.output[out_index] := l.output[out_index] + net.input[in_index]
                    end;
                l.output[out_index] := l.output[out_index] / (l.h * l.w)
            end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}
end;

procedure backward_avgpool_layer(var l: TAvgPoolLayer; const net: PNetworkState);
var
    b, i, k, out_index, in_index: longint;
    t:int64;
begin
    for b := 0 to l.batch -1 do
        for k := 0 to l.c -1 do
            begin
                out_index := k+b * l.c;
                for i := 0 to l.h * l.w -1 do
                    begin
                        in_index := i+l.h * l.w * (k+b * l.c);
                        net.delta[in_index] := net.delta[in_index] + (l.delta[out_index] / (l.h * l.w))
                    end
            end
end;



end.

