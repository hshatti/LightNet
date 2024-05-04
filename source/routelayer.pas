unit RouteLayer;

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
  PRouteLayer = ^TRouteLayer;
  TRouteLayer = TLayer;

function make_route_layer(const batch, n: longint; const input_layers, input_sizes: TArray<longint>; const groups, group_id: longint):TRouteLayer;
procedure resize_route_layer(var l: TRouteLayer; const net: PNetwork);
procedure forward_route_layer(var l: TRouteLayer; const state: PNetworkState);
procedure backward_route_layer(var l: TRouteLayer; const state: PNetworkState);
{$ifdef GPU}
procedure forward_route_layer_gpu(const l: route_layer; state: network_state);
procedure backward_route_layer_gpu(const l: route_layer; state: network_state);
{$endif}

implementation

// not converted to yolo4 yet
// note Route layer is just a tensors Concatenation?

function make_route_layer(const batch, n: longint; const input_layers, input_sizes: TArray<longint>; const groups, group_id: longint):TRouteLayer;
var
    l: TRouteLayer;
    i: longint;
    outputs: longint;
begin
    write(ErrOutput, 'route ');
    l := Default(TRouteLayer);
    l.&type := ltROUTE;
    l.batch := batch;
    l.n := n;
    l.input_layers := input_layers;
    l.input_sizes := input_sizes;
    l.groups := groups;
    l.group_id := group_id;
    l.wait_stream_id := -1;
    outputs := 0;
    for i := 0 to n -1 do
        begin
            write(ErrOutput, ' ', input_layers[i]);
            outputs := outputs + input_sizes[i]
        end;
    outputs := outputs div groups;
    l.outputs := outputs;
    l.inputs := outputs;
    l.delta := TSingles.Create(outputs * batch);
    l.output := TSingles.Create(outputs * batch);
    l.forward := forward_route_layer;
    l.backward := backward_route_layer;
{$ifdef GPU}
    l.forward_gpu := forward_route_layer_gpu;
    l.backward_gpu := backward_route_layer_gpu;
    l.delta_gpu := cuda_make_array(l.delta, outputs * batch);
    l.output_gpu := cuda_make_array(l.output, outputs * batch);
{$endif}
    exit(l)
end;

procedure resize_route_layer(var l: TRouteLayer; const net: PNetwork);
var
    i: longint;
    first, next: TLayer;
    index: longint;
begin
    first := net.layers[l.input_layers[0]];
    l.out_w := first.out_w;
    l.out_h := first.out_h;
    l.out_c := first.out_c;
    l.outputs := first.outputs;
    l.input_sizes[0] := first.outputs;
    for i := 1 to l.n -1 do
        begin
            index := l.input_layers[i];
            next := net.layers[index];
            l.outputs := l.outputs + next.outputs;
            l.input_sizes[i] := next.outputs;
            if (next.out_w = first.out_w) and (next.out_h = first.out_h) then
                l.out_c := l.out_c + next.out_c
            else
                begin
                    writeln(format('Error: Different size of input layers: %d x %d, %d x %d', [next.out_w, next.out_h, first.out_w, first.out_h]));
                    raise Exception.Create('Error!')
                end
        end;
    l.out_c := l.out_c div l.groups;
    l.outputs := l.outputs div l.groups;
    l.inputs := l.outputs;
    l.delta.reAllocate(l.outputs * l.batch);
    l.output.reAllocate(l.outputs * l.batch);
{$ifdef GPU}
    cuda_free(l.output_gpu);
    cuda_free(l.delta_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch);
    l.delta_gpu := cuda_make_array(l.delta, l.outputs * l.batch)
{$endif}
end;

procedure forward_route_layer(var l: TRouteLayer; const state: PNetworkState);
var
    i: longint;
    j: longint;
    offset: longint;
    index: longint;
    input: PSingle;
    input_size: longint;
    part_input_size: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    offset := 0;
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            input := state.net.layers[index].output;
            input_size := l.input_sizes[i];
            part_input_size := input_size div l.groups;
            for j := 0 to l.batch -1 do
                copy_cpu(part_input_size, input+j * input_size+part_input_size * l.group_id, 1, l.output+offset+j * l.outputs, 1);
            offset := offset + part_input_size
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_route_layer(var l: TRouteLayer; const state: PNetworkState);
var
    i: longint;
    j: longint;
    offset: longint;
    index: longint;
    delta: PSingle;
    input_size: longint;
    part_input_size: longint;
begin
    offset := 0;
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            delta := state.net.layers[index].delta;
            input_size := l.input_sizes[i];
            part_input_size := input_size div l.groups;
            for j := 0 to l.batch -1 do
                axpy_cpu(part_input_size, 1, l.delta+offset+j * l.outputs, 1, delta+j * input_size+part_input_size * l.group_id, 1);
            offset := offset + part_input_size
        end
end;
{$ifdef GPU}
procedure forward_route_layer_gpu(const l: route_layer; state: network_state);
var
    i: longint;
    j: longint;
    offset: longint;
    index: longint;
    input: PSingle;
    input_size: longint;
    part_input_size: longint;
begin
    if l.stream >= 0 then
        switch_stream(l.stream);
    if (l.wait_stream_id >= 0) then
        wait_stream(l.wait_stream_id);
    offset := 0;
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            input := state.net.layers[index].output_gpu;
            input_size := l.input_sizes[i];
            part_input_size := input_size div l.groups;
            for j := 0 to l.batch -1 do
                simple_copy_ongpu(part_input_size, input+j * input_size+part_input_size * l.group_id, l.output_gpu+offset+j * l.outputs);
            offset := offset + part_input_size
        end
end;

procedure backward_route_layer_gpu(const l: route_layer; state: network_state);
var
    i: longint;
    j: longint;
    offset: longint;
    index: longint;
    delta: PSingle;
    input_size: longint;
    part_input_size: longint;
begin
    offset := 0;
    for i := 0 to l.n -1 do
        begin
            index := l.input_layers[i];
            delta := state.net.layers[index].delta_gpu;
            input_size := l.input_sizes[i];
            part_input_size := input_size div l.groups;
            for j := 0 to l.batch -1 do
                axpy_ongpu(part_input_size, 1, l.delta_gpu+offset+j * l.outputs, 1, delta+j * input_size+part_input_size * l.group_id, 1);
            offset := offset + part_input_size
        end
end;
{$endif}

end.

