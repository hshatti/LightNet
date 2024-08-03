unit CostLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, lightnet, blas;

type
  TCostLayer = TLayer ;

  function get_cost_type(s: string):TCostType;

  function get_cost_string(a: TCostType):string;

  function make_cost_layer(const batch, inputs:longint; const costType: TCostType; const scale: single):TCostLayer;

  procedure resize_cost_layer(var l: TCostLayer; const inputs: longint);

  procedure forward_cost_layer(var l: TCostLayer; const state: PNetworkState);

  procedure backward_cost_layer(var l: TCostLayer; const state: PNetworkState);

  {$ifdef GPU}
  //procedure pull_cost_layer(l: cost_layer);
  //
  //procedure push_cost_layer(l: cost_layer);
  //
  //function float_abs_compare(const a, b: single):longint;

  procedure forward_cost_layer_gpu(l: cost_layer; net: network);

  procedure backward_cost_layer_gpu(const l: cost_layer; net: network);
  {$endif}

implementation

function get_cost_type(s: string):TCostType;
begin
    if (CompareStr(s, 'seg') = 0) then
        exit(ctSEG);
    if CompareStr(s, 'sse') = 0 then
        exit(ctSSE);
    if CompareStr(s, 'masked') = 0 then
        exit(ctMASKED);
    if CompareStr(s, 'smooth') = 0 then
        exit(ctSMOOTH);
    if CompareStr(s, 'L1') = 0 then
        exit(ctL1);
    if CompareStr(s, 'wgan') = 0 then
        exit(ctWGAN);
    writeln('Couldn''t find cost type ', s, ' going with SSE'#10'');
    exit(ctSSE)
end;

function get_cost_string(a: TCostType):string;
begin
    case a of
        ctSEG:
            exit('seg');
        ctSSE:
            exit('sse');
        ctMASKED:
            exit('masked');
        ctSMOOTH:
            exit('smooth');
        ctL1:
            exit('L1');
        ctWGAN:
            exit('wgan')
    end;
    exit('sse')
end;

function make_cost_layer(const batch, inputs:longint; const costType: TCostType; const scale: single):TCostLayer;

begin
    writeln(format('cost                                           %4d', [inputs]));
    result := Default(TCostLayer);
    result.&type := ltCOST;
    result.scale := scale;
    result.batch := batch;
    result.inputs := inputs;
    result.outputs := inputs;
    result.CostType := costType;
    result.delta := TSingles.Create(inputs * batch);
    result.output := TSingles.Create(inputs * batch);
    result.cost := [0];//TSingles.Create(1);
    result.forward := forward_cost_layer;
    result.backward := backward_cost_layer;
  {$ifdef GPU}
    result.forward_gpu := forward_cost_layer_gpu;
    result.backward_gpu := backward_cost_layer_gpu;
    result.delta_gpu := cuda_make_array(l.output, inputs * batch);
    result.output_gpu := cuda_make_array(l.delta, inputs * batch)
  {$endif}
end;

procedure resize_cost_layer(var l: TCostLayer; const inputs: longint);
begin
    l.inputs := inputs;
    l.outputs := inputs;
    //setLength(l.delta, inputs * l.batch);
    l.delta.reAllocate(inputs * l.batch);
    //setLength(l.output, inputs * l.batch);
    l.output.reAllocate(inputs * l.batch);
  {$ifdef GPU}
    cuda_free(l.delta_gpu);
    cuda_free(l.output_gpu);
    l.delta_gpu := cuda_make_array(l.delta, inputs * l.batch);
    l.output_gpu := cuda_make_array(l.output, inputs * l.batch)
  {$endif}
end;

procedure forward_cost_layer(var l: TCostLayer; const state: PNetworkState);
var
    i: longint;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}
    if not assigned(state.truth) then
        exit();
    if l.costtype = ctMASKED then
        begin
            for i := 0 to l.batch * l.inputs -1 do
                if state.truth[i] = SECRET_NUM then
                    state.input[i] := SECRET_NUM
        end;
    if (l.costType = ctSMOOTH) then
        smooth_l1_cpu(l.batch * l.inputs, @state.input[0], @state.truth[0], @l.delta[0], @l.output[0])
    else
        if l.CostType = ctL1 then
            l1_cpu(l.batch * l.inputs, @state.input[0], @state.truth[0], @l.delta[0], @l.output[0])
    else
        l2_cpu(l.batch * l.inputs, @state.input[0], @state.truth[0], @l.delta[0], @l.output[0]);
    l.cost[0] := sum_array(@l.output[0], l.batch * l.inputs);
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_cost_layer(var l: TCostLayer; const state: PNetworkState);
begin
    axpy_cpu(l.batch * l.inputs, l.scale, @l.delta[0], 1, @state.delta[0], 1)
end;

{$ifdef GPU}
procedure pull_cost_layer(l: cost_layer);
begin
    cuda_pull_array(l.delta_gpu, l.delta, l.batch * l.inputs)
end;

procedure push_cost_layer(l: cost_layer);
begin
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.inputs)
end;

function float_abs_compare(const a, b: single):longint;
var fa,fb:single;
begin
    fa:=a;fb:=b;
    if fa < 0 then
        fa := -a;
    if fb < 0 then
        fb := -b;
    exit(longint(fa > fb)-longint(fa < fb))
end;

procedure forward_cost_layer_gpu(l: cost_layer; net: network);
var
    n: longint;
    thresh: single;
begin
    if not assigned(net.truth) then
        exit();
    if l.smooth<>0 then
        begin
            scal_gpu(l.batch * l.inputs, (1-l.smooth), net.truth_gpu, 1);
            add_gpu(l.batch * l.inputs, l.smooth * 1.0 / l.inputs, net.truth_gpu, 1)
        end;
    if l.cost_type = SMOOTH then
        smooth_l1_gpu(l.batch * l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu)
    else
        if l.cost_type = L1 then
            l1_gpu(l.batch * l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu)
    else
        if l.cost_type = WGAN then
            wgan_gpu(l.batch * l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu)
    else
        l2_gpu(l.batch * l.inputs, net.input_gpu, net.truth_gpu, l.delta_gpu, l.output_gpu);
    if l.cost_type = SEG and l.noobject_scale <> 1 then
        begin
            scale_mask_gpu(l.batch * l.inputs, l.delta_gpu, 0, net.truth_gpu, l.noobject_scale);
            scale_mask_gpu(l.batch * l.inputs, l.output_gpu, 0, net.truth_gpu, l.noobject_scale)
        end;
    if l.cost_type = MASKED then
        mask_gpu(l.batch * l.inputs, net.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
    if l.ratio then
        begin
            cuda_pull_array(l.delta_gpu, l.delta, l.batch * l.inputs);
            TTools.QuickSort(@l.delta[0],0, l.batch * l.inputs-1, float_abs_compare);
            n := (1-l.ratio) * l.batch * l.inputs;
            thresh := l.delta[n];
            thresh := 0;
            writeln(thresh);
            supp_gpu(l.batch * l.inputs, thresh, l.delta_gpu, 1)
        end;
    if l.thresh<>0 then
        supp_gpu(l.batch * l.inputs, l.thresh * 1.0 / l.inputs, l.delta_gpu, 1);
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.inputs);
    l.cost[0] := sum_array(l.output, l.batch * l.inputs)
end;

procedure backward_cost_layer_gpu(const l: cost_layer; net: network);
begin
    axpy_gpu(l.batch * l.inputs, l.scale, l.delta_gpu, 1, net.delta_gpu, 1)
end;

{$endif}
end.

