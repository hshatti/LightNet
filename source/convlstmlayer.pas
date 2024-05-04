unit ConvLSTMLayer;

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
  sysutils, lightnet, ConvolutionalLayer, activations;

type
  PConvLSTMLayer = ^TConvLSTMLayer;
  TConvLSTMLayer = TLayer;

function make_conv_lstm_layer(batch:longint; const h, w, c, output_filters, groups, steps, size, stride, dilation, pad: longint; const activation: TActivation; const batch_normalize, peephole, xnor, bottleneck, train: boolean):TConvLSTMLayer;
function make_history_layer(const batch, h, w, c, history_size, steps: longint; const train: boolean):TLayer;
procedure forward_history_layer(var l: TLayer; const state: PNetworkState);
procedure backward_history_layer(var l: TLayer; const state: PNetworkState);
{$ifdef GPU}
procedure forward_history_layer_gpu(const l: layer; state: network_state);
procedure backward_history_layer_gpu(const l: layer; state: network_state);
{$endif}
procedure update_conv_lstm_layer(const l: TConvLSTMLayer; const arg:TUpdateArgs);// batch: longint; learning_rate: single; momentum: single; decay: single);
procedure resize_conv_lstm_layer(var l: TConvLSTMLayer; const w, h: longint);
procedure free_state_conv_lstm(const l: TConvLSTMLayer);
procedure randomize_state_conv_lstm(const l: TConvLSTMLayer);
procedure remember_state_conv_lstm(const l: TConvLSTMLayer);
procedure restore_state_conv_lstm(const l: TConvLSTMLayer);
procedure forward_conv_lstm_layer(var l: TConvLSTMLayer; const state: PNetworkState);
procedure backward_conv_lstm_layer(var l: TConvLSTMLayer; const state: PNetworkState);
{$ifdef GPU}
procedure pull_conv_lstm_layer(l: layer);
procedure push_conv_lstm_layer(l: layer);
procedure update_conv_lstm_layer_gpu(l: layer; batch: longint; learning_rate: single; momentum: single; decay: single; loss_scale: single);
procedure forward_conv_lstm_layer_gpu(l: layer; state: network_state);
procedure backward_conv_lstm_layer_gpu(l: layer; state: network_state);
{$endif}

implementation
uses blas;
procedure increment_layer(const l: PConvLSTMLayer; const steps: longint);
var
    num: longint;
begin
    num := l.outputs * l.batch * steps;
    l.output := l.output + num;
    l.delta := l.delta + num;
    l.x := l.x + num;
    l.x_norm := l.x_norm + num;
  {$ifdef GPU}
    l.output_gpu := l.output_gpu + num;
    l.delta_gpu := l.delta_gpu + num;
    l.x_gpu := l.x_gpu + num;
    l.x_norm_gpu := l.x_norm_gpu + num
  {$endif}
end;

function make_conv_lstm_layer(batch: longint; const h, w, c, output_filters,
  groups, steps, size, stride, dilation, pad: longint;
  const activation: TActivation; const batch_normalize, peephole, xnor,
  bottleneck, train: boolean): TConvLSTMLayer;
var
    outputs: longint;
begin
    writeln(ErrOutput, format('CONV_LSTM Layer: %d x %d x %d image, %d filters', [h, w, c, output_filters]));
    batch := batch div steps;
    result := default(TConvLSTMLayer);
    result.train := train;
    result.batch := batch;
    result.&type := ltConvLSTM;
    result.bottleneck := bottleneck;
    result.steps := steps;
    result.size := size;
    result.stride := stride;
    result.dilation := dilation;
    result.pad := pad;
    result.h := h;
    result.w := w;
    result.c := c;
    result.groups := groups;
    result.out_c := output_filters;
    result.inputs := h * w * c;
    result.xnor := xnor;
    result.peephole := peephole;
    setLength(result.uf, 1);
    result.uf[0] := make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.uf[0].batch := batch;
    if result.workspace_size < result.uf[0].workspace_size then
        result.workspace_size := result.uf[0].workspace_size;
    setLength(result.ui, 1);
    result.ui[0] := make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.ui[0].batch := batch;
    if result.workspace_size < result.ui[0].workspace_size then
        result.workspace_size := result.ui[0].workspace_size;
    setLength(result.ug, 1);
    result.ug[0] := make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.ug[0].batch := batch;
    if result.workspace_size < result.ug[0].workspace_size then
        result.workspace_size := result.ug[0].workspace_size;
    setLength(result.uo, 1);
    result.uo[0] := make_convolutional_layer(batch, steps, h, w, c, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
    result.uo[0].batch := batch;
    if result.workspace_size < result.uo[0].workspace_size then
        result.workspace_size := result.uo[0].workspace_size;
    if result.bottleneck then
        begin
            setLength(result.wf, 1);
            setLength(result.wi, 1);
            setLength(result.wg, 1);
            setLength(result.wo, 1);
             result.wf[0] := make_convolutional_layer(batch, steps, h, w, output_filters * 2, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.wf[0].batch := batch;
            if result.workspace_size < result.wf[0].workspace_size then
                result.workspace_size := result.wf[0].workspace_size
        end
    else
        begin
            setLength(result.wf, 1);
             result.wf[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.wf[0].batch := batch;
            if result.workspace_size < result.wf[0].workspace_size then
                result.workspace_size := result.wf[0].workspace_size;
            setLength(result.wi, 1);
             result.wi[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.wi[0].batch := batch;
            if result.workspace_size < result.wi[0].workspace_size then
                result.workspace_size := result.wi[0].workspace_size;
            setLength(result.wg, 1);
             result.wg[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.wg[0].batch := batch;
            if result.workspace_size < result.wg[0].workspace_size then
                result.workspace_size := result.wg[0].workspace_size;
            setLength(result.wo, 1);
             result.wo[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.wo[0].batch := batch;
            if result.workspace_size < result.wo[0].workspace_size then
                result.workspace_size := result.wo[0].workspace_size
        end;
    setLength(result.vf, 1);
    if result.peephole then
        begin
             result.vf[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.vf[0].batch := batch;
            if result.workspace_size < result.vf[0].workspace_size then
                result.workspace_size := result.vf[0].workspace_size
        end;
    setLength(result.vi, 1);
    if result.peephole then
        begin
             result.vi[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.vi[0].batch := batch;
            if result.workspace_size < result.vi[0].workspace_size then
                result.workspace_size := result.vi[0].workspace_size
        end;
    setLength(result.vo, 1);
    if result.peephole then
        begin
             result.vo[0] := make_convolutional_layer(batch, steps, h, w, output_filters, output_filters, groups, size, stride, stride, dilation, pad, activation, batch_normalize, false, xnor, false, false, 0, 0, nil, 0, false, train);
            result.vo[0].batch := batch;
            if result.workspace_size < result.vo[0].workspace_size then
                result.workspace_size := result.vo[0].workspace_size
        end;
    result.batch_normalize := batch_normalize;
    result.out_h := result.uo[0].out_h;
    result.out_w := result.uo[0].out_w;
    result.outputs := result.uo[0].outputs;
    outputs := result.outputs;
    result.inputs := w * h * c;
    if not result.bottleneck then
        assert(result.wo[0].outputs = result.uo[0].outputs);
    assert(result.wf[0].outputs = result.uf[0].outputs);
    result.output := TSingles.Create(outputs * batch);
    result.forward := forward_conv_lstm_layer;
    result.update := update_conv_lstm_layer;
    result.backward := backward_conv_lstm_layer;
    result.prev_state_cpu := TSingles.Create(batch * outputs);
    result.prev_cell_cpu := TSingles.Create(batch * outputs);
    result.cell_cpu := TSingles.Create(batch * outputs * steps);
    result.f_cpu := TSingles.Create(batch * outputs);
    result.i_cpu := TSingles.Create(batch * outputs);
    result.g_cpu := TSingles.Create(batch * outputs);
    result.o_cpu := TSingles.Create(batch * outputs);
    result.c_cpu := TSingles.Create(batch * outputs);
    result.stored_c_cpu := TSingles.Create(batch * outputs);
    result.h_cpu := TSingles.Create(batch * outputs);
    result.stored_h_cpu := TSingles.Create(batch * outputs);
    result.temp_cpu := TSingles.Create(batch * outputs);
    result.temp2_cpu := TSingles.Create(batch * outputs);
    result.temp3_cpu := TSingles.Create(batch * outputs);
    result.dc_cpu := TSingles.Create(batch * outputs);
    result.dh_cpu := TSingles.Create(batch * outputs);
{$ifdef GPU}
    result.forward_gpu := forward_conv_lstm_layer_gpu;
    result.backward_gpu := backward_conv_lstm_layer_gpu;
    result.update_gpu := update_conv_lstm_layer_gpu;
    result.output_gpu := cuda_make_array(0, batch * outputs * steps);
    result.delta_gpu := cuda_make_array(0, batch * result.outputs * steps);
    result.prev_state_gpu := cuda_make_array(0, batch * outputs);
    result.prev_cell_gpu := cuda_make_array(0, batch * outputs);
    result.cell_gpu := cuda_make_array(0, batch * outputs * steps);
    result.f_gpu := cuda_make_array(0, batch * outputs);
    result.i_gpu := cuda_make_array(0, batch * outputs);
    result.g_gpu := cuda_make_array(0, batch * outputs);
    result.o_gpu := cuda_make_array(0, batch * outputs);
    result.c_gpu := cuda_make_array(0, batch * outputs);
    if result.bottleneck then
        begin
            result.bottelneck_hi_gpu := cuda_make_array(0, batch * outputs * 2);
            result.bottelneck_delta_gpu := cuda_make_array(0, batch * outputs * 2)
        end;
    result.h_gpu := cuda_make_array(0, batch * outputs);
    result.stored_c_gpu := cuda_make_array(0, batch * outputs);
    result.stored_h_gpu := cuda_make_array(0, batch * outputs);
    result.temp_gpu := cuda_make_array(0, batch * outputs);
    result.temp2_gpu := cuda_make_array(0, batch * outputs);
    result.temp3_gpu := cuda_make_array(0, batch * outputs);
    result.dc_gpu := cuda_make_array(0, batch * outputs);
    result.dh_gpu := cuda_make_array(0, batch * outputs);
    result.last_prev_state_gpu := cuda_make_array(0, result.batch * result.outputs);
    result.last_prev_cell_gpu := cuda_make_array(0, result.batch * result.outputs);
{$endif}
    result.bflops := result.uf[0].bflops+result.ui[0].bflops+result.ug[0].bflops+result.uo[0].bflops+result.wf[0].bflops+result.wi[0].bflops+result.wg[0].bflops+result.wo[0].bflops+result.vf[0].bflops+result.vi[0].bflops+result.vo[0].bflops;
    if result.peephole then
        result.bflops := result.bflops + (12 * result.outputs * result.batch / 1000000000)
    else
        result.bflops := result.bflops + (9 * result.outputs * result.batch / 1000000000);
end;

function make_history_layer(const batch, h, w, c, history_size, steps: longint; const train: boolean):TLayer;
begin
    result := default(TLayer);
    result.train := train;
    result.batch := batch;
    result.&type := ltHISTORY;
    result.steps := steps;
    result.history_size := history_size;
    result.h := h;
    result.w := w;
    result.c := c;
    result.out_h := h;
    result.out_w := w;
    result.out_c := c * history_size;
    result.inputs := h * w * c;
    result.outputs := h * w * c * history_size;
    result.forward := forward_history_layer;
    result.backward := backward_history_layer;
    writeln(ErrOutput, format('HISTORY b = %d, s = %2d, steps = %2d   %4d x%4d x%4d -> %4d x%4d x%4d ', [result.batch div result.steps, result.history_size, result.steps, w, h, c, result.out_w, result.out_h, result.out_c]));
    result.output := TSingles.Create(result.batch * result.outputs);
    result.delta := TSingles.Create(result.batch * result.outputs);
    result.prev_state_cpu := TSingles.Create(result.batch * result.outputs);
{$ifdef GPU}
    result.forward_gpu := forward_history_layer_gpu;
    result.backward_gpu := backward_history_layer_gpu;
    result.output_gpu := cuda_make_array(0, result.batch * result.outputs);
    result.delta_gpu := cuda_make_array(0, result.batch * result.outputs);
    result.prev_state_gpu := cuda_make_array(0, result.batch * result.outputs);
{$endif}
end;

procedure forward_history_layer(var l: TLayer; const state: PNetworkState);
var
    prev_output: PSingle;
    batch, i, shift_size, output_sift, b, input_start, output_start: longint;
    input, output: PSingle;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    if l.steps = 1 then
        begin
            copy_cpu(l.inputs * l.batch, state.input, 1, l.output, 1);
            exit()
        end;
    batch := l.batch div l.steps;
    prev_output := l.prev_state_cpu;
    for i := 0 to l.steps -1 do
        begin
            shift_size := l.inputs * (l.history_size-1);
            output_sift := l.inputs;
            for b := 0 to batch -1 do
                begin
                    input_start := b * l.inputs+i * l.inputs * batch;
                    output_start := b * l.outputs+i * l.outputs * batch;
                    input := state.input+input_start;
                    output := l.output+output_start;
                    copy_cpu(shift_size, prev_output+b * l.outputs, 1, output+output_sift, 1);
                    copy_cpu(l.inputs, input, 1, output, 1)
                end;
            prev_output := l.output+i * l.outputs * batch
        end;
    output_start := (l.steps-1) * l.outputs * batch;
    copy_cpu(batch * l.outputs, l.output+output_start, 1, l.prev_state_cpu, 1);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_history_layer(var l: TLayer; const state: PNetworkState);
var
    batch, i, b, input_start, output_start: longint;
    state_delta, l_delta: PSingle;
begin
    if l.steps = 1 then
        begin
            axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, state.delta, 1);
            exit()
        end;
    batch := l.batch div l.steps;
    for i := 0 to l.steps -1 do
        begin
            for b := 0 to batch -1 do
                begin
                    input_start := b * l.inputs+i * l.inputs * batch;
                    output_start := b * l.outputs+i * l.outputs * batch;
                    state_delta := state.delta+input_start;
                    l_delta := l.delta+output_start;
                    axpy_cpu(l.inputs, 1, l_delta, 1, state_delta, 1)
                end
        end
end;

{$ifdef GPU}
procedure forward_history_layer_gpu(const l: layer; state: network_state);
var
    batch: longint;
    prev_output: PSingle;
    i: longint;
    shift_size: longint;
    output_sift: longint;
    b: longint;
    input_start: longint;
    output_start: longint;
    input: PSingle;
    output: PSingle;
    h: longint;
begin
    if l.steps = 1 then
        begin
            simple_copy_ongpu(l.inputs * l.batch, state.input, l.output_gpu);
            exit()
        end;
    batch := l.batch div l.steps;
    prev_output := l.prev_state_gpu;
    for i := 0 to l.steps -1 do
        begin
            shift_size := l.inputs * (l.history_size-1);
            output_sift := l.inputs;
            for b := 0 to batch -1 do
                begin
                    input_start := b * l.inputs+i * l.inputs * batch;
                    output_start := b * l.outputs+i * l.outputs * batch;
                    input := state.input+input_start;
                    output := l.output_gpu+output_start;
                    simple_copy_ongpu(shift_size, prev_output+b * l.outputs, output+output_sift);
                    simple_copy_ongpu(l.inputs, input, output);
                    for h := 1 to l.history_size -1 do

                end;
            prev_output := l.output_gpu+i * l.outputs * batch
        end;
    output_start := (l.steps-1) * l.outputs * batch;
    simple_copy_ongpu(batch * l.outputs, l.output_gpu+output_start, l.prev_state_gpu)
end;

procedure backward_history_layer_gpu(const l: layer; state: network_state);
var
    batch: longint;
    i: longint;
    b: longint;
    input_start: longint;
    output_start: longint;
    state_delta: PSingle;
    l_delta: PSingle;
begin
    if l.steps = 1 then
        begin
            axpy_ongpu(l.inputs * l.batch, 1, l.delta_gpu, 1, state.delta, 1);
            exit()
        end;
    batch := l.batch div l.steps;
    for i := 0 to l.steps -1 do
        begin
            for b := 0 to batch -1 do
                begin
                    input_start := b * l.inputs+i * l.inputs * batch;
                    output_start := b * l.outputs+i * l.outputs * batch;
                    state_delta := state.delta+input_start;
                    l_delta := l.delta_gpu+output_start;
                    axpy_ongpu(l.inputs, 1, l_delta, 1, state_delta, 1)
                end
        end
end;
{$endif}

procedure update_conv_lstm_layer(const l: TConvLSTMLayer; const arg:TUpdateArgs);// batch: longint; learning_rate: single; momentum: single; decay: single);
begin
    if l.peephole then
        begin
            update_convolutional_layer( l.vf[0], arg);
            update_convolutional_layer( l.vi[0], arg);
            update_convolutional_layer( l.vo[0], arg)
        end;
    update_convolutional_layer( l.wf[0], arg);
    update_convolutional_layer( l.wi[0], arg);
    update_convolutional_layer( l.wg[0], arg);
    update_convolutional_layer( l.wo[0], arg);
    update_convolutional_layer( l.uf[0], arg);
    update_convolutional_layer( l.ui[0], arg);
    update_convolutional_layer( l.ug[0], arg);
    update_convolutional_layer( l.uo[0], arg)
end;

procedure resize_conv_lstm_layer(var l: TConvLSTMLayer; const w, h: longint);
var
    outputs, steps, batch: longint;
begin
    if l.peephole then
        begin
            resize_convolutional_layer(l.vf[0], w, h);
            if l.workspace_size < l.vf[0].workspace_size then
                l.workspace_size := l.vf[0].workspace_size;
            resize_convolutional_layer(l.vi[0], w, h);
            if l.workspace_size < l.vi[0].workspace_size then
                l.workspace_size := l.vi[0].workspace_size;
            resize_convolutional_layer(l.vo[0], w, h);
            if l.workspace_size < l.vo[0].workspace_size then
                l.workspace_size := l.vo[0].workspace_size
        end;
    resize_convolutional_layer(l.wf[0], w, h);
    if l.workspace_size < l.wf[0].workspace_size then
        l.workspace_size := l.wf[0].workspace_size;
    resize_convolutional_layer(l.wi[0], w, h);
    if l.workspace_size < l.wi[0].workspace_size then
        l.workspace_size := l.wi[0].workspace_size;
    resize_convolutional_layer(l.wg[0], w, h);
    if l.workspace_size < l.wg[0].workspace_size then
        l.workspace_size := l.wg[0].workspace_size;
    resize_convolutional_layer(l.wo[0], w, h);
    if l.workspace_size < l.wo[0].workspace_size then
        l.workspace_size := l.wo[0].workspace_size;
    resize_convolutional_layer(l.uf[0], w, h);
    if l.workspace_size < l.uf[0].workspace_size then
        l.workspace_size := l.uf[0].workspace_size;
    resize_convolutional_layer(l.ui[0], w, h);
    if l.workspace_size < l.ui[0].workspace_size then
        l.workspace_size := l.ui[0].workspace_size;
    resize_convolutional_layer(l.ug[0], w, h);
    if l.workspace_size < l.ug[0].workspace_size then
        l.workspace_size := l.ug[0].workspace_size;
    resize_convolutional_layer(l.uo[0], w, h);
    if l.workspace_size < l.uo[0].workspace_size then
        l.workspace_size := l.uo[0].workspace_size;
    l.w := w;
    l.h := h;
    l.out_h := l.wo[0].out_h;
    l.out_w := l.wo[0].out_w;
    l.outputs := l.wo[0].outputs;
    outputs := l.outputs;
    l.inputs := w * h * l.c;
    steps := l.steps;
    batch := l.batch;
    assert(l.wo[0].outputs = l.uo[0].outputs);
    l.output.reAllocate(outputs * batch * steps);
    l.prev_state_cpu.reAllocate(batch * outputs);
    l.prev_cell_cpu.reAllocate(batch * outputs);
    l.cell_cpu.reAllocate(batch * outputs * steps);
    l.f_cpu.reAllocate(batch * outputs);
    l.i_cpu.reAllocate(batch * outputs);
    l.g_cpu.reAllocate(batch * outputs);
    l.o_cpu.reAllocate(batch * outputs);
    l.c_cpu.reAllocate(batch * outputs);
    l.h_cpu.reAllocate(batch * outputs);
    l.temp_cpu.reAllocate(batch * outputs);
    l.temp2_cpu.reAllocate(batch * outputs);
    l.temp3_cpu.reAllocate(batch * outputs);
    l.dc_cpu.reAllocate(batch * outputs);
    l.dh_cpu.reAllocate(batch * outputs);
    l.stored_c_cpu.reAllocate(batch * outputs);
    l.stored_h_cpu.reAllocate(batch * outputs);
{$ifdef GPU}
    if l.output_gpu then
        cudaFree(l.output_gpu);
    l.output_gpu := cuda_make_array(0, batch * outputs * steps);
    if l.delta_gpu then
        cudaFree(l.delta_gpu);
    l.delta_gpu := cuda_make_array(0, batch * outputs * steps);
    if l.prev_state_gpu then
        cudaFree(l.prev_state_gpu);
    l.prev_state_gpu := cuda_make_array(0, batch * outputs);
    if l.prev_cell_gpu then
        cudaFree(l.prev_cell_gpu);
    l.prev_cell_gpu := cuda_make_array(0, batch * outputs);
    if l.cell_gpu then
        cudaFree(l.cell_gpu);
    l.cell_gpu := cuda_make_array(0, batch * outputs * steps);
    if l.f_gpu then
        cudaFree(l.f_gpu);
    l.f_gpu := cuda_make_array(0, batch * outputs);
    if l.i_gpu then
        cudaFree(l.i_gpu);
    l.i_gpu := cuda_make_array(0, batch * outputs);
    if l.g_gpu then
        cudaFree(l.g_gpu);
    l.g_gpu := cuda_make_array(0, batch * outputs);
    if l.o_gpu then
        cudaFree(l.o_gpu);
    l.o_gpu := cuda_make_array(0, batch * outputs);
    if l.c_gpu then
        cudaFree(l.c_gpu);
    l.c_gpu := cuda_make_array(0, batch * outputs);
    if l.h_gpu then
        cudaFree(l.h_gpu);
    l.h_gpu := cuda_make_array(0, batch * outputs);
    if l.temp_gpu then
        cudaFree(l.temp_gpu);
    l.temp_gpu := cuda_make_array(0, batch * outputs);
    if l.temp2_gpu then
        cudaFree(l.temp2_gpu);
    l.temp2_gpu := cuda_make_array(0, batch * outputs);
    if l.temp3_gpu then
        cudaFree(l.temp3_gpu);
    l.temp3_gpu := cuda_make_array(0, batch * outputs);
    if l.dc_gpu then
        cudaFree(l.dc_gpu);
    l.dc_gpu := cuda_make_array(0, batch * outputs);
    if l.dh_gpu then
        cudaFree(l.dh_gpu);
    l.dh_gpu := cuda_make_array(0, batch * outputs);
    if l.stored_c_gpu then
        cudaFree(l.stored_c_gpu);
    l.stored_c_gpu := cuda_make_array(0, batch * outputs);
    if l.stored_h_gpu then
        cudaFree(l.stored_h_gpu);
    l.stored_h_gpu := cuda_make_array(0, batch * outputs);
    if l.last_prev_state_gpu then
        cudaFree(l.last_prev_state_gpu);
    l.last_prev_state_gpu := cuda_make_array(0, batch * outputs);
    if l.last_prev_cell_gpu then
        cudaFree(l.last_prev_cell_gpu);
    l.last_prev_cell_gpu := cuda_make_array(0, batch * outputs)
{$endif}
end;

procedure free_state_conv_lstm(const l: TConvLSTMLayer);
var
    i: longint;
begin
    for i := 0 to l.outputs * l.batch -1 do
        l.h_cpu[i] := 0;
    for i := 0 to l.outputs * l.batch -1 do
        l.c_cpu[i] := 0;
{$ifdef GPU}
    cuda_push_array(l.h_gpu, l.h_cpu, l.outputs * l.batch);
    cuda_push_array(l.c_gpu, l.c_cpu, l.outputs * l.batch)
{$endif}
end;

procedure randomize_state_conv_lstm(const l: TConvLSTMLayer);
var
    i: longint;
begin
    for i := 0 to l.outputs * l.batch -1 do
        l.h_cpu[i] := rand_uniform(-1, 1);
    for i := 0 to l.outputs * l.batch -1 do
        l.c_cpu[i] := rand_uniform(-1, 1);
{$ifdef GPU}
    cuda_push_array(l.h_gpu, l.h_cpu, l.outputs * l.batch);
    cuda_push_array(l.c_gpu, l.c_cpu, l.outputs * l.batch)
{$endif}
end;

procedure remember_state_conv_lstm(const l: TConvLSTMLayer);
begin
    move(l.c_cpu[0], l.stored_c_cpu[0], l.outputs * l.batch * sizeof(single));
    move(l.h_cpu[0], l.stored_h_cpu[0],  l.outputs * l.batch * sizeof(single));
{$ifdef GPU}
    copy_ongpu(l.outputs * l.batch, l.c_gpu, 1, l.stored_c_gpu, 1);
    copy_ongpu(l.outputs * l.batch, l.h_gpu, 1, l.stored_h_gpu, 1)
{$endif}
end;

procedure restore_state_conv_lstm(const l: TConvLSTMLayer);
begin
    move(l.stored_c_cpu[0], l.c_cpu[0], l.outputs * l.batch * sizeof(single));
    move(l.stored_h_cpu[0], l.h_cpu[0], l.outputs * l.batch * sizeof(single));
{$ifdef GPU}
    copy_ongpu(l.outputs * l.batch, l.stored_c_gpu, 1, l.c_gpu, 1);
    copy_ongpu(l.outputs * l.batch, l.stored_h_gpu, 1, l.h_gpu, 1)
{$endif}
end;

procedure forward_conv_lstm_layer(var l: TConvLSTMLayer;
  const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    vf,vi,vo,wf,wi,wg,wo,uf,ui,ug,uo: TLayer;
begin
    s := default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    vf :=  l.vf[0];
    vi :=  l.vi[0];
    vo :=  l.vo[0];
    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];
    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];
    if state.train then
        begin
            if l.peephole then
                begin
                    fill_cpu(l.outputs * l.batch * l.steps, 0, vf.delta, 1);
                    fill_cpu(l.outputs * l.batch * l.steps, 0, vi.delta, 1);
                    fill_cpu(l.outputs * l.batch * l.steps, 0, vo.delta, 1)
                end;
            fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
            fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            if l.peephole then
                begin
                    assert(l.outputs = vf.out_w * vf.out_h * vf.out_c);
                    s.input := l.c_cpu;
                    forward_convolutional_layer(vf, @s);
                    forward_convolutional_layer(vi, @s)
                end;
            assert(l.outputs = wf.out_w * wf.out_h * wf.out_c);
            assert((wf.c = l.out_c) and (wi.c = l.out_c) and (wg.c = l.out_c) and (wo.c = l.out_c));
            s.input := l.h_cpu;
            forward_convolutional_layer(wf, @s);
            forward_convolutional_layer(wi, @s);
            forward_convolutional_layer(wg, @s);
            forward_convolutional_layer(wo, @s);
            assert(l.inputs = uf.w * uf.h * uf.c);
            assert((uf.c = l.c) and (ui.c = l.c) and (ug.c = l.c) and (uo.c = l.c));
            s.input := state.input;
            forward_convolutional_layer(uf, @s);
            forward_convolutional_layer(ui, @s);
            forward_convolutional_layer(ug, @s);
            forward_convolutional_layer(uo, @s);
            copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);
            if l.peephole then
                axpy_cpu(l.outputs * l.batch, 1, vf.output, 1, l.f_cpu, 1);
            copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);
            if l.peephole then
                axpy_cpu(l.outputs * l.batch, 1, vi.output, 1, l.i_cpu, 1);
            copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);
            activate_array(l.f_cpu, l.outputs * l.batch, acLOGISTIC);
            activate_array(l.i_cpu, l.outputs * l.batch, acLOGISTIC);
            activate_array(l.g_cpu, l.outputs * l.batch, acTANH);
            copy_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
            mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
            mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.c_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);
            if l.peephole then
                begin
                    s.input := l.c_cpu;
                    forward_convolutional_layer(vo, @s)
                end;
            copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);
            if l.peephole then
                axpy_cpu(l.outputs * l.batch, 1, vo.output, 1, l.o_cpu, 1);
            activate_array(l.o_cpu, l.outputs * l.batch, acLOGISTIC);
            copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.h_cpu, 1);
            activate_array(l.h_cpu, l.outputs * l.batch, acTANH);
            mul_cpu(l.outputs * l.batch, l.o_cpu, 1, l.h_cpu, 1);
            if l.state_constrain<>0 then
                constrain_cpu(l.outputs * l.batch, l.state_constrain, l.c_cpu);
            fix_nan_and_inf_cpu(l.c_cpu, l.outputs * l.batch);
            fix_nan_and_inf_cpu(l.h_cpu, l.outputs * l.batch);
            copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.cell_cpu, 1);
            copy_cpu(l.outputs * l.batch, l.h_cpu, 1, l.output, 1);
            state.input := state.input + (l.inputs * l.batch);
            l.output := l.output + (l.outputs * l.batch);
            l.cell_cpu := l.cell_cpu + (l.outputs * l.batch);
            if l.peephole then
                begin
                    increment_layer( @vf, 1);
                    increment_layer( @vi, 1);
                    increment_layer( @vo, 1)
                end;
            increment_layer( @wf, 1);
            increment_layer( @wi, 1);
            increment_layer( @wg, 1);
            increment_layer( @wo, 1);
            increment_layer( @uf, 1);
            increment_layer( @ui, 1);
            increment_layer( @ug, 1);
            increment_layer( @uo, 1)
        end
end;

procedure backward_conv_lstm_layer(var l: TConvLSTMLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    vf, vi, vo, wf, wi, wg, wo, uf, ui, ug, uo: TConvLSTMLayer;
begin
    s := default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;
    vf :=  l.vf[0];
    vi :=  l.vi[0];
    vo :=  l.vo[0];
    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];
    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];
    if l.peephole then
        begin
            increment_layer( @vf, l.steps-1);
            increment_layer( @vi, l.steps-1);
            increment_layer( @vo, l.steps-1)
        end;
    increment_layer( @wf, l.steps-1);
    increment_layer( @wi, l.steps-1);
    increment_layer( @wg, l.steps-1);
    increment_layer( @wo, l.steps-1);
    increment_layer( @uf, l.steps-1);
    increment_layer( @ui, l.steps-1);
    increment_layer( @ug, l.steps-1);
    increment_layer( @uo, l.steps-1);
    state.input := state.input + (l.inputs * l.batch * (l.steps-1));
    if assigned(state.delta) then
        state.delta := state.delta + (l.inputs * l.batch * (l.steps-1));
    l.output := l.output + (l.outputs * l.batch * (l.steps-1));
    l.cell_cpu := l.cell_cpu + (l.outputs * l.batch * (l.steps-1));
    l.delta := l.delta + (l.outputs * l.batch * (l.steps-1));
    i := l.steps-1;
    while i >= 0 do begin
        if i <> 0 then
            copy_cpu(l.outputs * l.batch, l.cell_cpu-l.outputs * l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if i <> 0 then
            copy_cpu(l.outputs * l.batch, l.output-l.outputs * l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.output, 1, l.h_cpu, 1);
        if i = 0 then
            l.dh_cpu := nil
        else
            l.dh_cpu := l.delta-l.outputs * l.batch;
        copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);
        if l.peephole then
            axpy_cpu(l.outputs * l.batch, 1, vf.output, 1, l.f_cpu, 1);
        copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);
        if l.peephole then
            axpy_cpu(l.outputs * l.batch, 1, vi.output, 1, l.i_cpu, 1);
        copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);
        copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);
        if l.peephole then
            axpy_cpu(l.outputs * l.batch, 1, vo.output, 1, l.o_cpu, 1);
        activate_array(l.f_cpu, l.outputs * l.batch, acLOGISTIC);
        activate_array(l.i_cpu, l.outputs * l.batch, acLOGISTIC);
        activate_array(l.g_cpu, l.outputs * l.batch, acTANH);
        activate_array(l.o_cpu, l.outputs * l.batch, acLOGISTIC);
        copy_cpu(l.outputs * l.batch, l.delta, 1, l.temp3_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs * l.batch, acTANH);
        copy_cpu(l.outputs * l.batch, l.temp3_cpu, 1, l.temp2_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.o_cpu, 1, l.temp2_cpu, 1);
        gradient_array(l.temp_cpu, l.outputs * l.batch, acTANH, l.temp2_cpu);
        axpy_cpu(l.outputs * l.batch, 1, l.dc_cpu, 1, l.temp2_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.temp_cpu, 1);
        activate_array(l.temp_cpu, l.outputs * l.batch, acTANH);
        mul_cpu(l.outputs * l.batch, l.temp3_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.o_cpu, l.outputs * l.batch, acLOGISTIC, l.temp_cpu);
        if l.peephole then
            begin
                copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vo.delta, 1);
                s.input := l.cell_cpu;
                backward_convolutional_layer(vo, @s)
            end;
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input := l.prev_state_cpu;
        backward_convolutional_layer(wo, @s);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_convolutional_layer(uo, @s);
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.g_cpu, l.outputs * l.batch, acTANH, l.temp_cpu);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input := l.prev_state_cpu;
        backward_convolutional_layer(wg, @s);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_convolutional_layer(ug, @s);
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.i_cpu, l.outputs * l.batch, acLOGISTIC, l.temp_cpu);
        if l.peephole then
            begin
                copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vi.delta, 1);
                s.input := l.prev_cell_cpu;
                backward_convolutional_layer(vi, @s)
            end;
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input := l.prev_state_cpu;
        backward_convolutional_layer(wi, @s);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_convolutional_layer(ui, @s);
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs * l.batch, acLOGISTIC, l.temp_cpu);
        if l.peephole then
            begin
                copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, vf.delta, 1);
                s.input := l.prev_cell_cpu;
                backward_convolutional_layer(vf, @s)
            end;
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input := l.prev_state_cpu;
        backward_convolutional_layer(wf, @s);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_convolutional_layer(uf, @s);
        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.temp_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, l.dc_cpu, 1);
        state.input := state.input - (l.inputs * l.batch);
        if assigned(state.delta) then
            state.delta := state.delta - (l.inputs * l.batch);
        l.output := l.output - (l.outputs * l.batch);
        l.cell_cpu := l.cell_cpu - (l.outputs * l.batch);
        l.delta := l.delta - (l.outputs * l.batch);
        if l.peephole then
            begin
                increment_layer( @vf, -1);
                increment_layer( @vi, -1);
                increment_layer( @vo, -1)
            end;
        increment_layer( @wf, -1);
        increment_layer( @wi, -1);
        increment_layer( @wg, -1);
        increment_layer( @wo, -1);
        increment_layer( @uf, -1);
        increment_layer( @ui, -1);
        increment_layer( @ug, -1);
        increment_layer( @uo, -1);
        dec(i)
    end
end;
{$ifdef GPU}
procedure pull_conv_lstm_layer(l: layer);
begin
    if l.peephole then
        begin
            pull_convolutional_layer( l.vf[0]);
            pull_convolutional_layer( l.vi[0]);
            pull_convolutional_layer( l.vo[0])
        end;
    pull_convolutional_layer( * (l.wf));
    if not l.bottleneck then
        begin
            pull_convolutional_layer( l.wi[0]);
            pull_convolutional_layer( l.wg[0]);
            pull_convolutional_layer( l.wo[0])
        end;
    pull_convolutional_layer( l.uf[0]);
    pull_convolutional_layer( l.ui[0]);
    pull_convolutional_layer( l.ug[0]);
    pull_convolutional_layer( l.uo[0])
end;

procedure push_conv_lstm_layer(l: layer);
begin
    if l.peephole then
        begin
            push_convolutional_layer( * (l.vf));
            push_convolutional_layer( * (l.vi));
            push_convolutional_layer( * (l.vo))
        end;
    push_convolutional_layer( * (l.wf));
    if not l.bottleneck then
        begin
            push_convolutional_layer( * (l.wi));
            push_convolutional_layer( * (l.wg));
            push_convolutional_layer( * (l.wo))
        end;
    push_convolutional_layer( * (l.uf));
    push_convolutional_layer( * (l.ui));
    push_convolutional_layer( * (l.ug));
    push_convolutional_layer( * (l.uo))
end;

procedure update_conv_lstm_layer_gpu(l: layer; batch: longint; learning_rate: single; momentum: single; decay: single; loss_scale: single);
begin
    if l.peephole then
        begin
            update_convolutional_layer_gpu( * (l.vf), batch, learning_rate, momentum, decay, loss_scale);
            update_convolutional_layer_gpu( * (l.vi), batch, learning_rate, momentum, decay, loss_scale);
            update_convolutional_layer_gpu( * (l.vo), batch, learning_rate, momentum, decay, loss_scale)
        end;
    update_convolutional_layer_gpu( * (l.wf), batch, learning_rate, momentum, decay, loss_scale);
    if not l.bottleneck then
        begin
            update_convolutional_layer_gpu( * (l.wi), batch, learning_rate, momentum, decay, loss_scale);
            update_convolutional_layer_gpu( * (l.wg), batch, learning_rate, momentum, decay, loss_scale);
            update_convolutional_layer_gpu( * (l.wo), batch, learning_rate, momentum, decay, loss_scale)
        end;
    update_convolutional_layer_gpu( * (l.uf), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu( * (l.ui), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu( * (l.ug), batch, learning_rate, momentum, decay, loss_scale);
    update_convolutional_layer_gpu( * (l.uo), batch, learning_rate, momentum, decay, loss_scale)
end;

procedure forward_conv_lstm_layer_gpu(l: layer; state: network_state);
var
    s: network_state;
    i: longint;
    vf: layer;
    vi: layer;
    vo: layer;
    wf: layer;
    wi: layer;
    wg: layer;
    wo: layer;
    uf: layer;
    ui: layer;
    ug: layer;
    uo: layer;
begin
    s := [0];
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    if not state.train then
        s.index := state.index;
    vf :=  * (l.vf);
    vi :=  * (l.vi);
    vo :=  * (l.vo);
    wf :=  * (l.wf);
    wi :=  * (l.wi);
    wg :=  * (l.wg);
    wo :=  * (l.wo);
    uf :=  * (l.uf);
    ui :=  * (l.ui);
    ug :=  * (l.ug);
    uo :=  * (l.uo);
    if state.train then
        begin
            if l.peephole then
                begin
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, vf.delta_gpu, 1);
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, vi.delta_gpu, 1);
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, vo.delta_gpu, 1)
                end;
            fill_ongpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
            if not l.bottleneck then
                begin
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
                    fill_ongpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1)
                end;
            fill_ongpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
            fill_ongpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            if l.peephole then
                begin
                    assert(l.outputs = vf.out_w * vf.out_h * vf.out_c);
                    s.input := l.c_gpu;
                    forward_convolutional_layer_gpu(vf, s);
                    forward_convolutional_layer_gpu(vi, s)
                end;
            if l.bottleneck then
                begin
                    simple_copy_ongpu(l.outputs * l.batch, l.h_gpu, l.bottelneck_hi_gpu);
                    simple_copy_ongpu(l.outputs * l.batch, state.input, l.bottelneck_hi_gpu+l.outputs * l.batch);
                    s.input := l.bottelneck_hi_gpu;
                    forward_convolutional_layer_gpu(wf, s);
                    activate_array_ongpu(wf.output_gpu, l.outputs * l.batch, l.lstm_activation);
                    s.input := wf.output_gpu
                end
            else
                begin
                    assert(l.outputs = wf.out_w * wf.out_h * wf.out_c);
                    assert((wf.c = l.out_c) and (wi.c = l.out_c) and (wg.c = l.out_c) and (wo.c = l.out_c));
                    s.input := l.h_gpu;
                    forward_convolutional_layer_gpu(wf, s);
                    forward_convolutional_layer_gpu(wi, s);
                    forward_convolutional_layer_gpu(wg, s);
                    forward_convolutional_layer_gpu(wo, s);
                    s.input := state.input
                end;
            assert(l.inputs = uf.w * uf.h * uf.c);
            assert((uf.c = l.c) and (ui.c = l.c) and (ug.c = l.c) and (uo.c = l.c));
            forward_convolutional_layer_gpu(uf, s);
            forward_convolutional_layer_gpu(ui, s);
            forward_convolutional_layer_gpu(ug, s);
            forward_convolutional_layer_gpu(uo, s);
            add_3_arrays_activate(ifthen((l.bottleneck), NULL, wf.output_gpu), uf.output_gpu, ifthen((l.peephole), vf.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.f_gpu);
            add_3_arrays_activate(ifthen((l.bottleneck), NULL, wi.output_gpu), ui.output_gpu, ifthen((l.peephole), vi.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.i_gpu);
            add_3_arrays_activate(ifthen((l.bottleneck), NULL, wg.output_gpu), ug.output_gpu, NULL, l.outputs * l.batch, l.lstm_activation, l.g_gpu);
            sum_of_mults(l.f_gpu, l.c_gpu, l.i_gpu, l.g_gpu, l.outputs * l.batch, l.c_gpu);
            if l.peephole then
                begin
                    s.input := l.c_gpu;
                    forward_convolutional_layer_gpu(vo, s)
                end;
            add_3_arrays_activate(ifthen((l.bottleneck), NULL, wo.output_gpu), uo.output_gpu, ifthen((l.peephole), vo.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.o_gpu);
            activate_and_mult(l.c_gpu, l.o_gpu, l.outputs * l.batch, l.lstm_activation, l.h_gpu);
            fix_nan_and_inf(l.c_gpu, l.outputs * l.batch);
            fix_nan_and_inf(l.h_gpu, l.outputs * l.batch);
            if l.state_constrain then
                constrain_ongpu(l.outputs * l.batch, l.state_constrain, l.c_gpu, 1);
            if state.train then
                simple_copy_ongpu(l.outputs * l.batch, l.c_gpu, l.cell_gpu);
            simple_copy_ongpu(l.outputs * l.batch, l.h_gpu, l.output_gpu);
            if l.shortcut then
                if l.bottleneck then
                    axpy_ongpu(l.outputs * l.batch div 2, 1, wf.output_gpu, 1, l.output_gpu, 1);
            state.input := state.input + (l.inputs * l.batch);
            l.output_gpu := l.output_gpu + (l.outputs * l.batch);
            l.cell_gpu := l.cell_gpu + (l.outputs * l.batch);
            if l.peephole then
                begin
                    increment_layer( and vf, 1);
                    increment_layer( and vi, 1);
                    increment_layer( and vo, 1)
                end;
            increment_layer( and wf, 1);
            increment_layer( and wi, 1);
            increment_layer( and wg, 1);
            increment_layer( and wo, 1);
            increment_layer( and uf, 1);
            increment_layer( and ui, 1);
            increment_layer( and ug, 1);
            increment_layer( and uo, 1)
        end
end;

procedure backward_conv_lstm_layer_gpu(l: layer; state: network_state);
var
    last_output: PSingle;
    last_cell: PSingle;
    s: network_state;
    i: longint;
    vf: layer;
    vi: layer;
    vo: layer;
    wf: layer;
    wi: layer;
    wg: layer;
    wo: layer;
    uf: layer;
    ui: layer;
    ug: layer;
    uo: layer;
    sequence: longint;
begin
    last_output := l.output_gpu+l.outputs * l.batch * (l.steps-1);
    last_cell := l.cell_gpu+l.outputs * l.batch * (l.steps-1);
    s := [0];
    s.train := state.train;
    s.workspace := state.workspace;
    s.net := state.net;
    vf :=  * (l.vf);
    vi :=  * (l.vi);
    vo :=  * (l.vo);
    wf :=  * (l.wf);
    wi :=  * (l.wi);
    wg :=  * (l.wg);
    wo :=  * (l.wo);
    uf :=  * (l.uf);
    ui :=  * (l.ui);
    ug :=  * (l.ug);
    uo :=  * (l.uo);
    if l.peephole then
        begin
            increment_layer( and vf, l.steps-1);
            increment_layer( and vi, l.steps-1);
            increment_layer( and vo, l.steps-1)
        end;
    increment_layer( and wf, l.steps-1);
    increment_layer( and wi, l.steps-1);
    increment_layer( and wg, l.steps-1);
    increment_layer( and wo, l.steps-1);
    increment_layer( and uf, l.steps-1);
    increment_layer( and ui, l.steps-1);
    increment_layer( and ug, l.steps-1);
    increment_layer( and uo, l.steps-1);
    state.input := state.input + (l.inputs * l.batch * (l.steps-1));
    if state.delta then
        state.delta := state.delta + (l.inputs * l.batch * (l.steps-1));
    l.output_gpu := l.output_gpu + (l.outputs * l.batch * (l.steps-1));
    l.cell_gpu := l.cell_gpu + (l.outputs * l.batch * (l.steps-1));
    l.delta_gpu := l.delta_gpu + (l.outputs * l.batch * (l.steps-1));
    sequence := get_sequence_value(state.net);
    i := l.steps-1;
    while i >= 0 do begin
        if i <> 0 then
            simple_copy_ongpu(l.outputs * l.batch, l.cell_gpu-l.outputs * l.batch, l.prev_cell_gpu)
        else
            if state.net.current_subdivision mod sequence <> 0 then
                simple_copy_ongpu(l.outputs * l.batch, l.last_prev_cell_gpu, l.prev_cell_gpu);
        simple_copy_ongpu(l.outputs * l.batch, l.cell_gpu, l.c_gpu);
        if i <> 0 then
            simple_copy_ongpu(l.outputs * l.batch, l.output_gpu-l.outputs * l.batch, l.prev_state_gpu)
        else
            if state.net.current_subdivision mod sequence <> 0 then
                simple_copy_ongpu(l.outputs * l.batch, l.last_prev_state_gpu, l.prev_state_gpu);
        simple_copy_ongpu(l.outputs * l.batch, l.output_gpu, l.h_gpu);
        l.dh_gpu := ifthen((i = 0), 0, l.delta_gpu-l.outputs * l.batch);
        add_3_arrays_activate(ifthen((l.bottleneck), NULL, wf.output_gpu), uf.output_gpu, ifthen((l.peephole), vf.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.f_gpu);
        add_3_arrays_activate(ifthen((l.bottleneck), NULL, wi.output_gpu), ui.output_gpu, ifthen((l.peephole), vi.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.i_gpu);
        add_3_arrays_activate(ifthen((l.bottleneck), NULL, wg.output_gpu), ug.output_gpu, NULL, l.outputs * l.batch, l.lstm_activation, l.g_gpu);
        add_3_arrays_activate(ifthen((l.bottleneck), NULL, wo.output_gpu), uo.output_gpu, ifthen((l.peephole), vo.output_gpu, NULL), l.outputs * l.batch, LOGISTIC, l.o_gpu);
        simple_copy_ongpu(l.outputs * l.batch, l.delta_gpu, l.temp3_gpu);
        simple_copy_ongpu(l.outputs * l.batch, l.c_gpu, l.temp_gpu);
        activate_array_ongpu(l.temp_gpu, l.outputs * l.batch, l.lstm_activation);
        simple_copy_ongpu(l.outputs * l.batch, l.temp3_gpu, l.temp2_gpu);
        mul_ongpu(l.outputs * l.batch, l.o_gpu, 1, l.temp2_gpu, 1);
        gradient_array_ongpu(l.temp_gpu, l.outputs * l.batch, l.lstm_activation, l.temp2_gpu);
        axpy_ongpu(l.outputs * l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);
        simple_copy_ongpu(l.outputs * l.batch, l.c_gpu, l.temp_gpu);
        activate_array_ongpu(l.temp_gpu, l.outputs * l.batch, l.lstm_activation);
        mul_ongpu(l.outputs * l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.o_gpu, l.outputs * l.batch, LOGISTIC, l.temp_gpu);
        if l.peephole then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, vo.delta_gpu);
                s.input := l.cell_gpu;
                backward_convolutional_layer_gpu(vo, s)
            end;
        if not l.bottleneck then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, wo.delta_gpu);
                s.input := l.prev_state_gpu;
                s.delta := l.temp3_gpu;
                fill_ongpu(l.outputs * l.batch, 0, l.temp3_gpu, 1);
                backward_convolutional_layer_gpu(wo, s)
            end;
        simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, uo.delta_gpu);
        if l.bottleneck then
            begin
                s.input := wf.output_gpu;
                s.delta := wf.delta_gpu
            end
        else
            begin
                s.input := state.input;
                s.delta := state.delta
            end;
        backward_convolutional_layer_gpu(uo, s);
        simple_copy_ongpu(l.outputs * l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs * l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.g_gpu, l.outputs * l.batch, l.lstm_activation, l.temp_gpu);
        if not l.bottleneck then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, wg.delta_gpu);
                s.input := l.prev_state_gpu;
                s.delta := l.temp3_gpu;
                backward_convolutional_layer_gpu(wg, s)
            end;
        simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, ug.delta_gpu);
        if l.bottleneck then
            begin
                s.input := wf.output_gpu;
                s.delta := wf.delta_gpu
            end
        else
            begin
                s.input := state.input;
                s.delta := state.delta
            end;
        backward_convolutional_layer_gpu(ug, s);
        simple_copy_ongpu(l.outputs * l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs * l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.i_gpu, l.outputs * l.batch, LOGISTIC, l.temp_gpu);
        if l.peephole then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, vi.delta_gpu);
                s.input := l.prev_cell_gpu;
                backward_convolutional_layer_gpu(vi, s)
            end;
        if not l.bottleneck then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, wi.delta_gpu);
                s.input := l.prev_state_gpu;
                s.delta := l.temp3_gpu;
                backward_convolutional_layer_gpu(wi, s)
            end;
        simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, ui.delta_gpu);
        if l.bottleneck then
            begin
                s.input := wf.output_gpu;
                s.delta := wf.delta_gpu
            end
        else
            begin
                s.input := state.input;
                s.delta := state.delta
            end;
        backward_convolutional_layer_gpu(ui, s);
        simple_copy_ongpu(l.outputs * l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs * l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_ongpu(l.f_gpu, l.outputs * l.batch, LOGISTIC, l.temp_gpu);
        if l.peephole then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, vf.delta_gpu);
                s.input := l.prev_cell_gpu;
                backward_convolutional_layer_gpu(vf, s)
            end;
        simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, uf.delta_gpu);
        if l.bottleneck then
            begin
                s.input := wf.output_gpu;
                s.delta := wf.delta_gpu
            end
        else
            begin
                s.input := state.input;
                s.delta := state.delta
            end;
        backward_convolutional_layer_gpu(uf, s);
        if l.bottleneck then
            begin
                simple_copy_ongpu(l.outputs * l.batch, l.prev_state_gpu, l.bottelneck_hi_gpu);
                simple_copy_ongpu(l.outputs * l.batch, state.input, l.bottelneck_hi_gpu+l.outputs * l.batch);
                fill_ongpu(l.outputs * l.batch * 2, 0, l.bottelneck_delta_gpu, 1);
                s.input := l.bottelneck_hi_gpu;
                s.delta := l.bottelneck_delta_gpu;
                if l.shortcut then
                    axpy_ongpu(l.outputs * l.batch div 2, 1, l.delta_gpu, 1, wf.delta_gpu, 1);
                gradient_array_ongpu(wf.output_gpu, l.outputs * l.batch, l.lstm_activation, wf.delta_gpu);
                reset_nan_and_inf(wf.delta_gpu, l.outputs * l.batch);
                constrain_ongpu(l.outputs * l.batch, 1, wf.delta_gpu, 1)
            end
        else
            begin
                s.input := l.prev_state_gpu;
                simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, wf.delta_gpu);
                s.delta := l.temp3_gpu
            end;
        backward_convolutional_layer_gpu(wf, s);
        if l.bottleneck then
            begin
                reset_nan_and_inf(l.bottelneck_delta_gpu, l.outputs * l.batch * 2);
                if l.dh_gpu then
                    axpy_ongpu(l.outputs * l.batch, l.time_normalizer, l.bottelneck_delta_gpu, 1, l.dh_gpu, 1);
                axpy_ongpu(l.outputs * l.batch, 1, l.bottelneck_delta_gpu+l.outputs * l.batch, 1, state.delta, 1)
            end
        //else
        // if l.dh_gpu then
        //    axpy_ongpu(l.outputs*l.batch, l.time_normalizer, l.temp3_gpu, 1, l.dh_gpu, 1);
            ;
        simple_copy_ongpu(l.outputs * l.batch, l.temp2_gpu, l.temp_gpu);
        mul_ongpu(l.outputs * l.batch, l.f_gpu, 1, l.temp_gpu, 1);
        simple_copy_ongpu(l.outputs * l.batch, l.temp_gpu, l.dc_gpu);
        reset_nan_and_inf(l.dc_gpu, l.outputs * l.batch);
        if i <> 0 then
            reset_nan_and_inf(l.dh_gpu, l.outputs * l.batch);
        state.input := state.input - (l.inputs * l.batch);
        if state.delta then
            state.delta := state.delta - (l.inputs * l.batch);
        l.output_gpu := l.output_gpu - (l.outputs * l.batch);
        l.cell_gpu := l.cell_gpu - (l.outputs * l.batch);
        l.delta_gpu := l.delta_gpu - (l.outputs * l.batch);
        if l.peephole then
            begin
                increment_layer( and vf, -1);
                increment_layer( and vi, -1);
                increment_layer( and vo, -1)
            end;
        increment_layer( and wf, -1);
        increment_layer( and wi, -1);
        increment_layer( and wg, -1);
        increment_layer( and wo, -1);
        increment_layer( and uf, -1);
        increment_layer( and ui, -1);
        increment_layer( and ug, -1);
        increment_layer( and uo, -1);
        &ced(i)
    end;
    simple_copy_ongpu(l.outputs * l.batch, last_output, l.last_prev_state_gpu);
    simple_copy_ongpu(l.outputs * l.batch, last_cell, l.last_prev_cell_gpu)
end;
{$endif}

end.

