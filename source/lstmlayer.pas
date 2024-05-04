unit LSTMLayer;

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
  SysUtils, lightnet, blas, ConnectedLayer, Activations;

type
    TLSTMLayer = TLayer;

function make_lstm_layer(batch: longint ;const inputs, outputs, steps:longint; const batch_normalize: boolean):TLSTMLayer;
procedure update_lstm_layer(const l: TLSTMLayer; const a: TUpdateArgs);
procedure forward_lstm_layer(var l: TLSTMLayer; const state: PNetworkState);
procedure backward_lstm_layer(var l: TLSTMLayer; const state: PNetworkState);

{$ifdef GPU}
procedure update_lstm_layer_gpu(l: TLSTMLayer; a: update_args);
procedure forward_lstm_layer_gpu(l: TLSTMLayer; state: TNetwork);
procedure backward_lstm_layer_gpu(l: TLSTMLayer; state: TNetwork);
{$endif}

implementation

procedure increment_layer(var l: TConnectedlayer; const steps: longint);
var
    num: longint;
begin
    num := l.outputs * l.batch * steps;
    inc(l.output, num);
    inc(l.delta, num);
    inc(l.x, num);
    inc(l.x_norm, num);
{$ifdef GPU}
    inc(l.output_gpu, num);
    inc(l.delta_gpu, num);
    inc(l.x_gpu, num);
    inc(l.x_norm_gpu, num);
{$endif}
end;

function make_lstm_layer(batch: longint ;const inputs, outputs, steps:longint; const batch_normalize: boolean):TLSTMLayer;
begin
    writeln(ErrOutput, format('LSTM Layer: %d inputs, %d outputs', [inputs, outputs]));
    batch := batch div steps;
    result := Default(TLSTMLayer);
    result.batch := batch;
    result.&type := ltLSTM;
    result.steps := steps;
    result.inputs := inputs;

    result.out_w := 1;
    result.out_h := 1;
    result.out_c := outputs;

    setLength(result.uf,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.uf[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize);
    result.uf[0].batch := batch;
    if result.uf[0].workspace_size> result.workspace_size then result.workspace_size := result.uf[0].workspace_size;

    setLength(result.ui,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.ui[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize);
    result.ui[0].batch := batch;
    if result.ui[0].workspace_size> result.workspace_size then result.workspace_size := result.ui[0].workspace_size;

    setLength(result.ug,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.ug[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize);
    result.ug[0].batch := batch;
    if result.ug[0].workspace_size> result.workspace_size then result.workspace_size := result.ug[0].workspace_size;

    setLength(result.uo,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.uo[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize);
    result.uo[0].batch := batch;
    if result.uo[0].workspace_size> result.workspace_size then result.workspace_size := result.uo[0].workspace_size;

    setLength(result.wf,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.wf[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize);
    result.wf[0].batch := batch;
    if result.wf[0].workspace_size> result.workspace_size then result.workspace_size := result.wf[0].workspace_size;

    setLength(result.wi,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.wi[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize);
    result.wi[0].batch := batch;
    if result.wi[0].workspace_size> result.workspace_size then result.workspace_size := result.wi[0].workspace_size;

    setLength(result.wg,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.wg[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize);
    result.wg[0].batch := batch;
    if result.wg[0].workspace_size> result.workspace_size then result.workspace_size := result.wg[0].workspace_size;

    setLength(result.wo,1);// := AllocMem(sizeof(TConnectedLayer));
    write(ErrOutput, #9#9);
    result.wo[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize);
    result.wo[0].batch := batch;
    if result.wo[0].workspace_size> result.workspace_size then result.workspace_size := result.wo[0].workspace_size;

    result.batch_normalize := batch_normalize;
    result.outputs := outputs;

    result.output := TSingles.Create(outputs * batch * steps);
    result.state := TSingles.Create(outputs * batch);

    result.forward := forward_lstm_layer;
    result.backward := backward_lstm_layer;
    result.update := update_lstm_layer;

    result.prev_state_cpu := TSingles.Create(batch * outputs);
    result.prev_cell_cpu := TSingles.Create(batch * outputs);
    result.cell_cpu := TSingles.Create(batch * outputs * steps);

    result.f_cpu := TSingles.Create(batch * outputs);
    result.i_cpu := TSingles.Create(batch * outputs);
    result.g_cpu := TSingles.Create(batch * outputs);
    result.o_cpu := TSingles.Create(batch * outputs);
    result.c_cpu := TSingles.Create(batch * outputs);
    result.h_cpu := TSingles.Create(batch * outputs);

    result.temp_cpu := TSingles.Create(batch * outputs);
    result.temp2_cpu := TSingles.Create(batch * outputs);
    result.temp3_cpu := TSingles.Create(batch * outputs);
    result.dc_cpu := TSingles.Create(batch * outputs);
    result.dh_cpu := TSingles.Create(batch * outputs);
{$ifdef GPU}
    result.forward_gpu := forward_lstm_layer_gpu;
    result.backward_gpu := backward_lstm_layer_gpu;
    result.update_gpu := update_lstm_layer_gpu;
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
    result.h_gpu := cuda_make_array(0, batch * outputs);
    result.temp_gpu := cuda_make_array(0, batch * outputs);
    result.temp2_gpu := cuda_make_array(0, batch * outputs);
    result.temp3_gpu := cuda_make_array(0, batch * outputs);
    result.dc_gpu := cuda_make_array(0, batch * outputs);
    result.dh_gpu := cuda_make_array(0, batch * outputs);
  {$ifdef CUDNN}
    cudnnSetTensor4dDescriptor(result.wf.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wf.out_c, result.wf.out_h, result.wf.out_w);
    cudnnSetTensor4dDescriptor(result.wi.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wi.out_c, result.wi.out_h, result.wi.out_w);
    cudnnSetTensor4dDescriptor(result.wg.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wg.out_c, result.wg.out_h, result.wg.out_w);
    cudnnSetTensor4dDescriptor(result.wo.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wo.out_c, result.wo.out_h, result.wo.out_w);
    cudnnSetTensor4dDescriptor(result.uf.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.uf.out_c, result.uf.out_h, result.uf.out_w);
    cudnnSetTensor4dDescriptor(result.ui.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.ui.out_c, result.ui.out_h, result.ui.out_w);
    cudnnSetTensor4dDescriptor(result.ug.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.ug.out_c, result.ug.out_h, result.ug.out_w);
    cudnnSetTensor4dDescriptor(result.uo.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.uo.out_c, result.uo.out_h, result.uo.out_w);
  {$endif}
{$endif}
end;

procedure update_lstm_layer(const l: TLSTMLayer; const a: TUpdateArgs);
begin
    update_connected_layer(l.wf[0], a);
    update_connected_layer(l.wi[0], a);
    update_connected_layer(l.wg[0], a);
    update_connected_layer(l.wo[0], a);
    update_connected_layer(l.uf[0], a);
    update_connected_layer(l.ui[0], a);
    update_connected_layer(l.ug[0], a);
    update_connected_layer(l.uo[0], a)
end;

procedure forward_lstm_layer(var l: TLSTMLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    wf, wi, wg, wo, uf, ui, ug, uo: TConnectedLayer;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    s := Default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;

    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];

    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];

    fill_cpu(l.outputs * l.batch * l.steps, 0, wf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wi.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wg.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wo.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, uf.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ui.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ug.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uo.delta, 1);
    if state.train then
        fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);

    for i := 0 to l.steps -1 do
        begin
            s.input := l.h_cpu;
            forward_connected_layer(wf, @s);
            forward_connected_layer(wi, @s);
            forward_connected_layer(wg, @s);
            forward_connected_layer(wo, @s);

            s.input := state.input;
            forward_connected_layer(uf, @s);
            forward_connected_layer(ui, @s);
            forward_connected_layer(ug, @s);
            forward_connected_layer(uo, @s);

            copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);

            copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);

            copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);

            copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);

            activate_array(l.f_cpu, l.outputs * l.batch, acLOGISTIC);
            activate_array(l.i_cpu, l.outputs * l.batch, acLOGISTIC);
            activate_array(l.g_cpu, l.outputs * l.batch, acTANH);
            activate_array(l.o_cpu, l.outputs * l.batch, acLOGISTIC);

            copy_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
            mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
            mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.c_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, l.temp_cpu, 1, l.c_cpu, 1);

            copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.h_cpu, 1);
            activate_array(l.h_cpu, l.outputs * l.batch, acTANH);
            mul_cpu(l.outputs * l.batch, l.o_cpu, 1, l.h_cpu, 1);

            copy_cpu(l.outputs * l.batch, l.c_cpu, 1, l.cell_cpu, 1);
            copy_cpu(l.outputs * l.batch, l.h_cpu, 1, l.output, 1);

            state.input := state.input + (l.inputs * l.batch);
            l.output := l.output + (l.outputs * l.batch);
            l.cell_cpu := l.cell_cpu + (l.outputs * l.batch);

            increment_layer( wf, 1);
            increment_layer( wi, 1);
            increment_layer( wg, 1);
            increment_layer( wo, 1);

            increment_layer( uf, 1);
            increment_layer( ui, 1);
            increment_layer( ug, 1);
            increment_layer( uo, 1)
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_lstm_layer(var l: TLSTMLayer; const state: PNetworkState);
var
    s: TNetworkState;
    i: longint;
    wf, wi, wg, wo, uf, ui, ug, uo: TConnectedLayer;
begin
    s := default(TNetworkState);
    s.train := state.train;
    s.workspace := state.workspace;

    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];

    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];

    increment_layer( wf, l.steps-1);
    increment_layer( wi, l.steps-1);
    increment_layer( wg, l.steps-1);
    increment_layer( wo, l.steps-1);

    increment_layer( uf, l.steps-1);
    increment_layer( ui, l.steps-1);
    increment_layer( ug, l.steps-1);
    increment_layer( uo, l.steps-1);

    state.input := state.input + (l.inputs * l.batch * (l.steps-1));
    if assigned(state.delta) then
        state.delta := state.delta + (l.inputs * l.batch * (l.steps-1));

    l.output := l.output + (l.outputs * l.batch * (l.steps-1));
    l.cell_cpu := l.cell_cpu + (l.outputs * l.batch * (l.steps-1));
    l.delta := l.delta + (l.outputs * l.batch * (l.steps-1));

    for i := l.steps-1 downto 0 do begin
        if i <> 0 then
            copy_cpu(l.outputs * l.batch, l.cell_cpu-l.outputs * l.batch, 1, l.prev_cell_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.cell_cpu, 1, l.c_cpu, 1);
        if i <> 0 then
            copy_cpu(l.outputs * l.batch, l.output-l.outputs * l.batch, 1, l.prev_state_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.output, 1, l.h_cpu, 1);
        if (i = 0) then
          l.dh_cpu :=  nil
        else
          l.dh_cpu := l.delta-l.outputs * l.batch;

        copy_cpu(l.outputs * l.batch, wf.output, 1, l.f_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uf.output, 1, l.f_cpu, 1);

        copy_cpu(l.outputs * l.batch, wi.output, 1, l.i_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ui.output, 1, l.i_cpu, 1);

        copy_cpu(l.outputs * l.batch, wg.output, 1, l.g_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, ug.output, 1, l.g_cpu, 1);

        copy_cpu(l.outputs * l.batch, wo.output, 1, l.o_cpu, 1);
        axpy_cpu(l.outputs * l.batch, 1, uo.output, 1, l.o_cpu, 1);

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
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wo.delta, 1);
        s.input := l.prev_state_cpu;
        s.delta := l.dh_cpu;
        backward_connected_layer(wo, @s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uo.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_connected_layer(uo, @s);

        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.i_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.g_cpu, l.outputs * l.batch, acTANH, l.temp_cpu);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wg.delta, 1);
        s.input := l.prev_state_cpu;
        s.delta := l.dh_cpu;
        backward_connected_layer(wg, @s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ug.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_connected_layer(ug, @s);

        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.g_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.i_cpu, l.outputs * l.batch, acLOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wi.delta, 1);
        s.input := l.prev_state_cpu;
        s.delta := l.dh_cpu;
        backward_connected_layer(wi, @s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, ui.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_connected_layer(ui, @s);

        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.prev_cell_cpu, 1, l.temp_cpu, 1);
        gradient_array(l.f_cpu, l.outputs * l.batch, acLOGISTIC, l.temp_cpu);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, wf.delta, 1);
        s.input := l.prev_state_cpu;
        s.delta := l.dh_cpu;
        backward_connected_layer(wf, @s);

        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, uf.delta, 1);
        s.input := state.input;
        s.delta := state.delta;
        backward_connected_layer(uf, @s);

        copy_cpu(l.outputs * l.batch, l.temp2_cpu, 1, l.temp_cpu, 1);
        mul_cpu(l.outputs * l.batch, l.f_cpu, 1, l.temp_cpu, 1);
        copy_cpu(l.outputs * l.batch, l.temp_cpu, 1, l.dc_cpu, 1);

        state.input := state.input - (l.inputs * l.batch);
        if assigned(state.delta) then
            state.delta := state.delta - (l.inputs * l.batch);
        l.output := l.output - (l.outputs * l.batch);
        l.cell_cpu := l.cell_cpu - (l.outputs * l.batch);
        l.delta := l.delta - (l.outputs * l.batch);

        increment_layer( wf, -1);
        increment_layer( wi, -1);
        increment_layer( wg, -1);
        increment_layer( wo, -1);

        increment_layer( uf, -1);
        increment_layer( ui, -1);
        increment_layer( ug, -1);
        increment_layer( uo, -1);
    end
end;

{$ifdef GPU}
procedure update_lstm_layer_gpu(l: TLSTMLayer; a: update_args);
begin
    update_connected_layer_gpu( * (l.wf), a);
    update_connected_layer_gpu( * (l.wi), a);
    update_connected_layer_gpu( * (l.wg), a);
    update_connected_layer_gpu( * (l.wo), a);
    update_connected_layer_gpu( * (l.uf), a);
    update_connected_layer_gpu( * (l.ui), a);
    update_connected_layer_gpu( * (l.ug), a);
    update_connected_layer_gpu( * (l.uo), a)
end;

procedure forward_lstm_layer_gpu(l: TLSTMLayer; state: TNetwork);
var
    s: TNetwork;
    i: longint;
    wf, wi, wg, wo, uf, ui, ug, uo: TConnectedLayer;
begin
    s := default(TNetwork);
    s.train := state.train;
    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];
    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];
    fill_gpu(l.outputs * l.batch * l.steps, 0, wf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wi.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wg.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wo.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uf.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ui.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ug.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uo.delta_gpu, 1);
    if state.train then
        fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
    for i := 0 to l.steps -1 do
        begin
            s.input_gpu := l.h_gpu;
            forward_connected_layer_gpu(wf, s);
            forward_connected_layer_gpu(wi, s);
            forward_connected_layer_gpu(wg, s);
            forward_connected_layer_gpu(wo, s);
            s.input_gpu := state.input_gpu;
            forward_connected_layer_gpu(uf, s);
            forward_connected_layer_gpu(ui, s);
            forward_connected_layer_gpu(ug, s);
            forward_connected_layer_gpu(uo, s);
            copy_gpu(l.outputs * l.batch, wf.output_gpu, 1, l.f_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
            copy_gpu(l.outputs * l.batch, wi.output_gpu, 1, l.i_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);
            copy_gpu(l.outputs * l.batch, wg.output_gpu, 1, l.g_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);
            copy_gpu(l.outputs * l.batch, wo.output_gpu, 1, l.o_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);
            activate_array_gpu(l.f_gpu, l.outputs * l.batch, acLOGISTIC);
            activate_array_gpu(l.i_gpu, l.outputs * l.batch, acLOGISTIC);
            activate_array_gpu(l.g_gpu, l.outputs * l.batch, acTANH);
            activate_array_gpu(l.o_gpu, l.outputs * l.batch, acLOGISTIC);
            copy_gpu(l.outputs * l.batch, l.i_gpu, 1, l.temp_gpu, 1);
            mul_gpu(l.outputs * l.batch, l.g_gpu, 1, l.temp_gpu, 1);
            mul_gpu(l.outputs * l.batch, l.f_gpu, 1, l.c_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, l.temp_gpu, 1, l.c_gpu, 1);
            copy_gpu(l.outputs * l.batch, l.c_gpu, 1, l.h_gpu, 1);
            activate_array_gpu(l.h_gpu, l.outputs * l.batch, acTANH);
            mul_gpu(l.outputs * l.batch, l.o_gpu, 1, l.h_gpu, 1);
            copy_gpu(l.outputs * l.batch, l.c_gpu, 1, l.cell_gpu, 1);
            copy_gpu(l.outputs * l.batch, l.h_gpu, 1, l.output_gpu, 1);
            state.input_gpu := state.input_gpu + (l.inputs * l.batch);
            l.output_gpu := l.output_gpu + (l.outputs * l.batch);
            l.cell_gpu := l.cell_gpu + (l.outputs * l.batch);
            increment_layer( wf, 1);
            increment_layer( wi, 1);
            increment_layer( wg, 1);
            increment_layer( wo, 1);
            increment_layer( uf, 1);
            increment_layer( ui, 1);
            increment_layer( ug, 1);
            increment_layer( uo, 1)
        end
end;

procedure backward_lstm_layer_gpu(l: TLSTMLayer; state: TNetwork);
var
    s: TNetwork;
    i: longint;
    wf, wi, wg, wo, uf, ui, ug, uo: TConnectedLayer;
begin
    s := default(TNetwork);
    s.train := state.train;
    wf :=  l.wf[0];
    wi :=  l.wi[0];
    wg :=  l.wg[0];
    wo :=  l.wo[0];
    uf :=  l.uf[0];
    ui :=  l.ui[0];
    ug :=  l.ug[0];
    uo :=  l.uo[0];
    increment_layer( wf, l.steps-1);
    increment_layer( wi, l.steps-1);
    increment_layer( wg, l.steps-1);
    increment_layer( wo, l.steps-1);
    increment_layer( uf, l.steps-1);
    increment_layer( ui, l.steps-1);
    increment_layer( ug, l.steps-1);
    increment_layer( uo, l.steps-1);
    state.input_gpu := state.input_gpu + (l.inputs * l.batch * (l.steps-1));
    if assigned(state.delta_gpu )then
        state.delta_gpu := state.delta_gpu + (l.inputs * l.batch * (l.steps-1));
    l.output_gpu := l.output_gpu + (l.outputs * l.batch * (l.steps-1));
    l.cell_gpu := l.cell_gpu + (l.outputs * l.batch * (l.steps-1));
    l.delta_gpu := l.delta_gpu + (l.outputs * l.batch * (l.steps-1));
    for i := l.steps-1 downto 0 do begin
        if i <> 0 then
            copy_gpu(l.outputs * l.batch, l.cell_gpu-l.outputs * l.batch, 1, l.prev_cell_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.cell_gpu, 1, l.c_gpu, 1);
        if i <> 0 then
            copy_gpu(l.outputs * l.batch, l.output_gpu-l.outputs * l.batch, 1, l.prev_state_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.output_gpu, 1, l.h_gpu, 1);
        l.dh_gpu := ifthen((i = 0), 0, l.delta_gpu-l.outputs * l.batch);
        copy_gpu(l.outputs * l.batch, wf.output_gpu, 1, l.f_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, uf.output_gpu, 1, l.f_gpu, 1);
        copy_gpu(l.outputs * l.batch, wi.output_gpu, 1, l.i_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, ui.output_gpu, 1, l.i_gpu, 1);
        copy_gpu(l.outputs * l.batch, wg.output_gpu, 1, l.g_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, ug.output_gpu, 1, l.g_gpu, 1);
        copy_gpu(l.outputs * l.batch, wo.output_gpu, 1, l.o_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, uo.output_gpu, 1, l.o_gpu, 1);
        activate_array_gpu(l.f_gpu, l.outputs * l.batch, acLOGISTIC);
        activate_array_gpu(l.i_gpu, l.outputs * l.batch, acLOGISTIC);
        activate_array_gpu(l.g_gpu, l.outputs * l.batch, acTANH);
        activate_array_gpu(l.o_gpu, l.outputs * l.batch, acLOGISTIC);
        copy_gpu(l.outputs * l.batch, l.delta_gpu, 1, l.temp3_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.c_gpu, 1, l.temp_gpu, 1);
        activate_array_gpu(l.temp_gpu, l.outputs * l.batch, acTANH);
        copy_gpu(l.outputs * l.batch, l.temp3_gpu, 1, l.temp2_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.o_gpu, 1, l.temp2_gpu, 1);
        gradient_array_gpu(l.temp_gpu, l.outputs * l.batch, acTANH, l.temp2_gpu);
        axpy_gpu(l.outputs * l.batch, 1, l.dc_gpu, 1, l.temp2_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.c_gpu, 1, l.temp_gpu, 1);
        activate_array_gpu(l.temp_gpu, l.outputs * l.batch, acTANH);
        mul_gpu(l.outputs * l.batch, l.temp3_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.o_gpu, l.outputs * l.batch, acLOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, wo.delta_gpu, 1);
        s.input_gpu := l.prev_state_gpu;
        s.delta_gpu := l.dh_gpu;
        backward_connected_layer_gpu(wo, s);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, uo.delta_gpu, 1);
        s.input_gpu := state.input_gpu;
        s.delta_gpu := state.delta_gpu;
        backward_connected_layer_gpu(uo, s);
        copy_gpu(l.outputs * l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.i_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.g_gpu, l.outputs * l.batch, acTANH, l.temp_gpu);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, wg.delta_gpu, 1);
        s.input_gpu := l.prev_state_gpu;
        s.delta_gpu := l.dh_gpu;
        backward_connected_layer_gpu(wg, s);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, ug.delta_gpu, 1);
        s.input_gpu := state.input_gpu;
        s.delta_gpu := state.delta_gpu;
        backward_connected_layer_gpu(ug, s);
        copy_gpu(l.outputs * l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.g_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.i_gpu, l.outputs * l.batch, acLOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, wi.delta_gpu, 1);
        s.input_gpu := l.prev_state_gpu;
        s.delta_gpu := l.dh_gpu;
        backward_connected_layer_gpu(wi, s);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, ui.delta_gpu, 1);
        s.input_gpu := state.input_gpu;
        s.delta_gpu := state.delta_gpu;
        backward_connected_layer_gpu(ui, s);
        copy_gpu(l.outputs * l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.prev_cell_gpu, 1, l.temp_gpu, 1);
        gradient_array_gpu(l.f_gpu, l.outputs * l.batch, acLOGISTIC, l.temp_gpu);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, wf.delta_gpu, 1);
        s.input_gpu := l.prev_state_gpu;
        s.delta_gpu := l.dh_gpu;
        backward_connected_layer_gpu(wf, s);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, uf.delta_gpu, 1);
        s.input_gpu := state.input_gpu;
        s.delta_gpu := state.delta_gpu;
        backward_connected_layer_gpu(uf, s);
        copy_gpu(l.outputs * l.batch, l.temp2_gpu, 1, l.temp_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.f_gpu, 1, l.temp_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.temp_gpu, 1, l.dc_gpu, 1);
        state.input_gpu := state.input_gpu - (l.inputs * l.batch);
        if assigned(state.delta_gpu) then
            state.delta_gpu := state.delta_gpu - (l.inputs * l.batch);
        l.output_gpu := l.output_gpu - (l.outputs * l.batch);
        l.cell_gpu := l.cell_gpu - (l.outputs * l.batch);
        l.delta_gpu := l.delta_gpu - (l.outputs * l.batch);
        increment_layer( wf, -1);
        increment_layer( wi, -1);
        increment_layer( wg, -1);
        increment_layer( wo, -1);
        increment_layer( uf, -1);
        increment_layer( ui, -1);
        increment_layer( ug, -1);
        increment_layer( uo, -1);
    end
end;
{$endif}

end.

