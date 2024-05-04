unit GruLayer;

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
  SysUtils, darknet, Activations, ConnectedLayer, blas;

type
  TGRULayer = TLayer;

function make_gru_layer(batch:longint; const inputs, outputs, steps:longint;const batch_normalize: boolean):TGRULayer;
procedure update_gru_layer(const l: TGRULayer; const a: TUpdateArgs);
procedure forward_gru_layer(var l: TGRULayer; const state: PNetworkState);
procedure backward_gru_layer(var l: TGRULayer; const state: PNetworkState);

{$ifdef GPU}
procedure pull_gru_layer(l: TGRULayer);
procedure push_gru_layer(l: TGRULayer);
procedure update_gru_layer_gpu(l: TGRULayer; a: update_args);
procedure forward_gru_layer_gpu(l: TGRULayer; net: TNetwork);
procedure backward_gru_layer_gpu(l: TGRULayer; net: TNetwork);
{$endif}

implementation

procedure increment_layer(var l: TGRUlayer; const steps: longint);
var
    num: longint;
begin
    num := l.outputs * l.batch * steps;
    l.output := @l.output[num];
    l.delta := @l.delta[num];
    l.x := @l.x[num];
    l.x_norm := @l.x_norm[num];
{$ifdef GPU}
    l.output_gpu := l.output_gpu + num;
    l.delta_gpu := l.delta_gpu + num;
    l.x_gpu := l.x_gpu + num;
    l.x_norm_gpu := l.x_norm_gpu + num
{$endif}
end;

function make_gru_layer(batch: longint; const inputs, outputs, steps: longint;
  const batch_normalize: boolean): TGRULayer;
begin
    writeln(ErrOutput, format('GRU Layer: %d inputs, %d outputs', [inputs, outputs]));
    batch := batch div steps;
    result := default(TGRULayer);
    result.batch := batch;
    result.&type := ltGRU;
    result.steps := steps;
    result.inputs := inputs;


    setLength(result.uz, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.uz[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize{, adam});
    result.uz[0].batch := batch;

    setLength(result.wz, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.wz[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize{, adam});
    result.wz[0].batch := batch;

    setLength(result.ur, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.ur[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize{, adam});
    result.ur[0].batch := batch;

    setLength(result.wr, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.wr[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize{, adam});
    result.wr[0].batch := batch;

    setLength(result.uh, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.uh[0] := make_connected_layer(batch , steps, inputs, outputs, acLINEAR, batch_normalize{, adam});
    result.uh[0].batch := batch;

    setLength(result.wh, 1);// := AllocMem(sizeof(TGRULayer));
    write(ErrOutput, #9#9);
    result.wh[0] := make_connected_layer(batch , steps, outputs, outputs, acLINEAR, batch_normalize{, adam});
    result.wh[0].batch := batch;

    {
    setLength(result.input_z_layer, 1);
    write(ErrOutput, #9#9);
    result.input_z_layer[0] := make_connected_layer(batch, steps, inputs, outputs, acLINEAR, batch_normalize);
    result.input_z_layer.batch := batch;

    setLength(result.state_z_layer, 1);
    write(ErrOutput, #9#9);
    result.state_z_layer[0] := make_connected_layer(batch, steps, outputs, outputs, acLINEAR, batch_normalize);
    result.state_z_layer.batch := batch;



    setLength(result.input_r_layer, 1);
    write(ErrOutput, #9#9);
    result.input_r_layer[0] := make_connected_layer(batch, steps, inputs, outputs, acLINEAR, batch_normalize);
    result.input_r_layer.batch := batch;

    setLength(result.state_r_layer, 1);
    write(ErrOutput, #9#9);
    result.state_r_layer[0] := make_connected_layer(batch, steps, outputs, outputs, acLINEAR, batch_normalize);
    result.state_r_layer.batch := batch;



    setLength(result.input_h_layer, 1);
    write(ErrOutput, #9#9);
    result.input_h_layer[0] := make_connected_layer(batch, steps, inputs, outputs, acLINEAR, batch_normalize);
    result.input_h_layer.batch := batch;

    setLength(result.state_h_layer, 1);
    write(ErrOutput, #9#9);
    result.state_h_layer[0] := make_connected_layer(batch, steps, outputs, outputs, acLINEAR, batch_normalize);
    result.state_h_layer.batch := batch;
    }
    result.batch_normalize := batch_normalize;
    result.outputs := outputs;
    result.output := TSingles.Create(outputs * batch * steps);
    result.delta := TSingles.Create(outputs * batch * steps);
    result.state := TSingles.Create(outputs * batch);
    result.prev_state := TSingles.Create(outputs * batch);
    result.forgot_state := TSingles.Create(outputs * batch);
    result.forgot_delta := TSingles.Create(outputs * batch);
    result.r_cpu := TSingles.Create(outputs * batch);
    result.z_cpu := TSingles.Create(outputs * batch);
    result.h_cpu := TSingles.Create(outputs * batch);
    result.forward := forward_gru_layer;
    result.backward := backward_gru_layer;
    result.update := update_gru_layer;
{$ifdef GPU}
    result.forward_gpu := forward_gru_layer_gpu;
    result.backward_gpu := backward_gru_layer_gpu;
    result.update_gpu := update_gru_layer_gpu;
    result.forgot_state_gpu := cuda_make_array(0, batch * outputs);
    result.forgot_delta_gpu := cuda_make_array(0, batch * outputs);
    result.prev_state_gpu := cuda_make_array(0, batch * outputs);
    result.state_gpu := cuda_make_array(0, batch * outputs);
    result.output_gpu := cuda_make_array(0, batch * outputs * steps);
    result.delta_gpu := cuda_make_array(0, batch * outputs * steps);
    result.r_gpu := cuda_make_array(0, batch * outputs);
    result.z_gpu := cuda_make_array(0, batch * outputs);
    result.h_gpu := cuda_make_array(0, batch * outputs);
  {$ifdef CUDNN}
    cudnnSetTensor4dDescriptor(result.uz.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.uz.out_c, result.uz.out_h, result.uz.out_w);
    cudnnSetTensor4dDescriptor(result.uh.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.uh.out_c, result.uh.out_h, result.uh.out_w);
    cudnnSetTensor4dDescriptor(result.ur.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.ur.out_c, result.ur.out_h, result.ur.out_w);
    cudnnSetTensor4dDescriptor(result.wz.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wz.out_c, result.wz.out_h, result.wz.out_w);
    cudnnSetTensor4dDescriptor(result.wh.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wh.out_c, result.wh.out_h, result.wh.out_w);
    cudnnSetTensor4dDescriptor(result.wr.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, result.wr.out_c, result.wr.out_h, result.wr.out_w);
  {$endif}
{$endif}
end;

procedure update_gru_layer(const l: TGRULayer; const a: TUpdateArgs);
begin
    update_connected_layer(l.ur[0], a);
    update_connected_layer(l.uz[0], a);
    update_connected_layer(l.uh[0], a);
    update_connected_layer(l.wr[0], a);
    update_connected_layer(l.wz[0], a);
    update_connected_layer(l.wh[0], a)
end;

procedure forward_gru_layer(var l: TGRULayer; const state : PNetworkState);
var
    s: TNetworkState;
    i: longint;
    uz, ur, uh, wz, wr, wh: TGRULayer;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    s:=default(TNetworkState);
    //s := net[0];
    s.train := state.train;
    s.workspace:= state.workspace;
    uz := l.uz[0];
    ur := l.ur[0];
    uh := l.uh[0];

    wz := l.wz[0];
    wr := l.wr[0];
    wh := l.wh[0];

    fill_cpu(l.outputs * l.batch * l.steps, 0, uz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, ur.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, uh.delta, 1);

    fill_cpu(l.outputs * l.batch * l.steps, 0, wz.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wr.delta, 1);
    fill_cpu(l.outputs * l.batch * l.steps, 0, wh.delta, 1);
    if state.train then
        begin
            fill_cpu(l.outputs * l.batch * l.steps, 0, l.delta, 1);
            copy_cpu(l.outputs * l.batch, l.state, 1, l.prev_state, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            s.input := l.state;
            forward_connected_layer(wz, @s);
            forward_connected_layer(wr, @s);

            s.input := state.input;
            forward_connected_layer(uz, @s);
            forward_connected_layer(ur, @s);
            forward_connected_layer(uh, @s);


            copy_cpu(l.outputs * l.batch, uz.output, 1, l.z_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, wz.output, 1, l.z_cpu, 1);

            copy_cpu(l.outputs * l.batch, ur.output, 1, l.r_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, wr.output, 1, l.r_cpu, 1);

            activate_array(l.z_cpu, l.outputs * l.batch, acLOGISTIC);
            activate_array(l.r_cpu, l.outputs * l.batch, acLOGISTIC);

            copy_cpu(l.outputs * l.batch, l.state, 1, l.forgot_state, 1);
            mul_cpu(l.outputs * l.batch, l.r_cpu, 1, l.forgot_state, 1);

            s.input := l.forgot_state;
            forward_connected_layer(wh, @s);

            copy_cpu(l.outputs * l.batch, uh.output, 1, l.h_cpu, 1);
            axpy_cpu(l.outputs * l.batch, 1, wh.output, 1, l.h_cpu, 1);

            if l.tanh then
                activate_array(l.h_cpu, l.outputs * l.batch, acTANH)
            else
                activate_array(l.h_cpu, l.outputs * l.batch, acLOGISTIC);

            weighted_sum_cpu(l.state, l.h_cpu, l.z_cpu, l.outputs * l.batch, l.output);

            copy_cpu(l.outputs * l.batch, l.output, 1, l.state, 1);

            state.input := state.input + (l.inputs * l.batch);
            l.output := l.output + (l.outputs * l.batch);
            increment_layer(uz, 1);
            increment_layer(ur, 1);
            increment_layer(uh, 1);

            increment_layer(wz, 1);
            increment_layer(wr, 1);
            increment_layer(wh, 1)
        end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_gru_layer(var l: TGRULayer; const state: PNetworkState);
begin

end;
{$ifdef GPU}
procedure pull_gru_layer(l: TGRULayer);
begin

end;

procedure push_gru_layer(l: TGRULayer);
begin

end;

procedure update_gru_layer_gpu(l: TGRULayer; a: update_args);
begin
    update_connected_layer_gpu( * (l.ur), a);
    update_connected_layer_gpu( * (l.uz), a);
    update_connected_layer_gpu( * (l.uh), a);
    update_connected_layer_gpu( * (l.wr), a);
    update_connected_layer_gpu( * (l.wz), a);
    update_connected_layer_gpu( * (l.wh), a)
end;

procedure forward_gru_layer_gpu(l: TGRULayer; net: TNetwork);
var
    s: TNetwork;
    i: longint;
    uz: TGRULayer;
    ur: TGRULayer;
    uh: TGRULayer;
    wz: TGRULayer;
    wr: TGRULayer;
    wh: TGRULayer;
begin
    s := [0];
    s.train := net.train;
    uz := l.uz[0];
    ur := l.ur[0];
    uh := l.uh[0];
    wz := l.wz[0];
    wr := l.wr[0];
    wh := l.wh[0];
    fill_gpu(l.outputs * l.batch * l.steps, 0, uz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, ur.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, uh.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wz.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wr.delta_gpu, 1);
    fill_gpu(l.outputs * l.batch * l.steps, 0, wh.delta_gpu, 1);
    if net.train then
        begin
            fill_gpu(l.outputs * l.batch * l.steps, 0, l.delta_gpu, 1);
            copy_gpu(l.outputs * l.batch, l.state_gpu, 1, l.prev_state_gpu, 1)
        end;
    for i := 0 to l.steps -1 do
        begin
            s.input_gpu := l.state_gpu;
            forward_connected_layer_gpu(wz, s);
            forward_connected_layer_gpu(wr, s);
            s.input_gpu := net.input_gpu;
            forward_connected_layer_gpu(uz, s);
            forward_connected_layer_gpu(ur, s);
            forward_connected_layer_gpu(uh, s);
            copy_gpu(l.outputs * l.batch, uz.output_gpu, 1, l.z_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
            copy_gpu(l.outputs * l.batch, ur.output_gpu, 1, l.r_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
            activate_array_gpu(l.z_gpu, l.outputs * l.batch, LOGISTIC);
            activate_array_gpu(l.r_gpu, l.outputs * l.batch, LOGISTIC);
            copy_gpu(l.outputs * l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
            mul_gpu(l.outputs * l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
            s.input_gpu := l.forgot_state_gpu;
            forward_connected_layer_gpu(wh, s);
            copy_gpu(l.outputs * l.batch, uh.output_gpu, 1, l.h_gpu, 1);
            axpy_gpu(l.outputs * l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
            if l.tanh then
                activate_array_gpu(l.h_gpu, l.outputs * l.batch, TANH)
            else
                activate_array_gpu(l.h_gpu, l.outputs * l.batch, LOGISTIC);
            weighted_sum_gpu(l.state_gpu, l.h_gpu, l.z_gpu, l.outputs * l.batch, l.output_gpu);
            copy_gpu(l.outputs * l.batch, l.output_gpu, 1, l.state_gpu, 1);
            net.input_gpu := net.input_gpu + (l.inputs * l.batch);
            l.output_gpu := l.output_gpu + (l.outputs * l.batch);
            increment_layer(uz, 1);
            increment_layer(ur, 1);
            increment_layer(uh, 1);
            increment_layer(wz, 1);
            increment_layer(wr, 1);
            increment_layer(wh, 1)
        end
end;

procedure backward_gru_layer_gpu(l: TGRULayer; net: TNetwork);
var
    s: TNetwork;
    i: longint;
    uz:, ur, uh, wz, wr, wh: TGRULayer;
    end_state, prev_delta_gpu: TSingles;
begin
    s := default(TNetwork);
    s.train := net.train;
    uz := l.uz[0];
    ur := l.ur[0];
    uh := l.uh[0];
    wz := l.wz[0];
    wr := l.wr[0];
    wh := l.wh[0];
    increment_layer(uz, l.steps-1);
    increment_layer(ur, l.steps-1);
    increment_layer(uh, l.steps-1);
    increment_layer(wz, l.steps-1);
    increment_layer(wr, l.steps-1);
    increment_layer(wh, l.steps-1);
    net.input_gpu := net.input_gpu + (l.inputs * l.batch * (l.steps-1));
    if net.delta_gpu then
        net.delta_gpu := net.delta_gpu + (l.inputs * l.batch * (l.steps-1));
    l.output_gpu := l.output_gpu + (l.outputs * l.batch * (l.steps-1));
    l.delta_gpu := l.delta_gpu + (l.outputs * l.batch * (l.steps-1));
    end_state := l.output_gpu;
    i := l.steps-1;
    while i >= 0 do begin
        if i <> 0 then
            copy_gpu(l.outputs * l.batch, l.output_gpu-l.outputs * l.batch, 1, l.state_gpu, 1)
        else
            copy_gpu(l.outputs * l.batch, l.prev_state_gpu, 1, l.state_gpu, 1);
        prev_delta_gpu := ifthen((i = 0), 0, l.delta_gpu-l.outputs * l.batch);
        copy_gpu(l.outputs * l.batch, uz.output_gpu, 1, l.z_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, wz.output_gpu, 1, l.z_gpu, 1);
        copy_gpu(l.outputs * l.batch, ur.output_gpu, 1, l.r_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, wr.output_gpu, 1, l.r_gpu, 1);
        activate_array_gpu(l.z_gpu, l.outputs * l.batch, LOGISTIC);
        activate_array_gpu(l.r_gpu, l.outputs * l.batch, LOGISTIC);
        copy_gpu(l.outputs * l.batch, uh.output_gpu, 1, l.h_gpu, 1);
        axpy_gpu(l.outputs * l.batch, 1, wh.output_gpu, 1, l.h_gpu, 1);
        if l.tanh then
            activate_array_gpu(l.h_gpu, l.outputs * l.batch, TANH)
        else
            activate_array_gpu(l.h_gpu, l.outputs * l.batch, LOGISTIC);
        weighted_delta_gpu(l.state_gpu, l.h_gpu, l.z_gpu, prev_delta_gpu, uh.delta_gpu, uz.delta_gpu, l.outputs * l.batch, l.delta_gpu);
        if l.tanh then
            gradient_array_gpu(l.h_gpu, l.outputs * l.batch, TANH, uh.delta_gpu)
        else
            gradient_array_gpu(l.h_gpu, l.outputs * l.batch, LOGISTIC, uh.delta_gpu);
        copy_gpu(l.outputs * l.batch, uh.delta_gpu, 1, wh.delta_gpu, 1);
        copy_gpu(l.outputs * l.batch, l.state_gpu, 1, l.forgot_state_gpu, 1);
        mul_gpu(l.outputs * l.batch, l.r_gpu, 1, l.forgot_state_gpu, 1);
        fill_gpu(l.outputs * l.batch, 0, l.forgot_delta_gpu, 1);
        s.input_gpu := l.forgot_state_gpu;
        s.delta_gpu := l.forgot_delta_gpu;
        backward_connected_layer_gpu(wh, s);
        if prev_delta_gpu then
            mult_add_into_gpu(l.outputs * l.batch, l.forgot_delta_gpu, l.r_gpu, prev_delta_gpu);
        mult_add_into_gpu(l.outputs * l.batch, l.forgot_delta_gpu, l.state_gpu, ur.delta_gpu);
        gradient_array_gpu(l.r_gpu, l.outputs * l.batch, LOGISTIC, ur.delta_gpu);
        copy_gpu(l.outputs * l.batch, ur.delta_gpu, 1, wr.delta_gpu, 1);
        gradient_array_gpu(l.z_gpu, l.outputs * l.batch, LOGISTIC, uz.delta_gpu);
        copy_gpu(l.outputs * l.batch, uz.delta_gpu, 1, wz.delta_gpu, 1);
        s.input_gpu := l.state_gpu;
        s.delta_gpu := prev_delta_gpu;
        backward_connected_layer_gpu(wr, s);
        backward_connected_layer_gpu(wz, s);
        s.input_gpu := net.input_gpu;
        s.delta_gpu := net.delta_gpu;
        backward_connected_layer_gpu(uh, s);
        backward_connected_layer_gpu(ur, s);
        backward_connected_layer_gpu(uz, s);
        net.input_gpu := net.input_gpu - (l.inputs * l.batch);
        if net.delta_gpu then
            net.delta_gpu := net.delta_gpu - (l.inputs * l.batch);
        l.output_gpu := l.output_gpu - (l.outputs * l.batch);
        l.delta_gpu := l.delta_gpu - (l.outputs * l.batch);
        increment_layer(uz, -1);
        increment_layer(ur, -1);
        increment_layer(uh, -1);
        increment_layer(wz, -1);
        increment_layer(wr, -1);
        increment_layer(wh, -1);
        dec(i)
    end;
    copy_gpu(l.outputs * l.batch, end_state, 1, l.state_gpu, 1)
end;
{$endif}

end.

