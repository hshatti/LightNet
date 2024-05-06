unit BatchNormLayer;
{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX64}
          {$asmmode intel}
  {$endif}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, lightnet, blas;

  type
    PBatchNormLayer =^TBatchNormLayer;
    TBatchNormLayer = TLayer;

    function make_batchnorm_layer(const batch, w, h, c: longint; const train:boolean):TBatchNormLayer;
    procedure resize_batchnorm_layer(var l: TBatchNormLayer; const w, h: longint);
    procedure forward_batchnorm_layer(var l: TBatchNormLayer; const state: PNetworkState);
    procedure backward_batchnorm_layer(var l: TBatchNormLayer; const state: PNetworkState);
    procedure update_batchnorm_layer(const l:TBatchNormLayer; const args: TUpdateArgs);// batch:longint; const learning_rate, momentum, decay:single);


{$ifdef GPU}
    procedure pull_batchnorm_layer(const l: TBatchNormLayer);
    procedure push_batchnorm_layer(const l: TBatchNormLayer);
    procedure forward_batchnorm_layer_gpu(var l: TBatchNormLayer; const net: PNetworkState);
    procedure backward_batchnorm_layer_gpu(const l: TBatchNormLayer; const net: TNetwork);
    procedure update_batchnorm_layer_gpu(l: TBatchNormLayer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);

{$endif}

implementation

function make_batchnorm_layer(const batch, w, h, c: longint; const train:boolean): TBatchNormLayer;
var
    i: longint;
begin
    writeln(format('Batch Normalization Layer: %d x %d x %d image', [w, h, c]));
    result:=Default(TBatchNormLayer);
    result.&type := ltBATCHNORM;
    result.batch := batch;
    result.train := train;
    result.h := h; result.out_h := h;
    result.w := w; result.out_w := w;
    result.c := c; result.out_c := c;

    result.n := result.c;

    //setLength(result.output, h * w * c * batch);
    result.output := TSingles.Create(h * w * c * batch);

    //setLength(result.delta, h * w * c * batch);
    result.delta := TSingles.Create( h * w * c * batch);

    result.inputs := w * h * c;
    result.outputs := result.inputs;

    //setLength(result.biases, c);
    result.biases := TSingles.Create(c);

    //setLength(result.bias_updates, c);
    result.bias_updates := TSingles.Create(c);

    //setLength(result.scales, c);
    result.scales := TSingles.Create(c);

    //setLength(result.scale_updates, c);
    result.scale_updates := TSingles.Create(c);

    for i := 0 to c -1 do
        result.scales[i] := 1;

    //setLength(result.mean, c);
    result.mean := TSingles.Create(c);

    //setLength(result.variance, c);
    result.variance := TSingles.Create(c);

    //setLength(result.rolling_mean, c);
    result.rolling_mean := TSingles.Create(c);

    //setLength(result.rolling_variance, c);
    result.rolling_variance := TSingles.Create(c);

    result.mean_delta := TSingles.Create(c);
    result.variance_delta := TSingles.Create(c);

    result.x := TSingles.Create(result.batch*result.outputs);
    result.x_norm := TSingles.Create(result.batch*result.outputs);

    result.forward  := forward_batchnorm_layer;
    result.backward := backward_batchnorm_layer;
    result.update := update_batchnorm_layer;
{$ifdef GPU}
    result.forward_gpu   := forward_batchnorm_layer_gpu;
    result.backward_gpu  := backward_batchnorm_layer_gpu;
    result.output_gpu    := cuda_make_array(result.output, h * w * c * batch);
    result.biases_gpu    := cuda_make_array(result.biases, c);
    result.scales_gpu           := cuda_make_array(result.scales, c);
    if train then begin
      result.delta_gpu     := cuda_make_array(result.delta, h * w * c * batch);
      result.bias_updates_gpu     := cuda_make_array(result.bias_updates, c);
      result.scale_updates_gpu    := cuda_make_array(result.scale_updates, c);
      result.mean_delta_gpu       := cuda_make_array(result.mean, c);
      result.variance_delta_gpu   := cuda_make_array(result.variance, c)
    end;
    result.mean_gpu             := cuda_make_array(result.mean, c);
    result.variance_gpu         := cuda_make_array(result.variance, c);
    result.rolling_mean_gpu     := cuda_make_array(result.mean, c);
    result.rolling_variance_gpu := cuda_make_array(result.variance, c);
    if train then begin
      result.x_gpu                := cuda_make_array(result.output, result.batch * result.outputs);
  {$ifndef CUDNN}
      result.x_norm_gpu           := cuda_make_array(result.output, result.batch * result.outputs);
  {$endif}
    end;
  {$ifdef CUDNN}
    cudnnCreateTensorDescriptor( @result.normTensorDesc);
    cudnnCreateTensorDescriptor( @result.dstTensorDesc);
    cudnnSetTensor4dDescriptor(result.dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, result.batch, result.out_c, result.out_h, result.out_w);
    cudnnSetTensor4dDescriptor(result.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, result.out_c, 1, 1);
  {$endif}
{$endif}
end;


procedure resize_batchnorm_layer(var l: TBatchNormLayer; const w, h: longint);
var
    output_size: longint;
begin
    l.out_h :=h; l.h := h;
    l.out_w :=w; l.w := w;
    l.inputs := h * w * l.c;
    l.outputs:= l.inputs;

    output_size := l.outputs * l.batch;

    l.output.reAllocate(output_size);
    l.delta.reAllocate(output_size);
{$ifdef GPU}
    cuda_free(l.output_gpu);
    l.output_gpu := cuda_make_array(l.output, output_size);
    if l.train then
        begin
            cuda_free(l.delta_gpu);
            l.delta_gpu := cuda_make_array(l.delta, output_size);
            cuda_free(l.x_gpu);
            l.x_gpu := cuda_make_array(l.output, output_size);
        {$ifndef CUDNN}
            cuda_free(l.x_norm_gpu);
            l.x_norm_gpu := cuda_make_array(l.output, output_size)
        {$endif}
        end;
  {$ifdef CUDNN}
    CHECK_CUDNN(cudnnDestroyTensorDescriptor(l.normDstTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.normDstTensorDesc));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w))
  {$endif}
{$endif}
end;


procedure forward_batchnorm_layer(var l: TBatchNormLayer; const state: PNetworkState);
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    if l.&type = ltBATCHNORM then
        copy_cpu(l.outputs * l.batch, state.input, 1, l.output, 1);
    if l.&type = ltCONNECTED then begin
        l.out_c :=l.outputs;
        l.out_h :=1; l.out_w:=1;
    end;

    if state.train then
        begin
            mean_cpu(@l.output[0], l.batch, l.out_c, l.out_h * l.out_w, @l.mean[0]);
            variance_cpu(@l.output[0], @l.mean[0], l.batch, l.out_c, l.out_h * l.out_w, @l.variance[0]);

            scal_cpu(l.out_c, 0.9, @l.rolling_mean[0], 1);
            axpy_cpu(l.out_c, 0.1, @l.mean[0], 1, @l.rolling_mean[0], 1);
            scal_cpu(l.out_c, 0.9, @l.rolling_variance[0], 1);
            axpy_cpu(l.out_c, 0.1, @l.variance[0], 1, @l.rolling_variance[0], 1);

            copy_cpu(l.outputs * l.batch, l.output, 1, l.x, 1);
            normalize_cpu(@l.output[0], @l.mean[0], @l.variance[0], l.batch, l.out_c, l.out_h * l.out_w);
            copy_cpu(l.outputs * l.batch, @l.output[0], 1, l.x_norm, 1)
        end
    else
        normalize_cpu(@l.output[0], @l.rolling_mean[0], @l.rolling_variance[0], l.batch, l.out_c, l.out_h * l.out_w);

    scale_add_bias(@l.output[0], @l.scales[0], @l.biases[0], l.batch, l.out_c, l.out_h * l.out_w);

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}
    //scale_bias(@l.output[0], @l.scales[0], l.batch, l.out_c, l.out_h * l.out_w);
    //add_bias(@l.output[0], @l.biases[0], l.batch, l.out_c, l.out_h * l.out_w)
end;

procedure backward_batchnorm_layer(var l: TBatchNormLayer; const state: PNetworkState);

begin
    //if not net.train then
    //    begin
    //        l.mean := l.rolling_mean;
    //        l.variance := l.rolling_variance
    //    end;
    backward_scale_cpu(@l.x_norm[0], @l.delta[0], l.batch, l.out_c, l.out_w * l.out_h, @l.scale_updates[0]);
    scale_bias(@l.delta[0], @l.scales[0], l.batch, l.out_c, l.out_h * l.out_w);
    mean_delta_cpu(@l.delta[0], @l.variance[0], l.batch, l.out_c, l.out_w * l.out_h, @l.mean_delta[0]);
    variance_delta_cpu(@l.x[0], @l.delta[0], @l.mean[0], @l.variance[0], l.batch, l.out_c, l.out_w * l.out_h, @l.variance_delta[0]);
    normalize_delta_cpu(@l.x[0], @l.mean[0], @l.variance[0], @l.mean_delta[0], @l.variance_delta[0], l.batch, l.out_c, l.out_w * l.out_h, @l.delta[0]);
    if l.&type = ltBATCHNORM then
        copy_cpu(l.outputs * l.batch, l.delta, 1, state.delta, 1)
    //backward_bias(@l.bias_updates[0], @l.delta[0], l.batch, l.out_c, l.out_w * l.out_h);
end;

procedure update_batchnorm_layer(const l: TBatchNormLayer; const args: TUpdateArgs);
begin
    //int size = l.nweights;
    axpy_cpu(l.c, args.learning_rate / args.batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.c, args.momentum, l.bias_updates, 1);
    axpy_cpu(l.c, args.learning_rate / args.batch, l.scale_updates, 1, l.scales, 1);
    scal_cpu(l.c, args.momentum, l.scale_updates, 1);
end;

{$ifdef GPU}
procedure pull_batchnorm_layer(const l: TBatchNormLayer);
begin
    cuda_pull_array(l.biases_gpu, l.biases, l.out_c);
    cuda_pull_array(l.scales_gpu, l.scales, l.out_c);
    cuda_pull_array(l.rolling_mean_gpu, l.rolling_mean, l.out_c);
    cuda_pull_array(l.rolling_variance_gpu, l.rolling_variance, l.out_c)
end;

procedure push_batchnorm_layer(const l: TBatchNormLayer);
begin
    cuda_push_array(l.biases_gpu, l.biases, l.out_c);
    cuda_push_array(l.scales_gpu, l.scales, l.out_c);
    cuda_push_array(l.rolling_mean_gpu, l.rolling_mean, l.out_c);
    cuda_push_array(l.rolling_variance_gpu, l.rolling_variance, l.out_c)
end;

procedure forward_batchnorm_layer_gpu(var l: TBatchNormLayer;
  var net: TNetwork);
var
    one: single;
    zero: single;
begin
    if l.&type = BATCHNORM then
        copy_gpu(l.outputs * l.batch, net.input_gpu, 1, l.output_gpu, 1);
    copy_gpu(l.outputs * l.batch, l.output_gpu, 1, l.x_gpu, 1);
    if net.train then
        begin
  {$ifdef CUDNN}
            one := 1;
            zero := 0;
            cudnnBatchNormalizationForwardTraining(cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, @one, @zero, l.dstTensorDesc, l.x_gpu, l.dstTensorDesc, l.output_gpu, l.normTensorDesc, l.scales_gpu, l.biases_gpu, 0.01, l.rolling_mean_gpu, l.rolling_variance_gpu, 0.00001, l.mean_gpu, l.variance_gpu);
  {$else}
            fast_mean_gpu(l.output_gpu, l.batch, l.out_c, l.out_h * l.out_w, l.mean_gpu);
            fast_variance_gpu(l.output_gpu, l.mean_gpu, l.batch, l.out_c, l.out_h * l.out_w, l.variance_gpu);
            scal_gpu(l.out_c, 0.99, l.rolling_mean_gpu, 1);
            axpy_gpu(l.out_c, 0.01, l.mean_gpu, 1, l.rolling_mean_gpu, 1);
            scal_gpu(l.out_c, 0.99, l.rolling_variance_gpu, 1);
            axpy_gpu(l.out_c, 0.01, l.variance_gpu, 1, l.rolling_variance_gpu, 1);
            copy_gpu(l.outputs * l.batch, l.output_gpu, 1, l.x_gpu, 1);
            normalize_gpu(l.output_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_h * l.out_w);
            copy_gpu(l.outputs * l.batch, l.output_gpu, 1, l.x_norm_gpu, 1);
            scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h * l.out_w);
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w * l.out_h)
  {$endif}
        end
    else
        begin
            normalize_gpu(l.output_gpu, l.rolling_mean_gpu, l.rolling_variance_gpu, l.batch, l.out_c, l.out_h * l.out_w);
            scale_bias_gpu(l.output_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h * l.out_w);
            add_bias_gpu(l.output_gpu, l.biases_gpu, l.batch, l.out_c, l.out_w * l.out_h)
        end
end;

procedure backward_batchnorm_layer_gpu(var l: TBatchNormLayer;
  var net: TNetwork);
var
    one: single;
    zero: single;
begin
    if not net.train then
        begin
            l.mean_gpu := l.rolling_mean_gpu;
            l.variance_gpu := l.rolling_variance_gpu
        end;
  {$ifdef CUDNN}
    one := 1;
    zero := 0;
    cudnnBatchNormalizationBackward(cudnn_handle(), CUDNN_BATCHNORM_SPATIAL, @one, @zero, @one, @one, l.dstTensorDesc, l.x_gpu, l.dstTensorDesc, l.delta_gpu, l.dstTensorDesc, l.x_norm_gpu, l.normTensorDesc, l.scales_gpu, l.scale_updates_gpu, l.bias_updates_gpu, 0.00001, l.mean_gpu, l.variance_gpu);
    copy_gpu(l.outputs * l.batch, l.x_norm_gpu, 1, l.delta_gpu, 1);
  {$else}
    backward_bias_gpu(l.bias_updates_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w * l.out_h);
    backward_scale_gpu(l.x_norm_gpu, l.delta_gpu, l.batch, l.out_c, l.out_w * l.out_h, l.scale_updates_gpu);
    scale_bias_gpu(l.delta_gpu, l.scales_gpu, l.batch, l.out_c, l.out_h * l.out_w);
    fast_mean_delta_gpu(l.delta_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w * l.out_h, l.mean_delta_gpu);
    fast_variance_delta_gpu(l.x_gpu, l.delta_gpu, l.mean_gpu, l.variance_gpu, l.batch, l.out_c, l.out_w * l.out_h, l.variance_delta_gpu);
    normalize_delta_gpu(l.x_gpu, l.mean_gpu, l.variance_gpu, l.mean_delta_gpu, l.variance_delta_gpu, l.batch, l.out_c, l.out_w * l.out_h, l.delta_gpu);
  {$endif}
    if l.&type = BATCHNORM then
        copy_gpu(l.outputs * l.batch, l.delta_gpu, 1, net.delta_gpu, 1)
end;

procedure update_batchnorm_layer_gpu(l: layer; batch: longint; learning_rate_init: single; momentum: single; decay: single; loss_scale: single);
var
    learning_rate: single;
begin
    learning_rate := learning_rate_init * l.learning_rate_scale / loss_scale;
    axpy_ongpu(l.c, learning_rate / batch, l.bias_updates_gpu, 1, l.biases_gpu, 1);
    scal_ongpu(l.c, momentum, l.bias_updates_gpu, 1);
    axpy_ongpu(l.c, learning_rate / batch, l.scale_updates_gpu, 1, l.scales_gpu, 1);
    scal_ongpu(l.c, momentum, l.scale_updates_gpu, 1)
end;

{$endif}

end.

