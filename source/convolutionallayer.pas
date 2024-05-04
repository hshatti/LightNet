unit ConvolutionalLayer;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$asmmode intel}
{$else}
{$excessprecision off}
{$endif}
{$pointermath on}

interface

uses
  SysUtils, math, darknet, blas, col2im, gemm, box, image, Activations, BatchNormLayer {$ifdef AI2}, XNORLayer{$endif};

Type
   PConvolutionalLayer = ^TConvolutionalLayer ;
   TConvolutionalLayer = TLayer ;

procedure binarize_weights(const weights: PSingle; const n, size: longint; const binary: PSingle);
procedure binarize_cpu(const input: PSingle; const n: longint; const binary: PSingle);
procedure binarize_input(const input: PSingle; const n, size: longint; const binary: PSingle);
function convolutional_out_height(const l: TConvolutionalLayer):longint;
function convolutional_out_width(const l: TConvolutionalLayer):longint;
function get_convolutional_image(const l: TConvolutionalLayer):TImageData;
function get_convolutional_delta(const l: TConvolutionalLayer):TImageData;
function get_workspace_size32(const l: TConvolutionalLayer):size_t;
function get_workspace_size16(const l: TConvolutionalLayer):size_t;
function get_convolutional_workspace_size(const l: TConvolutionalLayer):size_t;
{$ifdef GPU}
{$ifdef CUDNN}
procedure create_convolutional_cudnn_tensors(l: Player);
procedure cudnn_convolutional_setup(l: Player; cudnn_preference: longint; workspace_size_specify: size_t);
{$endif}
{$endif}
procedure free_convolutional_batchnorm(var l: TConvolutionalLayer);
function make_convolutional_layer(const batch, steps, h, w, c, n:longint; groups, size, stride_x, stride_y, dilation, padding:longint; const activation: TActivation; const batch_normalize, binary, xnor, adam, use_bin_output:boolean;const index, antialiasing: longint; const share_layer: PConvolutionalLayer; const assisted_excitation:longint;const deform, train: boolean):TConvolutionalLayer;
procedure denormalize_convolutional_layer(const l: TConvolutionalLayer);
procedure test_convolutional_layer();
procedure resize_convolutional_layer(var l: TConvolutionalLayer; const w, h: longint);
procedure set_specified_workspace_limit(const l: PConvolutionalLayer; const workspace_size_limit: size_t);
procedure gemm_nn_custom(M: longint; N: longint; K: longint; ALPHA: single; A: PSingle; lda: longint; B: PSingle; ldb: longint; C: PSingle; ldc: longint);
procedure get_mean_array(const src: PSingle; const size: size_t; const filters: size_t; const mean_arr: PSingle);
procedure bit_to_float(src: Pbyte; dst: PSingle; size: size_t; filters: size_t; mean_arr: PSingle);
procedure binary_align_weights(const l: PConvolutionalLayer);
function binary_transpose_align_input(k: longint; n: longint; b: PSingle; t_bit_input: PPByte; ldb_align: size_t; bit_align: longint):size_t;
procedure assisted_excitation_forward(var l: TConvolutionalLayer; const state: PNetworkState);
procedure forward_convolutional_layer(var l: TConvolutionalLayer; const state: PNetworkState);
procedure backward_convolutional_layer(var l: TConvolutionalLayer; const state: PNetworkState);
procedure update_convolutional_layer(const l: TConvolutionalLayer; const arg :TUpdateArgs);
function get_convolutional_weight(const l: TConvolutionalLayer; const i: longint):TImageData;
procedure rgbgr_weights(const l: TConvolutionalLayer);
procedure rescale_weights(const l: TConvolutionalLayer; const scale, trans: single);
function get_weights(const l: TConvolutionalLayer):TArray<TImageData>;
function visualize_convolutional_layer(const l: TConvolutionalLayer; const window: string; const prev_weights: TArray<TImageData>):TArray<TImageData>;

implementation


procedure swap_binary(var l: TConvolutionalLayer);
var
    swap: PSingle;
begin
    swap := l.weights;
    l.weights := l.binary_weights;
    l.binary_weights := swap;
{$ifdef GPU}
    swap := l.weights_gpu;
    l.weights_gpu := l.binary_weights_gpu;
    l.binary_weights_gpu := swap
{$endif}
end;

procedure binarize_weights(const weights: PSingle; const n, size: longint; const binary: PSingle);
var
    i: longint;
    f: longint;
    mean: single;
begin
    for f := 0 to n -1 do
        begin
            mean := 0;
            for i := 0 to size -1 do
                mean := mean + abs(weights[f * size+i]);
            mean := mean / size;
            for i := 0 to size -1 do
                if weights[f * size+i] > 0 then
                    binary[f * size+i] := mean
                else
                    binary[f * size+i] := -mean
        end
end;

procedure binarize_cpu(const input: PSingle; const n: longint; const binary: PSingle);
var
    i: longint;
begin
    for i := 0 to n -1 do
        if input[i] > 0 then
            binary[i] := 1
        else
            binary[i] := -1
end;

procedure binarize_input(const input: PSingle; const n, size: longint; const binary: PSingle);
var
    i, s: longint;
    mean: single;
begin
    for s := 0 to size -1 do
        begin
            mean := 0;
            for i := 0 to n -1 do
                mean := mean + abs(input[i * size+s]);
            mean := mean / n;
            for i := 0 to n -1 do
                if input[i * size+s] > 0 then
                    binary[i * size+s] := mean
                else
                    binary[i * size+s] := -mean
        end
end;

function convolutional_out_height(const l: TConvolutionalLayer):longint;
begin
    exit((l.h+2 * l.pad-l.size) div l.stride_y+1)
end;

function convolutional_out_width(const l: TConvolutionalLayer):longint;
begin
    exit((l.w+2 * l.pad-l.size) div l.stride_x+1)
end;

function get_convolutional_image(const l: TConvolutionalLayer):TImageData;
var
    h, w, c: longint;
begin
    h := convolutional_out_height(l);
    w := convolutional_out_width(l);
    c := l.n;
    exit(float_to_image(w, h, c, l.output))
end;

function get_convolutional_delta(const l: TConvolutionalLayer):TImageData;
var
    h, w, c: longint;
begin
    h := convolutional_out_height(l);
    w := convolutional_out_width(l);
    c := l.n;
    exit(float_to_image(w, h, c, l.delta))
end;

function get_workspace_size32(const l: TConvolutionalLayer):size_t;
var
    most: size_t;
    s: size_t;
    re_packed_input_size: size_t;
    workspace_size: size_t;
begin
{$ifdef CUDNN}
    if gpu_index >= 0 then
        begin
            most := 0;
            s := 0;
            CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), l.srcTensorDesc, l.weightDesc, l.convDesc, l.dstTensorDesc, l.fw_algo,  and s));
            if s > most then
                most := s;
            CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(), l.srcTensorDesc, l.ddstTensorDesc, l.convDesc, l.dweightDesc, l.bf_algo,  and s));
            if (s > most) and l.train then
                most := s;
            CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(), l.weightDesc, l.ddstTensorDesc, l.convDesc, l.dsrcTensorDesc, l.bd_algo,  and s));
            if s > most and l.train then
                most := s;
            exit(most)
        end;
{$endif}
    if l.xnor then
        begin
            re_packed_input_size := l.c * l.w * l.h * sizeof(single);
            workspace_size := size_t(l.bit_align) * l.size * l.size * l.c * sizeof(single);
            if workspace_size < re_packed_input_size then
                workspace_size := re_packed_input_size;
            exit(workspace_size)
        end;
    exit(size_t(l.out_h) * l.out_w * l.size * l.size * (l.c div l.groups) * sizeof(single))
end;

function get_workspace_size16(const l: TConvolutionalLayer):size_t;
var
    most: size_t;
    s: size_t;
begin
{$if defined(CUDNN) and defined(CUDNN_HALF)}
    if gpu_index >= 0 then
        begin
            most := 0;
            s := 0;
            CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), l.srcTensorDesc16, l.weightDesc16, l.convDesc, l.dstTensorDesc16, l.fw_algo16,  @s));
            if s > most then
                most := s;
            CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle(), l.srcTensorDesc16, l.ddstTensorDesc16, l.convDesc, l.dweightDesc16, l.bf_algo16,  @s));
            if s > most and l.train then
                most := s;
            CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle(), l.weightDesc16, l.ddstTensorDesc16, l.convDesc, l.dsrcTensorDesc16, l.bd_algo16,  @s));
            if s > most and l.train then
                most := s;
            exit(most)
        end;
{$endif}
    exit(0)
end;

function get_convolutional_workspace_size(const l: TConvolutionalLayer):size_t;
var
    workspace_size: size_t;
    workspace_size16: size_t;
begin
    workspace_size := get_workspace_size32(l);
    workspace_size16 := get_workspace_size16(l);
    if (workspace_size16 > workspace_size) then
        workspace_size := workspace_size16;
    exit(workspace_size)
end;

{$ifdef GPU}
{$ifdef CUDNN}
procedure create_convolutional_cudnn_tensors(l: Player);
begin
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.normTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.normDstTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.srcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.dstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor( @l.weightDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.dsrcTensorDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.ddstTensorDesc));
    CHECK_CUDNN(cudnnCreateFilterDescriptor( @l.dweightDesc));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.normDstTensorDescF16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.srcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.dstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor( @l.weightDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.dsrcTensorDesc16));
    CHECK_CUDNN(cudnnCreateTensorDescriptor( @l.ddstTensorDesc16));
    CHECK_CUDNN(cudnnCreateFilterDescriptor( @l.dweightDesc16));
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor( @l.convDesc))
end;

procedure cudnn_convolutional_setup(l: Player; cudnn_preference: longint; workspace_size_specify: size_t);
var
    data_type: cudnnDataType_t;
    free_memory: size_t;
    total_memory: size_t;
    requested_algo_count: longint;
    found_conv_algorithm: longint;
    min_time: single;
    conv_fwd_results: array[0..99] of cudnnConvolutionFwdAlgoPerf_t;
    i: longint;
    conv_bwd_data_results: array[0..99] of cudnnConvolutionBwdDataAlgoPerf_t;
    conv_bwd_filter_results: array[0..99] of cudnnConvolutionBwdFilterAlgoPerf_t;
    forward_algo: longint;
    backward_algo: longint;
    backward_filter: longint;
begin

{.$if (CUDNN_MAJOR >= 7)}
    data_type := CUDNN_DATA_FLOAT;
    if (l.groups < 1) then
        l.groups := 1;
    if (l.stride_x < 1) then
        l.stride_x := 1;
    if l.stride_y < 1 then
        l.stride_y := 1;
    CHECK_CUDNN(cudnnSetConvolutionGroupCount(l.convDesc, l.groups));
    CHECK_CUDNN(cudnnSetConvolutionMathType(l.convDesc, CUDNN_TENSOR_OP_MATH));
  {.$if ((CUDNN_MAJOR*10) + CUDNN_MINOR) >= 72}
   //CHECK_CUDNN(cudnnSetConvolutionMathType(l->convDesc, CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION)); // reduces the speed of regular and group convolution
  {.$endif}
{.$else}
    if l.groups > 1 then
        error('CUDNN < 7 doesn''t support groups, please upgrade!', DARKNET_LOC);
{.$endif}
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dsrcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.ddstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l.batch, l.out_c, l.out_h, l.out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.dweightDesc, data_type, CUDNN_TENSOR_NCHW, l.n, l.c div l.groups, l.size, l.size));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc, CUDNN_TENSOR_NCHW, data_type, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc, CUDNN_TENSOR_NCHW, data_type, l.batch, l.out_c, l.out_h, l.out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.weightDesc, data_type, CUDNN_TENSOR_NCHW, l.n, l.c div l.groups, l.size, l.size));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dsrcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.ddstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.out_c, l.out_h, l.out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.dweightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l.n, l.c div l.groups, l.size, l.size));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.srcTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.c, l.h, l.w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.dstTensorDesc16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.out_c, l.out_h, l.out_w));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(l.weightDesc16, CUDNN_DATA_HALF, CUDNN_TENSOR_NCHW, l.n, l.c div l.groups, l.size, l.size));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDescF16, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, l.batch, l.out_c, l.out_h, l.out_w));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, l.out_c, 1, 1));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(l.normDstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, l.batch, l.out_c, l.out_h, l.out_w));
{.$if CUDNN_MAJOR >= 6}
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x, l.dilation, l.dilation, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
{.$else}
//    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(l.convDesc, l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x, l.dilation, l.dilation, CUDNN_CROSS_CORRELATION));
{.$endif}
{$ifdef CUDNN_MAJOR >= 8}
    if cudnn_preference = cudnn_smallest then
        workspace_size_specify := 0;
    requested_algo_count := 0; returned_algo_count := 0;
    found_conv_algorithm := 0;
    min_time := 1000000;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle(),  and requested_algo_count));
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle(), l.srcTensorDesc, l.weightDesc, l.convDesc, l.dstTensorDesc, requested_algo_count,  and returned_algo_count, conv_fwd_results));
    CHECK_CUDA(cudaMemGetInfo( and free_memory,  and total_memory));
    found_conv_algorithm := 0;
    min_time := 1000000;
    for i := 0 to returned_algo_count -1 do
        if (conv_fwd_results[i].status = CUDNN_STATUS_SUCCESS) and conv_fwd_results[i].algo <> CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED and conv_fwd_results[i].memory < free_memory and ((conv_fwd_results[i].memory <= workspace_size_specify) or (cudnn_preference = cudnn_fastest)) and conv_fwd_results[i].time < min_time then
            begin
                found_conv_algorithm := 1;
                l.fw_algo := conv_fwd_results[i].algo;
                min_time := conv_fwd_results[i].time
            end;
    if not found_conv_algorithm then
        error('Error: cuDNN hasn''t found FWD algo for convolution', DARKNET_LOC);
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(cudnn_handle(),  and requested_algo_count));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle(), l.weightDesc, l.ddstTensorDesc, l.convDesc, l.dsrcTensorDesc, requested_algo_count,  and returned_algo_count,  and conv_bwd_data_results[0]));
    CHECK_CUDA(cudaMemGetInfo( and free_memory,  and total_memory));
    found_conv_algorithm := 0;
    min_time := 1000000;
    for i := 0 to returned_algo_count -1 do
        if (conv_bwd_data_results[i].status = CUDNN_STATUS_SUCCESS) and conv_bwd_data_results[i].memory < free_memory and ((conv_bwd_data_results[i].memory <= workspace_size_specify) or (cudnn_preference = cudnn_fastest)) and conv_bwd_data_results[i].time < min_time then
            begin
                found_conv_algorithm := 1;
                l.bd_algo := conv_bwd_data_results[i].algo;
                min_time := conv_bwd_data_results[i].time
            end;
    if not found_conv_algorithm then
        error('Error: cuDNN hasn''t found BWD-data algo for convolution', DARKNET_LOC);
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(cudnn_handle(),  and requested_algo_count));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle(), l.srcTensorDesc, l.ddstTensorDesc, l.convDesc, l.dweightDesc, requested_algo_count,  and returned_algo_count,  and conv_bwd_filter_results[0]));
    CHECK_CUDA(cudaMemGetInfo( and free_memory,  and total_memory));
    found_conv_algorithm := 0;
    min_time := 1000000;
    for i := 0 to returned_algo_count -1 do
        if (conv_bwd_filter_results[i].status = CUDNN_STATUS_SUCCESS) and (conv_bwd_filter_results[i].memory < free_memory) and ((conv_bwd_filter_results[i].memory <= workspace_size_specify) or (cudnn_preference = cudnn_fastest)) and conv_bwd_filter_results[i].time < min_time then
            begin
                found_conv_algorithm := 1;
                l.bf_algo := conv_bwd_filter_results[i].algo;
                min_time := conv_bwd_filter_results[i].time
            end;
    if not found_conv_algorithm then
        error('Error: cuDNN hasn''t found BWD-filter algo for convolution', DARKNET_LOC);
{$else}
    forward_algo := CUDNN_CONVOLUTION_FWD_PREFER_FASTEST;
    backward_algo := CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST;
    backward_filter := CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST;
    if cudnn_preference = cudnn_smallest then
        begin
            forward_algo := CUDNN_CONVOLUTION_FWD_NO_WORKSPACE;
            backward_algo := CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE;
            backward_filter := CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE;
            printf(' CUDNN-slow ')
        end;
    if cudnn_preference = cudnn_specify then
        begin
            forward_algo := CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT;
            backward_algo := CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT;
            backward_filter := CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT
        end;
    CHECK_CUDNN(cudnnGetConvolutionForwardAlgorithm(cudnn_handle(), l.srcTensorDesc, l.weightDesc, l.convDesc, l.dstTensorDesc, cudnnConvolutionFwdPreference_t(forward_algo), workspace_size_specify,  and l.fw_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardDataAlgorithm(cudnn_handle(), l.weightDesc, l.ddstTensorDesc, l.convDesc, l.dsrcTensorDesc, cudnnConvolutionBwdDataPreference_t(backward_algo), workspace_size_specify,  and l.bd_algo));
    CHECK_CUDNN(cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle(), l.srcTensorDesc, l.ddstTensorDesc, l.convDesc, l.dweightDesc, cudnnConvolutionBwdFilterPreference_t(backward_filter), workspace_size_specify,  and l.bf_algo));
{$endif}
    //if data_type = CUDNN_DATA_HALF then begin
      l.fw_algo16 := CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
      l.bd_algo16 := CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
      l.bf_algo16 := CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1
    //end
end;
{$endif}
{$endif}

procedure free_convolutional_batchnorm(var l: TConvolutionalLayer);
begin
    if not assigned(l.share_layer) then
        begin
            if assigned(l.scales) then
                FreeMemAndNil(l.scales);
            if assigned(l.scale_updates) then
                FreeMemAndNil(l.scale_updates);
            if assigned(l.mean) then
                FreeMemAndNil(l.mean);
            if assigned(l.variance) then
                FreeMemAndNil(l.variance);
            if assigned(l.mean_delta) then
                FreeMemAndNil(l.mean_delta);
            if assigned(l.variance_delta) then
                FreeMemAndNil(l.variance_delta);
            if assigned(l.rolling_mean) then
                FreeMemAndNil(l.rolling_mean);
            if assigned(l.rolling_variance) then
                FreeMemAndNil(l.rolling_variance);
            if assigned(l.x) then
                FreeMemAndNil(l.x);
            if assigned(l.x_norm) then
                FreeMemAndNil(l.x_norm);
{$ifdef GPU}
            if l.scales_gpu then
                cuda_free(l.scales_gpu);
            if l.scale_updates_gpu then
                cuda_free(l.scale_updates_gpu);
            if l.mean_gpu then
                cuda_free(l.mean_gpu);
            if l.variance_gpu then
                cuda_free(l.variance_gpu);
            if l.mean_delta_gpu then
                cuda_free(l.mean_delta_gpu);
            if l.variance_delta_gpu then
                cuda_free(l.variance_delta_gpu);
            if l.rolling_mean_gpu then
                cuda_free(l.rolling_mean_gpu);
            if l.rolling_variance_gpu then
                cuda_free(l.rolling_variance_gpu);
            if l.x_gpu then
                cuda_free(l.x_gpu);
            if l.x_norm_gpu then
                cuda_free(l.x_norm_gpu)
{$endif}
        end
end;

function make_convolutional_layer(const batch, steps, h, w, c, n:longint; groups, size, stride_x, stride_y, dilation, padding:longint; const activation: TActivation; const batch_normalize, binary, xnor, adam, use_bin_output:boolean;const index, antialiasing: longint; const share_layer: PConvolutionalLayer; const assisted_excitation:longint;const deform, train: boolean):TConvolutionalLayer;
var
    total_batch, i, blur_stride_x, blur_stride_y, out_h, out_w, align, src_align, k: longint;
    scale: single;
    new_c, in_re_packed_input_size, k_aligned, t_bit_input_size: size_t;
    size2, blur_size, blur_pad, blur_nweights: longint;
begin
    total_batch := batch * steps;
    result := default(TConvolutionalLayer);
    result.&type := ltCONVOLUTIONAL;
    result.train := train;
    if xnor then
        groups := 1;
    if groups < 1 then
        groups := 1;
    blur_stride_x := stride_x;
    blur_stride_y := stride_y;
    result.antialiasing := antialiasing;
    if antialiasing>0 then begin
        stride_x        := 1;
        stride_y        := 1;
        result.stride   := 1;
        result.stride_x := 1;
        result.stride_y := 1
    end;
    result.wait_stream_id := -1;
    result.deform := deform;
    result.assisted_excitation := assisted_excitation;
    result.share_layer := PLayer(share_layer);
    result.index := index;
    result.h := h;
    result.w := w;
    result.c := c;
    result.groups := groups;
    result.n := n;
    result.binary := binary;
    result.xnor := xnor;
    result.use_bin_output := use_bin_output;
    result.batch := batch;
    result.steps := steps;
    result.stride := stride_x;
    result.stride_x := stride_x;
    result.stride_y := stride_y;
    result.dilation := dilation;
    result.size := size;
    result.pad := padding;
    result.batch_normalize := batch_normalize;
    result.learning_rate_scale := 1;
    result.nweights := (c div groups) * n * size * size;
    if assigned(result.share_layer) then
        begin
            if (result.size <> result.share_layer.size) or (result.nweights <> result.share_layer.nweights) or (result.c <> result.share_layer.c) or (result.n <> result.share_layer.n) then
                raise Exception.create('Layer size, nweights, channels or filters don''t match for the share_layer');
            result.weights := result.share_layer.weights;
            result.weight_updates := result.share_layer.weight_updates;
            result.biases := result.share_layer.biases;
            result.bias_updates := result.share_layer.bias_updates
        end
    else
        begin
            result.weights := TSingles.Create(result.nweights);
            result.biases := TSingles.Create(n);
            if train then
                begin
                    result.weight_updates := TSingles.Create(result.nweights);
                    result.bias_updates := TSingles.Create(n);
                    result.weights_ema := TSingles.Create(result.nweights);
                    result.biases_ema := TSingles.Create(n)
                end
        end;
    scale := sqrt(2 / (size * size * c / groups));
    if (result.activation = acNORM_CHAN) or (result.activation = acNORM_CHAN_SOFTMAX) or (result.activation = acNORM_CHAN_SOFTMAX_MAXVAL) then
        for i := 0 to result.nweights -1 do
            result.weights[i] := 1
    else
        for i := 0 to result.nweights -1 do
            result.weights[i] := scale * rand_uniform(-1, 1);
    out_h := convolutional_out_height(result);
    out_w := convolutional_out_width(result);
    result.out_h := out_h;
    result.out_w := out_w;
    result.out_c := n;
    result.outputs := result.out_h * result.out_w * result.out_c;
    result.inputs := result.w * result.h * result.c;
    result.activation := activation;
    result.output := TSingles.Create(total_batch * result.outputs);
{$ifndef GPU}
    if train then
        result.delta := TSingles.Create(total_batch * result.outputs);
{$endif}
    result.forward := forward_convolutional_layer;
    result.backward := backward_convolutional_layer;
    result.update := update_convolutional_layer;
    if binary then
        begin
            result.binary_weights := TSingles.Create(result.nweights);
            setLength(result.cweights, result.nweights);
            result.scales := TSingles.Create(n)
        end;
    if xnor then
        begin
            result.binary_weights := TSingles.Create(result.nweights);
            result.binary_input := TSingles.Create(result.inputs * result.batch);
            align := 32;
            src_align := result.out_h * result.out_w;
            result.bit_align := src_align+(align-src_align mod align);
            result.mean_arr := TSingles.Create(result.n);
            new_c := result.c div 32;
            in_re_packed_input_size := new_c * result.w * result.h+1;
            result.bin_re_packed_input := AllocMem(in_re_packed_input_size*sizeof(UInt32));
            result.lda_align := 256;
            k := result.size * result.size * result.c;
            k_aligned := k+(result.lda_align-k mod result.lda_align);
            t_bit_input_size := k_aligned * result.bit_align div 8;
            setLength(result.t_bit_input, t_bit_input_size)
            //setLength(result.t_bit_input, t_bit_input_size);
        end;
    if batch_normalize then
        begin
            if assigned(result.share_layer) then
                begin
                    result.scales := result.share_layer.scales;
                    result.scale_updates := result.share_layer.scale_updates;
                    result.mean := result.share_layer.mean;
                    result.variance := result.share_layer.variance;
                    result.mean_delta := result.share_layer.mean_delta;
                    result.variance_delta := result.share_layer.variance_delta;
                    result.rolling_mean := result.share_layer.rolling_mean;
                    result.rolling_variance := result.share_layer.rolling_variance
                end
            else
                begin
                    result.scales := TSingles.Create(n);
                    for i := 0 to n -1 do
                        result.scales[i] := 1;
                    if train then
                        begin
                            result.scales_ema := TSingles.Create(n);
                            result.scale_updates := TSingles.Create(n);
                            result.mean := TSingles.Create(n);
                            result.variance := TSingles.Create(n);
                            result.mean_delta := TSingles.Create(n);
                            result.variance_delta := TSingles.Create(n)
                        end;
                    result.rolling_mean := TSingles.Create(n);
                    result.rolling_variance := TSingles.Create(n)
                end;
{$ifndef GPU}
            if train then
                begin
                    result.x := TSingles.Create(total_batch * result.outputs);
                    result.x_norm := TSingles.Create(total_batch * result.outputs)
                end
{$endif}
        end;
{$ifndef GPU}
    if result.activation in [acSWISH, acMISH, acHARD_MISH] then
        result.activation_input := TSingles.Create(total_batch * result.outputs);
{$endif}
    if adam then
        begin
            result.adam := 1;
            result.m := TSingles.Create(result.nweights);
            result.v := TSingles.Create(result.nweights);
            result.bias_m := TSingles.Create(n);
            result.scale_m := TSingles.Create(n);
            result.bias_v := TSingles.Create(n);
            result.scale_v := TSingles.Create(n)
        end;
{$ifdef GPU}
    result.forward_gpu := forward_convolutional_layer_gpu;
    result.backward_gpu := backward_convolutional_layer_gpu;
    result.update_gpu := update_convolutional_layer_gpu;
    if gpu_index >= 0 then
        begin
            if train and ((result.activation = SWISH) or (result.activation = MISH) or (result.activation = HARD_MISH)) then
                result.activation_input_gpu := cuda_make_array(result.activation_input, total_batch * result.outputs);
            if result.deform then
                result.weight_deform_gpu := cuda_make_array(NULL, result.nweights);
            if adam then
                begin
                    result.m_gpu := cuda_make_array(result.m, result.nweights);
                    result.v_gpu := cuda_make_array(result.v, result.nweights);
                    result.bias_m_gpu := cuda_make_array(result.bias_m, n);
                    result.bias_v_gpu := cuda_make_array(result.bias_v, n);
                    result.scale_m_gpu := cuda_make_array(result.scale_m, n);
                    result.scale_v_gpu := cuda_make_array(result.scale_v, n)
                end;
            if result.share_layer then
                begin
                    result.weights_gpu := result.share_layer.weights_gpu;
                    result.weight_updates_gpu := result.share_layer.weight_updates_gpu;
                    result.weights_gpu16 := result.share_layer.weights_gpu16;
                    result.weight_updates_gpu16 := result.share_layer.weight_updates_gpu16;
                    result.biases_gpu := result.share_layer.biases_gpu;
                    result.bias_updates_gpu := result.share_layer.bias_updates_gpu
                end
            else
                begin
                    result.weights_gpu := cuda_make_array(result.weights, result.nweights);
                    if train then
                        result.weight_updates_gpu := cuda_make_array(result.weight_updates, result.nweights);
{$ifdef CUDNN_HALF}
                    result.weights_gpu16 := cuda_make_array(NULL, result.nweights div 2+1);
                    if train then
                        result.weight_updates_gpu16 := cuda_make_array(NULL, result.nweights div 2+1);
{$endif}
                    result.biases_gpu := cuda_make_array(result.biases, n);
                    if train then
                        result.bias_updates_gpu := cuda_make_array(result.bias_updates, n)
                end;
            result.output_gpu := cuda_make_array(result.output, total_batch * out_h * out_w * n);
            if train then
                result.delta_gpu := cuda_make_array(result.delta, total_batch * out_h * out_w * n);
            if binary then
                result.binary_weights_gpu := cuda_make_array(result.weights, result.nweights);
            if xnor then
                begin
                    result.binary_weights_gpu := cuda_make_array(result.weights, result.nweights);
                    result.mean_arr_gpu := cuda_make_array(0, result.n);
                    result.binary_input_gpu := cuda_make_array(0, result.inputs * result.batch)
                end;
            if batch_normalize then
                begin
                    if result.share_layer then
                        begin
                            result.scales_gpu := result.share_layer.scales_gpu;
                            result.scale_updates_gpu := result.share_layer.scale_updates_gpu;
                            result.mean_gpu := result.share_layer.mean_gpu;
                            result.variance_gpu := result.share_layer.variance_gpu;
                            result.rolling_mean_gpu := result.share_layer.rolling_mean_gpu;
                            result.rolling_variance_gpu := result.share_layer.rolling_variance_gpu;
                            result.mean_delta_gpu := result.share_layer.mean_delta_gpu;
                            result.variance_delta_gpu := result.share_layer.variance_delta_gpu
                        end
                    else
                        begin
                            result.scales_gpu := cuda_make_array(result.scales, n);
                            if train then
                                begin
                                    result.scale_updates_gpu := cuda_make_array(result.scale_updates, n);
                                    result.mean_gpu := cuda_make_array(result.mean, n);
                                    result.variance_gpu := cuda_make_array(result.variance, n);
                                    result.m_cbn_avg_gpu := cuda_make_array(result.mean, n);
                                    result.v_cbn_avg_gpu := cuda_make_array(result.variance, n);
{$ifndef CUDNN}
                                    result.mean_delta_gpu := cuda_make_array(result.mean, n);
                                    result.variance_delta_gpu := cuda_make_array(result.variance, n)
{$endif}
                                end;
                            result.rolling_mean_gpu := cuda_make_array(result.mean, n);
                            result.rolling_variance_gpu := cuda_make_array(result.variance, n)
                        end;
                    if train then
                        begin
                            result.x_gpu := cuda_make_array(result.output, total_batch * out_h * out_w * n);
{$ifndef CUDNN}
                            result.x_norm_gpu := cuda_make_array(result.output, total_batch * out_h * out_w * n)
{$endif}
                        end
                end;
            if result.assisted_excitation then
                begin
                    size2 := result.out_w * result.out_h * result.batch;
                    result.gt_gpu := cuda_make_array(NULL, size2);
                    result.a_avg_gpu := cuda_make_array(NULL, size2)
                end;
{$ifdef CUDNN}
            create_convolutional_cudnn_tensors( @result);
            cudnn_convolutional_setup( @result, cudnn_fastest, 0)
{$endif}
        end;
{$endif}
    result.workspace_size := get_convolutional_workspace_size(result);
    result.bflops := (2.0 * result.nweights * result.out_h * result.out_w) / 1000000000;
    if result.xnor then
        result.bflops := result.bflops / 32;
    if result.xnor and result.use_bin_output then
        write(ErrOutput, 'convXB')
    else
        if result.xnor then
            write(ErrOutput, 'convX ')
    else
        if assigned(result.share_layer) then
            write(ErrOutput, 'convS ')
    else
        if result.assisted_excitation<>0 then
            write(ErrOutput, 'convAE')
    else
        write(ErrOutput, 'conv  ');
    if groups > 1 then
        write(ErrOutput,format( '%5d/%4d ', [n, groups]))
    else
        write(ErrOutput, format('%5d      ', [n]));
    if stride_x <> stride_y then
        write(ErrOutput, format('%2dx%2d/%2dx%2d ', [size, size, stride_x, stride_y]))
    else
        begin
            if dilation > 1 then
                write(ErrOutput, format('%2d x%2d/%2d(%1d)', [size, size, stride_x, dilation]))
            else
                write(ErrOutput, format('%2d x%2d/%2d   ', [size, size, stride_x]))
        end;
    writeln(ErrOutput, format('%4d x%4d x%4d -> %4d x%4d x%4d %5.3f BF', [w, h, c, result.out_w, result.out_h, result.out_c, result.bflops]));
    if result.antialiasing>0 then
        begin
            write('AA:  ');
            setLength(result.input_layer, 1);
            blur_size := 3;
            blur_pad := blur_size div 2;
            if result.antialiasing = 2 then
                begin
                    blur_size := 2;
                    blur_pad := 0
                end;
            result.input_layer[0] := make_convolutional_layer(batch, steps, out_h, out_w, n, n, n, blur_size, blur_stride_x, blur_stride_y, 1, blur_pad, acLINEAR, false, false, false, false, false, index, 0, nil, 0, false, train);
            blur_nweights := n * blur_size * blur_size;
            if blur_size = 2 then begin
                i := 0;
                while i < blur_nweights do begin
                    result.input_layer[0].weights[i+0] := 1 / 4;
                    result.input_layer[0].weights[i+1] := 1 / 4;
                    result.input_layer[0].weights[i+2] := 1 / 4;
                    result.input_layer[0].weights[i+3] := 1 / 4;
                    i := i + (blur_size * blur_size)
                end
            end
            else begin
                i := 0;
                while i < blur_nweights do begin
                    result.input_layer[0].weights[i+0] := 1 / 16;
                    result.input_layer[0].weights[i+1] := 2 / 16;
                    result.input_layer[0].weights[i+2] := 1 / 16;
                    result.input_layer[0].weights[i+3] := 2 / 16;
                    result.input_layer[0].weights[i+4] := 4 / 16;
                    result.input_layer[0].weights[i+5] := 2 / 16;
                    result.input_layer[0].weights[i+6] := 1 / 16;
                    result.input_layer[0].weights[i+7] := 2 / 16;
                    result.input_layer[0].weights[i+8] := 1 / 16;
                    i := i + (blur_size * blur_size)
                end;
            end;
            for i := 0 to n -1 do
                result.input_layer[0].biases[i] := 0;
{$ifdef GPU}
            if gpu_index >= 0 then
                begin
                    result.input_antialiasing_gpu := cuda_make_array(NULL, result.batch * result.outputs);
                    push_convolutional_layer( * (result.input_layer))
                end
{$endif}
        end;
end;

procedure denormalize_convolutional_layer(const l: TConvolutionalLayer);
var
    i: longint;
    j: longint;
    scale: single;
begin
    for i := 0 to l.n -1 do
        begin
            scale := l.scales[i] / sqrt(l.rolling_variance[i]+0.00001);
            for j := 0 to l.nweights -1 do
                l.weights[i * l.nweights+j] := l.weights[i * l.nweights+j] * scale;
            l.biases[i] := l.biases[i] - (l.rolling_mean[i] * scale);
            l.scales[i] := 1;
            l.rolling_mean[i] := 0;
            l.rolling_variance[i] := 1
        end
end;

procedure test_convolutional_layer();
var
    l: TConvolutionalLayer;
    data: array of single;
    state: TNetworkState;
begin
    l := make_convolutional_layer(1, 1, 5, 5, 3, 2, 1, 5, 2, 2, 1, 1, acLEAKY, false, false, false, false, false, 0, 0, nil, 0, false, false);
    l.batch_normalize := true;
    data := [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3];
    state := default(TNetworkState);
    state.input := @data[0];
    forward_convolutional_layer(l, @state)
end;

procedure resize_convolutional_layer(var l: TConvolutionalLayer; const w,
  h: longint);
var
    total_batch: longint;
    old_w: longint;
    old_h: longint;
    out_w: longint;
    out_h: longint;
    size: longint;
    free_byte: size_t;
    total_byte: size_t;
begin
    total_batch := l.batch * l.steps;
    old_w := l.w;
    old_h := l.h;
    l.w := w;
    l.h := h;
    out_w := convolutional_out_width(l);
    out_h := convolutional_out_height(l);
    l.out_w := out_w;
    l.out_h := out_h;
    l.outputs := l.out_h * l.out_w * l.out_c;
    l.inputs := l.w * l.h * l.c;
    l.output.reAllocate( total_batch * l.outputs);
    if l.train then
        begin
            l.delta.ReAllocate(total_batch * l.outputs);
            if l.batch_normalize then
                begin
                    l.x.ReAllocate(total_batch * l.outputs);
                    l.x_norm.ReAllocate(total_batch * l.outputs)
                end
        end;
    if l.xnor then
        //l.binary_input.reallocate(l.inputs*l.batch)
       ;
    if (l.activation = acSWISH) or (l.activation = acMISH) or (l.activation = acHARD_MISH) then
        l.activation_input.ReAllocate(total_batch * l.outputs);
{$ifdef GPU}
    if old_w < w or old_h < h or l.dynamic_minibatch then
        begin
            if l.train then
                begin
                    cuda_free(l.delta_gpu);
                    l.delta_gpu := cuda_make_array(l.delta, total_batch * l.outputs)
                end;
            cuda_free(l.output_gpu);
            l.output_gpu := cuda_make_array(l.output, total_batch * l.outputs);
            if l.batch_normalize then
                begin
                    cuda_free(l.x_gpu);
                    l.x_gpu := cuda_make_array(l.output, total_batch * l.outputs);
{$ifndef CUDNN}
                    cuda_free(l.x_norm_gpu);
                    l.x_norm_gpu := cuda_make_array(l.output, total_batch * l.outputs)
{$endif}
                end;
            if l.xnor then
                begin
                    cuda_free(l.binary_input_gpu);
                    l.binary_input_gpu := cuda_make_array(0, l.inputs * l.batch)
                end;
            if l.activation = acSWISH or l.activation = acMISH or l.activation = acHARD_MISH then
                begin
                    cuda_free(l.activation_input_gpu);
                    l.activation_input_gpu := cuda_make_array(l.activation_input, total_batch * l.outputs)
                end;
            if l.assisted_excitation then
                begin
                    cuda_free(l.gt_gpu);
                    cuda_free(l.a_avg_gpu);
                    size := l.out_w * l.out_h * l.batch;
                    l.gt_gpu := cuda_make_array(NULL, size);
                    l.a_avg_gpu := cuda_make_array(NULL, size)
                end
        end;
{$ifndef CUDNN}
    cudnn_convolutional_setup(l, cudnn_fastest, 0);
{$endif}
{$endif}
    l.workspace_size := get_convolutional_workspace_size(l);
{$ifdef CUDNN}
    CHECK_CUDA(cudaMemGetInfo( and free_byte,  and total_byte));
    if l.workspace_size > free_byte or l.workspace_size >= total_byte div 2 then
        begin
            writeln(format(' used slow CUDNN algo without Workspace! Need memory: %d, available: %d',[ l.workspace_size, ifthen((free_byte < total_byte div 2), free_byte, total_byte div 2)]));
            cudnn_convolutional_setup(l, cudnn_smallest, 0);
            l.workspace_size := get_convolutional_workspace_size( * l)
        end
{$endif}
end;

procedure set_specified_workspace_limit(const l: PConvolutionalLayer; const workspace_size_limit: size_t);
var
    free_byte: size_t;
    total_byte: size_t;
begin
{$ifdef CUDNN}
    CHECK_CUDA(cudaMemGetInfo( and free_byte,  and total_byte));
    cudnn_convolutional_setup(l, cudnn_specify, workspace_size_limit);
    l.workspace_size := get_convolutional_workspace_size(l[0])
{$endif}
end;

//procedure add_bias(output: PSingle; biases: PSingle; batch: longint; n: longint; size: longint);
//var
//    i, j, b: longint;
//begin
//    for b := 0 to batch -1 do
//        for i := 0 to n -1 do
//            for j := 0 to size -1 do
//                output[(b * n+i) * size+j] := output[(b * n+i) * size+j] + biases[i]
//end;
//
//procedure scale_bias(output: PSingle; scales: PSingle; batch: longint; n: longint; size: longint);
//var
//    i, j, b: longint;
//begin
//    for b := 0 to batch -1 do
//        for i := 0 to n -1 do
//            for j := 0 to size -1 do
//                output[(b * n+i) * size+j] := output[(b * n+i) * size+j] * scales[i]
//end;
//
//procedure backward_bias(bias_updates: PSingle; delta: PSingle; batch: longint; n: longint; size: longint);
//var
//    i, b: longint;
//begin
//    for b := 0 to batch -1 do
//        for i := 0 to n -1 do
//            bias_updates[i] := bias_updates[i] + sum_array(delta+size * (i+b * n), size)
//end;

procedure gemm_nn_custom(M: longint; N: longint; K: longint; ALPHA: single; A: PSingle; lda: longint; B: PSingle; ldb: longint; C: PSingle; ldc: longint);
var
    i, j, kk: longint;
    A_PART: single;
begin
    for i := 0 to M -1 do
        for k := 0 to K -1 do
            begin
                A_PART := ALPHA * A[i * lda+kk];
                for j := 0 to N -1 do
                    C[i * ldc+j] := C[i * ldc+j] + (A_PART * B[kk * ldb+j])
            end
end;

procedure get_mean_array(const src: PSingle; const size: size_t; const filters: size_t; const mean_arr: PSingle);
var
    i: size_t;
    counter: size_t;
begin
    counter := 0;
    i := 0;
    while i < size do begin
        mean_arr[counter] := abs(src[i]);
        inc(counter);
        i := i + (size div filters)
    end
end;

procedure bit_to_float(src: Pbyte; dst: PSingle; size: size_t; filters: size_t; mean_arr: PSingle);
var
    i: size_t;
    mean_val: single;
begin
    FillDWord(dst[0], size, 0);//* sizeof(float));
    for i := 0 to size -1 do
        begin
            mean_val := 1;
            if mean_arr <> nil then
                mean_val := abs(mean_arr[i div (size div filters)]);

            if get_bit(src, i) then
                dst[i] := mean_val
            else
                dst[i] := -mean_val
        end
end;

procedure binary_align_weights(const l: PConvolutionalLayer);
var
    m, k: longint;
    new_lda, align_weights_size, i, j: size_t;
    fil, chan, items_per_filter, items_per_channel, c_pack: longint;
    src: single;
    align_weights:TArray<single>;
{$ifdef GPU}
    status: cudaError_t;
{$endif}
begin
    m := l.n;
    k := l.size * l.size * l.c;
    new_lda := k+(l.lda_align-k mod l.lda_align);
    l.new_lda := new_lda;
    binarize_weights(l.weights, m, k, l.binary_weights);
    align_weights_size := new_lda * m;
    l.align_bit_weights_size := align_weights_size div 8+1;
    //align_weights := single(xcalloc(align_weights_size, sizeof(float)));
    setLength(align_weights, align_weights_size);
    setLength(l.align_bit_weights, l.align_bit_weights_size);
    //setLength(l.align_bit_weights, l.align_bit_weights_size);
    for i := 0 to m -1 do
        for j := 0 to k -1 do
            align_weights[i * new_lda+j] := l.binary_weights[i * k+j];
    if l.c mod 32 = 0 then
        begin
            items_per_filter := l.c * l.size * l.size;
            for fil := 0 to l.n -1 do
                chan := 0;
                while chan < l.c do begin
                    items_per_channel := l.size * l.size;
                    for i := 0 to items_per_channel -1 do
                        begin
                            for c_pack := 0 to 32 -1 do
                                begin
                                    src := l.binary_weights[fil * items_per_filter+(chan+c_pack) * items_per_channel+i];
                                    align_weights[fil * new_lda+chan * items_per_channel+i * 32+c_pack] := src
                                end
                        end;
                    chan := chan + 32
                end;
            float_to_bit(@align_weights[0], PByte(l.align_bit_weights), align_weights_size);
            if gpu_index >= 0 then
                for i := 0 to align_weights_size div 8 -1 do
                    l.align_bit_weights[i] := not l.align_bit_weights[i] ;
            get_mean_array(l.binary_weights, m * k, l.n, l.mean_arr)
        end
    else
        begin
            float_to_bit(@align_weights[0], PByte(l.align_bit_weights), align_weights_size);
            get_mean_array(l.binary_weights, m * k, l.n, l.mean_arr)
        end;
{$ifdef GPU}
    l.align_workspace_size := l.bit_align * l.size * l.size * l.c;
    status := cudaMalloc(PPointer(@l.align_workspace_gpu), l.align_workspace_size * sizeof(float));
    status := cudaMalloc(PPointer(@l.transposed_align_workspace_gpu), l.align_workspace_size * sizeof(float));
    CHECK_CUDA(status);
    status := cudaMalloc(() and l.align_bit_weights_gpu, l.align_bit_weights_size);
    CHECK_CUDA(status);
    status := cudaMemcpy(l.align_bit_weights_gpu, l.align_bit_weights, l.align_bit_weights_size, cudaMemcpyHostToDevice);
    CHECK_CUDA(status);
    status := cudaMemcpy(l.binary_weights_gpu, l.binary_weights, m * k * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA(status);
    cuda_push_array(l.mean_arr_gpu, l.mean_arr, l.n);
    CHECK_CUDA(cudaDeviceSynchronize());
{$endif}
    //free(align_weights)
end;

function binary_transpose_align_input(k: longint; n: longint; b: PSingle; t_bit_input: PPByte; ldb_align: size_t; bit_align: longint):size_t;
var
    new_ldb, t_intput_size, t_bit_input_size: size_t;
begin
    new_ldb := k + (ldb_align - k mod ldb_align); // (k / 8 + 1) * 8;
    //printf("\n n = %d, bit_align = %d \n", n, bit_align);
    t_intput_size := new_ldb * bit_align;// n;
    t_bit_input_size := t_intput_size div 8;// +1;

    fillchar(t_bit_input[0][0], t_bit_input_size, 0);
    //int src_size = k * bit_align;

    // b - [bit_align, k] - [l.bit_align, l.size*l.size*l.c] = src_size
    // t_input - [bit_align, k] - [n', k]
    // t_bit_input - [new_ldb, n] - [k', n]

    //transpose_bin(t_input, *t_bit_input, k, n, bit_align, new_ldb, 8);
    transpose_bin(PUint32(b), PUInt32(t_bit_input^), k, n, bit_align, new_ldb, 8);
    exit(t_intput_size);


end;

procedure assisted_excitation_forward(var l: TConvolutionalLayer; const state: PNetworkState);
var
    iteration_num, b, w, h, c, t, left, right, top, bottom: longint;
    alpha: single;
    a_avg, g: TArray<single>;
    Ps :PSingle;
    truth: TBox;
begin
    iteration_num := state.net.seen[0] div (state.net.batch * state.net.subdivisions);
    alpha := (1+cos(3.141592 * iteration_num / state.net.max_batches));
    if l.assisted_excitation > 1 then
        begin
            if iteration_num > l.assisted_excitation then
                alpha := 0
            else
                alpha := (1+cos(3.141592 * iteration_num / l.assisted_excitation))
        end;
    setLength(a_avg, l.out_w * l.out_h * l.batch);
    setLength(g, l.out_w * l.out_h * l.batch);
    l.max_boxes := state.net.num_boxes;
    l.truths := l.max_boxes * (4+1);
    for b := 0 to l.batch -1 do
        begin
            for t := 0 to state.net.num_boxes -1 do
                begin
                    truth := float_to_box_stride(state.truth+t * (4+1)+b * l.truths, 1);
                    if truth.x=0 then
                        break;
                    left := floor((truth.x-truth.w / 2) * l.out_w);
                    right := ceil((truth.x+truth.w / 2) * l.out_w);
                    top := floor((truth.y-truth.h / 2) * l.out_h);
                    bottom := ceil((truth.y+truth.h / 2) * l.out_h);
                    for w := left to right do
                        for h := top to bottom -1 do
                            g[w+l.out_w * h+l.out_w * l.out_h * b] := 1
                end
        end;
    for b := 0 to l.batch -1 do
        for w := 0 to l.out_w -1 do
            for h := 0 to l.out_h -1 do
                begin
                    Ps := @a_avg[w+l.out_w * (h+l.out_h * b)];
                    for c := 0 to l.out_c -1 do
                        Ps^ := Ps^ + l.output[w+l.out_w * (h+l.out_h * (c+l.out_c * b))];
                    Ps^ := Ps^ / l.out_c
                end;
    for b := 0 to l.batch -1 do
        for w := 0 to l.out_w -1 do
            for h := 0 to l.out_h -1 do
                for c := 0 to l.out_c -1 do begin
                    Ps := @l.output[w+l.out_w * (h+l.out_h * (c+l.out_c * b))];
                    Ps^ := Ps^ + alpha * g[w+l.out_w * (h+l.out_h * b)] * a_avg[w+l.out_w * (h+l.out_h * b)];
                end
end;

procedure forward_convolutional_layer(var l: TConvolutionalLayer; const state: PNetworkState);
var
    out_h, out_w, i, j, m, k, n,  ldb_align, re_packed_input_size, new_k: longint;
    a, b, c: TSingles;
    new_ldb, t_intput_size, new_c, in_re_packed_input_size: size_t;
    im: TSingles;
    s: TNetworkState;

begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    out_h := convolutional_out_height(l);
    out_w := convolutional_out_width(l);
    fill_cpu(l.outputs * l.batch, 0, l.output, 1);

    if l.xnor and (not assigned(l.align_bit_weights) or state.train) then
        begin
            if not assigned(l.align_bit_weights) or state.train then
                binarize_weights(l.weights, l.n, l.nweights, l.binary_weights);
            swap_binary( l);
            binarize_cpu(state.input, l.c * l.h * l.w * l.batch, l.binary_input);
            state.input := l.binary_input
        end;
    m := l.n div l.groups;
    k := l.size * l.size * l.c div l.groups;
    n := out_h * out_w;
    //u := 0;
    //inc(u);
    for i := 0 to l.batch -1 do
        for j := 0 to l.groups -1 do
            begin
                a := l.weights+j * l.nweights div l.groups;
                b := state.workspace;
                c := l.output+(i * l.groups+j) * n * m;
                if l.xnor and assigned(l.align_bit_weights) and not state.train and (l.stride_x = l.stride_y) then
                    begin
                        filldword(b[0], l.bit_align * l.size * l.size * l.c ,0);
                        if l.c mod 32 = 0 then
                            begin
                                ldb_align := l.lda_align;
                                new_ldb := k+(ldb_align-k mod ldb_align);
                                re_packed_input_size := l.c * l.w * l.h;
                                filldword(state.workspace[0], re_packed_input_size,0);
                                new_c := l.c div 32;
                                in_re_packed_input_size := new_c * l.w * l.h+1;
                                filldword(l.bin_re_packed_input[0], in_re_packed_input_size ,0);
                                repack_input(state.input, state.workspace, l.w, l.h, l.c);
                                float_to_bit(state.workspace, PByte(l.bin_re_packed_input), l.c * l.w * l.h);
                                im2col_cpu_custom(PSingle(l.bin_re_packed_input), new_c, l.h, l.w, l.size, l.stride, l.pad, state.workspace);
                                new_k := l.size * l.size * l.c div 32;
                                transpose_uint32(Puint32(state.workspace), Puint32(l.t_bit_input), new_k, n, n, new_ldb);
                                gemm_nn_custom_bin_mean_transposed(m, n, k, 1, PByte(l.align_bit_weights), new_ldb, PByte(l.t_bit_input), new_ldb, c, n, l.mean_arr)
                            end
                        else
                            begin
                                im2col_cpu_custom_bin(state.input, l.c, l.h, l.w, l.size, l.stride, l.pad, state.workspace, l.bit_align);
                                ldb_align := l.lda_align;
                                new_ldb := k + (ldb_align - k mod ldb_align);
                                t_intput_size := binary_transpose_align_input(k, n, state.workspace, PPByte(&l.t_bit_input), ldb_align, l.bit_align);

                                // 5x times faster than gemm()-float32
                                gemm_nn_custom_bin_mean_transposed(m, n, k, 1, PByte(l.align_bit_weights), new_ldb, PByte(l.t_bit_input), new_ldb, c, n, l.mean_arr);

                            end;
                        add_bias(l.output, l.biases, l.batch, l.n, out_h * out_w);
                        case l.activation of
                           acSWISH :
                                activate_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.output);
                           acMISH :
                                activate_array_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
                           acHARD_MISH :
                                activate_array_hard_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
                           acNORM_CHAN :
                                activate_array_normalize_channels(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output);
                           acNORM_CHAN_SOFTMAX :
                                activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, false);
                           acNORM_CHAN_SOFTMAX_MAXVAL :
                                activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, true)
                        else
                            //activate_array_cpu_custom(l.output, m * n * l.batch, l.activation);
                            activate_array(l.output, m * n * l.batch, l.activation)
                        end;
                        {$ifdef USE_TELEMETRY}
                        if benchmark then metrics.forward.finish(l.&type);
                        {$endif}
                        exit()
                    end
                else
                    begin
                        im := state.input+(i * l.groups+j) * (l.c div l.groups) * l.h * l.w;
                        if (l.size = 1) and (l.stride = 1) and (l.dilation = 1) then
                            b := im
                        else
                            //im2col_cpu(im, l.c div l.groups, l.h, l.w, l.size, l.stride_x, l.pad, b);
                            im2col_cpu_ext(im, l.c div l.groups, l.h, l.w, l.size, l.size, l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x, l.dilation, l.dilation, b);
                        sgemm(0, 0, m, n, k, 1, a, k, b, n, 1, c, n);
                    end
            end;
    if l.batch_normalize then
        forward_batchnorm_layer(l, state)
    else
        add_bias(l.output, l.biases, l.batch, l.n, out_h * out_w);
    case l.activation of
       acSWISH :
          activate_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.output);
       acMISH :
          activate_array_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
       acHARD_MISH :
            activate_array_hard_mish(l.output, l.outputs * l.batch, l.activation_input, l.output);
       acNORM_CHAN :
            activate_array_normalize_channels(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output);
       acNORM_CHAN_SOFTMAX :
            activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, false);
       acNORM_CHAN_SOFTMAX_MAXVAL :
            activate_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.output, true);
       else
        //activate_array_cpu_custom(l.output, l.outputs * l.batch, l.activation);
            activate_array(l.output, l.outputs * l.batch, l.activation);
    end;
    if l.binary or l.xnor then
        swap_binary( l);
    if (l.assisted_excitation<>0) and state.train then
        assisted_excitation_forward(l, state);
    if l.antialiasing<>0 then
        begin
            s := default(TNetworkState);
            s.train := state.train;
            s.workspace := state.workspace;
            s.net := state.net;
            s.input := l.output;
            forward_convolutional_layer( l.input_layer[0], @s);
            move(l.input_layer[0].output[0], l.output[0], l.input_layer[0].outputs * l.input_layer[0].batch * sizeof(single))
        end;

    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;

procedure backward_convolutional_layer(var l: TConvolutionalLayer;
  const state: PNetworkState);
var
    i,j, m, n, k: longint;
    a, b, c, im: PSingle;
begin
    m := l.n div l.groups;
    n := l.size * l.size * l.c div l.groups;
    k := l.out_w * l.out_h;
    if (l.activation = acSWISH) then
        gradient_array_swish(l.output, l.outputs * l.batch, l.activation_input, l.delta)
    else
        if l.activation = acMISH then
            gradient_array_mish(l.outputs * l.batch, l.activation_input, l.delta)
    else
        if l.activation = acHARD_MISH then
            gradient_array_hard_mish(l.outputs * l.batch, l.activation_input, l.delta)
    else
        if (l.activation = acNORM_CHAN_SOFTMAX) or (l.activation = acNORM_CHAN_SOFTMAX_MAXVAL) then
            gradient_array_normalize_channels_softmax(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta)
    else
        if l.activation = acNORM_CHAN then
            gradient_array_normalize_channels(l.output, l.outputs * l.batch, l.batch, l.out_c, l.out_w * l.out_h, l.delta)
    else
        gradient_array(l.output, l.outputs * l.batch, l.activation, l.delta);
    if l.batch_normalize then
        backward_batchnorm_layer(l, state)
    else
        backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
    for i := 0 to l.batch -1 do
        for j := 0 to l.groups -1 do
            begin
                a := l.delta+(i * l.groups+j) * m * k;
                b := state.workspace;
                c := l.weight_updates+j * l.nweights div l.groups;
                im := state.input+(i * l.groups+j) * (l.c div l.groups) * l.h * l.w;
                im2col_cpu_ext(im, l.c div l.groups, l.h, l.w, l.size, l.size, l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x, l.dilation, l.dilation, b);
                sgemm(0, 1, m, n, k, 1, a, k, b, k, 1, c, n);
                if assigned(state.delta) then
                    begin
                        a := l.weights+j * l.nweights div l.groups;
                        b := l.delta+(i * l.groups+j) * m * k;
                        c := state.workspace;
                        sgemm(1, 0, n, k, m, 1, a, n, b, k, 0, c, k);
                        col2im_cpu_ext(state.workspace, l.c div l.groups, l.h, l.w, l.size, l.size, l.pad * l.dilation, l.pad * l.dilation, l.stride_y, l.stride_x, l.dilation, l.dilation, state.delta+(i * l.groups+j) * (l.c div l.groups) * l.h * l.w)
                    end
            end
end;

procedure update_convolutional_layer(const l: TConvolutionalLayer; const arg :TUpdateArgs);
var
    learning_rate: single;
begin
    learning_rate := arg.learning_rate * l.learning_rate_scale;
    axpy_cpu(l.nweights, -arg.decay * arg.batch, l.weights, 1, l.weight_updates, 1);
    axpy_cpu(l.nweights, learning_rate / arg.batch, l.weight_updates, 1, l.weights, 1);
    scal_cpu(l.nweights, arg.momentum, l.weight_updates, 1);
    axpy_cpu(l.n, learning_rate / arg.batch, l.bias_updates, 1, l.biases, 1);
    scal_cpu(l.n, arg.momentum, l.bias_updates, 1);
    if assigned(l.scales) then
        begin
            axpy_cpu(l.n, learning_rate / arg.batch, l.scale_updates, 1, l.scales, 1);
            scal_cpu(l.n, arg.momentum, l.scale_updates, 1)
        end
end;

function get_convolutional_weight(const l: TConvolutionalLayer; const i: longint):TImageData;
var
    h, w, c: longint;
begin
    h := l.size;
    w := l.size;
    c := l.c div l.groups;
    exit(float_to_image(w, h, c, l.weights+i * h * w * c))
end;

procedure rgbgr_weights(const l: TConvolutionalLayer);
var
    i: longint;
    im: TImageData;
begin
    for i := 0 to l.n -1 do
        begin
            im := get_convolutional_weight(l, i);
            if im.c = 3 then
                rgbgr_image(im)
        end
end;

procedure rescale_weights(const l: TConvolutionalLayer; const scale, trans: single);
var
    i: longint;
    im: TImageData;
    sum: single;
begin
    for i := 0 to l.n -1 do
        begin
            im := get_convolutional_weight(l, i);
            if im.c = 3 then
                begin
                    scale_image(im, scale);
                    sum := sum_array(@im.data[0], im.w * im.h * im.c);
                    l.biases[i] := l.biases[i] + (sum * trans)
                end
        end
end;

function get_weights(const l: TConvolutionalLayer):TArray<TImageData>;
var
    i: longint;
begin
    setLength(result, l.n);
    for i := 0 to l.n -1 do
        begin
            result[i] := copy_image(get_convolutional_weight(l, i));
            normalize_image(result[i])
        end;
end;

function visualize_convolutional_layer(const l: TConvolutionalLayer;
  const window: string; const prev_weights: TArray<TImageData>): TArray<TImageData>;
var
    delta, dc: TImageData;
    buff : string;
begin
    result := get_weights(l);
    show_images(result, l.n, window);
    delta := get_convolutional_image(l);
    dc := collapse_image_layers(delta, 1);
    buff := format('%s: Output', [window]);
    show_image(dc, buff,1);
    free_image(dc);
end;

end.

