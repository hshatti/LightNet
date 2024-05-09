unit parser;

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
  Classes, SysUtils, math, lightnet, utils, cfg, blas, data, tree
  , nnetwork
  , Activations
  , LocalLayer
  , DeConvolutionalLayer
  , ConvolutionalLayer
  , ConvLSTMLayer
  , CRNNLayer
  , GruLayer
  , RNNLayer
  , ConnectedLayer
  , LSTMLayer
  , SoftmaxLayer
  , GaussianYoloLayer
  , SAMLayer
  , YoloLayer
  , iSegLayer
  , DetectionLayer
  , CostLayer
  , CropLayer
  , MaxpoolLayer
  , AvgPoolLayer
  , DropoutLayer
  , NormalizationLayer
  , BatchNormLayer
  , ShortcutLayer
  , L2NormLayer
  , LogisticLayer
  , UpSampleLayer
  , RouteLayer
  , ActivationLayer
  , RegionLayer
  , RepresentationLayer
  , ReOrgLayer
  , ReOrgOldLayer
  , ScaleChannelLayer;

const
  MAJOR_VERSION = 0;
  MINOR_VERSION = 2;
  PATCH_VERSION = 5;

type
  TSizeParams = record
      batch, inputs, h, w, c, index, time_steps :longint;
      train:boolean;
      net : PNetwork;
  end;

function string_to_layer_type(const &type: string):TLayerType;
procedure parse_data(const data: string; const a: TSingles; const n: longint);
function parse_local(const options: TCFGSection; params: TSizeParams):TLocalLayer;
function parse_deconvolutional(const options: TCFGSection; const params: TSizeParams):TDeConvolutionalLayer;
function parse_convolutional(const options: TCFGSection; const params: TSizeParams):TConvolutionalLayer;
function parse_crnn(const options: TCFGSection; const params: TSizeParams):TCRNNLayer;
function parse_rnn(const options: TCFGSection; const params: TSizeParams):TRNNLayer;
function parse_gru(const options: TCFGSection; const params: TSizeParams):TGRULayer;
function parse_lstm(const options: TCFGSection; const params: TSizeParams):TLSTMLayer;
function parse_connected(const options: TCFGSection; const params: TSizeParams):TConnectedLayer;
function parse_softmax(const options: TCFGSection; const params: TSizeParams):TSoftmaxLayer;
function parse_yolo_mask( a: string; const num: TIntegers):TArray<longint>;
function parse_yolo(const options: TCFGSection; const params: TSizeParams):TYoloLayer;
function parse_iseg(const options: TCFGSection; const params: TSizeParams):TISEGlayer;
function parse_region(const options: TCFGSection; const params: TSizeParams):TRegionLayer;
function parse_detection(const options: TCFGSection; const params: TSizeParams):TDetectionLayer;
function parse_cost(const options: TCFGSection; const params: TSizeParams):TCostLayer;
function parse_crop(const options: TCFGSection; const params: TSizeParams):TCropLayer;
function parse_reorg(const options: TCFGSection; const params: TSizeParams):TReOrgLayer;
function parse_maxpool(const options: TCFGSection; const params: TSizeParams):TMaxpoolLayer;
function parse_avgpool(const options: TCFGSection; const params: TSizeParams):TAvgPoolLayer;
function parse_dropout(const options: TCFGSection; const params: TSizeParams):TDropoutLayer;
function parse_normalization(const options: TCFGSection; const params: TSizeParams):TNormalizationLayer;
function parse_batchnorm(const options: TCFGSection; const params: TSizeParams):TBatchNormLayer;
function parse_l2norm(const options: TCFGSection; const params: TSizeParams):TL2Norm;
function parse_logistic(const options: TCFGSection; const params: TSizeParams):TLogisticLayer;
function parse_activation(const options: TCFGSection; const params: TSizeParams):TActivationLayer;
function parse_upsample(const options: TCFGSection; const params: TSizeParams; const net: TNetwork):TUpSampleLayer;
function parse_route(const options: TCFGSection; const params: TSizeParams):TRouteLayer;
function get_policy(const s: string):TLearningRatePolicy;
procedure parse_net_options(const options: TCFGSection; const net: PNetwork);
function is_network(const s: TCFGSection):boolean;
function read_cfg(const filename: string):TCFGList;
function parse_network_cfg_custom(const filename: string; const batch, time_steps: longint):TNetwork;
function parse_network_cfg(const filename: string):TNetwork;
procedure save_convolutional_weights_binary(const l: TConvolutionalLayer; var fp: file);
procedure save_convolutional_weights(const l: TConvolutionalLayer; var fp: file);
procedure save_batchnorm_weights(const l: TBatchNormLayer; var fp: file);
procedure save_connected_weights(const l: TConnectedLayer; var fp: file);
procedure save_weights_upto(const net: TNetwork; const filename: string; cutoff: longint; const save_ema: longint);
procedure save_weights(const net: TNetwork; const filename: string);
procedure transpose_matrix(const a: TSingles; const rows, cols: longint);
procedure load_connected_weights(var l: TConnectedLayer; var fp: file ; const transpose: boolean);
procedure load_batchnorm_weights(var l: TBatchNormLayer; var fp: file);
procedure load_convolutional_weights_binary(var l: TConvolutionalLayer; var fp: file);
procedure load_convolutional_weights(var l: TConvolutionalLayer; var fp: file);
procedure load_weights_upto(const net: PNetwork; const filename: string; const cutoff: longint);
procedure load_weights(const net: PNetwork; const filename: string);
function load_network(const cfg, weights: string; const clear: boolean):TArray<TNetwork>;
function read_data_cfg(const filename: string): TCFGSection;
function get_metadata(const filename: string):TMetadata;

implementation

procedure empty_func (var l :TDropoutLayer; const state: PNetworkState);
begin
end;

function string_to_layer_type(const &type: string):TLayerType;
begin
    if &type = '[shortcut]'     then exit(ltSHORTCUT);
    if &type = '[scale_channels]' then exit(ltScaleChannels);
    if &type = '[sam]'          then exit(ltSAM);
    if &type = '[crop]'         then exit(ltCROP);
    if &type = '[cost]'         then exit(ltCOST);
    if &type = '[detection]'    then exit(ltDETECTION);
    if &type = '[region]'       then exit(ltREGION);
    if &type = '[yolo]'         then exit(ltYOLO);
    if &type = '[Gaussian_yolo]' then exit(ltGaussianYOLO);
    if &type = '[iseg]'         then exit(ltISEG);
    if &type = '[local]'        then exit(ltLOCAL);
    if (&type = '[conv]') or
              (&type = '[convolutional]') then exit(ltCONVOLUTIONAL);
    if &type = '[activation]'   then exit(ltACTIVE);
    if (&type = '[net]') or
              (&type = '[network]') then exit(ltNETWORK);
    if &type = '[crnn]'         then exit(ltCRNN);
    if &type = '[gru]'          then exit(ltGRU);
    if &type = '[lstm]'         then exit(ltLSTM);
    if &type = '[conv_lstm]'    then exit(ltConvLSTM);
    if &type = '[history]'      then exit(ltHISTORY);
    if &type = '[rnn]'          then exit(ltRNN);
    if (&type = '[conn]')       or (&type = '[connected]') then exit(ltCONNECTED);
    if (&type = '[max]')        or (&type = '[maxpool]') then exit(ltMAXPOOL);
    if (&type = '[local_avg]')  or (&type = '[local_avgpool]') then exit(ltLOCAL_AVGPOOL);
    if &type = '[reorg3d]'      then exit(ltREORG);
    if &type = '[reorg]'        then exit(ltREORG_OLD);
    if (&type = '[avg]')        or (&type = '[avgpool]') then exit(ltAVGPOOL);
    if &type = '[dropout]'      then exit(ltDROPOUT);
    if &type = '[logistic]'     then exit(ltLOGXENT);
    if (&type = '[lrn]') or (&type = '[normalization]') then exit(ltNORMALIZATION);
    if &type = '[batchnorm]'    then exit(ltBATCHNORM);
    if (&type = '[soft]') or (&type = '[softmax]') then exit(ltSOFTMAX);
    if &type = '[constrative]'  then exit(ltCONTRASTIVE);
    if &type = '[route]'        then exit(ltROUTE);
    if &type = '[upsample]'     then exit(ltUPSAMPLE);
    if (&type = '[empty]') or (&type = '[silence]') then exit(ltEMPTY);
    if &type = '[implicit]'       then exit(ltIMPLICIT);
    if &type = '[l2norm]'       then exit(ltL2NORM);
    if (&type = '[deconv]') or
              (&type = '[deconvolutional]') then exit(ltDECONVOLUTIONAL);
    exit(ltBLANK)
end;

procedure parse_data(const data: string; const a: TSingles; const n: longint);
var
   s:TStringArray;
   i,j:longint;
begin
  s:=data.Split([',']);
  if length(s)>n then
    j:=n
  else
    j:=length(s);
  for i:=0 to j-1 do
    a[i]:=StrToFloat(s[i]);

end;

function parse_local(const options: TCFGSection; params: TSizeParams): TLocalLayer;
var
    n, size, stride, pad, batch, h, w, c: longint;
    activation_s: string;
    activation: TActivation;
begin
    n := options.getInt('filters', 1);
    size := options.getInt( 'size', 1);
    stride := options.getInt( 'stride', 1);
    pad := options.getInt( 'pad', 0);
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if (h * w * c)=0 then
        raise Exception.Create('Layer before local layer must output image.');
    result := make_local_layer(batch, h, w, c, n, size, stride, pad, activation);

end;

function parse_deconvolutional(const options: TCFGSection; const params: TSizeParams
  ): TDeConvolutionalLayer;
var
    n, size, stride, batch, h, w, c, padding: longint;
    batch_normalize, pad :boolean;
    activation_s: string;
    activation: TActivation;
begin
    n := options.getInt( 'filters', 1);
    size := options.getInt( 'size', 1);
    stride := options.getInt( 'stride', 1);
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if (h * w * c)=0 then
        raise Exception.Create('Layer before deconvolutional layer must output image.');
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    pad := options.getBool( 'pad', false, true);
    padding := options.getInt('padding', 0, true);
    if pad then
        padding := size div 2;
    result := make_deconvolutional_layer(batch, h, w, c, n, size, stride, {padding, }activation{, batch_normalize, params.net.adam});
    //exit(l)
end;

function parse_convolutional(const options: TCFGSection; const params: TSizeParams
  ): TConvolutionalLayer;
var
    n: longint;
    groups: longint;
    size: longint;
    stride: longint;
    stride_x: longint;
    stride_y: longint;
    dilation: longint;
    antialiasing: longint;
    pad: boolean;
    padding: longint;
    activation_s: string;
    activation: TActivation;
    assisted_excitation: longint;
    share_index: longint;
    share_layer: PConvolutionalLayer;
    batch: longint;
    h: longint;
    w: longint;
    c: longint;
    batch_normalize: longint;
    cbn: boolean;
    binary: boolean;
    xnor: boolean;
    use_bin_output: boolean;
    sway: longint;
    rotate: longint;
    stretch: longint;
    stretch_sway: longint;
    deform: boolean;
    layer: TConvolutionalLayer;
begin
    n := options.getInt( 'filters', 1);
    groups := options.getInt('groups', 1, true);
    size := options.getInt( 'size', 1);
    stride := -1;
    stride_x := options.getInt('stride_x', -1, true);
    stride_y := options.getInt('stride_y', -1, true);
    if (stride_x < 1) or (stride_y < 1) then
        begin
            stride := options.getInt( 'stride', 1);
            if stride_x < 1 then
                stride_x := stride;
            if stride_y < 1 then
                stride_y := stride
        end
    else
        stride := options.getInt('stride', 1, true);
    dilation := options.getInt('dilation', 1, true);
    antialiasing := options.getInt('antialiasing', 0, true);
    if size = 1 then
        dilation := 1;
    pad := options.getBool( 'pad', false, true);
    padding := options.getInt('padding', 0, true);
    if pad then
        padding := size div 2;
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    assisted_excitation := options.getInt('assisted_excitation', 0, true);
    share_index := options.getInt('share_index', -1000000000, true);
    share_layer := nil;
    if share_index >= 0 then
        share_layer :=  @params.net.layers[share_index]
    else
        if share_index <> -1000000000 then
            share_layer := @params.net.layers[params.index+share_index];
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if not ((h<>0) and (w<>0) and (c<>0)) then
        raise Exception.Create('Layer before convolutional layer must output image.');
        //error('Layer before convolutional layer must output image.', DARKNET_LOC);
    batch_normalize := options.getInt('batch_normalize', 0, true);
    cbn := options.getBool( 'cbn', false, true);
    if cbn then
        batch_normalize := 2;
    binary := options.getBool( 'binary', false, true);
    xnor := options.getBool( 'xnor', false, true);
    use_bin_output := options.getBool( 'bin_output', false, true);
    sway := options.getInt('sway', 0, true);
    rotate := options.getInt('rotate', 0, true);
    stretch := options.getInt('stretch', 0, true);
    stretch_sway := options.getInt('stretch_sway', 0, true);
    if (sway+rotate+stretch+stretch_sway) > 1 then
        raise Exception.Create('Error: should be used only 1 param: sway=1, rotate=1 or stretch=1 in the [convolutional] layer');
    deform := (sway<>0) or (rotate<>0) or (stretch<>0) or (stretch_sway<>0);
    if deform and (size = 1) then
        raise Exception.Create('Error: params (sway=1, rotate=1 or stretch=1) should be used only with size >=3 in the [convolutional] layer');
    layer := make_convolutional_layer(batch, 1, h, w, c, n, groups, size, stride_x, stride_y, dilation, padding, activation, batch_normalize in [1,2], binary, xnor, params.net.adam, use_bin_output, params.index, antialiasing, share_layer, assisted_excitation, deform, params.train);
    layer.flipped := options.getBool( 'flipped', false, true);
    layer.dot := options.getFloat( 'dot', 0, true);
    layer.sway := sway;
    layer.rotate := rotate;
    layer.stretch := stretch;
    layer.stretch_sway := stretch_sway;
    layer.angle := options.getFloat( 'angle', 15, true);
    layer.grad_centr := options.getInt('grad_centr', 0, true);
    layer.reverse := options.getBool( 'reverse', false, true);
    layer.coordconv := options.getInt('coordconv', 0, true);
    layer.stream := options.getInt('stream', -1, true);
    layer.wait_stream_id := options.getInt('wait_stream', -1, true);
    if params.net.adam then
        begin
            layer.B1 := params.net.B1;
            layer.B2 := params.net.B2;
            layer.eps := params.net.eps
        end;
    exit(layer)
end;

function parse_crnn(const options: TCFGSection; const params: TSizeParams
  ): TCRNNLayer;
var
    size: longint;
    stride: longint;
    dilation: longint;
    pad: boolean;
    padding: longint;
    output_filters: longint;
    hidden_filters: longint;
    groups: longint;
    activation_s:string;
    activation: TActivation;
    batch_normalize: boolean;
    xnor: boolean;
    l: TCRNNLayer;
begin
    size := options.getInt('size', 3, true);
    stride := options.getInt('stride', 1, true);
    dilation := options.getInt('dilation', 1, true);
    pad := options.getBool( 'pad', false, true);
    padding := options.getInt('padding', 0, true);
    if pad then
        padding := size div 2;
    output_filters := options.getInt( 'output', 1);
    hidden_filters := options.getInt( 'hidden', 1);
    groups := options.getInt('groups', 1, true);
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    xnor := options.getBool( 'xnor', false, true);
    l := make_crnn_layer(params.batch, params.h, params.w, params.c, hidden_filters, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, xnor, params.train);
    l.shortcut := options.getBool( 'shortcut', false, true);
    exit(l)
end;

function parse_rnn(const options: TCFGSection; const params: TSizeParams): TRNNLayer;
var
    output, hidden, logistic: longint;
    batch_normalize:boolean;
    activation_s: string;
    activation: TActivation;
    //l: layer;
begin
    output := options.getInt( 'output', 1);
    hidden := options.getInt( 'hidden',1);
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    logistic := options.getInt('logistic', 0, true);
    result := make_rnn_layer(params.batch, params.inputs, hidden, output, params.time_steps, activation, batch_normalize, logistic);
    result.shortcut := options.getBool( 'shortcut', false, true);
    //exit(l)
end;

function parse_gru(const options: TCFGSection; const params: TSizeParams): TGRULayer;
var
    output: longint;
    batch_normalize: boolean;
    //l: Tlayer;
begin
    output := options.getInt( 'output', 1);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    result := make_gru_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize{, params.net.adam});
    result.tanh := options.getBool( 'tanh', false, true);
    //exit(l)
end;

function parse_lstm(const options: TCFGSection; const params: TSizeParams
  ): TLSTMLayer;
var
    output: longint;
    batch_normalize: boolean;
    //l: layer;
begin
    output := options.getInt( 'output', 1);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    result := make_lstm_layer(params.batch, params.inputs, output, params.time_steps, batch_normalize{, params.net.adam});
    //exit(l)
end;

function parse_conv_lstm(const options: TCFGSection; const params: TSizeParams):TConvLSTMLayer;
var
    size: longint;
    stride: longint;
    dilation: longint;
    pad: boolean;
    padding: longint;
    output_filters: longint;
    groups: longint;
    activation_s: string;
    activation: TActivation;
    batch_normalize: boolean;
    xnor: boolean;
    peephole: boolean;
    bottleneck: boolean;
    l: TConvLSTMLayer;
    lstm_activation_s: string;
begin
    size := options.getInt('size', 3, true);
    stride := options.getInt('stride', 1, true);
    dilation := options.getInt('dilation', 1, true);
    pad := options.getBool( 'pad', false, true);
    padding := options.getInt('padding', 0, true);
    if pad then
        padding := size div 2;
    output_filters := options.getInt( 'output', 1);
    groups := options.getInt('groups', 1, true);
    activation_s := options.getStr( 'activation', 'linear');
    activation := get_activation(activation_s);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    xnor := options.getBool( 'xnor', false, true);
    peephole := options.getBool( 'peephole', false, true);
    bottleneck := options.getBool( 'bottleneck', false, true);
    l := make_conv_lstm_layer(params.batch, params.h, params.w, params.c, output_filters, groups, params.time_steps, size, stride, dilation, padding, activation, batch_normalize, peephole, xnor, bottleneck, params.train);
    l.state_constrain := options.getInt('state_constrain', params.time_steps * 32, true);
    l.shortcut := options.getBool( 'shortcut', false, true);
    lstm_activation_s := options.getStr( 'lstm_activation', 'tanh');
    l.lstmActivation := get_activation(lstm_activation_s);
    l.time_normalizer := options.getFloat( 'time_normalizer', 1.0, true);
    exit(l)
end;

function parse_history(const options: TCFGSection; const params: TSizeParams):TLayer;
var
    history_size: longint;
    l: TLayer;
begin
    history_size := options.getInt( 'history_size', 4);
    l := make_history_layer(params.batch, params.h, params.w, params.c, history_size, params.time_steps, params.train);
    exit(l)
end;

function parse_connected(const options: TCFGSection; const params: TSizeParams
  ): TConnectedLayer;
var
    output: longint;
    activation_s: string;
    activation: TActivation;
    batch_normalize: boolean;
    //l: layer;
begin
    output := options.getInt( 'output', 1);
    activation_s := options.getStr( 'activation', 'logistic');
    activation := get_activation(activation_s);
    batch_normalize := options.getBool( 'batch_normalize', false, true);
    result := make_connected_layer(params.batch, 1, params.inputs, output, activation, batch_normalize{, params.net.adam});
    //exit(l)
end;

function parse_softmax(const options: TCFGSection; const params: TSizeParams
  ): TSoftmaxLayer;
var
    groups: longint;
    //result: layer;
    tree_file: string;
begin
    groups := options.getInt('groups', 1, true);
    result := make_softmax_layer(params.batch, params.inputs, groups);
    result.temperature := options.getFloat( 'temperature', 1, true);
    tree_file := options.getStr( 'tree', '');
    if tree_file<>'' then
        result.softmax_tree := [read_tree(tree_file)];
    result.w := params.w;
    result.h := params.h;
    result.c := params.c;
    result.spatial := trunc(options.getFloat( 'spatial', 0, true));
    result.noloss := options.getBool( 'noloss', false, true);
    //exit(result)
end;

function parse_contrastive(const options: TCFGSection; const params: TSizeParams):TContrastiveLayer;
var
    classes: longint;
    yolo_layer: PYoloLayer;
    yolo_layer_id: longint;
    layer: TContrastiveLayer;
begin
    classes := options.getInt( 'classes', 1000);
    yolo_layer := nil;
    yolo_layer_id := options.getInt('yolo_layer', 0, true);
    if yolo_layer_id < 0 then
        yolo_layer_id := params.index+yolo_layer_id;
    if yolo_layer_id <> 0 then
        yolo_layer := @params.net.layers[yolo_layer_id];
    if yolo_layer.&type <> ltYOLO then
        begin
            writeln(format(' Error: [contrastive] layer should point to the [yolo] layer instead of %d layer! ', [yolo_layer_id]));
            Exception.Create('Error!')
        end;
    layer := make_contrastive_layer(params.batch, params.w, params.h, params.c, classes, params.inputs, Pointer(yolo_layer));
    layer.temperature := options.getFloat( 'temperature', 1, true);
    layer.steps := params.time_steps;
    layer.cls_normalizer := options.getFloat( 'cls_normalizer', 1, true);
    layer.max_delta := options.getFloat( 'max_delta', MaxSingle, true);
    layer.contrastive_neg_max := options.getInt('contrastive_neg_max', 3, true);
    exit(layer)
end;

function parse_yolo_mask( a: string; const num: TIntegers):TArray<longint>;
var
    //mask: TIntegers;
    i: longint;
    vals:TStringArray;
begin
    result := nil;
    if a<>'' then
        begin
            i:=pos('#',a);
            if i>0 then
                a:=copy(a,1, i-1) ;
            vals := a.Split([',']);

            //len := length(a);
            //n := 1;
            //for i := 0 to len -1 do
            //    if a[i] = ',' then
            //        inc(n);
            //result := TIntegers.Create(length(vals));//, sizeof(int));
            setLength(result, length(vals));
            for i := 0 to length(vals) -1 do
              TryStrToInt(trim(vals[i]), result[i]);
            //    begin
            //        val := atoi(a);
            //        result[i] := val;
            //        a := strchr(a, ',')+1
            //    end;
            num[0] := length(vals)
        end;
    // test parse_yolo_mask
    //exit(mask)
end;

function get_classes_multipliers(const cpc: string; const classes: longint; const max_delta: single):TArray<Single>;
var
    classes_multipliers: TArray<Single>;
    classes_counters: longint;
    counters_per_class: TArray<Longint>;
    max_counter: single;
    i: longint;
begin
    classes_multipliers := nil;
    if cpc <> '' then
        begin
            classes_counters := classes;
            counters_per_class := parse_yolo_mask(cpc,  @classes_counters);
            if classes_counters <> classes then
                begin
                    writeln(format(' number of values in counters_per_class = %d doesn''t match with classes = %d ', [classes_counters, classes]));
                    raise Exception.Create('Error!')
                end;
            max_counter := 0;
            for i := 0 to classes_counters -1 do
                begin
                    if counters_per_class[i] < 1 then
                        counters_per_class[i] := 1;
                    if max_counter < counters_per_class[i] then
                        max_counter := counters_per_class[i]
                end;
            setLength(classes_multipliers, classes_counters);
            for i := 0 to classes_counters -1 do
                begin
                    classes_multipliers[i] := max_counter / counters_per_class[i];
                    if classes_multipliers[i] > max_delta then
                        classes_multipliers[i] := max_delta
                end;
            //free(counters_per_class);
            write(' classes_multipliers: ');
            for i := 0 to classes_counters -1 do
                write(format('%.1f, ', [classes_multipliers[i]]));
            writeln('')
        end;
    exit(classes_multipliers)
end;


function parse_yolo(const options: TCFGSection; const params: TSizeParams
  ): TYoloLayer;
var
    classes: longint;
    total: longint;
    num: longint;
    a: string;
    mask: TArray<longint>;
    max_boxes: longint;
    l: TYoloLayer;
    cpc: string;
    iou_loss: string;
    iou_thresh_kind_str: string;
    nms_kind: string;
    embedding_layer_id: longint;
    le: TLayer;
    map_file: string;
    i: longint;
    vals:TArray<string>;
begin
    classes := options.getInt( 'classes', 20);
    total := options.getInt( 'num', 1);
    num := total;
    a := options.getStr( 'mask', '');
    mask := parse_yolo_mask(a, @num);
    max_boxes := options.getInt('max', 200, true);
    l := make_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if l.outputs <> params.inputs then
        raise Exception.Create('Error: l.outputs == params.inputs, filters= in the [convolutional]-layer doesn''t correspond to classes= or mask= in [yolo]-layer');
    l.show_details := options.getInt('show_details', 1, true);
    l.max_delta := options.getFloat( 'max_delta', MaxSingle, true);
    cpc := options.getStr( 'counters_per_class', '');
    l.classes_multipliers := get_classes_multipliers(cpc, classes, l.max_delta);
    l.label_smooth_eps := options.getFloat( 'label_smooth_eps', 0.0, true);
    l.scale_x_y := options.getFloat( 'scale_x_y', 1, true);
    l.objectness_smooth := options.getBool( 'objectness_smooth', false, true);
    l.new_coords := options.getBool( 'new_coords', false, true);
    l.iou_normalizer := options.getFloat( 'iou_normalizer', 0.75, true);
    l.obj_normalizer := options.getFloat( 'obj_normalizer', 1, true);
    l.cls_normalizer := options.getFloat( 'cls_normalizer', 1, true);
    l.delta_normalizer := options.getFloat( 'delta_normalizer', 1, true);
    iou_loss := options.getStr( 'iou_loss', 'mse', true);
    if iou_loss= 'mse' then
        l.iou_loss := ilMSE
    else
        if iou_loss= 'giou' then
            l.iou_loss := ilGIOU
    else
        if iou_loss= 'diou' then
            l.iou_loss := ilDIOU
    else
        if iou_loss= 'ciou' then
            l.iou_loss := ilCIOU
    else
        l.iou_loss := ilIOU;
    writeln(ErrOutput, format('[yolo] params: iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale_x_y: %2.2f', [iou_loss, ord(l.iou_loss), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y]));
    iou_thresh_kind_str := options.getStr( 'iou_thresh_kind', 'iou', true);
    if iou_thresh_kind_str= 'iou' then
        l.iou_thresh_kind :=ilIOU
    else
        if iou_thresh_kind_str= 'giou' then
            l.iou_thresh_kind := ilGIOU
    else
        if iou_thresh_kind_str= 'diou' then
            l.iou_thresh_kind := ilDIOU
    else
        if iou_thresh_kind_str= 'ciou' then
            l.iou_thresh_kind := ilCIOU
    else
        begin
            writeln(ErrOutput, format(' Wrong iou_thresh_kind = %s ', [iou_thresh_kind_str]));
            l.iou_thresh_kind := ilIOU
        end;
    l.beta_nms := options.getFloat( 'beta_nms', 0.6, true);
    nms_kind := options.getStr( 'nms_kind', 'default', true);
    if nms_kind = 'default' then
        l.nms_kind := nmsDEFAULT_NMS
    else
        begin
            if nms_kind = 'greedynms' then
                l.nms_kind := nmsGREEDY_NMS
            else
                if nms_kind = 'diounms' then
                    l.nms_kind := nmsDIOU_NMS
            else
                l.nms_kind := nmsDEFAULT_NMS;
            writeln(format('nms_kind: %s (%d), beta = %f ', [nms_kind, ord(l.nms_kind), l.beta_nms]))
        end;
    l.jitter := options.getFloat( 'jitter', 0.2);
    l.resize := options.getFloat( 'resize', 1.0, true);
    l.focal_loss := options.getBool( 'focal_loss', false, true);
    l.ignore_thresh := options.getFloat( 'ignore_thresh', 0.5);
    l.truth_thresh := options.getFloat( 'truth_thresh', 1);
    l.iou_thresh := options.getFloat( 'iou_thresh', 1, true);
    l.random := options.getBool( 'random', false, true);
    l.track_history_size := options.getInt('track_history_size', 5, true);
    l.sim_thresh := options.getFloat( 'sim_thresh', 0.8, true);
    l.dets_for_track := options.getInt('dets_for_track', 1, true);
    l.dets_for_show := options.getInt('dets_for_show', 1, true);
    l.track_ciou_norm := options.getFloat( 'track_ciou_norm', 0.01, true);
    embedding_layer_id := options.getInt('embedding_layer', 999999, true);
    if embedding_layer_id < 0 then
        embedding_layer_id := params.index+embedding_layer_id;
    if embedding_layer_id <> 999999 then
        begin
            write(format(' embedding_layer_id = %d, ', [embedding_layer_id]));
            le := params.net.layers[embedding_layer_id];
            l.embedding_layer_id := embedding_layer_id;
            l.embedding_output := TSingles.Create(le.batch * le.outputs);
            l.embedding_size := le.n div l.n;
            writeln(format(' embedding_size = %d ', [l.embedding_size]));
            if le.n mod l.n <> 0 then
                writeln(format(' Warning: filters=%d number in embedding_layer=%d isn''t divisable by number of anchors %d ', [le.n, embedding_layer_id, l.n]))
        end;
    map_file := options.getStr( 'map', '');
    if map_file<>'' then
        l.map := read_map(map_file);
    a := options.getStr( 'anchors', '');
    if a<>'' then
        begin
            i:=pos('#',a);
            if i>0 then
                a:= copy(a,1,i-1);
            vals := a.Split([',']);
           for i:= 0 to lightnet.min(length(vals), total * 2)- 1 do
                TryStrToFloat(vals[i],l.biases[i]);
        end;
    exit(l)
end;

function parse_gaussian_yolo_mask(a: string; const num: PLongint):TArray<longint>;
var
    vals :TArray<string>;
    i:longint;
begin
    result := nil;
    if a<>'' then
        begin
            i:=pos('#',a);
            if i>0 then
                a:=copy(a,1, i-1);
            vals := a.Split([',']);

            //len := length(a);
            //n := 1;
            //for i := 0 to len -1 do
            //    if a[i] = ',' then
            //        inc(n);
            //result := TIntegers.Create(length(vals));//, sizeof(int));
            setLength(result, length(vals));
            for i := 0 to length(vals) -1 do
              TryStrToInt(trim(vals[i]), result[i]);
            //    begin
            //        val := atoi(a);
            //        result[i] := val;
            //        a := strchr(a, ',')+1
            //    end;
            num[0] := length(vals)
        end;
end;

function parse_gaussian_yolo(options: TCFGSection; params: TSizeParams):TGaussianYoloLayer;
var
    classes: longint;
    max_boxes: longint;
    total: longint;
    num: longint;
    a: string;
    mask: TArray<longint>;
    l: TGaussianYoloLayer;
    cpc: string;
    iou_loss: string;
    iou_thresh_kind_str: string;
    nms_kind: string;
    yolo_point: string;
    map_file: string;
    len: longint;
    n: longint;
    i: longint;
    vals: TArray<string>;
begin
    classes := options.getInt( 'classes', 20);
    max_boxes := options.getInt('max', 200, true);
    total := options.getInt( 'num', 1);
    num := total;
    a := options.getStr( 'mask', '');
    mask := parse_gaussian_yolo_mask(a, @num);
    l := make_gaussian_yolo_layer(params.batch, params.w, params.h, num, total, mask, classes, max_boxes);
    if l.outputs <> params.inputs then
        raise Exception.Create('Error: l.outputs == params.inputs, filters= in the [convolutional]-layer doesn''t correspond to classes= or mask= in [Gaussian_yolo]-layer');
    l.max_delta := options.getFloat( 'max_delta', MaxSingle, true);
    cpc := options.getStr( 'counters_per_class', '');
    l.classes_multipliers := get_classes_multipliers(cpc, classes, l.max_delta);
    l.label_smooth_eps := options.getFloat( 'label_smooth_eps', 0.0, true);
    l.scale_x_y := options.getFloat( 'scale_x_y', 1, true);
    l.objectness_smooth := options.getBool( 'objectness_smooth', false, true);
    l.uc_normalizer := options.getFloat( 'uc_normalizer', 1.0, true);
    l.iou_normalizer := options.getFloat( 'iou_normalizer', 0.75, true);
    l.obj_normalizer := options.getFloat( 'obj_normalizer', 1.0, true);
    l.cls_normalizer := options.getFloat( 'cls_normalizer', 1, true);
    l.delta_normalizer := options.getFloat( 'delta_normalizer', 1, true);
    iou_loss := options.getStr( 'iou_loss', 'mse', true);
    if iou_loss = 'mse' then
        l.iou_loss := ilMSE
    else
        if iou_loss = 'giou' then
            l.iou_loss := ilGIOU
    else
        if iou_loss = 'diou' then
            l.iou_loss := ilDIOU
    else
        if iou_loss = 'ciou' then
            l.iou_loss := ilCIOU
    else
        l.iou_loss := ilIOU;
    iou_thresh_kind_str := options.getStr( 'iou_thresh_kind', 'iou', true);
    if iou_thresh_kind_str = 'iou' then
        l.iou_thresh_kind := ilIOU
    else
        if iou_thresh_kind_str = 'giou' then
            l.iou_thresh_kind := ilGIOU
    else
        if iou_thresh_kind_str = 'diou' then
            l.iou_thresh_kind := ilDIOU
    else
        if iou_thresh_kind_str = 'ciou' then
            l.iou_thresh_kind := ilCIOU
    else
        begin
            writeln(ErrOutput, format(' Wrong iou_thresh_kind = %s ', [iou_thresh_kind_str]));
            l.iou_thresh_kind := ilIOU
        end;
    l.beta_nms := options.getFloat( 'beta_nms', 0.6, true);
    nms_kind := options.getStr( 'nms_kind', 'default', true);
    if nms_kind = 'default' then
        l.nms_kind := nmsDEFAULT_NMS
    else
        begin
            if nms_kind = 'greedynms' then
                l.nms_kind := nmsGREEDY_NMS
            else
                if nms_kind = 'diounms' then
                    l.nms_kind := nmsDIOU_NMS
            else
                if nms_kind = 'cornersnms' then
                    l.nms_kind := nmsCORNERS_NMS
            else
                l.nms_kind := nmsDEFAULT_NMS;
            writeln(format('nms_kind: %s (%d), beta = %f ', [nms_kind, ord(l.nms_kind), l.beta_nms]))
        end;
    yolo_point := options.getStr( 'yolo_point', 'center', true);
    if yolo_point = 'left_top' then
        l.yolo_point := ypYOLO_LEFT_TOP
    else
        if yolo_point = 'right_bottom' then
            l.yolo_point := ypYOLO_RIGHT_BOTTOM
    else
        l.yolo_point := ypYOLO_CENTER;
    writeln(ErrOutput, format('[Gaussian_yolo] iou loss: %s (%d), iou_norm: %2.2f, obj_norm: %2.2f, cls_norm: %2.2f, delta_norm: %2.2f, scale: %2.2f, point: %d', [iou_loss, ord(l.iou_loss), l.iou_normalizer, l.obj_normalizer, l.cls_normalizer, l.delta_normalizer, l.scale_x_y, ord(l.yolo_point)]));
    l.jitter := options.getFloat( 'jitter', 0.2);
    l.resize := options.getFloat( 'resize', 1.0, true);
    l.ignore_thresh := options.getFloat( 'ignore_thresh', 0.5);
    l.truth_thresh := options.getFloat( 'truth_thresh', 1);
    l.iou_thresh := options.getFloat( 'iou_thresh', 1, true);
    l.random := options.getBool( 'random', false, true);
    map_file := options.getStr( 'map', '');
    if map_file<>'' then
        l.map := read_map(map_file);
    a := options.getStr( 'anchors', '');
    if a<>'' then
        begin
           i:=pos('#',a);
           if i>0 then
               a:= copy(a,1,i-1);
           vals := a.Split([',']);
           for i:= 0 to length(vals)- 1 do
                TryStrToFloat(vals[i],l.biases[i]);
        end;
    exit(l)
end;

function parse_iseg(const options: TCFGSection; const params: TSizeParams
  ): TISEGlayer;
var
    classes: longint;
    ids: longint;
    //l: layer;
begin
    classes := options.getInt( 'classes', 20);
    ids := options.getInt( 'ids', 32);
    result := make_iseg_layer(params.batch, params.w, params.h, classes, ids);
    assert(result.outputs = params.inputs);
    //exit(l)
end;

function parse_region(const options: TCFGSection; const params: TSizeParams
  ): TRegionLayer;
var
    coords, classes, num, n, i: longint;
    //l: layer;
    tree_file: string;
    map_file: string;
    a: string;
    bias: TStringArray;
    max_boxes: longint;
begin
    coords := options.getInt( 'coords', 4);
    classes := options.getInt( 'classes', 20);
    num := options.getInt( 'num', 1);
    max_boxes := options.getInt('max', 200, true);

    result := make_region_layer(params.batch, params.w, params.h, num, classes, coords, max_boxes);
    if result.outputs <> params.inputs then
        raise Exception.create('Error: l.outputs == params.inputs, filters= in the [convolutional]-layer doesn''t correspond to classes= or num= in [region]-layer');
    assert(result.outputs = params.inputs);

    result.log := options.getInt('log', 0, true);
    result.sqrt := options.getBool( 'sqrt', false, true);

    result.softmax := options.getBool( 'softmax', false);
    result.focal_loss := options.getBool( 'focal_loss', false, true);
    //result.background := options.getInt('background', false, true);
    //result.max_boxes := options.getInt('max', 30, true);
    result.jitter := options.getFloat( 'jitter', 0.2);
    result.rescore := options.getBool( 'rescore', false, true);

    result.thresh := options.getFloat( 'thresh', 0.5);
    result.classfix := options.getInt('classfix', 0, true);
    result.absolute := options.getInt('absolute', 0, true);
    result.random := options.getBool( 'random', false, true);

    result.coord_scale := options.getFloat( 'coord_scale', 1);
    result.object_scale := options.getFloat( 'object_scale', 1);
    result.noobject_scale := options.getFloat( 'noobject_scale', 1);
    result.mask_scale := options.getFloat( 'mask_scale', 1);
    result.class_scale := options.getFloat( 'class_scale', 1);
    result.bias_match := options.getBool( 'bias_match', false, true);

    tree_file := options.getStr( 'tree', '');
    if tree_file<>'' then
        result.softmax_tree := [read_tree(tree_file)];
    map_file := options.getStr( 'map', '');
    if map_file<>'' then
        result.map := read_map(map_file);
    a := options.getStr( 'anchors', '');
    if a<>'' then
        begin
          bias := a.Split([',']);
          for i:=0 to length(a)-1 do
            trystrToFloat(trim(bias[i]),result.biases[i]);
            //len := strlen(a);
            //n := 1;
            //for i := 0 to len -1 do
            //    if a[i] = ',' then
            //        &cni(n);
            //for i := 0 to n -1 do
            //    begin
            //        bias := atof(a);
            //        result.biases[i] := bias;
            //        a := strchr(a, ',')+1
            //    end
        end;
    //exit(l)
end;

function parse_detection(const options: TCFGSection; const params: TSizeParams
  ): TDetectionLayer;
var
    coords, classes, num, side: longint;
    rescore: boolean;
    //layer: detection_layer;
begin
    coords := options.getInt( 'coords', 1);
    classes := options.getInt( 'classes', 1);
    rescore := options.getBool( 'rescore', false);
    num := options.getInt( 'num', 1);
    side := options.getInt( 'side', 7);
    result := make_detection_layer(params.batch, params.inputs, num, side, classes, coords, rescore);

    result.softmax := options.getBool( 'softmax', false);
    result.sqrt := options.getBool( 'sqrt', false);

    result.max_boxes := options.getInt('max', 200, true);
    result.coord_scale := options.getFloat( 'coord_scale', 1);
    result.forced := options.getBool( 'forced', false);
    result.object_scale := options.getFloat( 'object_scale', 1);
    result.noobject_scale := options.getFloat( 'noobject_scale', 1);
    result.class_scale := options.getFloat( 'class_scale', 1);
    result.jitter := options.getFloat( 'jitter', 0.2);
    result.resize := options.getFloat( 'resize', 1.0, true);
    result.random := options.getBool( 'random', false, true);
    result.reorg := options.getInt('reorg', 0, true);
    //exit(layer)
end;

function parse_cost(const options: TCFGSection; const params: TSizeParams
  ): TCostLayer;
var
    type_s: string;
    &type: TCostType;
    scale: single;
    //layer: cost_layer;
begin
    type_s := options.getStr( 'type', 'sse');
    &type := get_cost_type(type_s);
    scale := options.getFloat( 'scale', 1, true);
    result := make_cost_layer(params.batch, params.inputs, &type, scale);
    result.ratio := options.getFloat( 'ratio', 0, true);
    //result.noobject_scale := options.getFloat( 'noobj', 1, true);
    //result.thresh := options.getFloat( 'thresh', 0, true);
    //exit(layer)
end;

function parse_crop(const options: TCFGSection; const params: TSizeParams
  ): TCropLayer;
var
    crop_height, crop_width, batch, h, w, c: longint;
    flip, noadjust: boolean;
    angle, saturation, exposure: single;
    //l: crop_layer;
begin
    crop_height := options.getInt( 'crop_height', 1);
    crop_width := options.getInt( 'crop_width', 1);
    flip := options.getBool( 'flip', false);
    angle := options.getFloat( 'angle', 0);
    saturation := options.getFloat( 'saturation', 1);
    exposure := options.getFloat( 'exposure', 1);

    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if (h * w * c)=0 then
        raise Exception.Create('Layer before crop layer must output image.');
    noadjust := options.getBool( 'noadjust', false, true);
    result := make_crop_layer(batch, h, w, c, crop_height, crop_width, flip, angle, saturation, exposure);
    result.shift := options.getFloat( 'shift', 0);
    result.noadjust := noadjust;
    //exit(result)
end;

function parse_reorg(const options: TCFGSection; const params: TSizeParams
  ): TReOrgLayer;
var
    stride, extra, batch, h, w, c: longint;
    reverse, flatten : boolean;
    //layer: layer;
begin
    stride := options.getInt( 'stride', 1);
    reverse := options.getBool( 'reverse', false, true);

    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;

    if (h * w * c)=0 then
        raise Exception.Create('Layer before reorg layer must output image.');

    result := make_reorg_layer(batch, w, h, c, stride, reverse{, flatten, extra});
    //exit(layer)
end;

function parse_reorg_old(const options: TCFGSection; const params: TSizeParams):TReOrgLayer;
var
    stride, extra, batch, h, w, c: longint;
    reverse, flatten : boolean;
    //layer: layer;
begin
    writeln(#10' reorg_old ');
    stride := options.getInt( 'stride', 1);
    reverse := options.getBool( 'reverse', false, true);
    //flatten := options.getBool( 'flatten', false, true);
    //extra := options.getInt('extra', 0, true);
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if (h * w * c)=0 then
        raise Exception.Create('Layer before reorg layer must output image.');
    result := make_reorg_old_layer(batch, w, h, c, stride, reverse{, flatten, extra});
    //exit(layer)
end;

function parse_local_avgpool(const options: TCFGSection; const params: TSizeParams):TMaxpoolLayer;
const avgpool: boolean = true;
var
    stride: longint;
    stride_x: longint;
    stride_y: longint;
    size: longint;
    padding: longint;
    maxpool_depth: longint;
    out_channels: longint;
    antialiasing: longint;
    batch: longint;
    h: longint;
    w: longint;
    c: longint;
    layer: TMaxpoolLayer;
begin
    stride := options.getInt( 'stride', 1);
    stride_x := options.getInt('stride_x', stride, true);
    stride_y := options.getInt('stride_y', stride, true);
    size := options.getInt( 'size', stride);
    padding := options.getInt('padding', size-1, true);
    maxpool_depth := 0;
    out_channels := 1;
    antialiasing := 0;
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if not ((h<>0) and (w<>0) and (c<>0)) then
        raise Exception.Create('Layer before [local_avgpool] layer must output image.');
    layer := make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    exit(layer)
end;

function parse_maxpool(const options: TCFGSection; const params: TSizeParams
  ): TMaxpoolLayer;
const avgpool: boolean = false;
var
    stride: longint;
    stride_x: longint;
    stride_y: longint;
    size: longint;
    padding: longint;
    maxpool_depth: longint;
    out_channels: longint;
    antialiasing: longint;
    batch: longint;
    h: longint;
    w: longint;
    c: longint;
    layer: TMaxpoolLayer;
begin
    stride := options.getInt( 'stride', 1);
    stride_x := options.getInt('stride_x', stride, true);
    stride_y := options.getInt('stride_y', stride, true);
    size := options.getInt( 'size', stride);
    padding := options.getInt('padding', size-1, true);
    maxpool_depth := options.getInt('maxpool_depth', 0, true);
    out_channels := options.getInt('out_channels', 1, true);
    antialiasing := options.getInt('antialiasing', 0, true);
    h := params.h;
    w := params.w;
    c := params.c;
    batch := params.batch;
    if (h * w * c=0) then
        raise Exception.Create('Layer before [maxpool] layer must output image.');
    layer := make_maxpool_layer(batch, h, w, c, size, stride_x, stride_y, padding, maxpool_depth, out_channels, antialiasing, avgpool, params.train);
    layer.maxpool_zero_nonmax := options.getInt('maxpool_zero_nonmax', 0, true);
    exit(layer)
end;

function parse_avgpool(const options: TCFGSection; const params: TSizeParams
  ): TAvgPoolLayer;
var
    batch, w, h, c: longint;
    //layer: avgpool_layer;
begin
    w := params.w;
    h := params.h;
    c := params.c;
    batch := params.batch;
    if (h * w * c)=0 then
        raise Exception.Create('Layer before avgpool layer must output image.');
    result := make_avgpool_layer(batch, w, h, c);
    //exit(layer)
end;

function parse_dropout(const options: TCFGSection; const params: TSizeParams
  ): TDropoutLayer;
var
    probability: single;
    dropblock: boolean;
    dropblock_size_rel: single;
    dropblock_size_abs: longint;
    layer: TDropoutLayer;
begin
    probability := options.getFloat( 'probability', 0.2);
    dropblock := options.getBool( 'dropblock', false, true);
    dropblock_size_rel := options.getFloat( 'dropblock_size_rel', 0, true);
    dropblock_size_abs := options.getInt('dropblock_size_abs', 0, true);
    if (dropblock_size_abs > params.w) or (dropblock_size_abs > params.h) then
        begin
            writeln(' [dropout] - dropblock_size_abs = %d that is bigger than layer size %d x %d ', dropblock_size_abs, params.w, params.h);
            dropblock_size_abs := min(params.w, params.h)
        end;
    if dropblock and not (dropblock_size_rel<>0) and not (dropblock_size_abs<>0) then
        begin
            writeln(' [dropout] - None of the parameters (dropblock_size_rel or dropblock_size_abs) are set, will be used: dropblock_size_abs = 7 ');
            dropblock_size_abs := 7
        end;
    if (dropblock_size_rel<>0) and (dropblock_size_abs<>0) then
        begin
            writeln(' [dropout] - Both parameters are set, only the parameter will be used: dropblock_size_abs = %d ', dropblock_size_abs);
            dropblock_size_rel := 0
        end;
    layer := make_dropout_layer(params.batch, params.inputs, probability, dropblock, dropblock_size_rel, dropblock_size_abs, params.w, params.h, params.c);
    layer.out_w := params.w;
    layer.out_h := params.h;
    layer.out_c := params.c;
    exit(layer)
end;

function parse_normalization(const options: TCFGSection; const params: TSizeParams
  ): TNormalizationLayer;
var
    alpha, beta, kappa: single;
    size: longint;
    //l: layer;
begin
    alpha := options.getFloat( 'alpha', 0.0001);
    beta := options.getFloat( 'beta', 0.75);
    kappa := options.getFloat( 'kappa', 1);
    size := options.getInt( 'size', 5);
    result := make_normalization_layer(params.batch, params.w, params.h, params.c, size, alpha, beta, kappa);
    //exit(l)
end;

function parse_batchnorm(const options: TCFGSection; const params: TSizeParams
  ): TBatchNormLayer;
//var
    //l: layer;
begin
    result := make_batchnorm_layer(params.batch, params.w, params.h, params.c, params.train);
    //exit(l)
end;

function parse_shortcut(const options: TCFGSection; const params: TSizeParams; const net: TNetwork):TShortcutLayer;
var
    activation_s, weights_type_str: string;
    activation: TActivation;
    weights_type: TWeightsType;
    weights_normalization_str: string;
    weights_normalization: TWeightsNormalization;
    l: string;
    len: longint;
    n: longint;
    i: longint;
    layers: TArray<longint>;
    sizes: TArray<longint>;
    layers_output: TArray<PSingle>;
    layers_delta: TArray<PSingle>;
    layers_output_gpu: TArray<PSingle>;
    layers_delta_gpu: TArray<PSingle>;
    index: longint;
    sIndex :TArray<string>;
    s: TShortcutLayer;
begin
    activation_s := options.getStr( 'activation', 'linear');
    activation := get_activation(activation_s);
    weights_type_str := options.getStr( 'weights_type', 'none', true);
    weights_type := wtNO_WEIGHTS;
    if (weights_type_str = 'per_feature') or (weights_type_str = 'per_layer') then
        weights_type := wtPER_FEATURE
    else
        if weights_type_str = 'per_channel' then
            weights_type := wtPER_CHANNEL
    else
        if (weights_type_str <> 'none') then
            begin
                writeln(format('Error: Incorrect weights_type = %s '#10' Use one of: none, per_feature, per_channel ', [weights_type_str]));
                raise Exception.Create('Error!')
            end;
    weights_normalization_str := options.getStr( 'weights_normalization', 'none', true);
    weights_normalization := wnNO_NORMALIZATION;
    if (weights_normalization_str = 'relu') or (weights_normalization_str = 'avg_relu') then
        weights_normalization := wnRELU_NORMALIZATION
    else
        if weights_normalization_str = 'softmax' then
            weights_normalization := wnSOFTMAX_NORMALIZATION
    else
        if weights_type_str <> 'none' then
            begin
                writeln(format('Error: Incorrect weights_normalization = %s '#10' Use one of: none, relu, softmax ',[ weights_normalization_str]));
                raise Exception.Create('Error!')
            end;
    l := trim(options.getStr('from',''));
    len := length(l);
    if l='' then
        raise Exception.Create('Route Layer must specify input layers: from = ...');
    sIndex:=l.split([',']);
    n := length(sIndex);
    setLength(layers, n);
    setLength(sizes, n);
    //layers_output := AllocMem(n * sizeOf(PSingle));
    //layers_delta := AllocMem(n * sizeOf(PSingle));
    setLength(layers_output ,n);
    setLength(layers_delta ,n);
    setLength(layers_output_gpu , n);
    setLength(layers_delta_gpu , n);

    for i := 0 to n -1 do
        begin
            TryStrToInt(sIndex[i],index);
            if index < 0 then
                index := params.index+index;
            layers[i] := index;
            sizes[i] := params.net.layers[index].outputs;
            layers_output[i] := params.net.layers[index].output;
            layers_delta[i] := params.net.layers[index].delta
        end;
{$ifdef GPU}
    for i := 0 to n -1 do
        begin
            layers_output_gpu[i] := params.net.layers[layers[i]].output_gpu;
            layers_delta_gpu[i] := params.net.layers[layers[i]].delta_gpu
        end;
{$endif}
    s := make_shortcut_layer(params.batch, n, layers, sizes, params.w, params.h, params.c, layers_output, layers_delta, @layers_output_gpu[0], @layers_delta_gpu[0], weights_type, weights_normalization, activation, params.train);
    //free(layers_output_gpu);
    //free(layers_delta_gpu);
    for i := 0 to n -1 do
        begin
            index := layers[i];
            assert((params.w = net.layers[index].out_w) and (params.h = net.layers[index].out_h));
            if (params.w <> net.layers[index].out_w) or (params.h <> net.layers[index].out_h) or (params.c <> net.layers[index].out_c) then
                writeln(ErrOutput, format(' (%4d x%4d x%4d) + (%4d x%4d x%4d) ', [params.w, params.h, params.c, net.layers[index].out_w, net.layers[index].out_h, params.net.layers[index].out_c]))
        end;
    exit(s)
end;

function parse_scale_channels(const options: TCFGSection; const params: TSizeParams; const net: TNetwork):TScaleChannelLayer;
var
    l: string;
    index: longint;
    scale_wh: longint;
    batch: longint;
    from: TLayer;
    s :TScaleChannelLayer;
    activation_s: string;
    activation: TActivation;
begin
    l := options.getStr('from','');
    index := StrToInt(l);
    if index < 0 then
        index := params.index+index;
    scale_wh := options.getInt('scale_wh', 0, true);
    batch := params.batch;
    from := net.layers[index];
    s := make_scale_channels_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c, scale_wh);
    activation_s := options.getStr( 'activation', 'linear', true);
    activation := get_activation(activation_s);
    s.activation := activation;
    if (activation = acSWISH) or (activation = acMISH) then
        writeln(' [scale_channels] layer doesn''t support SWISH or MISH activations ');
    exit(s)
end;

function parse_sam(const options: TCFGSection; const params: TSizeParams; const net: TNetwork):TSAMLayer;
var
    l: string;
    index: longint;
    batch: longint;
    from: Tlayer;
    s: TSAMLayer;
    activation_s: string;
    activation: TActivation;
begin
    l := options.getStr('from','');
    index := StrToInt(l);
    if index < 0 then
        index := params.index+index;
    batch := params.batch;
    from := net.layers[index];
    s := make_sam_layer(batch, index, params.w, params.h, params.c, from.out_w, from.out_h, from.out_c);
    activation_s := options.getStr( 'activation', 'linear', true);
    activation := get_activation(activation_s);
    s.activation := activation;
    if (activation = acSWISH) or (activation = acMISH) then
        writeln(' [sam] layer doesn''t support SWISH or MISH activations ');
    exit(s)
end;

function parse_implicit(const options: TCFGSection; const params: TSizeParams; const net: TNetwork):TLayer;
var
    mean_init: single;
    std_init: single;
    filters: longint;
    atoms: longint;
    s: TLayer;
begin
    mean_init := options.getFloat( 'mean', 0.0);
    std_init := options.getFloat( 'std', 0.2);
    filters := options.getInt( 'filters', 128);
    atoms := options.getInt('atoms', 1, true);
    s := make_implicit_layer(params.batch, params.index, mean_init, std_init, filters, atoms);
    exit(s)
end;


function parse_l2norm(const options: TCFGSection; const params: TSizeParams): TL2Norm;
//var
    //l: layer;
begin
    result := make_l2norm_layer(params.batch, params.inputs);
    result.h := params.h; result.out_h := params.h;
    result.w := params.w;  result.out_w := params.w;
    result.c := params.c;  result.out_c := params.c;
    //exit(l)
end;

function parse_logistic(const options: TCFGSection; const params: TSizeParams
  ): TLogisticLayer;
//var
    //l: layer;
begin
    result := make_logistic_layer(params.batch, params.inputs);
    result.h := params.h; result.out_h := params.h;
    result.w := params.w; result.out_w := params.w;
    result.c := params.c; result.out_c := params.c;
    //exit(l)
end;

function parse_activation(const options: TCFGSection; const params: TSizeParams
  ): TActivationLayer;
var
    activation_s: string;
    activation: TActivation;
    //l: layer;
begin
    activation_s := options.getStr( 'activation', 'linear');
    activation := get_activation(activation_s);
    result := make_activation_layer(params.batch, params.inputs, activation);
    result.h := params.h; result.out_h := params.h;
    result.w := params.w; result.out_w := params.w;
    result.c := params.c; result.out_c := params.c;
    //exit(l)
end;

function parse_upsample(const options: TCFGSection; const params: TSizeParams;
  const net: TNetwork): TUpSampleLayer;
var
    stride: longint;
    //l: layer;
begin
    stride := options.getInt( 'stride', 2);
    result := make_upsample_layer(params.batch, params.w, params.h, params.c, stride);
    result.scale := options.getFloat( 'scale', 1, true);
    //exit(l)
end;

function parse_route(const options: TCFGSection; const params: TSizeParams
  ): TRouteLayer;
var
    l: string;
    vals:TArray<string>;
    n: longint;
    i: longint;
    layers: TArray<longint>;
    sizes: TArray<longint>;
    index: longint;
    batch: longint;
    groups: longint;
    group_id: longint;
    layer: TRouteLayer;
    first: TLayer;
    next: TLayer;
begin
    l := options.getStr('layers','');
    if l='' then
        raise Exception.Create('Route Layer must specify input layers');
    vals := l.split([',']);
    n:= length(vals);
    setLength(layers, n);
    setLength(sizes, n);
    for i := 0 to n -1 do
        begin
            tryStrToInt(vals[i], index);
            if index < 0 then
                index := params.index+index;
            layers[i] := index;
            sizes[i] := params.net.layers[index].outputs
        end;
    batch := params.batch;
    groups := options.getInt('groups', 1, true);
    group_id := options.getInt('group_id', 0, true);
    layer := make_route_layer(batch, n, layers, sizes, groups, group_id);
    first := params.net.layers[layers[0]];
    layer.out_w := first.out_w;
    layer.out_h := first.out_h;
    layer.out_c := first.out_c;
    for i := 1 to n -1 do
        begin
            index := layers[i];
            next := params.net.layers[index];
            if (next.out_w = first.out_w) and (next.out_h = first.out_h) then
                layer.out_c := layer.out_c + next.out_c
            else
                begin
                    writeln(ErrOutput, ' The width and height of the input layers are different.');
                    layer.out_h := 0;
                    layer.out_w := 0;
                    layer.out_c := 0
                end
        end;
    layer.out_c := layer.out_c div layer.groups;
    layer.w := first.w;
    layer.h := first.h;
    layer.c := layer.out_c;
    layer.stream := options.getInt('stream', -1, true);
    layer.wait_stream_id := options.getInt('wait_stream', -1, true);
    if n > 3 then
        write(ErrOutput, ' '#9'    ')
    else
        if n > 1 then
            write(ErrOutput, ' '#9'            ')
    else
        write(ErrOutput, ' '#9#9'            ');
    write(ErrOutput, '           ');
    if layer.groups > 1 then
        write(ErrOutput, format('%d/%d', [layer.group_id, layer.groups]))
    else
        write(ErrOutput, '   ');
    writeln(ErrOutput, format(' -> %4d x%4d x%4d ', [layer.out_w, layer.out_h, layer.out_c]));
    exit(layer)
end;

function get_policy(const s: string):TLearningRatePolicy;
begin
    if s = 'random' then
        exit(lrpCONSTANT);
    if s = 'poly' then
        exit(lrpPOLY);
    if s = 'constant' then
        exit(lrpCONSTANT);
    if s = 'step' then
        exit(lrpSTEP);
    if s = 'exp' then
        exit(lrpEXP);
    if s = 'sigmoid' then
        exit(lrpSIG);
    if s = 'steps' then
        exit(lrpSTEPS);
    if s = 'sgdr' then
        exit(lrpSGDR);
    writeln(ErrOutput, format('Couldn''t find policy %s, going with constant', [s]));
    exit(lrpCONSTANT)
end;

procedure parse_net_options(const options: TCFGSection; const net: PNetwork);
var
    subdivs: longint;
    mini_batch: longint;
    cutmix: boolean;
    mosaic: boolean;
    policy_s: string;
    device_name: string;
    l: string;
    p: string;
    s: string;
    lvals,pvals,svals: TArray<string>;
    compute_capability: longint;
    len: longint;
    n: longint;
    i: longint;
    steps : TArray<longint>;
    scales : TArray<single>;
    seq_scales : TArray<single>;
    scale: single;
    sequence_scale: single;
    step: longint;
begin
    net.max_batches := options.getInt( 'max_batches', 0);
    net.batch := options.getInt( 'batch', 1);
    net.learning_rate := options.getFloat( 'learning_rate', 0.001);
    net.learning_rate_min := options.getFloat( 'learning_rate_min', 0.00001, true);
    net.batches_per_cycle := options.getInt('sgdr_cycle', net.max_batches, true);
    net.batches_cycle_mult := options.getInt('sgdr_mult', 2, true);
    net.momentum := options.getFloat( 'momentum', 0.9);
    net.decay := options.getFloat( 'decay', 0.0001);
    subdivs := options.getInt( 'subdivisions', 1);
    net.time_steps := options.getInt('time_steps', 1, true);
    net.track := options.getInt('track', 0, true);
    net.augment_speed := options.getInt('augment_speed', 2, true);
    net.sequential_subdivisions := options.getInt('sequential_subdivisions', subdivs, true);
    net.init_sequential_subdivisions := net.sequential_subdivisions;
    if net.sequential_subdivisions > subdivs then begin
        net.init_sequential_subdivisions := subdivs; net.sequential_subdivisions := subdivs;
    end;
    net.try_fix_nan := options.getInt('try_fix_nan', 0, true);
    net.batch := net.batch div subdivs;
    mini_batch := net.batch;
    net.batch := net.batch * net.time_steps;
    net.subdivisions := subdivs;
    net.weights_reject_freq := options.getInt('weights_reject_freq', 0, true);
    net.equidistant_point := options.getInt('equidistant_point', 0, true);
    net.badlabels_rejection_percentage := options.getFloat( 'badlabels_rejection_percentage', 0, true);
    net.num_sigmas_reject_badlabels := options.getFloat( 'num_sigmas_reject_badlabels', 0, true);
    net.ema_alpha := options.getFloat( 'ema_alpha', 0, true);
    net.badlabels_reject_threshold [0]:= 0;
    net.delta_rolling_max[0] := 0;
    net.delta_rolling_avg[0] := 0;
    net.delta_rolling_std[0] := 0;
    net.seen[0] := 0;
    net.cur_iteration[0] := 0;
    net.cuda_graph_ready[0] := 0;
    net.use_cuda_graph := options.getInt('use_cuda_graph', 0, true);
    net.loss_scale := options.getFloat( 'loss_scale', 1, true);
    net.dynamic_minibatch := options.getInt('dynamic_minibatch', 0, true);
    net.optimized_memory := options.getInt('optimized_memory', 0, true);
    net.workspace_size_limit := trunc(1024 * 1024 * options.getFloat( 'workspace_size_limit_MB', 1024, true));
    net.adam := options.getBool( 'adam', false, true);
    if net.adam then
        begin
            net.B1 := options.getFloat( 'B1', 0.9);
            net.B2 := options.getFloat( 'B2', 0.999);
            net.eps := options.getFloat( 'eps', 0.000001)
        end;
    net.h := options.getInt('height', 0, true);
    net.w := options.getInt('width', 0, true);
    net.c := options.getInt('channels', 0, true);
    net.inputs := options.getInt('inputs', net.h * net.w * net.c, true);
    net.max_crop := options.getInt('max_crop', net.w * 2, true);
    net.min_crop := options.getInt('min_crop', net.w, true);
    net.flip := options.getInt('flip', 1, true);
    net.blur := options.getInt('blur', 0, true);
    net.gaussian_noise := options.getInt('gaussian_noise', 0, true);
    net.mixup := options.getInt('mixup', 0, true);
    cutmix := options.getBool( 'cutmix', false, true);
    mosaic := options.getBool( 'mosaic', false, true);
    if mosaic and cutmix then
        net.mixup := 4
    else
        if cutmix then
            net.mixup := 2
    else
        if mosaic then
            net.mixup := 3;
    net.letter_box := options.getInt('letter_box', 0, true);
    net.mosaic_bound := options.getInt('mosaic_bound', 0, true);
    net.contrastive := options.getBool( 'contrastive', false, true);
    net.contrastive_jit_flip := options.getBool( 'contrastive_jit_flip', false, true);
    net.contrastive_color := options.getBool( 'contrastive_color', false, true);
    net.unsupervised := options.getBool( 'unsupervised', false, true);
    if (net.contrastive) and (mini_batch < 2) then
        raise Exception.Create('Error: mini_batch size (batch/subdivisions) should be higher than 1 for Contrastive loss!');
    net.label_smooth_eps := options.getFloat( 'label_smooth_eps', 0.0, true);
    net.resize_step := options.getInt('resize_step', 32, true);
    net.attention := options.getInt('attention', 0, true);
    net.adversarial_lr := options.getFloat( 'adversarial_lr', 0, true);
    net.max_chart_loss := options.getFloat( 'max_chart_loss', 20.0, true);
    net.angle := options.getFloat( 'angle', 0, true);
    net.aspect := options.getFloat( 'aspect', 1, true);
    net.saturation := options.getFloat( 'saturation', 1, true);
    net.exposure := options.getFloat( 'exposure', 1, true);
    net.hue := options.getFloat( 'hue', 0, true);
    net.power := options.getFloat( 'power', 4, true);
    if (net.inputs=0) and (net.h * net.w * net.c=0) then
        raise Exception.create('No input parameters supplied');
    policy_s := options.getStr( 'policy', 'constant');
    net.policy := get_policy(policy_s);
    net.burn_in := options.getInt('burn_in', 0, true);
{$ifdef GPU}
    if net.gpu_index >= 0 then
        begin
            compute_capability := get_gpu_compute_capability(net.gpu_index, device_name);
{$ifdef CUDNN_HALF}
            if compute_capability >= 700 then
              net.cudnn_half := 1
            else
                net.cudnn_half := 0;
{$endif}
            writeln(ErrOutput, ' %d : compute_capability = %d, cudnn_half = %d, GPU: %s '#10'', net.gpu_index, compute_capability, net.cudnn_half, device_name)
        end
    else
        writeln(ErrOutput, ' GPU isn''t used '#10'');
{$endif}
    if net.policy = lrpSTEP then
        begin
            net.step := options.getInt( 'step', 1);
            net.scale := options.getFloat( 'scale', 1)
        end
    else
        if (net.policy = lrpSTEPS) or (net.policy = lrpSGDR) then
            begin
                l := options.getStr('steps','');
                p := options.getStr('scales','');
                s := options.getStr('seq_scales','');
                if (net.policy = lrpSTEPS) and ((l='') or (p='')) then
                    raise Exception.Create('STEPS policy must have steps and scales in cfg file');
                if l<>'' then
                    begin
                        lvals := l.split([',']);
                        svals := l.split([',']);
                        pvals := l.split([',']);
                        n:= length(lvals);
                        setLength(steps, n);// := TIntegers.Create(n);
                        setLength(scales, n);// := TSingles.Create(n);
                        setLength(seq_scales, n);// := TSingles.Create(n);
                        for i := 0 to n -1 do
                            begin
                                scale := 1.0;
                                if i<length(pvals) then
                                        trystrToFloat(pvals[i], scale);
                                sequence_scale := 1.0;
                                if i<length(svals) then
                                        tryStrToFloat(svals[i], sequence_scale);
                                tryStrToInt(lvals[i], step );
                                steps[i] := step;
                                scales[i] := scale;
                                seq_scales[i] := sequence_scale
                            end;
                        net.scales := scales;
                        net.steps := steps;
                        net.seq_scales := seq_scales;
                        net.num_steps := n
                    end
            end
    else
        if net.policy = lrpEXP then
            net.gamma := options.getFloat( 'gamma', 1)
    else
        if net.policy = lrpSIG then
            begin
                net.gamma := options.getFloat( 'gamma', 1);
                net.step := options.getInt( 'step', 1)
            end
    else
        if (net.policy = lrpPOLY) or (net.policy = lrpRANDOM) then

end;

function is_network(const s: TCFGSection): boolean;
begin
    exit((s.typeName = '[net]') or (s.typeName = '[network]'))
end;

function read_cfg(const filename: string): TCFGList;
var
    f: TextFile;
    line: string;
    nu: longint;
    //options: PList;
    current: PCFGSection;
begin
    if not FileExists(filename) then
        raise EFileNotFoundException(filename+': not found.');
    AssignFile(f, filename);
    reset(f);
    //f := fopen(filename, 'r');
    //if file = 0 then
    //    file_error(filename);
    nu := 0;
    current := nil;
    while not EOF(f) do
        begin
            readln(f,line);
            inc(nu);
            line := _strip(line);
            if line<>'' then
              case line[1] of
                  '[':
                      begin
                          current:=result.addSection(default(TCFGSection));
                          current.typeName := line
                      end;
                  '#', ';':
                      //free(line);
              else
                  if not current.addOptionLine(line) then
                      begin
                          writeln(ErrOutput, format('Config file error line %d, could parse: %s', [nu, line]));
                          //free(line)
                      end
              end
        end;
    CloseFile(f);
    //exit(options)
end;

procedure set_train_only_bn(const net: TNetwork);
var
    train_only_bn: boolean;
    i: longint;
begin
    train_only_bn := false;
    for i:= net.n-1 downto 0 do begin
        if net.layers[i].train_only_bn then
            train_only_bn := net.layers[i].train_only_bn;
        if train_only_bn then
            begin
                net.layers[i].train_only_bn := train_only_bn;
                if net.layers[i].&type = ltConvLSTM then
                    begin
                        net.layers[i].wf[0].train_only_bn := train_only_bn;
                        net.layers[i].wi[0].train_only_bn := train_only_bn;
                        net.layers[i].wg[0].train_only_bn := train_only_bn;
                        net.layers[i].wo[0].train_only_bn := train_only_bn;
                        net.layers[i].uf[0].train_only_bn := train_only_bn;
                        net.layers[i].ui[0].train_only_bn := train_only_bn;
                        net.layers[i].ug[0].train_only_bn := train_only_bn;
                        net.layers[i].uo[0].train_only_bn := train_only_bn;
                        if net.layers[i].peephole then
                            begin
                                net.layers[i].vf[0].train_only_bn := train_only_bn;
                                net.layers[i].vi[0].train_only_bn := train_only_bn;
                                net.layers[i].vo[0].train_only_bn := train_only_bn
                            end
                    end
                else
                    if net.layers[i].&type = ltCRNN then
                        begin
                            net.layers[i].input_layer[0].train_only_bn := train_only_bn;
                            net.layers[i].self_layer[0].train_only_bn := train_only_bn;
                            net.layers[i].output_layer[0].train_only_bn := train_only_bn
                        end
            end;
    end
end;

function parse_network_cfg_custom(const filename: string; const batch, time_steps: longint):TNetwork;
var
    sections: TCFGList;
    //n: PNode;
    //n_tmp: PNode;
    net: TNetwork;
    params: TSizeParams;
    options: TCFGSection;
    last_stop_backward: longint;
    avg_outputs: longint;
    avg_counter: longint;
    bflops: single;
    workspace_size: size_t;
    max_inputs: size_t;
    max_outputs: size_t;
    receptive_w, receptive_h: longint;
    receptive_w_scale, receptive_h_scale: longint;
    show_receptive_field: boolean;
    count: longint;
    count_tmp: longint;
    stopbackward: longint;
    old_params_train: boolean;
    l: TLayer;
    lt: TLayerType;
    k: longint;
    empty_layer: TLayer;
    dilation: longint;
    stride: longint;
    size, i: longint;
    route_l: TLayer;
    increase_receptive: longint;
    cur_receptive_w: longint;
    cur_receptive_h: longint;
    delta_size: size_t;
begin
    sections := read_cfg(filename);
    if sections.isEmpty then
        raise Exception.Create('Config file has no sections');
    net := make_network(sections.count-1);
    net.gpu_index := -1;
    if batch > 0 then   // could params.train := batch<=0
        params.train := false
    else
        params.train := true;
    options := sections.Sections[0];
    if not is_network(options) then
        raise Exception.Create('First section must be [net] or [network]');
    parse_net_options(options,  @net);
{$ifdef GPU}
    printf('net.optimized_memory = %d '#10'', net.optimized_memory);
    if net.optimized_memory >= 2 and params.train then
        pre_allocate_pinned_memory(size_t(1024) * 1024 * 1024 * 8);
{$endif}
    params.h := net.h;
    params.w := net.w;
    params.c := net.c;
    params.inputs := net.inputs;
    if batch > 0 then
        net.batch := batch;
    if time_steps > 0 then
        net.time_steps := time_steps;
    if net.batch < 1 then
        net.batch := 1;
    if net.time_steps < 1 then
        net.time_steps := 1;
    if net.batch < net.time_steps then
        net.batch := net.time_steps;
    params.batch := net.batch;
    params.time_steps := net.time_steps;
    params.net := @net;
    writeln(format('mini_batch = %d, batch = %d, time_steps = %d, train = %d ', [net.batch, net.batch * net.subdivisions, net.time_steps, longint(params.train)]));
    last_stop_backward := -1;
    avg_outputs := 0;
    avg_counter := 0;
    bflops := 0;
    workspace_size := 0;
    max_inputs := 0;
    max_outputs := 0;
    receptive_w := 1; receptive_h := 1;
    receptive_w_scale := 1; receptive_h_scale := 1;
    show_receptive_field := options.getBool( 'show_receptive_field', false, true);
    //n := n.next;
    count := 0;
    //free_section(s);
    //n_tmp := n;
    count_tmp := 0;
    if params.train then
        for i:=1 to sections.count-1 do
            begin
                options := sections.Sections[i];
                stopbackward := options.getInt('stopbackward', 0, true);
                if stopbackward = 1 then
                    begin
                        last_stop_backward := count_tmp;
                        writeln(format('last_stop_backward = %d ', [last_stop_backward]))
                    end;
                inc(count_tmp)
            end;
    old_params_train := params.train;
    writeln(ErrOutput, '   layer   filters  size/strd(dil)      input                output');
    for i:=1 to sections.Count-1 do     // sections[0] has network params thus we start from 1
        begin
            params.train := old_params_train;
            if count < last_stop_backward then
                params.train := false;
            params.index := count;
            write(ErrOutput, count:4,' ');
            options := sections.Sections[i];
            l := Default(TLayer);
            lt := string_to_layer_type(options.typeName);
            if lt = ltCONVOLUTIONAL then
                l := parse_convolutional(options, params)
            else
                if lt = ltLOCAL then
                    l := parse_local(options, params)
            else
                if lt = ltACTIVE then
                    l := parse_activation(options, params)
            else
                if lt = ltRNN then
                    l := parse_rnn(options, params)
            else
                if lt = ltGRU then
                    l := parse_gru(options, params)
            else
                if lt = ltLSTM then
                    l := parse_lstm(options, params)
            else
                if lt = ltConvLSTM then
                    l := parse_conv_lstm(options, params)
            else
                if lt = ltHISTORY then
                    l := parse_history(options, params)
            else
                if lt = ltCRNN then
                    l := parse_crnn(options, params)
            else
                if lt = ltCONNECTED then
                    l := parse_connected(options, params)
            else
                if lt = ltCROP then
                    l := parse_crop(options, params)
            else
                if lt = ltCOST then
                    begin
                        l := parse_cost(options, params);
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltREGION then
                    begin
                        l := parse_region(options, params);
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltYOLO then
                    begin
                        l := parse_yolo(options, params);
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltGaussianYOLO then
                    begin
                        l := parse_gaussian_yolo(options, params);
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltDETECTION then
                    l := parse_detection(options, params)
            else
                if lt = ltSOFTMAX then
                    begin
                        l := parse_softmax(options, params);
                        net.hierarchy := l.softmax_tree;
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltCONTRASTIVE then
                    begin
                        l := parse_contrastive(options, params);
                        l.keep_delta_gpu := true
                    end
            else
                if lt = ltNORMALIZATION then
                    l := parse_normalization(options, params)
            else
                if lt = ltBATCHNORM then
                    l := parse_batchnorm(options, params)
            else
                if lt = ltMAXPOOL then
                    l := parse_maxpool(options, params)
            else
                if lt = ltLOCAL_AVGPOOL then
                    l := parse_local_avgpool(options, params)
            else
                if lt = ltREORG then
                    l := parse_reorg(options, params)
            else
                if lt = ltREORG_OLD then
                    l := parse_reorg_old(options, params)
            else
                if lt = ltAVGPOOL then
                    l := parse_avgpool(options, params)
            else
                if lt = ltROUTE then
                    begin
                        l := parse_route(options, params);
                        for k := 0 to l.n -1 do
                            begin
                                net.layers[l.input_layers[k]].use_bin_output := false;
                                if count >= last_stop_backward then
                                    net.layers[l.input_layers[k]].keep_delta_gpu := true
                            end
                    end
            else
                if lt = ltUPSAMPLE then
                    l := parse_upsample(options, params, net)
            else
                if lt = ltSHORTCUT then
                    begin
                        l := parse_shortcut(options, params, net);
                        net.layers[count-1].use_bin_output := false;
                        net.layers[l.index].use_bin_output := false;
                        if count >= last_stop_backward then
                            net.layers[l.index].keep_delta_gpu := true
                    end
            else
                if lt = ltScaleChannels then
                    begin
                        l := parse_scale_channels(options, params, net);
                        net.layers[count-1].use_bin_output := false;
                        net.layers[l.index].use_bin_output := false;
                        net.layers[l.index].keep_delta_gpu := true
                    end
            else
                if lt = ltSAM then
                    begin
                        l := parse_sam(options, params, net);
                        net.layers[count-1].use_bin_output := false;
                        net.layers[l.index].use_bin_output := false;
                        net.layers[l.index].keep_delta_gpu := true
                    end
            else
                if lt = ltIMPLICIT then
                    l := parse_implicit(options, params, net)
            else
                if lt = ltDROPOUT then
                    begin
                        l := parse_dropout(options, params);
                        l.output := net.layers[count-1].output;
                        l.delta := net.layers[count-1].delta;
                    {$ifdef GPU}
                        l.output_gpu := net.layers[count-1].output_gpu;
                        l.delta_gpu := net.layers[count-1].delta_gpu;
                        l.keep_delta_gpu := 1
                    {$endif}
                    end
            else
                if lt = ltEMPTY then
                    begin
                        empty_layer := Default(TLayer);
                        l := empty_layer;
                        l.&type := ltEMPTY;
                        l.w := params.w; l.out_w := params.w;
                        l.h := params.h; l.out_h := params.h;
                        l.c := params.c; l.out_c := params.c;
                        l.batch := params.batch;
                        l.inputs := params.inputs; l.outputs := params.inputs;
                        l.output := net.layers[count-1].output;
                        l.delta := net.layers[count-1].delta;
                        l.forward := empty_func;
                        l.backward := empty_func;
                    {$ifdef GPU}
                        l.output_gpu := net.layers[count-1].output_gpu;
                        l.delta_gpu := net.layers[count-1].delta_gpu;
                        l.keep_delta_gpu := 1;
                        l.forward_gpu := empty_func;
                        l.backward_gpu := empty_func;
                    {$endif}
                        writeln(ErrOutput, 'empty')
                    end
            else
                writeln(ErrOutput, 'Type not recognized: ', options.typeName);
            if show_receptive_field then
                begin
                    dilation := max(1, l.dilation);
                    stride := max(1, l.stride);
                    size := max(1, l.size);
                    if (l.&type = ltUPSAMPLE) or (l.&type = ltREORG) then
                        begin
                            l.receptive_w := receptive_w;
                            l.receptive_h := receptive_h;
                            receptive_w_scale := receptive_w_scale div stride;
                            l.receptive_w_scale := receptive_w_scale;
                            receptive_h_scale := receptive_h_scale div stride;
                            l.receptive_h_scale := receptive_h_scale;
                        end
                    else
                        begin
                            if l.&type = ltROUTE then
                                begin
                                    receptive_w := 0; receptive_h := 0; receptive_w_scale := 0; receptive_h_scale := 0;
                                    for k := 0 to l.n -1 do
                                        begin
                                            route_l := net.layers[l.input_layers[k]];
                                            receptive_w := max(receptive_w, route_l.receptive_w);
                                            receptive_h := max(receptive_h, route_l.receptive_h);
                                            receptive_w_scale := max(receptive_w_scale, route_l.receptive_w_scale);
                                            receptive_h_scale := max(receptive_h_scale, route_l.receptive_h_scale)
                                        end
                                end
                            else
                                begin
                                    increase_receptive := size+(dilation-1) * 2-1;
                                    increase_receptive := max(0, increase_receptive);
                                    receptive_w := receptive_w + (increase_receptive * receptive_w_scale);
                                    receptive_h := receptive_h + (increase_receptive * receptive_h_scale);
                                    receptive_w_scale := receptive_w_scale * stride;
                                    receptive_h_scale := receptive_h_scale * stride
                                end;
                            l.receptive_w := receptive_w;
                            l.receptive_h := receptive_h;
                            l.receptive_w_scale := receptive_w_scale;
                            l.receptive_h_scale := receptive_h_scale
                        end;
                    cur_receptive_w := receptive_w;
                    cur_receptive_h := receptive_h;
                    writeln(ErrOutput, format('%4d - receptive field: %d x %d ', [count, cur_receptive_w, cur_receptive_h]))
                end;
{$ifdef GPU}
            l.optimized_memory := net.optimized_memory;
            if (net.optimized_memory = 1) and params.train and (l.&type <> ltDROPOUT) then
                if l.delta_gpu then
                    begin
                        cuda_free(l.delta_gpu);
                        l.delta_gpu := NULL
                    end
            else
                if (net.optimized_memory >= 2) and params.train and (l.&type <> DROPOUT) then
                    begin
                        if l.output_gpu then
                            begin
                                cuda_free(l.output_gpu);
                                l.output_gpu := cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs)
                            end;
                        if l.activation_input_gpu then
                            begin
                                cuda_free(l.activation_input_gpu);
                                l.activation_input_gpu := cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs)
                            end;
                        if l.x_gpu then
                            begin
                                cuda_free(l.x_gpu);
                                l.x_gpu := cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs)
                            end;
                        if (net.optimized_memory >= 3) and (l.&type <> ltDROPOUT) then
                            if l.delta_gpu then
                                cuda_free(l.delta_gpu);
                        if l.&type = ltCONVOLUTIONAL then
                            set_specified_workspace_limit( and l, net.workspace_size_limit)
                    end;
{$endif}
            l.clip := options.getFloat( 'clip', 0, true);
            l.dynamic_minibatch := net.dynamic_minibatch;
            l.onlyforward := options.getBool( 'onlyforward', false, true);
            l.dont_update := options.getBool( 'dont_update', false, true);
            l.burnin_update := options.getBool( 'burnin_update', false, true);
            l.stopbackward := options.getBool( 'stopbackward', false, true);
            l.train_only_bn := options.getBool( 'train_only_bn', false, true);
            l.dontload := options.getBool( 'dontload', false, true);
            l.dontloadscales := options.getBool( 'dontloadscales', false, true);
            l.learning_rate_scale := options.getFloat( 'learning_rate', 1, true);
            //option_unused(options);
            if l.stopbackward = true then
                writeln(' ------- previous layers are frozen ------- ');
            net.layers[count] := l;
            if l.workspace_size > workspace_size then
                workspace_size := l.workspace_size;
            if l.inputs > max_inputs then
                max_inputs := l.inputs;
            if l.outputs > max_outputs then
                max_outputs := l.outputs;
            //free_section(s);
            //n := n.next;
            inc(count);
            if i< Sections.Count()-1 then
                begin
                    if l.antialiasing<>0 then
                        begin
                            params.h := l.input_layer[0].out_h;
                            params.w := l.input_layer[0].out_w;
                            params.c := l.input_layer[0].out_c;
                            params.inputs := l.input_layer[0].outputs
                        end
                    else
                        begin
                            params.h := l.out_h;
                            params.w := l.out_w;
                            params.c := l.out_c;
                            params.inputs := l.outputs
                        end
                end;
            if l.bflops > 0 then
                bflops := bflops + l.bflops;
            if (l.w > 1) and (l.h > 1) then
                begin
                    avg_outputs := avg_outputs + l.outputs;
                    inc(avg_counter)
                end
        end;
    if last_stop_backward > -1 then
        begin
            for k := 0 to last_stop_backward -1 do
                begin
                    l := net.layers[k];
                    if l.keep_delta_gpu then
                        begin
                            if not assigned(l.delta) then
                                net.layers[k].delta := TSingles.Create(l.outputs * l.batch);
                        {$ifdef GPU}
                             if not l.delta_gpu then
                                net.layers[k].delta_gpu := single(cuda_make_array(NULL, l.outputs * l.batch))
                        {$endif}
                        end;
                    net.layers[k].onlyforward := true;
                    net.layers[k].train := false
                end
        end;
    //free_list(sections);
{$ifdef GPU}
    if net.optimized_memory and params.train then
        begin
            for k := 0 to net.n -1 do
                begin
                    l := net.layers[k];
                    if not l.keep_delta_gpu then
                        begin
                            delta_size := l.outputs * l.batch;
                            if net.max_delta_gpu_size < delta_size then
                                begin
                                    net.max_delta_gpu_size := delta_size;
                                    if net.global_delta_gpu then
                                        cuda_free(net.global_delta_gpu);
                                    if net.state_delta_gpu then
                                        cuda_free(net.state_delta_gpu);
                                    assert(net.max_delta_gpu_size > 0);
                                    net.global_delta_gpu := single(cuda_make_array(NULL, net.max_delta_gpu_size));
                                    net.state_delta_gpu := single(cuda_make_array(NULL, net.max_delta_gpu_size))
                                end;
                            if l.delta_gpu then
                                begin
                                    if net.optimized_memory >= 3 then

                                    else
                                        cuda_free(l.delta_gpu)
                                end;
                            l.delta_gpu := net.global_delta_gpu
                        end
                    else
                        if not l.delta_gpu then
                            l.delta_gpu := single(cuda_make_array(NULL, l.outputs * l.batch));
                    if net.optimized_memory >= 3 and l.&type <> DROPOUT then
                        if l.delta_gpu and l.keep_delta_gpu then
                            l.delta_gpu := cuda_make_array_pinned_preallocated(NULL, l.batch * l.outputs);
                    net.layers[k] := l
                end
        end;
{$endif}
    set_train_only_bn(net);
    net.outputs := get_network_output_size(net);
    net.output := get_network_output(net);
    avg_outputs := avg_outputs div avg_counter;
    writeln(ErrOutput, format('Total GFLOPS %5.3f ', [bflops]));
    writeln(ErrOutput, format('avg_outputs = %d ', [avg_outputs]));
{$ifdef GPU}
    get_cuda_stream();
    if gpu_index >= 0 then
        begin
            size := get_network_input_size(net) * net.batch;
            net.input_state_gpu := cuda_make_array(0, size);
            if cudaSuccess = cudaHostAlloc( and net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped) then
                net.input_pinned_cpu_flag := 1
            else
                begin
                    cudaGetLastError();
                    net.input_pinned_cpu := single(xcalloc(size, sizeof(float)))
                end;
             * net.max_input16_size := 0;
             * net.max_output16_size := 0;
            if net.cudnn_half then
                begin
                     * net.max_input16_size := max_inputs;
                    CHECK_CUDA(cudaMalloc(PPointer(net.input16_gpu),  * net.max_input16_size * sizeof(short)));
                     * net.max_output16_size := max_outputs;
                    CHECK_CUDA(cudaMalloc(PPointer(net.output16_gpu),  * net.max_output16_size * sizeof(short)))
                end;
            if workspace_size then
                begin
                    fprintf(ErrOutput, ' Allocate additional workspace_size = %1.2f MB '#10'', single(workspace_size) / 1000000);
                    net.workspace := cuda_make_array(0, workspace_size div sizeof(float)+1)
                end
            else
                net.workspace := single(xcalloc(1, workspace_size))
        end;
{$else}
    if workspace_size<>0 then
        setLength(net.workspace, workspace_size div sizeof(single)) ;//:= single(xcalloc(1, workspace_size));
{$endif}
    lt := net.layers[net.n-1].&type;
    if ((net.w mod 32 <> 0) or (net.h mod 32 <> 0)) and ((lt = ltYOLO) or (lt = ltREGION) or (lt = ltDETECTION)) then
        writeln(format(#10' Warning: width=%d and height=%d in cfg-file must be divisible by 32 for default networks Yolo v1/v2/v3!!! '#10, [net.w, net.h]));
    exit(net)
end;

function parse_network_cfg(const filename: string): TNetwork;
begin
    exit(parse_network_cfg_custom(filename, 0, 0))
end;

procedure save_convolutional_weights_binary(const l: TConvolutionalLayer; var fp: file);
var
    size, i, j, k, index: longint;
    mean: single;
    c: byte;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        pull_convolutional_layer(l);
  {$endif}
    binarize_weights(l.weights, l.n, l.c * l.size * l.size, l.binary_weights);
    size := l.c * l.size * l.size;
    BlockWrite(fp, l.biases[0], l.n * sizeof(single), i);
    if l.batch_normalize then
        begin
            BlockWrite(fp, l.scales[0], l.n * sizeof(single), i);
            BlockWrite(fp, l.rolling_mean[0], l.n * sizeof(single), i);
            BlockWrite(fp, l.rolling_variance[0], l.n * sizeof(single), i)
        end;
    for i := 0 to l.n -1 do
        begin
            mean := l.binary_weights[i * size];
            if mean < 0 then
                mean := -mean;
            BlockWrite(fp, mean, sizeof(single), j);
            for j := 0 to size div 8 -1 do
                begin
                    index := i * size+j * 8;
                    c := 0;
                    for k := 0 to 8 -1 do
                        begin
                            if j * 8+k >= size then
                                break;
                            if l.binary_weights[index+k] > 0 then
                                c := (c or 1 shl k)
                        end;
                    BlockWrite(fp, c, sizeof(byte), k)
                end
        end
end;

procedure save_shortcut_weights(const l: TShortcutLayer; var fp: file);
var
    i: longint;
    num: longint;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        begin
            pull_shortcut_layer(l);
            printf(''#10' pull_shortcut_layer '#10'')
        end;
  {$endif}
    for i := 0 to l.nweights -1 do
        writeln(format(' %f, ', [ l.weights[i] ]));
    writeln(format(' l.nweights = %d '#10, [l.nweights]));
    num := l.nweights;
    BlockWrite(fp, l.weights[0], sizeof(Single)* num, i)
end;

procedure save_implicit_weights(const l: TImplicitLayer; var fp: file);
var
    i: longint;
    num: longint;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        pull_implicit_layer(l);
  {$endif}
    num := l.nweights;
    BlockWrite(fp, l.weights[0], sizeof(Single)* num, i)
end;


procedure save_convolutional_weights(const l: TConvolutionalLayer; var fp: file);
var
    num, n: longint;
begin
    if l.binary then begin
        //save_convolutional_weights_binary(l, fp);
        //exit
    end;
{$ifdef GPU}
    if gpu_index >= 0 then
        pull_convolutional_layer(l);
{$endif}
    num := l.nweights;
    BlockWrite(fp, l.biases[0], sizeof(single)* l.n, n);
    if l.batch_normalize then
        begin
            BlockWrite(fp, l.scales[0], sizeof(single) * l.n, n);
            BlockWrite(fp, l.rolling_mean[0], sizeof(single) * l.n, n);
            BlockWrite(fp, l.rolling_variance[0], sizeof(single) * l.n, n)
        end;
    BlockWrite(fp, l.weights[0], sizeof(single) * num, n)
end;

procedure save_convolutional_weights_ema(const l: TConvolutionalLayer; var fp: file);
var
    num,i: longint;
begin
    if l.binary then
       ;
  {$ifdef GPU}
    if gpu_index >= 0 then
        pull_convolutional_layer(l);
  {$endif}
    num := l.nweights;
    BlockWrite(fp, l.biases_ema, sizeof(single)* l.n, i);
    if l.batch_normalize then
        begin
            BlockWrite(fp, l.scales_ema[0], sizeof(single)* l.n, i);
            BlockWrite(fp, l.rolling_mean[0], sizeof(single)* l.n, i);
            BlockWrite(fp, l.rolling_variance[0], sizeof(single)* l.n, i)
        end;
    BlockWrite(fp, l.weights_ema[0], sizeof(single) * num, i)
end;

procedure save_batchnorm_weights(const l: TBatchNormLayer; var fp: file);
var n:longint;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        pull_batchnorm_layer(l);
  {$endif}
    BlockWrite(fp, l.biases[0], sizeof(single) * l.c, n);
    BlockWrite(fp, l.scales[0], sizeof(single) * l.c, n);
    BlockWrite(fp, l.rolling_mean[0], sizeof(single) * l.c, n);
    BlockWrite(fp, l.rolling_variance[0], sizeof(single) * l.c, n)
end;

procedure save_connected_weights(const l: TConnectedLayer; var fp: file);
var n:longint;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        pull_connected_layer(l);
  {$endif}
    BlockWrite(fp, l.biases[0], sizeof(single) * l.outputs, n);
    BlockWrite(fp, l.weights[0], sizeof(single) * l.outputs * l.inputs, n);
    if l.batch_normalize then
        begin
            BlockWrite(fp, l.scales[0], sizeof(single) * l.outputs, n);
            BlockWrite(fp, l.rolling_mean[0], sizeof(single) * l.outputs, n);
            BlockWrite(fp, l.rolling_variance[0], sizeof(single) * l.outputs, n)
        end
end;

procedure save_weights_upto(const net: TNetwork; const filename: string; cutoff: longint; const save_ema: longint);
var
    fp: file;
    major: int32;
    minor: int32;
    revision: int32;
    i: longint;
    l: TLayer;
    locations: longint;
    size, o: longint;
begin
{$ifdef GPU}
    if (net.gpu_index >= 0) then
        cuda_set_device(net.gpu_index);
{$endif}
    writeln(ErrOutput, 'Saving weights to ', filename);
    //assert(fileExists(filename), 'File not fount:'+filename);
    assignfile(fp, filename);
    reWrite(fp, 1);
    major := MAJOR_VERSION;
    minor := MINOR_VERSION;
    revision := PATCH_VERSION;
    BlockWrite(fp, major, sizeof(int32)* 1, o);
    BlockWrite(fp, minor, sizeof(int32)* 1, o);
    BlockWrite(fp, revision, sizeof(int32)* 1, o);
    net.seen[0] := get_current_iteration(net) * net.batch * net.subdivisions;
    BlockWrite(fp, net.seen[0], sizeof(uint64)* 1, o);
    i := 0;
    while (i < net.n) and (i < cutoff) do begin
        l := net.layers[i];
        if (l.&type = ltCONVOLUTIONAL) and (l.share_layer = nil) then
            begin
                if save_ema<>0 then
                    save_convolutional_weights_ema(l, fp)
                else
                    save_convolutional_weights(l, fp)
            end;
        if (l.&type = ltSHORTCUT) and (l.nweights > 0) then
            save_shortcut_weights(l, fp);
        if l.&type = ltIMPLICIT then
            save_implicit_weights(l, fp);
        if l.&type = ltCONNECTED then
            save_connected_weights(l, fp);
        if l.&type = ltBATCHNORM then
            save_batchnorm_weights(l, fp);
        if l.&type = ltRNN then
            begin
                save_connected_weights(l.input_layer[0], fp);
                save_connected_weights(l.self_layer[0], fp);
                save_connected_weights(l.output_layer[0], fp)
            end;
        if l.&type = ltGRU then
            begin
                save_connected_weights(l.wz[0], fp);
                save_connected_weights(l.wr[0], fp);
                save_connected_weights(l.wh[0], fp);
                save_connected_weights(l.uz[0], fp);
                save_connected_weights(l.ur[0], fp);
                save_connected_weights(l.uh[0], fp)
            end;
        if l.&type = ltLSTM then
            begin
                save_connected_weights(l.wf[0], fp);
                save_connected_weights(l.wi[0], fp);
                save_connected_weights(l.wg[0], fp);
                save_connected_weights(l.wo[0], fp);
                save_connected_weights(l.uf[0], fp);
                save_connected_weights(l.ui[0], fp);
                save_connected_weights(l.ug[0], fp);
                save_connected_weights(l.uo[0], fp)
            end;
        if l.&type = ltConvLSTM then
            begin
                if l.peephole then
                    begin
                        save_convolutional_weights(l.vf[0], fp);
                        save_convolutional_weights(l.vi[0], fp);
                        save_convolutional_weights(l.vo[0], fp)
                    end;
                save_convolutional_weights(l.wf[0], fp);
                if not l.bottleneck then
                    begin
                        save_convolutional_weights(l.wi[0], fp);
                        save_convolutional_weights(l.wg[0], fp);
                        save_convolutional_weights(l.wo[0], fp)
                    end;
                save_convolutional_weights(l.uf[0], fp);
                save_convolutional_weights(l.ui[0], fp);
                save_convolutional_weights(l.ug[0], fp);
                save_convolutional_weights(l.uo[0], fp)
            end;
        if l.&type = ltCRNN then
            begin
                save_convolutional_weights(l.input_layer[0], fp);
                save_convolutional_weights(l.self_layer[0], fp);
                save_convolutional_weights(l.output_layer[0], fp)
            end;
        if l.&type = ltLOCAL then
            begin
{$ifdef GPU}
                if gpu_index >= 0 then
                    pull_local_layer(l);
{$endif}
                locations := l.out_w * l.out_h;
                size := l.size * l.size * l.c * l.n * locations;
                BlockWrite(fp, l.biases[0], sizeof(single) * l.outputs, o);
                BlockWrite(fp, l.weights[0], sizeof(single) * size, o)
            end;
        //fflush(fp);
        inc(i)
    end;
    closefile(fp)
end;

procedure save_weights(const net: TNetwork; const filename: string);
begin
    save_weights_upto(net, filename, net.n, 0)
end;

procedure transpose_matrix(const a: TSingles; const rows, cols: longint);
var
    x: longint;
    y: longint;
    transpose : TArray<Single>;
begin
    setLength(transpose ,rows * cols);
    for x := 0 to rows -1 do
        for y := 0 to cols -1 do
            transpose[y * rows+x] := a[x * cols+y];
    move(transpose[0], a[0], rows * cols * sizeof(single));
    //free(transpose)
end;

procedure load_connected_weights(var l: TConnectedLayer; var fp: file; const transpose: boolean);
var o :longint;
begin
    BlockRead(fp, l.biases[0], sizeof(single) * l.outputs, o);
    BlockRead(fp, l.weights[0], sizeof(single) * l.outputs * l.inputs, o);
    if transpose then
        transpose_matrix(l.weights, l.inputs, l.outputs);
    if l.batch_normalize and (not l.dontloadscales) then
        begin
            BlockRead(fp, l.scales[0], sizeof(single) * l.outputs, o);
            BlockRead(fp, l.rolling_mean[0], sizeof(single) * l.outputs, o);
            BlockRead(fp, l.rolling_variance[0], sizeof(single) * l.outputs, o)
        end;
  {$ifdef GPU}
    if gpu_index >= 0 then
        push_connected_layer(l)
  {$endif}
end;

procedure load_batchnorm_weights(var l: TBatchNormLayer; var fp: file);
var o :longint;
begin
    BlockRead(fp, l.biases[0], sizeof(single) * l.c, o);
    BlockRead(fp, l.scales[0], sizeof(single) * l.c, o);
    BlockRead(fp, l.rolling_mean[0], sizeof(single) * l.c, o);
    BlockRead(fp, l.rolling_variance[0], sizeof(single) * l.c, o);
  {$ifdef GPU}
    if gpu_index >= 0 then
        push_batchnorm_layer(l)
  {$endif}
end;

procedure load_convolutional_weights_binary(var l: TConvolutionalLayer; var fp: file);
var
    size: longint;
    i: longint;
    j: longint;
    k: longint;
    mean: single;
    index, o: longint;
    c : byte;
begin
    BlockRead(fp, l.biases, sizeof(single) * l.n, o);
    if l.batch_normalize and (not l.dontloadscales) then
        begin
            BlockRead(fp, l.scales[0], sizeof(single) * l.n, o);
            BlockRead(fp, l.rolling_mean[0], sizeof(single) * l.n, o);
            BlockRead(fp, l.rolling_variance[0], sizeof(single) * l.n, o)
        end;
    size := (l.c div l.groups) * l.size * l.size;
    for i := 0 to l.n -1 do
        begin
            mean := 0;
            BlockRead(fp, mean, sizeof(single)* 1, o);
            for j := 0 to size div 8 -1 do
                begin
                    index := i * size+j * 8;
                    c := 0;
                    BlockRead(fp, c, sizeof(byte)* 1, o);
                    for k := 0 to 8 -1 do
                        begin
                            if j * 8+k >= size then
                                break;
                            if (c and 1 shl k<>0) then
                                l.weights[index+k] := mean
                            else
                                l.weights[index+k] := -mean
                        end
                end
        end;
  {$ifdef GPU}
    if gpu_index >= 0 then
        push_convolutional_layer(l)
  {$endif}
end;

procedure load_convolutional_weights(var l: TConvolutionalLayer; var fp: file);
var
    num: longint;
    read_bytes: longint;
    i: longint;
begin
    if l.binary then
      ;
    num := l.nweights;
    BlockRead(fp, l.biases[0], sizeof(single)* l.n, read_bytes);
    if (read_bytes > 0) and (read_bytes < l.n) then    // todo [load_convolutional_weights] Read_Bytes should be divided by size of(single)?
        writeln(format(#10' Warning: Unexpected end of wights-file! l.biases - l.index = %d', [l.index]));
    if l.batch_normalize and (not l.dontloadscales) then
        begin
            BlockRead(fp, l.scales[0], sizeof(single) * l.n, read_bytes);
            if (read_bytes > 0) and (read_bytes < l.n) then
                writeln(format(#10' Warning: Unexpected end of wights-file! l.scales - l.index = %d', [l.index]));
            BlockRead(fp, l.rolling_mean[0], sizeof(single) * l.n, read_bytes);
            if (read_bytes > 0) and (read_bytes < l.n) then
                writeln(format(#10' Warning: Unexpected end of wights-file! l.rolling_mean - l.index = %d', [l.index]));
            BlockRead(fp, l.rolling_variance[0], sizeof(single) * l.n, read_bytes);
            if (read_bytes > 0) and (read_bytes < l.n) then
                writeln(format(#10' Warning: Unexpected end of wights-file! l.rolling_variance - l.index = %d', [l.index]));
            if false then
                begin
                    for i := 0 to l.n -1 do
                        write(format('%g, ', [l.rolling_mean[i]]));
                    writeln('');
                    for i := 0 to l.n -1 do
                        write(format('%g, ', [l.rolling_variance[i]]));
                    writeln('')
                end;
            if false then
                begin
                    fill_cpu(l.n, 0, l.rolling_mean, 1);
                    fill_cpu(l.n, 0, l.rolling_variance, 1)
                end
        end;
    BlockRead(fp, l.weights[0], sizeof(single) * num, read_bytes);
    if (read_bytes > 0) and (read_bytes < l.n) then
        writeln(format(#10' Warning: Unexpected end of wights-file! l.weights - l.index = %d', [l.index]));
    if l.flipped then
        transpose_matrix(l.weights, (l.c div l.groups) * l.size * l.size, l.n);
 {$ifdef GPU}
    if gpu_index >= 0 then
        push_convolutional_layer(l)
 {$endif}
end;

procedure load_shortcut_weights(var l: TShortcutLayer; var fp: file);
var
    num: longint;
    read_bytes: longint;
begin
    num := l.nweights;
    BlockRead(fp, l.weights[0], sizeof(single) * num, read_bytes);
    if (read_bytes > 0) and (read_bytes < num) then
        writeln(format(#10' Warning: Unexpected end of wights-file! l.weights - l.index = %d ', [l.index]));
  {$ifdef GPU}
    if gpu_index >= 0 then
        push_shortcut_layer(l)
  {$endif}
end;

procedure load_implicit_weights(var l: TImplicitLayer; var fp: file);
var
    num: longint;
    read_bytes: longint;
begin
    num := l.nweights;
    BlockRead(fp, l.weights[0], sizeof(single)* num, read_bytes);
    if (read_bytes > 0) and (read_bytes < num) then
        WriteLn(format(#10' Warning: Unexpected end of wights-file! l.weights - l.index = %d', [l.index]));
  {$ifdef GPU}
    if gpu_index >= 0 then
        push_implicit_layer(l)
  {$endif}
end;

procedure load_weights_upto(const net: PNetwork; const filename: string; const cutoff: longint);
var
    fp: file;
    major: int32;
    minor: int32;
    revision: int32;
    iseen: uint64;
    transpose: boolean;
    i: longint;
    l: TLayer;
    locations: longint;
    size, o: longint;
begin
  {$ifdef GPU}
    if net.gpu_index >= 0 then
        cuda_set_device(net.gpu_index);
  {$endif}
    write(ErrOutput, 'Loading weights from ', filename,'...');
    assert(fileExists(FileName),'File not fount :'+ filename);
    assignfile(fp, filename);
    reset(fp,1);
    BlockRead(fp, major, sizeof(int32) * 1, o);
    BlockRead(fp, minor, sizeof(int32) * 1, o);
    BlockRead(fp, revision, sizeof(int32) * 1, o);
    if (major * 10+minor) >= 2 then
        begin
            write(#10' seen size = 64');
            iseen := 0;
            BlockRead(fp, iseen, sizeof(uint64)* 1, o);
            net.seen[0] := iseen
        end
    else
        begin
            write(#10' seen size = 32');
            iseen := 0;
            BlockRead(fp, iseen, sizeof(uint32)* 1, o);
            net.seen[0] := iseen
        end;
    net.cur_iteration[0] := get_current_batch(net[0]);
    writeln(format(', trained: %.0f K-images (%.0f Kilo-batches_64) ', [net.seen[0] / 1000, net.seen[0] / 64000]));
    transpose := (major > 1000) or (minor > 1000);
    i := 0;
    while (i < net.n) and (i < cutoff) do begin
        l := net.layers[i];
        if l.dontload then
            continue;
        if (l.&type = ltCONVOLUTIONAL) and (l.share_layer = nil) then
            load_convolutional_weights(l, fp);
        if (l.&type = ltSHORTCUT) and (l.nweights > 0) then
            load_shortcut_weights(l, fp);
        if l.&type = ltIMPLICIT then
            load_implicit_weights(l, fp);
        if l.&type = ltCONNECTED then
            load_connected_weights(l, fp, transpose);
        if l.&type = ltBATCHNORM then
            load_batchnorm_weights(l, fp);
        if l.&type = ltCRNN then
            begin
                load_convolutional_weights(l.input_layer[0], fp);
                load_convolutional_weights(l.self_layer[0], fp);
                load_convolutional_weights(l.output_layer[0], fp)
            end;
        if l.&type = ltRNN then
            begin
                load_connected_weights(l.input_layer[0], fp, transpose);
                load_connected_weights(l.self_layer[0], fp, transpose);
                load_connected_weights(l.output_layer[0], fp, transpose)
            end;
        if l.&type = ltGRU then
            begin
                load_connected_weights(l.wz[0], fp, transpose);
                load_connected_weights(l.wr[0], fp, transpose);
                load_connected_weights(l.wh[0], fp, transpose);
                load_connected_weights(l.uz[0], fp, transpose);
                load_connected_weights(l.ur[0], fp, transpose);
                load_connected_weights(l.uh[0], fp, transpose)
            end;
        if l.&type = ltLSTM then
            begin
                load_connected_weights(l.wf[0], fp, transpose);
                load_connected_weights(l.wi[0], fp, transpose);
                load_connected_weights(l.wg[0], fp, transpose);
                load_connected_weights(l.wo[0], fp, transpose);
                load_connected_weights(l.uf[0], fp, transpose);
                load_connected_weights(l.ui[0], fp, transpose);
                load_connected_weights(l.ug[0], fp, transpose);
                load_connected_weights(l.uo[0], fp, transpose)
            end;
        if l.&type = ltConvLSTM then
            begin
                if l.peephole then
                    begin
                        load_convolutional_weights(l.vf[0], fp);
                        load_convolutional_weights(l.vi[0], fp);
                        load_convolutional_weights(l.vo[0], fp)
                    end;
                load_convolutional_weights(l.wf[0], fp);
                if not l.bottleneck then
                    begin
                        load_convolutional_weights(l.wi[0], fp);
                        load_convolutional_weights(l.wg[0], fp);
                        load_convolutional_weights(l.wo[0], fp)
                    end;
                load_convolutional_weights(l.uf[0], fp);
                load_convolutional_weights(l.ui[0], fp);
                load_convolutional_weights(l.ug[0], fp);
                load_convolutional_weights(l.uo[0], fp)
            end;
        if l.&type = ltLOCAL then
            begin
                locations := l.out_w * l.out_h;
                size := l.size * l.size * l.c * l.n * locations;
                BlockRead(fp, l.biases, sizeof(Single)* l.outputs, o);
                BlockRead(fp, l.weights, sizeof(Single)* size, o);
            {$ifdef GPU}
                if gpu_index >= 0 then
                    push_local_layer(l)
            {$endif}
            end;
        if eof(fp) then
            break;
        inc(i)
    end;
    writeln(ErrOutput, format('Done! Loaded %d layers from weights-file ', [i]));
    closeFile(fp)
end;

procedure load_weights(const net: PNetwork; const filename: string);
begin
    load_weights_upto(net, filename, net.n)
end;

function load_network_custom(cfg: string; weights: string; clear: boolean; const batch: longint):TArray<TNetwork>;
begin
    writeln(format(' Try to load cfg: %s, weights: %s, clear = %d ', [cfg, weights, longint(clear)]));
    setLength(result, 1);
    result[0] := parse_network_cfg_custom(cfg, batch, 1);
    if weights<>'' then
        begin
            writeln(' Try to load weights: ', weights);
            load_weights(@result[0], weights)
        end;
    fuse_conv_batchnorm(result[0]);
    if clear then
        begin
            result[0].seen[0] := 0;
            result[0].cur_iteration[0] := 0
        end;
end;

function load_network(const cfg, weights: string; const clear: boolean):TArray<TNetwork>;
begin
    writeln(format(' Try to load cfg: %s, clear = %d ', [cfg, longint(clear)]));
    setLength(result ,1);
    result[0] := parse_network_cfg(cfg);
    if weights<>'' then
        begin
            writeln(format(' Try to load weights: %s ', [weights]));
            load_weights(@result[0], weights)
        end;
    if clear then
        begin
            result[0].seen[0] := 0;
            result[0].cur_iteration[0] := 0
        end;
end;

function read_data_cfg(const filename: string): TCFGSection;
var
    f: TextFile;
    line: string;
    nu: longint;
begin
    if not FileExists(filename) then
        raise EFileNotFoundException.Create(filename+': was not found');
    result:=default(TCFGSection);
    AssignFile(f , filename);
    reset(f);
    nu := 0;
    result := Default(TCFGSection);//make_list();
    while not EOF(f) do
        begin
            readln(f,line);
            inc(nu);
            line:=trim(line);
            if (line='') or (line[1] in ['#', ';']) then
                  continue
            else
                if not result.addOptionLine(line) then
                    begin
                        writeln(ErrOutput, format('Config file error line %d, could parse: %s', [nu, line]));
                        //free(line)
                    end
        end;
    closeFile(f);
end;

function get_metadata(const filename: string):TMetadata;
var
    //m: TMetadata;
    options: TCFGSection;
    name_list: string;
begin
    result := default(TMetadata);
    options := read_data_cfg(filename);
    name_list := options.getStr('names', '');
    if name_list='' then
        name_list := options.getStr('labels', '');
    if name_list='' then
        writeln(ErrOutput, 'No names or labels found')
    else
        result.names := get_labels(name_list);
    result.classes := options.getInt('classes', 2);
    //free_list(options);
    if(name_list<>'') then begin
        writeln(format('Loaded - names_list: %s, classes = %d ', [name_list, result.classes]));
    end;

    //exit(result)
end;


end.

