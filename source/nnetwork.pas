unit nnetwork;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}
{$PointerMath on}
{$WriteableConst on}

interface
uses
  Classes, SysUtils, lightnet, data, matrix, imagedata, blas
  , AvgPoolLayer
  , BatchNormLayer
  , ConnectedLayer
  , ConvolutionalLayer
  , ConvLSTMLayer
  , CRNNLayer
  , CropLayer
  , CostLayer
  , DetectionLayer
  , DropoutLayer
  , GaussianYoloLayer
  , MaxpoolLayer
  , NormalizationLayer
  , RegionLayer
  , RouteLayer
  , ShortcutLayer
  , ReOrgLayer
  , ReOrgOldLayer
  , SAMLayer
  , ScaleChannelLayer
  , UpSampleLayer
  , YoloLayer
  ;

type

  PTrainArgs = ^TTrainArgs;
  TTrainArgs = record
      net : PNetwork;
      d   : TData ;
      err : TSingles;
  end;

  PSyncArgs = ^TSyncArgs;
  TSyncArgs = record
      nets : PPNetwork;
      n,j  :longint;
  end;

function get_base_args(const net: TNetwork):TLoadArgs;
function get_current_iteration(const net: TNetwork):int64;
function get_current_batch(const net: TNetwork):longint;
procedure reset_network_state(const net: PNetwork; const b: longint);
procedure reset_rnn(const net: Pnetwork);
function get_current_seq_subdivisions(const net: TNetwork):single;
function get_sequence_value(const net: TNetwork):longint;
function get_current_rate(const net: TNetwork):single;
function get_layer_string(const a: TLayerType):string;
function make_network(const n: longint):TNetwork;
procedure forward_network(const net: TNetwork; state: TNetworkState);
procedure update_network(const net: TNetwork);
function get_network_output(const net: TNetwork):PSingle;
function get_network_cost(const net: TNetwork):single;
function get_predicted_class_network(const net: TNetwork):longint;
procedure backward_network(const net: TNetwork; state: TNetworkState);
function train_network_datum(const net: TNetwork; x: Psingle; y: Psingle):single;
function train_network_sgd(net: TNetwork; const d: TData; const n: longint):single;
function train_network(const net: TNetwork; const d: TData):single;
function train_network_waitkey(net: TNetwork; const d: TData; const wait_key: boolean): single;
function train_network_batch(const net: TNetwork; const d: TData; const n: longint):single;
function recalculate_workspace_size(net: Pnetwork):longint;
procedure set_batch_network(net: Pnetwork; b: longint);
function resize_network(net: PNetwork; w: longint; h: longint):longint;
function get_network_output_size(const net: TNetwork):longint;
function get_network_input_size(const net: TNetwork):longint;
function get_network_detection_layer(const net: TNetwork):TDetectionLayer;
function get_network_image_layer(const net: TNetwork; i: longint):TImageData;
function get_network_layer(const net: PNetwork; const i: longint):PLayer;
function get_network_image(const net: TNetwork):TImageData;
procedure visualize_network(const net: TNetwork);
procedure top_predictions(const net: TNetwork; const k: longint; const index: Plongint);
function network_predict(const net: TNetwork; const input: PSingle):PSingle;  overload;
function network_predict(const net: TNetwork; const input: TArray<Single>):PSingle; overload;
function network_predict_ptr(const net: Pnetwork; const input: PSingle):PSingle;
{$ifdef CUDA_OPENGL_INTEGRATION}
function network_predict_gl_texture(net: Pnetwork; texture_id: uint32_t):PSingle;
{$endif}
function num_detections(const net: PNetwork; const thresh: single):longint;
function num_detections_batch(const net: PNetwork; const thresh: single; const batch: longint):longint;
function make_network_boxes(const net: PNetwork; const thresh: single; const num: PLongint):TArray<TDetection>;
function make_network_boxes_batch(const net: PNetwork; const thresh: single; const num: Plongint; const batch: longint):TArray<TDetection>;
procedure custom_get_region_detections(const l: PRegionLayer; const w, h, net_w, net_h: longint; const thresh: single; const map: Plongint; const hier: single; const relative: boolean; const dets: PDetection; const letter: boolean);
procedure fill_network_boxes(const net: Pnetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; const relative: boolean; dets: PDetection; const letter: boolean);
procedure fill_network_boxes_batch(const net: Pnetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; const relative: boolean; dets: PDetection; const letter: boolean; const batch: longint);
function get_network_boxes(const net: PNetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; relative: boolean; const num: Plongint; const letter: boolean):TArray<TDetection>;
procedure free_detections(dets: Pdetection; n: longint);
procedure free_batch_detections(det_num_pairs: PDetNumPair; const n: longint);
function detection_to_json(const dets: PDetection; const nboxes: longint; const classes: longint; const names: TArray<string>; const frame_id: int64; const filename: string):string;
function network_predict_image(const net: Pnetwork; const im: TImageData):PSingle;
function network_predict_batch(const net: PNetwork; const im: TImageData; const batch_size, w, h: longint; const thresh, hier: single; const map: Plongint; const relative, letter: boolean):TArray<TDetNumPair>;
function network_predict_image_letterbox(const net: PNetwork; const im: TImageData):PSingle;
function network_height(net: Pnetwork):longint;
function network_predict_data_multi(const net: TNetwork; const test: TData; const n: longint):TMatrix;
function network_predict_data(const net: TNetwork; const test: TData):TMatrix;
procedure print_network(const net: TNetwork);
procedure compare_networks(const n1, n2: TNetwork; const test: TData);
function network_accuracy(const net: TNetwork; const d: TData):single;
function network_accuracies(const net: TNetwork; const d: TData; const n: longint):PSingle;
function network_accuracy_multi(const net: TNetwork; const d: TData; const n: longint):single;
procedure free_network(net: TNetwork);
procedure free_network_ptr(const net: Pnetwork);
function relu(src: single):single;
function lrelu(src: single):single;
procedure fuse_conv_batchnorm(const net: TNetwork);
procedure forward_blank_layer(var l: TLayer; const state: PNetworkState);
procedure calculate_binary_weights(const net: TNetwork);
procedure copy_cudnn_descriptors(const src: TLayer; const dst: Player);
procedure copy_weights_net(const net_train: TNetwork; const net_map: PNetwork);
function combine_train_valid_networks(const net_train: TNetwork; const net_map: TNetwork):TNetwork;
procedure free_network_recurrent_state(const net: TNetwork);
procedure randomize_network_recurrent_state(const net: TNetwork);
procedure remember_network_recurrent_state(const net: TNetwork);
procedure restore_network_recurrent_state(const net: TNetwork);
function is_ema_initialized(const net: TNetwork):boolean;
procedure ema_update(const net: TNetwork; const ema_alpha: single);
procedure ema_apply(const net: TNetwork);
procedure reject_similar_weights(const net: TNetwork; const sim_threshold: single);
{$ifdef MSWINDOWS}
const VK_ESCAPE = $1B;
function GetKeyState(nVirtKey: Integer): Smallint; winapi;external 'kernel32.dll';
{$endif}

implementation
uses math ;

function get_base_args(const net: TNetwork):TLoadArgs;
begin
    result := Default(TLoadArgs);
    result.w := net.w;
    result.h := net.h;
    result.size := net.w;
    result.min := net.min_crop;
    result.max := net.max_crop;
    result.angle := net.angle;
    result.aspect := net.aspect;
    result.exposure := net.exposure;
    result.center := net.center;
    result.saturation := net.saturation;
    result.hue := net.hue;
end;

function get_current_iteration(const net: TNetwork):int64;
begin
    exit(net.cur_iteration[0])
end;

function get_current_batch(const net: TNetwork):longint;
begin
    result := net.seen[0] div (net.batch * net.subdivisions);
end;

procedure reset_network_state(const net: PNetwork; const b: longint);
var
    i: longint;
    l: TLayer;
begin
    {$ifdef GPU}
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.state_gpu then
                fill_ongpu(l.outputs, 0, l.state_gpu+l.outputs * b, 1);
            if l.h_gpu then
                fill_ongpu(l.outputs, 0, l.h_gpu+l.outputs * b, 1)
        end
    {$endif}
end;

procedure reset_rnn(const net: Pnetwork);
begin
    reset_network_state(net, 0)
end;

function get_current_seq_subdivisions(const net: TNetwork):single;
var
    sequence_subdivisions: longint;
    batch_num: longint;
    i: longint;
begin
    sequence_subdivisions := net.init_sequential_subdivisions;
    if net.num_steps > 0 then
        begin
            batch_num := get_current_batch(net);
            for i := 0 to net.num_steps -1 do
                begin
                    if net.steps[i] > batch_num then
                        break;
                    sequence_subdivisions := trunc(sequence_subdivisions * net.seq_scales[i])
                end
        end;
    if sequence_subdivisions < 1 then
        sequence_subdivisions := 1;
    if sequence_subdivisions > net.subdivisions then
        sequence_subdivisions := net.subdivisions;
    exit(sequence_subdivisions)
end;

function get_sequence_value(const net: TNetwork):longint;
begin
    result := 1;
    if net.sequential_subdivisions <> 0 then
        result := net.subdivisions div net.sequential_subdivisions;
    if result < 1 then
        result := 1;
    exit(result)
end;

function get_current_rate(const net: TNetwork):single;
var
    batch_num, i, last_iteration_start, cycle_size: longint;
    rate: single;
begin
    batch_num := get_current_batch(net);
    if batch_num < net.burn_in then
        exit(net.learning_rate * power(batch_num / net.burn_in, net.power));
    case net.policy of
        lrpCONSTANT:
            exit(net.learning_rate);
        lrpSTEP:
            exit(net.learning_rate * power(net.scale, batch_num div net.step));
        lrpSTEPS:
            begin
                rate := net.learning_rate;
                for i := 0 to net.num_steps -1 do
                    begin
                        if net.steps[i] > batch_num then
                            exit(rate);
                        rate := rate * net.scales[i]
                    end;
                exit(rate)
            end;
        lrpEXP:
            exit(net.learning_rate * power(net.gamma, batch_num));
        lrpPOLY:
            exit(net.learning_rate * power(1-batch_num / net.max_batches, net.power));
        lrpRANDOM:
            exit(net.learning_rate * power(rand_uniform(0, 1), net.power));
        lrpSIG:
            exit(net.learning_rate * (1.0 / (1.0+exp(net.gamma * (batch_num-net.step)))));
        lrpSGDR:
            begin
                last_iteration_start := 0;
                cycle_size := net.batches_per_cycle;
                while ((last_iteration_start+cycle_size) < batch_num) do
                    begin
                        last_iteration_start := last_iteration_start + cycle_size;
                        cycle_size := cycle_size * net.batches_cycle_mult
                    end;
                rate := net.learning_rate_min+0.5 * (net.learning_rate-net.learning_rate_min) * (1.0+cos((batch_num-last_iteration_start) * {3.14159265} Pi() / cycle_size));
                exit(rate)
            end
        else
            begin
                writeln(ErrOutput, 'Policy is weird!');
                exit(net.learning_rate)
            end
    end
end;

function get_layer_string(const a: TLayerType):string;
begin
    case a of
        ltCONVOLUTIONAL:
            exit('convolutional');
        ltACTIVE:
            exit('activation');
        ltLOCAL:
            exit('local');
        ltDECONVOLUTIONAL:
            exit('deconvolutional');
        ltCONNECTED:
            exit('connected');
        ltRNN:
            exit('rnn');
        ltGRU:
            exit('gru');
        ltLSTM:
            exit('lstm');
        ltCRNN:
            exit('crnn');
        ltMAXPOOL:
            exit('maxpool');
        ltREORG:
            exit('reorg');
        ltAVGPOOL:
            exit('avgpool');
        ltSOFTMAX:
            exit('softmax');
        ltDETECTION:
            exit('detection');
        ltREGION:
            exit('region');
        ltYOLO:
            exit('yolo');
        ltGaussianYOLO:
            exit('Gaussian_yolo');
        ltDROPOUT:
            exit('dropout');
        ltCROP:
            exit('crop');
        ltCOST:
            exit('cost');
        ltROUTE:
            exit('route');
        ltSHORTCUT:
            exit('shortcut');
        ltScaleChannels:
            exit('scale_channels');
        ltSAM:
            exit('sam');
        ltNORMALIZATION:
            exit('normalization');
        ltBATCHNORM:
            exit('batchnorm');
        else

    end;
    exit('none')
end;

function make_network(const n: longint):TNetwork;
begin
    result := default(TNetwork);
    result.n := n;
    setLength(result.layers, result.n);// := layer(xcalloc(result.n, sizeof(layer)));
    setLength(result.seen, 1);// := uint64_t(xcalloc(1, sizeof(uint64_t)));
    setLength(result.badlabels_reject_threshold, 1);// := single(xcalloc(1, sizeof(float)));
    setLength(result.delta_rolling_max, 1);// := single(xcalloc(1, sizeof(float)));
    setLength(result.delta_rolling_avg, 1);// := single(xcalloc(1, sizeof(float)));
    setLength(result.delta_rolling_std, 1);// := single(xcalloc(1, sizeof(float)));
    setLength(result.cur_iteration, 1);// := longint(xcalloc(1, sizeof(int)));
    setLength(result.total_bbox, 1);// := longint(xcalloc(1, sizeof(int)));
    setLength(result.rewritten_bbox, 1);// := longint(xcalloc(1, sizeof(int)));
    result.rewritten_bbox[0] :=0;
    result.total_bbox[0] := 0;
{$ifdef GPU}
    result.cuda_graph_ready := TIntegers.Create(1);//longint(xcalloc(1, sizeof(int)));
    result.input_gpu := single(xcalloc(1, sizeof(float * )));
    result.truth_gpu := single(xcalloc(1, sizeof(float * )));
    result.input16_gpu := single(xcalloc(1, sizeof(float * )));
    result.output16_gpu := single(xcalloc(1, sizeof(float * )));
    result.max_input16_size := size_t(xcalloc(1, sizeof(size_t)));
    result.max_output16_size := size_t(xcalloc(1, sizeof(size_t)));
{$endif}

end;

procedure forward_network(const net: TNetwork; state: TNetworkState);
var
    i: longint;
    l: PLayer;
begin
    {$ifdef USE_TELEMETRY}metrics := Default(TMetrics);{$endif}
    state.workspace := Pointer(net.workspace);
    for i := 0 to net.n -1 do
        begin
            state.index := i;
            l := @net.layers[i];
            if assigned(l.delta) and state.train and l.train then
                scal_cpu(l.outputs * l.batch, 0, l.delta, 1);
            l.forward(l[0], @state);
            if assigned(net.onForward) then
              net.onForward(i, @net);
            state.input := l.output
        end
end;

procedure update_network(const net: TNetwork);
var
    i: longint;
    update_batch: longint;
    rate: single;
    l: PLayer;
    arg:TUpdateArgs;
begin
    update_batch := net.batch * net.subdivisions;
    rate := get_current_rate(net);
    for i := 0 to net.n -1 do
        begin
            l := @net.layers[i];
            if not l.train then
                continue;
            if assigned(l.update) then begin
                arg.batch:=update_batch;
                arg.learning_rate:=rate;
                arg.momentum:=net.momentum;
                arg.decay:=net.decay;
                l.update(l[0], arg)
            end;
        end
end;

function get_network_output(const net: TNetwork):PSingle;
var
    i: longint;
begin
    result:=nil;
    {$ifdef GPU}
    if gpu_index >= 0 then
        exit(get_network_output_gpu(net));
    {$endif}
    for i:= net.n-1 downto 1 do begin
        if net.layers[i].&type <> ltCOST then
            exit(net.layers[i].output)
    end;
end;

function get_network_cost(const net: TNetwork):single;
var
    i, count: longint;
    sum: single;
begin
    sum := 0;
    count := 0;
    for i := 0 to net.n -1 do
//        if assigned(net.layers[i].cost) then
            begin
                sum := sum + net.layers[i].cost;
                inc(count)
            end;
    exit(sum / count)
end;

function get_predicted_class_network(const net: TNetwork):longint;
var
    &out: PSingle;
    k: longint;
begin
    &out := get_network_output(net);
    k := get_network_output_size(net);
    exit(max_index(&out, k))
end;

procedure backward_network(const net: TNetwork; state: TNetworkState);
var
    i: longint;
    original_input, original_delta: PSingle;
    prev: PLayer;
    l: PLayer;
begin
    original_input := state.input;
    original_delta := state.delta;
    state.workspace := @net.workspace[0];
    for i:= net.n-1 downto 0 do begin
        state.index := i;
        if i = 0 then
            begin
                state.input := original_input;
                state.delta := original_delta
            end
        else
            begin
                prev := @net.layers[i-1];
                state.input := prev.output;
                state.delta := prev.delta
            end;
        l := @net.layers[i];
        if l.stopbackward then
            break;
        if l.onlyforward then
            continue;
        l.backward(l[0], @state);
        if assigned(net.onBackward) then
          net.onBackward(i, @net);
    end
end;

function train_network_datum(const net: TNetwork; x: Psingle; y: Psingle):single;
var
    state: TNetworkState;
    error: single;
begin
  {$ifdef GPU}
    if gpu_index >= 0 then
        exit(train_network_datum_gpu(net, x, y));
  {$endif}
    state := default(TNetworkState);
    net.seen[0] :=  net.seen[0] + net.batch;
    state.index := 0;
    state.net := @net;
    state.input := x;
    state.delta := nil;
    state.truth := y;
    state.train := true;
    forward_network(net, state);
    backward_network(net, state);
    error := get_network_cost(net);
    if  state.net.total_bbox[0] > 0 then
        writeln(ErrOutput, format(' total_bbox = %d, rewritten_bbox = %f %% '#10'',  [state.net.total_bbox[0], 100 * state.net.rewritten_bbox[0] /  state.net.total_bbox[0]]));
    exit(error)
end;

function train_network_sgd(net: TNetwork; const d: TData; const n: longint):single;
var
    batch, i: longint;
    sum, err: single;
    X, y: TArray<Single>;
begin
    batch := net.batch;
    setLength(X, batch * d.X.cols);// := single(xcalloc(batch * d.X.cols, sizeof(float)));
    setLength(y, batch * d.y.cols);// := single(xcalloc(batch * d.y.cols, sizeof(float)));
    sum := 0;
    for i := 0 to n -1 do
        begin
            get_random_batch(d, batch, @X[0], @y[0]);
            net.current_subdivision := i;
            err := train_network_datum(net, @X[0], @y[0]);
            sum := sum + err
        end;
    //free(X);
    //free(y);
    exit(sum / (n * batch))
end;

function train_network(const net: TNetwork; const d: TData):single;
begin
    result := train_network_waitkey(net, d, false)
end;

function train_network_waitkey(net: TNetwork; const d: TData; const wait_key: boolean): single;
var
    batch, n, i: longint;
    sum, err: single;
    ema_start_point, ema_period, ema_apply_point, reject_stop_point: longint;
    sim_threshold: single;
    X,y : TArray<single>;
begin
    assert(d.X.rows mod net.batch = 0);
    batch := net.batch;
    n := d.X.rows div batch;
    setLength(X, batch * d.X.cols);//:= single(xcalloc(batch * d.X.cols, sizeof(float)));
    setLength(y, batch * d.y.cols);//:= single(xcalloc(batch * d.y.cols, sizeof(float)));
    sum := 0;
    for i := 0 to n -1 do
        begin
            get_next_batch(d, batch, i * batch, @X[0], @y[0]);
            net.current_subdivision := i;
            err := train_network_datum(net, @X[0], @y[0]);
            sum := sum + err;
            if wait_key then begin
                sleep(5);
                {$ifdef MSWINDOWS}
                if GetKeyState(VK_ESCAPE)<0 then break;
                {$endif}
            end;
                //wait_key_cv(5)
        end;
    inc(net.cur_iteration[0]);
{$ifdef GPU}
    update_network_gpu(net);
{$else}
    update_network(net);
{$endif}
    ema_start_point := net.max_batches div 2;
    if (net.ema_alpha<>0) and (net.cur_iteration[0] >= ema_start_point) then
        begin
            ema_period := trunc( (net.max_batches-ema_start_point-1000) * (1.0-net.ema_alpha));
            ema_apply_point := net.max_batches-1000;
            if not is_ema_initialized(net) then
                begin
                    ema_update(net, 0);
                    writeln(' EMA initialization ')
                end;
            if net.cur_iteration[0] = ema_apply_point then
                begin
                    ema_apply(net);
                    writeln(' ema_apply() ')
                end
            else
                if net.cur_iteration[0] < ema_apply_point then
                    begin
                        ema_update(net, net.ema_alpha);
                        writeln(format(' ema_update(), ema_alpha = %f ', [net.ema_alpha]))
                    end
        end;
    reject_stop_point := net.max_batches * 3 div 4;
    if (net.cur_iteration[0] < reject_stop_point) and (net.weights_reject_freq<>0) and (net.cur_iteration[0] mod net.weights_reject_freq = 0) then
        begin
            sim_threshold := 0.4;
            reject_similar_weights(net, sim_threshold)
        end;
    //free(X);
    //free(y);
    exit(sum / (n * batch))
end;

function train_network_batch(const net: TNetwork; const d: TData; const n: longint):single;
var
    i, j, batch, index: longint;
    state: TNetworkState;
    sum: single;
begin
    state := default(TNetworkState);
    state.index := 0;
    state.net := @net;
    state.train := true;
    state.delta := nil;
    sum := 0;
    batch := 2;
    for i := 0 to n -1 do
        begin
            for j := 0 to batch -1 do
                begin
                    index := random(d.X.rows);
                    state.input := @d.X.vals[index][0];
                    state.truth := @d.y.vals[index][0];
                    forward_network(net, state);
                    backward_network(net, state);
                    sum := sum + get_network_cost(net)
                end;
            update_network(net)
        end;
    exit(sum / (n * batch))
end;

function recalculate_workspace_size(net: Pnetwork):longint;
var
    i: longint;
    workspace_size: size_t;
    l: TLayer;
begin
{$ifdef GPU}
    cuda_set_device(net.gpu_index);
    if (gpu_index >= 0) then
        cuda_free(net.workspace);
{$endif}
    workspace_size := 0;
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                l.workspace_size := get_convolutional_workspace_size(l)
            else
                if l.&type = ltCONNECTED then
                    l.workspace_size := get_connected_workspace_size(l);
            if l.workspace_size > workspace_size then
                workspace_size := l.workspace_size;
            net.layers[i] := l
        end;
{$ifdef GPU}
    if gpu_index >= 0 then
        begin
            printf(''#10' try to allocate additional workspace_size = %1.2f MB '#10'', single(workspace_size) / 1000000);
            net.workspace := cuda_make_array(0, workspace_size div sizeof(float)+1);
            printf(' CUDA allocate done! '#10'')
        end
    else
        begin
            free(net.workspace);
            net.workspace := single(xcalloc(1, workspace_size))
        end;
{$else}
    //free(net.workspace);
    // todo check workspace size [div (sizeof(single)]
    setLength(net.workspace, workspace_size div sizeof(single));//:= single(xcalloc(1, workspace_size));
{$endif}
    exit(0)
end;

procedure set_batch_network(net: Pnetwork; b: longint);
var
    i: longint;
begin
    net.batch := b;
    for i := 0 to net.n -1 do
        begin
            net.layers[i].batch := b;
{$ifdef CUDNN}
            if net.layers[i].&type = ltCONVOLUTIONAL then
                cudnn_convolutional_setup(net.layers+i, cudnn_fastest, 0)
            else
                if net.layers[i].&type = ltMAXPOOL then
                    cudnn_maxpool_setup(net.layers+i)
{$endif}
        end;
    recalculate_workspace_size(net)
end;

function resize_network(net: PNetwork; w: longint; h: longint):longint;
var
    i: longint;
    inputs: longint;
    workspace_size: size_t;
    l: TLayer;
    size: longint;
begin
{$ifdef GPU}
    cuda_set_device(net.gpu_index);
    if (gpu_index >= 0) then
        begin
            cuda_free(net.workspace);
            if net.input_gpu then
                begin
                    cuda_free( * net.input_gpu);
                     * net.input_gpu := 0;
                    cuda_free( * net.truth_gpu);
                     * net.truth_gpu := 0
                end;
            if net.input_state_gpu then
                cuda_free(net.input_state_gpu);
            if net.input_pinned_cpu then
                begin
                    if net.input_pinned_cpu_flag then
                        cudaFreeHost(net.input_pinned_cpu)
                    else
                        free(net.input_pinned_cpu)
                end
        end;
{$endif}
    net.w := w;
    net.h := h;
    inputs := 0;
    workspace_size := 0;
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                resize_convolutional_layer( l, w, h)
            else
                if l.&type = ltCRNN then
                    resize_crnn_layer( l, w, h)
            else
                if l.&type = ltConvLSTM then
                    resize_conv_lstm_layer( l, w, h)
            else
                if l.&type = ltCROP then
                    resize_crop_layer( l, w, h)
            else
                if l.&type = ltMAXPOOL then
                    resize_maxpool_layer( l, w, h)
            else
                if l.&type = ltLOCAL_AVGPOOL then
                    resize_maxpool_layer( l, w, h)
            else
                if l.&type = ltBATCHNORM then
                    resize_batchnorm_layer( l, w, h)
            else
                if l.&type = ltREGION then
                    resize_region_layer( l, w, h)
            else
                if l.&type = ltYOLO then
                    resize_yolo_layer( l, w, h)
            else
                if l.&type = ltGaussianYOLO then
                    resize_gaussian_yolo_layer( l, w, h)
            else
                if l.&type = ltROUTE then
                    resize_route_layer( l, net)
            else
                if l.&type = ltSHORTCUT then
                    resize_shortcut_layer( l, w, h, net)
            else
                if l.&type = ltScaleChannels then
                    resize_scale_channels_layer( l, net)
            else
                if l.&type = ltSAM then
                    resize_sam_layer( l, w, h)
            else
                if l.&type = ltDROPOUT then
                    begin
                        resize_dropout_layer( l, inputs);
                        l.out_w :=w; l.w := w;
                        l.out_h :=h; l.h := h;
                        l.output := net.layers[i-1].output;
                        l.delta := net.layers[i-1].delta;
{$ifdef GPU}
                        l.output_gpu := net.layers[i-1].output_gpu;
                        l.delta_gpu := net.layers[i-1].delta_gpu
{$endif}
                    end
            else
                if l.&type = ltUPSAMPLE then
                    resize_upsample_layer( l, w, h)
            else
                if l.&type = ltREORG then
                    resize_reorg_layer( l, w, h)
            else
                if l.&type = ltREORG_OLD then
                    resize_reorg_old_layer( l, w, h)
            else
                if l.&type = ltAVGPOOL then
                    resize_avgpool_layer( l, w, h)
            else
                if l.&type = ltNORMALIZATION then
                    resize_normalization_layer( l, w, h)
            else
                if l.&type = ltCOST then
                    resize_cost_layer( l, inputs)
            else
                begin
                    writeln(ErrOutput, 'Resizing type ', longint(l.&type));
                    raise Exception.Create('Cannot resize this type of layer');
                    //error('Cannot resize this type of layer', DARKNET_LOC)
                end;
            if l.workspace_size > workspace_size then
                workspace_size := l.workspace_size;
            inputs := l.outputs;
            net.layers[i] := l;
            w := l.out_w;
            h := l.out_h
        end;
{$ifdef GPU}
    size := get_network_input_size(net[0]) * net.batch;
    if gpu_index >= 0 then
        begin
            writeln(format(' try to allocate additional workspace_size = %1.2f MB ', [workspace_size / 1000000]));
            net.workspace := cuda_make_array(0, workspace_size div sizeof(float)+1);
            net.input_state_gpu := cuda_make_array(0, size);
            if cudaSuccess = cudaHostAlloc( and net.input_pinned_cpu, size * sizeof(float), cudaHostRegisterMapped) then
                net.input_pinned_cpu_flag := 1
            else
                begin
                    cudaGetLastError();
                    net.input_pinned_cpu := single(xcalloc(size, sizeof(float)));
                    net.input_pinned_cpu_flag := 0
                end;
            printf(' CUDA allocate done! '#10'')
        end
    else
        begin
            free(net.workspace);
            net.workspace := single(xcalloc(1, workspace_size));
            if not net.input_pinned_cpu_flag then
                net.input_pinned_cpu := single(xrealloc(net.input_pinned_cpu, size * sizeof(float)))
        end;
{$else}
    //free(net.workspace);
    setlength(net.workspace, workspace_size div sizeof(single));
{$endif}
    exit(0)
end;

function get_network_output_size(const net: TNetwork):longint;
var
    i: longint;
begin
    //i := net.n-1;
    for i:=net.n-1 downto 1 do begin
    //while i > 0 do begin
        if net.layers[i].&type <> ltCOST then
            break;
        //dec(i)
    end;
    exit(net.layers[i].outputs)
end;

function get_network_input_size(const net: TNetwork):longint;
begin
    exit(net.layers[0].inputs)
end;

function get_network_detection_layer(const net: TNetwork):TDetectionLayer;
var
    i: longint;
    l: TDetectionLayer;
begin
    for i := 0 to net.n -1 do
        if net.layers[i].&type = ltDETECTION then
            exit(net.layers[i]);
    writeln(ErrOutput, 'Detection layer not found!!');
    l := default(TDetectionLayer);
    exit(l)
end;

function get_network_image_layer(const net: TNetwork; i: longint):TImageData;
var
    l: PLayer;
    def: TImageData;
begin
    l := @net.layers[i];
    if boolean(l.out_w) and boolean(l.out_h) and boolean(l.out_c) then
        exit(float_to_image(l.out_w, l.out_h, l.out_c, l.output));
    def := default(TImageData);
    exit(def)
end;

function get_network_layer(const net: PNetwork; const i: longint):PLayer;
begin
    exit(@net.layers[i])
end;

function get_network_image(const net: TNetwork):TImageData;
var
    i: longint;
    m, def: TImageData;
begin
    //i := net.n-1;
    for i:= net.n-1 downto 0 do begin //while (i >= 0) do begin
        m := get_network_image_layer(net, i);
        if m.h <> 0 then
            exit(m);
        //dec(i)
    end;
    def := default(TImageData);
    exit(def)
end;

procedure visualize_network(const net: TNetwork);
var
    prev: TArray<TImageData>;
    i: longint;
    buff: string;
    l: TLayer;
begin
    prev := nil;
    for i := 0 to net.n -1 do
        begin
            buff:='Layer '+intToStr(i);
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                prev := visualize_convolutional_layer(l, buff, prev)
        end
end;

procedure top_predictions(const net: TNetwork; const k: longint; const index: Plongint);
var
    size: longint;
    &out: PSingle;
begin
    size := get_network_output_size(net);
    &out := get_network_output(net);
    top_k(&out, size, k, index)
end;

function network_predict(const net: TNetwork; const input: PSingle):PSingle;  overload;
var
    state: TNetworkState;
begin
{$ifdef GPU}
    if gpu_index >= 0 then
        exit(network_predict_gpu(net, input));
{$endif}
    state := default(TNetworkState);
    state.net := @net;
    state.index := 0;
    state.input := input;
    state.truth := nil;
    state.train := false;
    state.delta := nil;
    forward_network(net, state);
    result := get_network_output(net);

end;

function network_predict(const net: TNetwork; const input: TArray<Single>):PSingle; overload;
begin
  network_predict(net, PSingle(input))
end;

function network_predict_ptr(const net: Pnetwork; const input: PSingle):PSingle;
begin
    exit(network_predict( net[0], input))
end;

{$ifdef CUDA_OPENGL_INTEGRATION}
function network_predict_gl_texture(net: Pnetwork; texture_id: uint32_t):PSingle;
begin
    if net.batch <> 1 then
        set_batch_network(net, 1);
    if gpu_index >= 0 then
        exit(network_predict_gpu_gl_texture( * net, texture_id));
    exit(NULL)
end;
{$endif}

function num_detections(const net: PNetwork; const thresh: single):longint;
var
    i, s: longint;
    l: PLayer;
begin
    s := 0;
    for i := 0 to net.n -1 do
        begin
            l := @net.layers[i];
            if l.&type = ltYOLO then
                s := s + yolo_num_detections(PYoloLayer(l), thresh);
            if l.&type = ltGaussianYOLO then
                s := s + gaussian_yolo_num_detections(PGaussianYoloLayer(l), thresh);
            if (l.&type = ltDETECTION) or (l.&type = ltREGION) then
                s := s + (l.w * l.h * l.n)
        end;
    exit(s)
end;

function num_detections_batch(const net: PNetwork; const thresh: single; const batch: longint):longint;
var
    i, s: longint;
    l: PYoloLayer;
begin
    s := 0;
    for i := 0 to net.n -1 do
        begin
            l := @net.layers[i];
            if l.&type = ltYOLO then
                s := s + yolo_num_detections_batch(l, thresh, batch);
            if (l.&type = ltDETECTION) or (l.&type = ltREGION) then
                s := s + (l.w * l.h * l.n)
        end;
    exit(s)
end;

function make_network_boxes(const net: PNetwork; const thresh: single; const num: PLongint):TArray<TDetection>;
var
    i: longint;
    l, l_tmp: PLayer;
    nboxes: longint;
begin
    l := @net.layers[net.n-1];
    for i := 0 to net.n -1 do
        begin
            l_tmp := @net.layers[i];
            if l_tmp.&type in [ltYOLO, ltGaussianYOLO, ltDETECTION, ltREGION] then
                begin
                    l := l_tmp;
                    break
                end
        end;
    nboxes := num_detections(net, thresh);
    if assigned(num) then
         num[0] := nboxes;
    setLength(result, nBoxes);// := detection(xcalloc(nboxes, sizeof(detection)));
    for i := 0 to nboxes -1 do
        begin
            setLength(result[i].prob, l.classes);
            if l.&type = ltGaussianYOLO then
                setLength(result[i].uc, 4)
            else
                result[i].uc := nil;
            if l.coords > 4 then
                setLength(result[i].mask, l.coords-4)
            else
                result[i].mask := nil;
            if assigned(l.embedding_output) then
                setLength(result[i].embeddings, l.embedding_size)
            else
                result[i].embeddings := nil;
            result[i].embedding_size := l.embedding_size
        end;
end;

function make_network_boxes_batch(const net: PNetwork; const thresh: single; const num: Plongint; const batch: longint):TArray<TDetection>;
var
    i, nboxes: longint;
    l, l_tmp: PLayer;
begin
    l := @net.layers[net.n-1];
    for i := 0 to net.n -1 do
        begin
            l_tmp := @net.layers[i];
            if (l_tmp.&type = ltYOLO) or (l_tmp.&type = ltGaussianYOLO) or (l_tmp.&type = ltDETECTION) or (l_tmp.&type = ltREGION) then
                begin
                    l := l_tmp;
                    break
                end
        end;
    nboxes := num_detections_batch(net, thresh, batch);
    assert(assigned(num));
    num[0] := nboxes;
    setLength(result, nboxes);
    for i := 0 to nboxes -1 do
        begin
            setLength(result[i].prob, l.classes);
            if l.&type = ltGaussianYOLO then
                setLength(result[i].uc, 4)
            else
                result[i].uc := nil;
            if l.coords > 4 then
                setLength(result[i].mask ,l.coords-4)
            else
                result[i].mask := nil;
            if assigned(l.embedding_output) then
                setLength(result[i].embeddings, l.embedding_size)
            else
                result[i].embeddings := nil;
            result[i].embedding_size := l.embedding_size
        end;

end;

procedure custom_get_region_detections(const l: PRegionLayer; const w, h,
  net_w, net_h: longint; const thresh: single; const map: Plongint;
  const hier: single; const relative: boolean; const dets: PDetection;
  const letter: boolean);
var
    probs: TArray<TArray<Single>>;
    i: longint;
    j: longint;
    highest_prob: single;
    boxes :TArray<TBox>;
begin
    setLength(boxes,l.w * l.h * l.n);
    setLength(probs, l.w * l.h * l.n, l.classes);
    //for j := 0 to l.w * l.h * l.n -1 do
    //    probs[j] := single(xcalloc(l.classes, sizeof(float)));
    get_region_boxes(l, 1, 1, thresh, probs, boxes, false, map);
    for j := 0 to l.w * l.h * l.n -1 do
        begin
            dets[j].classes := l.classes;
            dets[j].bbox := boxes[j];
            dets[j].objectness := 1;
            highest_prob := 0;
            dets[j].best_class_idx := -1;
            for i := 0 to l.classes -1 do
                begin
                    if probs[j][i] > highest_prob then
                        begin
                            highest_prob := probs[j][i];
                            dets[j].best_class_idx := i
                        end;
                    dets[j].prob[i] := probs[j][i]
                end
        end;
    //free(boxes);
    //free_ptrs(PPointer(probs), l.w * l.h * l.n);
    correct_yolo_boxes(dets, l.w * l.h * l.n, w, h, net_w, net_h, relative, letter)
end;

procedure fill_network_boxes(const net: Pnetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; const relative: boolean; dets: PDetection; const letter: boolean);
var
    prev_classes, j: longint;
    l: PLayer;
    count: longint;
begin
    prev_classes := -1;
    for j := 0 to net.n -1 do
        begin
            l := @net.layers[j];
            if l.&type = ltYOLO then
                begin
                    count := get_yolo_detections(PYoloLayer(l), w, h, net.w, net.h, thresh, map, relative, dets, letter);
                    dets := dets + count;
                    if prev_classes < 0 then
                        prev_classes := l.classes
                    else
                        if prev_classes <> l.classes then
                            writeln(format(' Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! ', [prev_classes, l.classes]))
                end;
            if l.&type = ltGaussianYOLO then
                begin
                    count := get_gaussian_yolo_detections(PGaussianYoloLayer(l), w, h, net.w, net.h, thresh, map, relative, dets, letter);
                    dets := dets + count
                end;
            if l.&type = ltREGION then
                begin
                    custom_get_region_detections(PRegionLayer(l), w, h, net.w, net.h, thresh, map, hier, relative, dets, letter);
                    dets := dets + (l.w * l.h * l.n)
                end;
            if l.&type = ltDETECTION then
                begin
                    get_detection_detections(PDetectionLayer(l), w, h, thresh, dets);
                    dets := dets + (l.w * l.h * l.n)
                end
        end
end;

procedure fill_network_boxes_batch(const net: Pnetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; const relative: boolean; dets: PDetection; const letter: boolean; const batch: longint);
var
    prev_classes, j: longint;
    l: PLayer;
    count: longint;
begin
    prev_classes := -1;
    for j := 0 to net.n -1 do
        begin
            l := @net.layers[j];
            if l.&type = ltYOLO then
                begin
                    count := get_yolo_detections_batch(PYoloLayer(l), w, h, net.w, net.h, thresh, map, relative, dets, letter, batch);
                    dets := dets + count;
                    if prev_classes < 0 then
                        prev_classes := l.classes
                    else
                        if prev_classes <> l.classes then
                            writeln(format(' Error: Different [yolo] layers have different number of classes = %d and %d - check your cfg-file! ', [prev_classes, l.classes]))
                end;
            if l.&type = ltREGION then
                begin
                    custom_get_region_detections(PRegionLayer(l), w, h, net.w, net.h, thresh, map, hier, relative, dets, letter);
                    dets := dets + (l.w * l.h * l.n)
                end;
            if l.&type = ltDETECTION then
                begin
                    get_detection_detections(PDetectionLayer(l), w, h, thresh, dets);
                    dets := dets + (l.w * l.h * l.n)
                end
        end
end;

function get_network_boxes(const net: PNetwork; const w, h: longint; const thresh, hier: single; const map: Plongint; relative: boolean; const num: Plongint; const letter: boolean):TArray<TDetection>;
begin
    result := make_network_boxes(net, thresh, num);
    fill_network_boxes(net, w, h, thresh, hier, map, relative, PDetection(result), letter);
end;

procedure free_detections(dets: Pdetection; n: longint);
var
    i: longint;
begin
    //for i := 0 to n -1 do
    //    begin
    //        free(dets[i].prob);
    //        if dets[i].uc then
    //            free(dets[i].uc);
    //        if dets[i].mask then
    //            free(dets[i].mask);
    //        if dets[i].embeddings then
    //            free(dets[i].embeddings)
    //    end;
    //free(dets)
end;

procedure free_batch_detections(det_num_pairs: PDetNumPair; const n: longint);
var
    i: longint;
begin
    for i := 0 to n -1 do
        free_detections(@det_num_pairs[i].dets[0], det_num_pairs[i].num);
    FreeMemAndNil(det_num_pairs)
end;

function detection_to_json(const dets: PDetection; const nboxes: longint; const classes: longint; const names: TArray<string>; const frame_id: int64; const filename: string):string;
var
    thresh: single;
    i: longint;
    j: longint;
    class_id: longint;
    show: boolean;
    //send_buf: string;
begin
    thresh := 0.005;   //get_network_boxes() has already filtred dets by actual threshold
    //send_buf := string(calloc(1024, sizeof(char)));
    //if not send_buf then
    //    exit(0);
    if filename<>'' then
        result := format('{'#10' "frame_id":%lld, '#10' "filename":"%s", '#10' "objects": [ '#10'', [frame_id, filename])
    else
        result := format('{'#10' "frame_id":%d, '#10' "objects": [ '#10, [frame_id]);

    class_id := -1;
    for i := 0 to nboxes -1 do
        for j := 0 to classes -1 do
            begin
                show := names[j] = 'dont_show';
                if (dets[i].prob[j] > thresh) and show then
                    begin
                        if class_id <> -1 then
                            result:= result + ', '#10;
                        class_id := j;
                        result := result + format('  {"class_id":%d, "name":"%s", "relative_coordinates":{"center_x":%f, "center_y":%f, "width":%f, "height":%f}, "confidence":%f}', [j, names[j], dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h, dets[i].prob[j]])
                    end
            end;
    result := result + #10' ] '#10
end;

function network_predict_image(const net: Pnetwork; const im: TImageData):PSingle;
var
    imr: TImageData;
begin
    if net.batch <> 1 then
        set_batch_network(net, 1);
    if (im.w = net.w) and (im.h = net.h) then
        result := network_predict(net[0], im.data)
    else
        begin
            imr := resize_image(im, net.w, net.h);
            result := network_predict(net[0], imr.data);
            free_image(imr)
        end;
end;

function network_predict_batch(const net: PNetwork; const im: TImageData; const batch_size, w, h: longint; const thresh, hier: single; const map: Plongint; const relative, letter: boolean):TArray<TDetNumPair>;
var
    num: longint;
    batch: longint;
    dets: TArray<TDetection>;
begin
    network_predict(net[0], im.data);

    setLength(result ,batch_size);
    for batch := 0 to batch_size -1 do
        begin
            dets := make_network_boxes_batch(net, thresh, @num, batch);
            fill_network_boxes_batch(net, w, h, thresh, hier, map, relative, @dets[0], letter, batch);
            result[batch].num := num;
            result[batch].dets := dets
        end;
    exit(result)
end;

function network_predict_image_letterbox(const net: PNetwork; const im: TImageData):PSingle;
var
    p: PSingle;
    imr: TImageData;
begin
    if net.batch <> 1 then
        set_batch_network(net, 1);
    if (im.w = net.w) and (im.h = net.h) then
        p := network_predict(net[0], im.data)
    else
        begin
            imr := letterbox_image(im, net.w, net.h);
            p := network_predict( net[0], imr.data);
            free_image(imr)
        end;
    exit(p)
end;

function network_width(net: Pnetwork):longint;
begin
    exit(net.w)
end;

function network_height(net: Pnetwork):longint;
begin
    exit(net.h)
end;

function network_predict_data_multi(const net: TNetwork; const test: TData; const n: longint):TMatrix;
var
    i, j, b, m, k: longint;
    //pred: TMatrix;
    X : TArray<single>;
    &out: PSingle;
begin
    k := get_network_output_size(net);
    result := make_matrix(test.X.rows, k);
    setLength(X, net.batch * test.X.rows);
    //float * X := single(xcalloc(net.batch * test.X.rows, sizeof(float)));
    i := 0;
    while i < test.X.rows do begin
        for b := 0 to net.batch -1 do
            begin
                if i+b = test.X.rows then
                    break;
                move(test.X.vals[i+b][0], X[b * test.X.cols], test.X.cols * sizeof(single))
            end;
        for m := 0 to n -1 do
            begin
                &out := network_predict(net, X);
                for b := 0 to net.batch -1 do
                    begin
                        if i+b = test.X.rows then
                            break;
                        for j := 0 to k -1 do
                            result.vals[i+b][j] := result.vals[i+b][j] + (&out[j+b * k] / n)
                    end
            end;
        i := i + net.batch
    end;
    //free(X);
    //exit(result)
end;

function network_predict_data(const net: TNetwork; const test: TData):TMatrix;
var
    i, j, b, k: longint;
    //pred: TMatrix;
    &out: PSingle;
    X: TArray<Single>;
begin
    k := get_network_output_size(net);
    result := make_matrix(test.X.rows, k);
    //float * X := single(xcalloc(net.batch * test.X.cols, sizeof(float)));
    setLength(X, net.batch * test.X.cols);
    i := 0;
    while i < test.X.rows do begin
        for b := 0 to net.batch -1 do
            begin
                if i+b = test.X.rows then
                    break;
                move(test.X.vals[i+b][0], X[b * test.X.cols], test.X.cols * sizeof(single))
            end;
        &out := network_predict(net, X);
        for b := 0 to net.batch -1 do
            begin
                if i+b = test.X.rows then
                    break;
                for j := 0 to k -1 do
                    result.vals[i+b][j] := &out[j+b * k]
            end;
        i := i + net.batch
    end;
    //free(X);
    exit(result)
end;

procedure print_network(const net: TNetwork);
var
    i, j, n: longint;
    l: TLayer;
    output: PSingle;
    mean, vari: single;
begin
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            output := l.output;
            n := l.outputs;
            mean := mean_array(output, n);
            vari := variance_array(output, n);
            writeln(ErrOutput, format('Layer %d - Mean: %f, Variance: %f', [i, mean, vari]));
            if n > 100 then
                n := 100;
            for j := 0 to n -1 do
                write(ErrOutput, output[j], ', ');
            if n = 100 then
                writeln(ErrOutput, '.....');
            writeln(ErrOutput, '')
        end
end;

procedure compare_networks(const n1, n2: TNetwork; const test: TData);
var
    g1, g2: TMatrix;
    i, a, b, c, d, truth, p1, p2: longint;
    num, den: single;
begin
    g1 := network_predict_data(n1, test);
    g2 := network_predict_data(n2, test);
    a := 0;b := 0; c := 0;d := 0;
    for i := 0 to g1.rows -1 do
        begin
            truth := max_index(PSingle(test.y.vals[i]), test.y.cols);
            p1 := max_index(PSingle(g1.vals[i]), g1.cols);
            p2 := max_index(PSingle(g2.vals[i]), g2.cols);
            if p1 = truth then
                begin
                    if p2 = truth then
                        inc(d)
                    else
                        inc(c)
                end
            else
                begin
                    if p2 = truth then
                        inc(b)
                    else
                        inc(a)
                end
        end;
    writeln(format('%5d %5d'#10'%5d %5d', [a, b, c, d]));
    num := sqr((abs(b-c)-1){, 2.});
    den := b+c;
    writeln(format('%f', [num / den]))
end;

function network_accuracy(const net: TNetwork; const d: TData):single;
var
    guess: TMatrix;
    acc: single;
begin
    guess := network_predict_data(net, d);
    acc := matrix_topk_accuracy(d.y, guess, 1);
    free_matrix(guess);
    exit(acc)
end;

function network_accuracies(const net: TNetwork; const d: TData; const n: longint):PSingle;
const
  acc:array[0..1] of single =(0,0);
var
    guess: TMatrix;
begin
    guess := network_predict_data(net, d);
    acc[0] := matrix_topk_accuracy(d.y, guess, 1);
    acc[1] := matrix_topk_accuracy(d.y, guess, n);
    free_matrix(guess);
    exit(@acc[0])
end;

function network_accuracy_multi(const net: TNetwork; const d: TData; const n: longint):single;
var
    guess: TMatrix;
begin
    guess := network_predict_data_multi(net, d, n);
    result := matrix_topk_accuracy(d.y, guess, 1);
    free_matrix(guess);
end;

procedure free_network(net: TNetwork);
var
    i: longint;
begin
    for i := 0 to net.n -1 do
        free_layer(net.layers[i]);
    //free(net.layers);
    //FreeMemAndNil(net.seq_scales);
    //freeMemAndNil(net.scales);
    //freeMemAndNil(net.steps);
    //freeMemAndNil(net.seen);
{$ifdef GPU}}
    freeMemAndNil(net.cuda_graph_ready);
{$endif}
    //freeMemAndNil(net.badlabels_reject_threshold);
    //freeMemAndNil(net.delta_rolling_max);
    //freeMemAndNil(net.delta_rolling_avg);
    //freeMemAndNil(net.delta_rolling_std);
    //freeMemAndNil(net.cur_iteration);
    //freeMemAndNil(net.total_bbox);
    //freeMemAndNil(net.rewritten_bbox);
{$ifdef GPU}
    if gpu_index >= 0 then
        cuda_free(net.workspace)
    else
        free(net.workspace);
    free_pinned_memory();
    if net.input_state_gpu then
        cuda_free(net.input_state_gpu);
    if net.input_pinned_cpu then
        begin
            if net.input_pinned_cpu_flag then
                cudaFreeHost(net.input_pinned_cpu)
            else
                free(net.input_pinned_cpu)
        end;
    if  * net.input_gpu then
        cuda_free( * net.input_gpu);
    if  * net.truth_gpu then
        cuda_free( * net.truth_gpu);
    if net.input_gpu then
        free(net.input_gpu);
    if net.truth_gpu then
        free(net.truth_gpu);
    if  * net.input16_gpu then
        cuda_free( * net.input16_gpu);
    if  * net.output16_gpu then
        cuda_free( * net.output16_gpu);
    if net.input16_gpu then
        free(net.input16_gpu);
    if net.output16_gpu then
        free(net.output16_gpu);
    if net.max_input16_size then
        free(net.max_input16_size);
    if net.max_output16_size then
        free(net.max_output16_size);
{$endif}
    //FreeMemAndNil(net.workspace)
end;

procedure free_network_ptr(const net: Pnetwork);
begin
    free_network(net[0])
end;

function relu(src: single):single;
begin
    if src > 0 then
        exit(src);
    exit(0)
end;

function lrelu(src: single):single;
var
    eps: single;
begin
    eps := 0.001;
    if src > eps then
        exit(src);
    exit(eps)
end;

procedure fuse_conv_batchnorm(const net: TNetwork);
var
    j: longint;
    l: PLayer;
    f: longint;
    precomputed: double;
    filter_size: size_t;
    i: longint;
    w_index: longint;
    layer_step: longint;
    chan: longint;
    sum, max_val: single;
    w: single;
    eps: single;
begin
    for j := 0 to net.n -1 do
        begin
            l :=  @net.layers[j];
            if l.&type = ltCONVOLUTIONAL then
                begin
                    if l.share_layer <> nil then
                        l.batch_normalize := false;
                    if l.batch_normalize then
                        begin
                            for f := 0 to l.n -1 do
                                begin
                                    l.biases[f] := l.biases[f] - l.scales[f] * l.rolling_mean[f] / (sqrt(l.rolling_variance[f]+0.00001));
                                    precomputed := l.scales[f] / (sqrt(l.rolling_variance[f])+0.00001);
                                    filter_size := l.size * l.size * l.c div l.groups;
                                    for i := 0 to filter_size -1 do
                                        begin
                                            w_index := f * filter_size+i;
                                            l.weights[w_index] := l.weights[w_index] * precomputed
                                        end
                                end;
                            free_convolutional_batchnorm(l[0]);
                            l.batch_normalize := false;
{$ifdef GPU}
                            if gpu_index >= 0 then
                                push_convolutional_layer(l[0])
{$endif}
                        end
                end
            else
                if (l.&type = ltSHORTCUT) and assigned(l.weights) and boolean(l.weights_normalization) then
                    begin
                        if l.nweights > 0 then
                            begin
                                for i := 0 to l.nweights -1 do
                                    write(format(' w = %f,', [l.weights[i]]));
                                writeln(format(' l.nweights = %d, j = %d ', [l.nweights, j]))
                            end;
                        layer_step := l.nweights div (l.n+1);
                        for chan := 0 to layer_step -1 do
                            begin
                                sum := 1; max_val := -MaxSingle;
                                if l.weights_normalization = wnSOFTMAX_NORMALIZATION then
                                    for i := 0 to (l.n+1) -1 do
                                        begin
                                            w_index := chan+i * layer_step;
                                            w := l.weights[w_index];
                                            if max_val < w then
                                                max_val := w
                                        end;
                                eps := 0.0001;
                                sum := eps;
                                for i := 0 to (l.n+1) -1 do
                                    begin
                                        w_index := chan+i * layer_step;
                                        w := l.weights[w_index];
                                        if l.weights_normalization = wnRELU_NORMALIZATION then
                                            sum := sum + lrelu(w)
                                        else
                                            if l.weights_normalization = wnSOFTMAX_NORMALIZATION then
                                                sum := sum + exp(w-max_val)
                                    end;
                                for i := 0 to (l.n+1) -1 do
                                    begin
                                        w_index := chan+i * layer_step;
                                        w := l.weights[w_index];
                                        if l.weights_normalization = wnRELU_NORMALIZATION then
                                            w := lrelu(w) / sum
                                        else
                                            if l.weights_normalization = wnSOFTMAX_NORMALIZATION then
                                                w := exp(w-max_val) / sum;
                                        l.weights[w_index] := w
                                    end
                            end;
                        l.weights_normalization := wnNO_NORMALIZATION;
{$ifdef GPU}
                        if gpu_index >= 0 then
                            push_shortcut_layer(l[0])
{$endif}
                    end
            else

        end
end;

procedure forward_blank_layer(var l: TLayer; const state: PNetworkState);
begin

end;

procedure calculate_binary_weights(const net: TNetwork);
var
    j: longint;
    l: Player;
    sc: Player;
begin
    for j := 0 to net.n -1 do
        begin
            l :=  @net.layers[j];
            if l.&type = ltCONVOLUTIONAL then
                if l.xnor then
                    begin
                        binary_align_weights(Pointer(l));
                        if net.layers[j].use_bin_output then
                            l.activation := acLINEAR;
{$ifdef GPU}
                        if ((j+1) < net.n) and (net.layers[j].&type = ltCONVOLUTIONAL) then
                            begin
                                sc :=  @net.layers[j+1];
                                if sc.&type = ltSHORTCUT and sc.w = sc.out_w and sc.h = sc.out_h and sc.c = sc.out_c then
                                    begin
                                        l.bin_conv_shortcut_in_gpu := net.layers[net.layers[j+1].index].output_gpu;
                                        l.bin_conv_shortcut_out_gpu := net.layers[j+1].output_gpu;
                                        net.layers[j+1].&type := ltBLANK;
                                        net.layers[j+1].forward_gpu := forward_blank_layer
                                    end
                            end
{$endif}
                    end
        end
end;

procedure copy_cudnn_descriptors(const src: TLayer; const dst: Player);
begin
{$ifdef GPU}
    dst.normTensorDesc := src.normTensorDesc;
    dst.normDstTensorDesc := src.normDstTensorDesc;
    dst.normDstTensorDescF16 := src.normDstTensorDescF16;
    dst.srcTensorDesc := src.srcTensorDesc;
    dst.dstTensorDesc := src.dstTensorDesc;
    dst.srcTensorDesc16 := src.srcTensorDesc16;
    dst.dstTensorDesc16 := src.dstTensorDesc16
{$endif}
end;

procedure copy_weights_net(const net_train: TNetwork; const net_map: PNetwork);
var
    k: longint;
    l: Player;
    tmp_layer, tmp_input_layer, tmp_self_layer, tmp_output_layer: TLayer;
begin
    for k := 0 to net_train.n -1 do
        begin
            l :=  @net_train.layers[k];
            copy_cudnn_descriptors(net_map.layers[k],  @tmp_layer);
            net_map.layers[k] := net_train.layers[k];
            copy_cudnn_descriptors(tmp_layer,  @net_map.layers[k]);
            if l.&type = ltCRNN then
                begin
                    copy_cudnn_descriptors( net_map.layers[k].input_layer[0],  @tmp_input_layer);
                    copy_cudnn_descriptors( net_map.layers[k].self_layer[0],  @tmp_self_layer);
                    copy_cudnn_descriptors( net_map.layers[k].output_layer[0],  @tmp_output_layer);
                    net_map.layers[k].input_layer := net_train.layers[k].input_layer;
                    net_map.layers[k].self_layer := net_train.layers[k].self_layer;
                    net_map.layers[k].output_layer := net_train.layers[k].output_layer;
                    copy_cudnn_descriptors(tmp_input_layer, @net_map.layers[k].input_layer[0]);
                    copy_cudnn_descriptors(tmp_self_layer, @net_map.layers[k].self_layer[0]);
                    copy_cudnn_descriptors(tmp_output_layer, @net_map.layers[k].output_layer[0])
                end
            else
                if assigned(l.input_layer) then
                    begin
                        copy_cudnn_descriptors(net_map.layers[k].input_layer[0],  @tmp_input_layer);
                        net_map.layers[k].input_layer := net_train.layers[k].input_layer;
                        copy_cudnn_descriptors(tmp_input_layer, @net_map.layers[k].input_layer[0])
                    end;
            net_map.layers[k].batch := 1;
            net_map.layers[k].steps := 1;
            net_map.layers[k].train := false
        end
end;

function combine_train_valid_networks(const net_train: TNetwork; const net_map: TNetwork):TNetwork;
var
    net_combined: TNetwork;
    old_layers: TArray<Tlayer>;
    k: longint;
    l: Player;
begin
    net_combined := make_network(net_train.n);
    old_layers := net_combined.layers;
    net_combined := net_train;
    net_combined.layers := old_layers;
    net_combined.batch := 1;
    for k := 0 to net_train.n -1 do
        begin
            l :=  @net_train.layers[k];
            net_combined.layers[k] := net_train.layers[k];
            net_combined.layers[k].batch := 1;
            if l.&type = ltCONVOLUTIONAL then
                begin
{$ifdef GPU}
                    net_combined.layers[k].normTensorDesc := net_map.layers[k].normTensorDesc;
                    net_combined.layers[k].normDstTensorDesc := net_map.layers[k].normDstTensorDesc;
                    net_combined.layers[k].normDstTensorDescF16 := net_map.layers[k].normDstTensorDescF16;
                    net_combined.layers[k].srcTensorDesc := net_map.layers[k].srcTensorDesc;
                    net_combined.layers[k].dstTensorDesc := net_map.layers[k].dstTensorDesc;
                    net_combined.layers[k].srcTensorDesc16 := net_map.layers[k].srcTensorDesc16;
                    net_combined.layers[k].dstTensorDesc16 := net_map.layers[k].dstTensorDesc16
{$endif}
                end
        end;
    exit(net_combined)
end;

procedure free_network_recurrent_state(const net: TNetwork);
var
    k: longint;
begin
    for k := 0 to net.n -1 do
        begin
            if net.layers[k].&type = ltConvLSTM then
                free_state_conv_lstm(net.layers[k]);
            if net.layers[k].&type = ltCRNN then
                free_state_crnn(net.layers[k])
        end
end;

procedure randomize_network_recurrent_state(const net: TNetwork);
var
    k: longint;
begin
    for k := 0 to net.n -1 do
        begin
            if net.layers[k].&type = ltConvLSTM then
                randomize_state_conv_lstm(net.layers[k]);
            if net.layers[k].&type = ltCRNN then
                free_state_crnn(net.layers[k])
        end
end;

procedure remember_network_recurrent_state(const net: TNetwork);
var
    k: longint;
begin
    for k := 0 to net.n -1 do
        if net.layers[k].&type = ltConvLSTM then
            remember_state_conv_lstm(net.layers[k])
end;

procedure restore_network_recurrent_state(const net: TNetwork);
var
    k: longint;
begin
    for k := 0 to net.n -1 do
        begin
            if net.layers[k].&type = ltConvLSTM then
                restore_state_conv_lstm(net.layers[k]);
            if net.layers[k].&type = ltCRNN then
                free_state_crnn(net.layers[k])
        end
end;

function is_ema_initialized(const net: TNetwork):boolean;
var
    i, k: longint;
    l: TLayer;
begin
    result := false;
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                begin
                    if assigned(l.weights_ema) then
                        for k := 0 to l.nweights -1 do
                            if l.weights_ema[k] <> 0 then
                                exit(true)
                end
        end;
end;

procedure ema_update(const net: TNetwork; const ema_alpha: single);
var
    i, k: longint;
    l: TLayer;
begin
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                begin
{$ifdef GPU}
                    if gpu_index >= 0 then
                        pull_convolutional_layer(l);
{$endif}
                    if assigned(l.weights_ema) then
                        for k := 0 to l.nweights -1 do
                            l.weights_ema[k] := ema_alpha * l.weights_ema[k]+(1-ema_alpha) * l.weights[k];
                    for k := 0 to l.n -1 do
                        begin
                            if assigned(l.biases_ema) then
                                l.biases_ema[k] := ema_alpha * l.biases_ema[k]+(1-ema_alpha) * l.biases[k];
                            if assigned(l.scales_ema) then
                                l.scales_ema[k] := ema_alpha * l.scales_ema[k]+(1-ema_alpha) * l.scales[k]
                        end
                end
        end
end;

procedure ema_apply(const net: TNetwork);
var
    i, k: longint;
    l: Tlayer;
begin
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if l.&type = ltCONVOLUTIONAL then
                begin
                    if assigned(l.weights_ema) then
                        for k := 0 to l.nweights -1 do
                            l.weights[k] := l.weights_ema[k];
                    for k := 0 to l.n -1 do
                        begin
                            if assigned(l.biases_ema) then
                                l.biases[k] := l.biases_ema[k];
                            if assigned(l.scales_ema) then
                                l.scales[k] := l.scales_ema[k]
                        end;
{$ifdef GPU}
                    if gpu_index >= 0 then
                        push_convolutional_layer(l)
{$endif}
                end
        end
end;

procedure reject_similar_weights(const net: TNetwork; const sim_threshold: single);
var
    i, k, j, max_sim_index, max_sim_index2, filter_size, w1, w2: longint;
    l: TLayer;
    max_sim, sim, scale: single;
begin
    for i := 0 to net.n -1 do
        begin
            l := net.layers[i];
            if i = 0 then
                continue;
            if net.n > i+1 then
                if net.layers[i+1].&type = ltYOLO then
                    continue;
            if net.n > i+2 then
                if net.layers[i+2].&type = ltYOLO then
                    continue;
            if net.n > i+3 then
                if net.layers[i+3].&type = ltYOLO then
                    continue;
            if (l.&type = ltCONVOLUTIONAL) and (l.activation <> acLINEAR) then
                begin
{$ifdef GPU}
                    if gpu_index >= 0 then
                        pull_convolutional_layer(l);
{$endif}
                    max_sim := -1000;
                    max_sim_index := 0;
                    max_sim_index2 := 0;
                    filter_size := l.size * l.size * l.c;
                    for k := 0 to l.n -1 do
                        for j := k+1 to l.n -1 do
                            begin
                                w1 := k;
                                w2 := j;
                                sim := cosine_similarity(@l.weights[filter_size * w1],  @l.weights[filter_size * w2], filter_size);
                                if sim > max_sim then
                                    begin
                                        max_sim := sim;
                                        max_sim_index := w1;
                                        max_sim_index2 := w2
                                    end
                            end;
                    writeln(format(' reject_similar_weights: i = %d, l.n = %d, w1 = %d, w2 = %d, sim = %f, thresh = %f ', [i, l.n, max_sim_index, max_sim_index2, max_sim, sim_threshold]));
                    if max_sim > sim_threshold then
                        begin
                            writeln(' rejecting... ');
                            scale := sqrt(2 / (l.size * l.size * l.c div l.groups));
                            for k := 0 to filter_size -1 do
                                l.weights[max_sim_index * filter_size+k] := scale * rand_uniform(-1, 1);
                            if assigned(l.biases) then
                                l.biases[max_sim_index] := 0.0;
                            if assigned(l.scales) then
                                l.scales[max_sim_index] := 1.0
                        end;
{$ifdef GPU}
                    if gpu_index >= 0 then
                        push_convolutional_layer(l)
{$endif}
                end
        end
end;

end.

