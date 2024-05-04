unit SoftmaxLayer;

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
  SysUtils, lightnet, blas;

type
  TSoftmaxLayer = TLayer;

  TContrastiveLayer = TLayer;

procedure softmax_tree(const input: Psingle; const batch, inputs: longint; const temp: single; const hierarchy: PTree; const output: PSingle);
function make_softmax_layer(const batch, inputs, groups: longint):TSoftmaxLayer;
procedure forward_softmax_layer(var l: TSoftmaxLayer; const net: PNetworkState);
procedure backward_softmax_layer(var l: TSoftmaxLayer; const net: PNetworkState);
{$ifdef GPU}
procedure pull_softmaxlayer_output(const layer: TSoftmaxLayer);
procedure backward_softmaxlayer_gpu(const layer: TSoftmaxLayer; const net: TNetwork);
{$endif}

function make_contrastive_layer(const batch, w, h, c:longint; classes:longint; const inputs: longint; const yolo_layer: Player):TContrastiveLayer;
procedure forward_contrastive_layer(var l: TContrastiveLayer; const state: PNetworkState);
procedure backward_contrastive_layer(var l: TContrastiveLayer; const state: PNetworkState);

{$ifdef GPU}
procedure pull_contrastive_layer_output(const l: contrastive_layer);
procedure push_contrastive_layer_output(const l: contrastive_layer);
procedure forward_contrastive_layer_gpu(l: contrastive_layer; state: network_state);
procedure backward_contrastive_layer_gpu(layer: contrastive_layer; state: network_state);
{$endif}

implementation

procedure softmax_tree(const input: Psingle; const batch, inputs: longint; const temp: single; const hierarchy: PTree; const output: PSingle);
var
    b, i, count, group_size: longint;
begin
    for b := 0 to batch -1 do
        begin
            count := 0;
            for i := 0 to hierarchy.groups -1 do
                begin
                    group_size := hierarchy.group_size[i];
                    softmax(input+b * inputs + count, group_size, temp, output+b * inputs+count, 1);
                    count := count + group_size
                end
        end
end;


function make_softmax_layer(const batch, inputs, groups: longint):TSoftmaxLayer;
//var
//    l: TSoftmaxLayer;
begin
    assert(inputs mod groups = 0);
    writeln(ErrOutput, 'softmax                                        ', inputs:4);
    result := default(TSoftmaxLayer);
    result.&type := ltSOFTMAX;
    result.batch := batch;
    result.groups := groups;
    result.inputs := inputs;
    result.outputs := inputs;
    result.loss := TSingles.Create(inputs * batch);
    result.output := TSingles.Create(inputs * batch);
    result.delta := TSingles.Create(inputs * batch);
    result.cost := TSingles.Create(1);

    result.forward := forward_softmax_layer;
    result.backward := backward_softmax_layer;
{$ifdef GPU}
    result.forward_gpu := forward_softmax_layer_gpu;
    result.backward_gpu := backward_softmax_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, inputs * batch);
    result.loss_gpu := cuda_make_array(result.loss, inputs * batch);
    result.delta_gpu := cuda_make_array(result.delta, inputs * batch);
{$endif}
    //exit(l)
end;

procedure forward_softmax_layer(var l: TSoftmaxLayer; const net: PNetworkState);
var
    i, count, group_size: longint;
begin
    if assigned(l.softmax_tree) then
        begin
            count := 0;
            for i := 0 to l.softmax_tree[0].groups -1 do
                begin
                    group_size := l.softmax_tree[0].group_size[i];
                    softmax_cpu(net.input+count, group_size, l.batch, l.inputs, 1, 0, 1, l.temperature, l.output+count);
                    count := count + group_size
                end
        end
    else
        softmax_cpu(net.input, l.inputs div l.groups, l.batch, l.inputs, l.groups, l.inputs div l.groups, 1, l.temperature, l.output);
    if assigned(net.truth) and not l.noloss then
        begin
            softmax_x_ent_cpu(l.batch * l.inputs, l.output, net.truth, l.delta, l.loss);
            l.cost[0] := sum_array(l.loss, l.batch * l.inputs)
        end
end;

procedure backward_softmax_layer(var l: TSoftmaxLayer; const net: PNetworkState);
begin
    axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, net.delta, 1)
end;

{$ifdef GPU}
procedure pull_softmax_layer_output(const layer: TSoftmaxLayer);
begin
    cuda_pull_array(layer.output_gpu, layer.output, layer.inputs * layer.batch)
end;

procedure forward_softmax_layer_gpu(var l: TSoftmaxLayer; const net: PNetworkState);
begin
    if l.softmax_tree then
        softmax_tree(net.input_gpu, 1, l.batch, l.inputs, l.temperature, l.output_gpu,  * l.softmax_tree)
    else
        begin
            if l.spatial then
                softmax_gpu(net.input_gpu, l.c, l.batch * l.c, l.inputs div l.c, l.w * l.h, 1, l.w * l.h, 1, l.output_gpu)
            else
                softmax_gpu(net.input_gpu, l.inputs div l.groups, l.batch, l.inputs, l.groups, l.inputs div l.groups, 1, l.temperature, l.output_gpu)
        end;
    if assigned(net.truth) and not l.noloss then
        begin
            softmax_x_ent_gpu(l.batch * l.inputs, l.output_gpu, net.truth_gpu, l.delta_gpu, l.loss_gpu);
            if l.softmax_tree then
                begin
                    mask_gpu(l.batch * l.inputs, l.delta_gpu, SECRET_NUM, net.truth_gpu, 0);
                    mask_gpu(l.batch * l.inputs, l.loss_gpu, SECRET_NUM, net.truth_gpu, 0)
                end;
            cuda_pull_array(l.loss_gpu, l.loss, l.batch * l.inputs);
            l.cost[0] := sum_array(l.loss, l.batch * l.inputs)
        end
end;

procedure backward_softmax_layer_gpu(const layer: TSoftmaxLayer; const net: TNetwork);
begin
    axpy_gpu(layer.batch * layer.inputs, 1, layer.delta_gpu, 1, net.delta_gpu, 1)
end;
{$endif}

function make_contrastive_layer(const batch, w, h, c:longint; classes:longint; const inputs: longint; const yolo_layer: Player):TContrastiveLayer;
var
    l: TContrastiveLayer;
    step: size_t;
    max_contr_size: longint;
begin
    l := Default(TContrastiveLayer);
    l.&type := ltCONTRASTIVE;
    l.batch := batch;
    l.inputs := inputs;
    l.w := w;
    l.h := h;
    l.c := c;
    l.temperature := 1;
    l.max_boxes := 0;
    if assigned(yolo_layer) then
        begin
            l.detection := 1;
            l.max_boxes := yolo_layer.max_boxes;
            l.labels := yolo_layer.labels;
            l.class_ids := yolo_layer.class_ids;
            l.n := yolo_layer.n;
            l.classes := yolo_layer.classes;
            classes := l.classes;
            l.embedding_size := l.inputs div (l.n * l.h * l.w);
            l.truths := yolo_layer.truths;
            if l.embedding_size <> yolo_layer.embedding_size then
                begin
                    writeln(format(' Error: [contrastive] embedding_size=%d isn''t equal to [yolo] embedding_size=%d. They should use the same [convolutional] layer ', [l.embedding_size, yolo_layer.embedding_size]));
                    raise Exception.create('Error!')
                end;
            if l.inputs mod (l.n * l.h * l.w) <> 0 then
                writeln (' Warning: filters= number in the previous (embedding) layer isn''t divisable by number of anchors ', l.n)
        end
    else
        begin
            l.detection := 0;
            setLength(l.labels, l.batch);
            l.n := 1;
            l.classes := classes;
            l.embedding_size := l.c
        end;
    l.outputs := inputs;
    l.loss := TSingles.Create(1);
    l.output := TSingles.Create(inputs * batch);
    l.delta := TSingles.Create(inputs * batch);
    l.cost := TSingles.Create(1);
    step := l.batch * l.n * l.h * l.w;
    l.cos_sim := nil;
    l.exp_cos_sim := nil;
    l.p_constrastive := nil;
    if l.detection=0 then
        begin
            l.cos_sim := TSingles.Create(step * step);
            l.exp_cos_sim := TSingles.Create(step * step);
            l.p_constrastive := TSingles.Create(step * step)
        end;
    l.forward := forward_contrastive_layer;
    l.backward := backward_contrastive_layer;
{$ifdef GPU}
    l.forward_gpu := forward_contrastive_layer_gpu;
    l.backward_gpu := backward_contrastive_layer_gpu;
    l.output_gpu := cuda_make_array(l.output, inputs * batch);
    l.delta_gpu := cuda_make_array(l.delta, inputs * batch);
    max_contr_size := (l.max_boxes * l.batch) * (l.max_boxes * l.batch) * sizeof(contrastive_params) div 4;
    writeln(format(' max_contr_size = %d MB ', [max_contr_size div (1024 * 1024)]));
    l.contrast_p_gpu := contrastive_params(cuda_make_array(NULL, max_contr_size));
{$endif}
    writeln(ErrOutput, format('contrastive %4d x%4d x%4d x emb_size %4d x batch: %4d  classes = %4d, step = %4zu ',[ w, h, l.n, l.embedding_size, batch, l.classes, step]));
    if l.detection<>0 then
        writeln(ErrOutput, 'detection ');
    exit(l)
end;

function clip_value(const val, max_val: single):single;
begin
    if val > max_val then
        exit(max_val)
    else
        if val < -max_val then
            exit(-max_val);
    exit(val)
end;

procedure forward_contrastive_layer(var l: TContrastiveLayer; const state: PNetworkState);
var
    truth_thresh: single;
    mini_batch: longint;
    b: longint;
    n: longint;
    w: longint;
    h: longint;
    max_truth: single;
    truth_prob: single;
    z: TArray<TArray<Single>>;
    max_sim_same, max_sim_diff : TArray<Single>;
    z_index: longint;
    b2: longint;
    n2: longint;
    h2: longint;
    w2: longint;
    contrast_p_index: longint;
    step: size_t;
    contrast_p_size: size_t;
    contrast_p : TArray<TContrastiveParams>;
    labels: TArray<longint>;
    z_index2: longint;
    time_step_i: longint;
    time_step_j: longint;
    sim: single;
    exp_sim: single;
    i: longint;
    good_sims, all_sims, same_sim, diff_sim: longint;
    contr_size: size_t;
    max_contr_size: longint;
    k: longint;
    P: single;
    q: longint;
    bd: longint;
    nd: longint;
    hd: longint;
    wd: longint;
    delta_index: longint;
    wh: longint;
begin

    if not state.train then
        exit();
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    truth_thresh := state.net.label_smooth_eps;
    mini_batch := l.batch div l.steps;
    fill_cpu(l.batch * l.inputs, 0, l.delta, 1);
    if l.detection=0 then
        begin
            for b := 0 to l.batch -1 do
                begin
                    if state.net.adversarial then
                        l.labels[b] := b mod 2
                    else
                        l.labels[b] := b div 2
                end;
            for b := 0 to l.batch -1 do
                for h := 0 to l.h -1 do
                    for w := 0 to l.w -1 do
                        begin
                            max_truth := 0;
                            for n := 0 to l.classes -1 do
                                begin
                                    truth_prob := state.truth[b * l.classes+n];
                                    if truth_prob > truth_thresh then
                                        begin
                                            max_truth := truth_prob;
                                            l.labels[b] := n
                                        end
                                end
                        end
        end;
    setLength(z, l.batch * l.n * l.h * l.w);
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            for h := 0 to l.h -1 do
                for w := 0 to l.w -1 do
                    begin
                        z_index := b * l.n * l.h * l.w+n * l.h * l.w+h * l.w+w;
                        if l.labels[z_index] < 0 then
                            continue;
                        setLength(z[z_index], l.embedding_size);
                        get_embedding(state.input, l.w, l.h, l.c, l.embedding_size, w, h, n, b, @z[z_index][0])
                    end;
    contrast_p_index := 0;
    step := l.batch * l.n * l.h * l.w;
    contrast_p_size := step;
    if l.detection=0 then
        contrast_p_size := l.batch * l.batch;
    setLength(contrast_p, contrast_p_size);
    setLength(max_sim_same, l.batch * l.inputs);
    setLength(max_sim_diff, l.batch * l.inputs);
    fill_cpu(l.batch * l.inputs, -10, @max_sim_same[0], 1);
    fill_cpu(l.batch * l.inputs, -10, @max_sim_diff[0], 1);
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            for h := 0 to l.h -1 do
                for w := 0 to l.w -1 do
                    begin
                        z_index := b * l.n * l.h * l.w+n * l.h * l.w+h * l.w+w;
                        if l.labels[z_index] < 0 then
                            continue;
                        for b2 := 0 to l.batch -1 do
                            for n2 := 0 to l.n -1 do
                                for h2 := 0 to l.h -1 do
                                    for w2 := 0 to l.w -1 do
                                        begin
                                            z_index2 := b2 * l.n * l.h * l.w+n2 * l.h * l.w+h2 * l.w+w2;
                                            if l.labels[z_index2] < 0 then
                                                continue;
                                            if z_index = z_index2 then
                                                continue;
                                            if l.detection<>0 then
                                                if l.class_ids[z_index] <> l.class_ids[z_index2] then
                                                    continue;
                                            time_step_i := b div mini_batch;
                                            time_step_j := b2 div mini_batch;
                                            if time_step_i <> time_step_j then
                                                continue;
                                            step := l.batch * l.n * l.h * l.w;
                                            sim := cosine_similarity(@z[z_index][0], @z[z_index2][0], l.embedding_size);
                                            exp_sim := exp(sim / l.temperature);
                                            if l.detection=0 then
                                                begin
                                                    l.cos_sim[z_index * step+z_index2] := sim;
                                                    l.exp_cos_sim[z_index * step+z_index2] := exp_sim
                                                end;
                                            if (l.labels[z_index] = l.labels[z_index2]) and (max_sim_same[z_index] < sim) then
                                                max_sim_same[z_index] := sim;
                                            if (l.labels[z_index] <> l.labels[z_index2]) and (max_sim_diff[z_index] < sim) then
                                                max_sim_diff[z_index] := sim;
                                            contrast_p[contrast_p_index].sim := sim;
                                            contrast_p[contrast_p_index].exp_sim := exp_sim;
                                            contrast_p[contrast_p_index].i := z_index;
                                            contrast_p[contrast_p_index].j := z_index2;
                                            contrast_p[contrast_p_index].time_step_i := time_step_i;
                                            contrast_p[contrast_p_index].time_step_j := time_step_j;
                                            inc(contrast_p_index);
                                            if (contrast_p_index+1) >= contrast_p_size then
                                                begin
                                                    contrast_p_size := contrast_p_index+1;
                                                    setLength(contrast_p , contrast_p_size)
                                                end;
                                            if (sim > 1.001) or (sim < -1.001) then
                                                writeln(format(' sim = %f, ', [sim]))
                                        end
                    end;
    good_sims := 0; all_sims := 0; same_sim := 0; diff_sim := 0;
    for i := 0 to l.batch * l.inputs -1 do
        if (max_sim_same[i] >= -1) and (max_sim_diff[i] >= -1) then
            begin
                if max_sim_same[i] >= -1 then
                    inc(same_sim);
                if max_sim_diff[i] >= -1 then
                    inc(diff_sim);
                inc(all_sims);
                if max_sim_diff[i] < max_sim_same[i] then
                    inc(good_sims)
            end;
    if all_sims > 0 then
        l.loss[0] := 100 * good_sims div all_sims
    else
         l.loss[0] := -1;
    writeln(format(' Contrast accuracy = %f %%, all = %d, good = %d, same = %d, diff = %d ', [l.loss[0], all_sims, good_sims, same_sim, diff_sim]));
    //free(max_sim_same);
    //free(max_sim_diff);
    contr_size := contrast_p_index;
    if l.detection<>0 then
        begin
{$ifdef GPU}
            max_contr_size := (l.max_boxes * l.batch) * (l.max_boxes * l.batch);
            if max_contr_size < contr_size then
                begin
                    writeln(format(' Error: too large number of bboxes: contr_size = %d > max_contr_size  = %d ', [contr_size, max_contr_size]));
                    Exception.Create('Error!')
                end;
            labels := nil;
            if contr_size > 2 then
                begin
                    cuda_push_array(single(l.contrast_p_gpu), single(contrast_p), contr_size * sizeof(contrastive_params) div 4);
                    P_constrastive_f_det_gpu(labels, l.embedding_size, l.temperature, l.contrast_p_gpu, contr_size);
                    cuda_pull_array(single(l.contrast_p_gpu), single(contrast_p), contr_size * sizeof(contrastive_params) div 4)
                end;
            for k := 0 to contr_size -1 do
                contrast_p[k].P := P_constrastive_f_det(k, l.labels, z, l.embedding_size, l.temperature, contrast_p, contr_size)
{$endif}
        end
    else
        for b := 0 to l.batch -1 do
            for n := 0 to l.n -1 do
                for h := 0 to l.h -1 do
                    for w := 0 to l.w -1 do
                        begin
                            z_index := b * l.n * l.h * l.w+n * l.h * l.w+h * l.w+w;
                            if l.labels[z_index] < 0 then
                                continue;
                            for b2 := 0 to l.batch -1 do
                                for n2 := 0 to l.n -1 do
                                    for h2 := 0 to l.h -1 do
                                        for w2 := 0 to l.w -1 do
                                            begin
                                                z_index2 := b2 * l.n * l.h * l.w+n2 * l.h * l.w+h2 * l.w+w2;
                                                if l.labels[z_index2] < 0 then
                                                    continue;
                                                if z_index = z_index2 then
                                                    continue;
                                                if l.detection<>0 then
                                                    if l.class_ids[z_index] <> l.class_ids[z_index2] then
                                                        continue;
                                                time_step_i := b div mini_batch;
                                                time_step_j := b2 div mini_batch;
                                                if time_step_i <> time_step_j then
                                                    continue;
                                                step := l.batch * l.n * l.h * l.w;
                                                P := -10;
                                                if l.detection<>0 then
                                                    P := P_constrastive_f(z_index, z_index2, @l.labels[0], @z[0], l.embedding_size, l.temperature, @contrast_p[0], contr_size)
                                                else
                                                    begin
                                                        P := P_constrastive(z_index, z_index2, @l.labels[0], step, @z[0], l.embedding_size, l.temperature, l.cos_sim, l.exp_cos_sim);
                                                        l.p_constrastive[z_index * step+z_index2] := P
                                                    end;
                                                for q := 0 to contr_size -1 do
                                                    if (contrast_p[q].i = z_index) and (contrast_p[q].j = z_index2) then
                                                        begin
                                                            contrast_p[q].P := P;
                                                            break
                                                        end
                                            end
                        end;
    bd := 0;
    for bd := 0 to l.batch -1 do
        for nd := 0 to l.n -1 do
            for hd := 0 to l.h -1 do
                for wd := 0 to l.w -1 do
                    begin
                        z_index := bd * l.n * l.h * l.w+nd * l.h * l.w+hd * l.w+wd;
                        step := l.batch * l.n * l.h * l.w;
                        if l.labels[z_index] < 0 then
                            continue;
                        delta_index := bd * l.embedding_size * l.n * l.h * l.w+nd * l.embedding_size * l.h * l.w+hd * l.w+wd;
                        wh := l.w * l.h;
                        if l.detection<>0 then
                            begin
                                grad_contrastive_loss_positive_f(z_index, @l.class_ids[0], @l.labels[0], step, @z[0], l.embedding_size, l.temperature, l.delta+delta_index, wh, @contrast_p[0], contr_size);
                                grad_contrastive_loss_negative_f(z_index, @l.class_ids[0], @l.labels[0], step, @z[0], l.embedding_size, l.temperature, l.delta+delta_index, wh, @contrast_p[0], contr_size, l.contrastive_neg_max)
                            end
                        else
                            begin
                                grad_contrastive_loss_positive(z_index, @l.labels[0], step, @z[0], l.embedding_size, l.temperature, l.cos_sim, l.p_constrastive, l.delta+delta_index, wh);
                                grad_contrastive_loss_negative(z_index, @l.labels[0], step, @z[0], l.embedding_size, l.temperature, l.cos_sim, l.p_constrastive, l.delta+delta_index, wh)
                            end
                    end;
    scal_cpu(l.inputs * l.batch, l.cls_normalizer, l.delta, 1);
    for i := 0 to l.inputs * l.batch -1 do
        l.delta[i] := clip_value(l.delta[i], l.max_delta);
    l.cost[0] := sqr(mag_array(l.delta, l.inputs * l.batch){, 2});
    if state.net.adversarial then
        writeln(format(' adversarial contrastive loss = %f '#10'', [l.cost[0]]))
    else
        writeln(format(' contrastive loss = %f '#10'', [l.cost[0]]));
    for b := 0 to l.batch -1 do
        for n := 0 to l.n -1 do
            for h := 0 to l.h -1 do
                for w := 0 to l.w -1 do
                    begin
                        z_index := b * l.n * l.h * l.w+n * l.h * l.w+h * l.w+w;
                        //if z[z_index] then
                            //free(z[z_index])
                    end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

    //free(contrast_p);
    //free(z)
end;

procedure backward_contrastive_layer(var l: TContrastiveLayer; const state: PNetworkState);
begin
    axpy_cpu(l.inputs * l.batch, 1, l.delta, 1, state.delta, 1)
end;

{$ifdef GPU}
procedure pull_contrastive_layer_output(const l: contrastive_layer);
begin
    cuda_pull_array(l.output_gpu, l.output, l.inputs * l.batch)
end;

procedure push_contrastive_layer_output(const l: contrastive_layer);
begin
    cuda_push_array(l.delta_gpu, l.delta, l.inputs * l.batch)
end;

procedure forward_contrastive_layer_gpu(l: contrastive_layer; state: network_state);
var
    num_truth: longint;
    cpu_state: network_state;
begin
    simple_copy_ongpu(l.batch * l.inputs, state.input, l.output_gpu);
    if not state.train then
        exit();
    float * in_cpu := single(xcalloc(l.batch * l.inputs, sizeof(float)));
    cuda_pull_array(l.output_gpu, l.output, l.batch * l.outputs);
    memcpy(in_cpu, l.output, l.batch * l.outputs * sizeof(float));
    float * truth_cpu := 0;
    if state.truth then
        begin
            num_truth := l.batch * l.classes;
            if l.detection then
                num_truth := l.batch * l.truths;
            truth_cpu := single(xcalloc(num_truth, sizeof(float)));
            cuda_pull_array(state.truth, truth_cpu, num_truth)
        end;
    cpu_state := state;
    cpu_state.net := state.net;
    cpu_state.index := state.index;
    cpu_state.train := state.train;
    cpu_state.truth := truth_cpu;
    cpu_state.input := in_cpu;
    forward_contrastive_layer(l, cpu_state);
    cuda_push_array(l.delta_gpu, l.delta, l.batch * l.outputs);
    free(in_cpu);
    if cpu_state.truth then
        free(cpu_state.truth)
end;

procedure backward_contrastive_layer_gpu(layer: contrastive_layer; state: network_state);
begin
    axpy_ongpu(layer.batch * layer.inputs, state.net.loss_scale, layer.delta_gpu, 1, state.delta, 1)
end;
{$endif}

end.

