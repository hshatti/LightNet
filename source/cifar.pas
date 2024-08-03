unit cifar;

{$ifdef FPC}
{$mode Delphi}
{$endif}

interface

uses
  SysUtils, lightnet, nnetwork, parser, utils, Data, blas, ImageData, matrix;

procedure train_cifar(cfgfile: string; weightfile: string);
procedure train_cifar_distill(cfgfile: string; weightfile: string);
procedure test_cifar_multi(filename: string; weightfile: string);
procedure test_cifar(filename: string; weightfile: string);
procedure extract_cifar(path : string ='data/cifar10');
procedure test_cifar_csv(filename: string; weightfile: string);
procedure test_cifar_csvtrain(filename: string; weightfile: string);
procedure eval_cifar_csv();
procedure run_cifar(const action:string; cfg :string='cfg/cifar.cfg' ;weightsFile :string='');

implementation

procedure train_cifar(cfgfile: string; weightfile: string);
var
    avg_loss: single;
    base: string;
    net: TNetwork;
    backup_directory: string;
    //classes: longint;
    N: longint;
    //labels: TArray<string>;
    epoch: longint;
    train: TData;
    time: clock_t;
    loss: single;
    buff: string;
begin
    randomize;
    avg_loss := -1;
    base := basecfg(cfgfile);
    writeln(base);
    net := parse_network_cfg(cfgfile);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    writeLn(format('Learning Rate: %g, Momentum: %g, Decay: %g', [net.learning_rate, net.momentum, net.decay]));
    backup_directory := 'backup/';
    //classes := 10;
    N := 50000;
    //labels := get_labels('data/cifar/labels.txt');
    epoch := (net.seen[0]) div N;
    train := load_all_cifar10();
    while (get_current_batch(net) < net.max_batches) or (net.max_batches = 0) do
        begin
            time := clock();
            loss := train_network_sgd(net, train, 1);
            if avg_loss = -1 then
                avg_loss := loss;
            avg_loss := avg_loss * 0.95+loss * 0.05;
            writeLn(format('%d, %.3f: %f, %f avg, %f rate, %.3f seconds, %d', [get_current_batch(net), net.seen[0] / N, loss, avg_loss, get_current_rate(net), (clock()-time)/CLOCKS_PER_SEC, net.seen[0]]));
            if net.seen[0] div N > epoch then
                begin
                    epoch := net.seen[0] div N;
                    buff := format('%s/%s_%d.weights', [backup_directory, base, epoch]);
                    save_weights(net, buff)
                end;
            if get_current_batch(net) mod 100 = 0 then
                begin
                    buff := format('%s/%s.backup', [backup_directory, base]);
                    save_weights(net, buff)
                end
        end;
    buff := format('%s/%s.weights', [backup_directory, base]);
    save_weights(net, buff);
    free_network(net);
end;

procedure train_cifar_distill(cfgfile: string; weightfile: string);
var
    avg_loss: single;
    base: string;
    net: TNetwork;
    backup_directory: string;
    classes: longint;
    N: longint;
    labels: TArray<string>;
    epoch: longint;
    train: TData;
    soft: TMatrix;
    weight: single;
    time: clock_t;
    loss: single;
    buff: string;
begin
    randomize;
    avg_loss := -1;
    base := basecfg(cfgfile);
    writeLn(base);
    net := parse_network_cfg(cfgfile);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    writeLn(format('Learning Rate: %g, Momentum: %g, Decay: %g', [net.learning_rate, net.momentum, net.decay]));
    backup_directory := 'backup/';
    classes := 10;
    N := 50000;
    labels := get_labels('data/cifar/labels.txt');
    epoch := (net.seen[0]) div N;
    train := load_all_cifar10();
    soft := csv_to_matrix('results/ensemble.csv');
    weight := 0.9;
    scale_matrix(soft, weight);
    scale_matrix(train.y, 1-weight);
    matrix_add_matrix(soft, train.y);
    while (get_current_batch(net) < net.max_batches) or (net.max_batches = 0) do
        begin
            time := clock();
            loss := train_network_sgd(net, train, 1);
            if avg_loss = -1 then
                avg_loss := loss;
            avg_loss := avg_loss * 0.95+loss * 0.05;
            writeLn(format('%d, %.3f: %f, %f avg, %f rate, %f seconds, %d', [get_current_batch(net), (net.seen[0]) / N, loss, avg_loss, get_current_rate(net), (clock()-time)/CLOCKS_PER_SEC, net.seen[0]]));
            if net.seen[0] div N > epoch then
                begin
                    epoch := net.seen[0] div N;
                    buff := format('%s/%s_%d.weights', [backup_directory, base, epoch]);
                    save_weights(net, buff)
                end;
            if get_current_batch(net) mod 100 = 0 then
                begin
                    buff := format('%s/%s.backup', [backup_directory, base]);
                    save_weights(net, buff)
                end
        end;
    buff := format('%s/%s.weights', [backup_directory, base]);
    save_weights(net, buff);
    free_network(net);
end;

procedure test_cifar_multi(filename: string; weightfile: string);
var
    net: TNetwork;
    avg_acc: single;
    test: TData;
    i: longint;
    im: TImageData;
    pred: TArray<Single>;
    p: TSingles;
    index: longint;
    class_id: longint;
begin
    net := parse_network_cfg(filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    set_batch_network( @net, 1);
    Randomize;
    avg_acc := 0;
    test := load_cifar10_data('data/cifar/cifar-10-batches-bin/test_batch.bin');
    setLength(pred, 10);
    for i := 0 to test.X.rows -1 do
        begin
            im := float_to_image(32, 32, 3, pointer(test.X.vals[i]));
            fillchar(pred[0], length(pred), #0);
            p := network_predict(net, im.data);
            axpy_cpu(10, 1, p, 1, @pred[0], 1);
            flip_image(im);
            p := network_predict(net, im.data);
            axpy_cpu(10, 1, p, 1, @pred[0], 1);
            index := max_index(pred);
            class_id := max_index(@test.y.vals[i][0], 10);
            if index = class_id then
                avg_acc := avg_acc + 1;
            free_image(im);
            writeLn(format('%4d: %.2f%%', [i, 100 * avg_acc / (i+1)]))
        end
end;

procedure test_cifar(filename: string; weightfile: string);
var
    net: TNetwork;
    time: clock_t;
    avg_acc: single;
    avg_top5: single;
    test: TData;
    acc: PSingle;
begin
    net := parse_network_cfg(filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    randomize;
    avg_acc := 0;
    avg_top5 := 0;
    test := load_cifar10_data('data/cifar/cifar-10-batches-bin/test_batch.bin');
    time := clock();
    acc := network_accuracies(net, test, 2);
    avg_acc := avg_acc + acc[0];
    avg_top5 := avg_top5 + acc[1];
    writeLn(format('top1: %f, %lf seconds, %d images', [avg_acc, (clock()-time)/CLOCKS_PER_SEC, test.X.rows]));
    free_data(test)
end;

procedure extract_cifar(path : string);
var
    labels: array of string;
    i: longint;
    train: TData;
    test: TData;
    im: TImageData;
    class_id: longint;
    buff: string;
begin
    labels := ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'];
    train := load_all_cifar10();
    test := load_cifar10_data(path+'/test_batch.bin');
    for i := 0 to train.X.rows -1 do
        begin
            im := float_to_image(32, 32, 3, Pointer(train.X.vals[i]));
            class_id := max_index(pointer(train.y.vals[i]), 10);
            buff := format('data/cifar/train/%d_%s', [i, labels[class_id]]);
            save_image_options(im, buff, imtPNG)
        end;
    for i := 0 to test.X.rows -1 do
        begin
            im := float_to_image(32, 32, 3, Pointer(test.X.vals[i]));
            class_id := max_index(Pointer(test.y.vals[i]), 10);
            buff := format('data/cifar/test/%d_%s', [i, labels[class_id]]);
            save_image_options(im, buff, imtPNG)
        end
end;

procedure test_cifar_csv(filename: string; weightfile: string);
var
    net: TNetwork;
    test: TData;
    pred: TMatrix;
    i: longint;
    im: TImageData;
    pred2: TMatrix;
begin
    net := parse_network_cfg(filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    randomize;
    test := load_cifar10_data('data/cifar/cifar-10-batches-bin/test_batch.bin');
    pred := network_predict_data(net, test);
    for i := 0 to test.X.rows -1 do
        begin
            im := float_to_image(32, 32, 3, Pointer(test.X.vals[i]));
            flip_image(im)
        end;
    pred2 := network_predict_data(net, test);
    scale_matrix(pred, 0.5);
    scale_matrix(pred2, 0.5);
    matrix_add_matrix(pred2, pred);
    matrix_to_csv(pred);
    writeln(stderr, format('Accuracy: %f', [matrix_topk_accuracy(test.y, pred, 1)]));
    free_data(test)
end;

procedure test_cifar_csvtrain(filename: string; weightfile: string);
var
    net: TNetwork;
    test: TData;
    pred: TMatrix;
    i: longint;
    im: TImageData;
    pred2: TMatrix;
begin
    net := parse_network_cfg(filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    randomize;
    test := load_all_cifar10();
    pred := network_predict_data(net, test);
    for i := 0 to test.X.rows -1 do
        begin
            im := float_to_image(32, 32, 3, Pointer(test.X.vals[i]));
            flip_image(im)
        end;
    pred2 := network_predict_data(net, test);
    scale_matrix(pred, 0.5);
    scale_matrix(pred2, 0.5);
    matrix_add_matrix(pred2, pred);
    matrix_to_csv(pred);
    writeLn(stderr, format('Accuracy: %f', [matrix_topk_accuracy(test.y, pred, 1)]));
    free_data(test)
end;

procedure eval_cifar_csv();
var
    test: TData;
    pred: TMatrix;
begin
    test := load_cifar10_data('data/cifar/cifar-10-batches-bin/test_batch.bin');
    pred := csv_to_matrix('results/combined.csv');
    WriteLn(stderr, format('%d %d', [pred.rows, pred.cols]));
    WriteLn(stderr, format('Accuracy: %f', [matrix_topk_accuracy(test.y, pred, 1)]));
    free_data(test);
    free_matrix(pred)
end;

procedure run_cifar(const action: string; cfg: string; weightsFile: string);
begin
    if action= 'train' then
        train_cifar(cfg, weightsFile)
    else if action= 'extract' then
      extract_cifar()
    else if action= 'distill' then
      train_cifar_distill(cfg, weightsFile)
    else if action= 'test' then
      test_cifar(cfg, weightsFile)
    else if action= 'multi' then
      test_cifar_multi(cfg, weightsFile)
    else if action= 'csv' then
      test_cifar_csv(cfg, weightsFile)
    else if action= 'csvtrain' then
      test_cifar_csvtrain(cfg, weightsFile)
    else if action= 'eval' then
      eval_cifar_csv()
end;



end.

