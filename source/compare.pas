unit compare;

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
  Classes, SysUtils, math, utils, darknet, parser, image, nnetwork, data, matrix;

type
  PSortableBBox = ^TSortableBBox;
  TSortableBBox = record
        net     : TNetwork;
        filename: string;
        &class  : longint;
        classes : longint;
        elo     : single;
        elos    : TSingles;
  end ;


procedure train_compare(const cfgfile, weightfile: string);
procedure validate_compare(const filename: string; const weightfile: string);
function elo_comparator(const a: TSortableBBox; const b: TSortableBBox):longint;
function bbox_comparator(const a: TSortableBBox; const b: TSortableBBox):longint;
procedure bbox_update(var a, b: TSortableBBox; const &class: longint; const isResult: boolean);
procedure bbox_fight(var net: TNetwork; var a, b: TSortableBBox; const classes: longint; const &class: longint);
procedure SortMaster3000(const filename: string; const weightfile: string);
procedure BattleRoyaleWithCheese(const filename: string; const weightfile: string);
procedure run_compare(const argc: longint; const argv: TArray<string>);


implementation

procedure train_compare(const cfgfile, weightfile: string);
var
    avg_loss: single;
    net: TNetwork;
    imgs: longint;
    //paths,
    plist: TStringarray;
    N: longint;
    time: clock_t;
    load_thread: TThread;
    train, buffer: TData;
    args: TLoadArgs;
    epoch: longint;
    i: longint;
    loss: single;
    buff: string;
    base, backup_directory: string;
begin
    //RandSeed:=0;
    randomize;
    avg_loss := -1;
    base := basecfg(cfgfile);
    backup_directory := './backup/'; // originally was '/home/pjreddie/backup/'
    writeln(base);
    net:=parse_network_cfg(cfgfile);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    writeln(format('Learning Rate: %g, Momentum: %g, Decay: %g', [net.learning_rate, net.momentum, net.decay]));
    imgs := 1024;
    plist := get_paths('data/compare.train.list');
    //paths := list_to_array(plist);
    N := length(plist);
    writeln(N);
    args := Default(TLoadArgs);
    args.w := net.w;
    args.h := net.h;
    args.paths := plist;
    args.classes := 20;
    args.n := imgs;
    args.m := N;
    args.d :=  @buffer;
    args.&type := dtCOMPARE_DATA;
    load_thread := load_data_in_thread(@args);
    epoch :=  net.seen[0] div N;
    i := 0;
    while true do
        begin
            inc(i);
            time := clock();
            load_thread.WaitFor;
            train := buffer;   // note copying a data value of buffer to train so no data racing will happen in buffer next lead thread?
            load_thread := load_data_in_thread(@args);
            writeln(format('Loaded: %f seconds', [(clock()-time)/CLOCKS_PER_SEC]));
            time := clock();
            loss := train_network(net, train);
            if avg_loss = -1 then
                avg_loss := loss;
            avg_loss := avg_loss * 0.9+loss * 0.1;
            writeln(format('%.3f: %f, %f avg, %f seconds, %d images', [net.seen[0] / N, loss, avg_loss, (clock()-time)/CLOCKS_PER_SEC,  net.seen[0]]));
            free_data(train);
            if i mod 100 = 0 then
                begin
                    buff:=format('%s/%s_%d_minor_%d.weights', [backup_directory, base, epoch, i]);
                    save_weights(net, buff)
                end;
            if  net.seen[0] div N > epoch then
                begin
                    epoch := net.seen[0] div N;
                    i := 0;
                    buff:=format('%s/%s_%d.weights', [backup_directory, base, epoch]);
                    save_weights(net, buff);
                    if epoch mod 22 = 0 then
                        net.learning_rate := net.learning_rate * 0.1
                end
        end;
    load_thread.WaitFor;
    free_data(buffer);
    free_network(net);
    //free_ptrs((paths), );
    //free_list(plist);
    //free(base)
end;

procedure validate_compare(const filename: string; const weightfile: string);
var
    i: longint;
    net: TNetwork;
    plist: TStringArray;
    N: longint;
    time: clock_t;
    correct: longint;
    total: longint;
    splits: longint;
    num: longint;
    val, buffer: TData;
    args: TLoadArgs;
    load_thread: TThread;
    part: TStringArray;
    pred: TMatrix;
    j, k: longint;
begin
    i := 0;
    net := parse_network_cfg( filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    //srand(time(0));
    randomize;
    plist := get_paths('data/compare.val.list');
    //paths := char(list_to_array(plist));
    N := length(plist) div 2 ;
    //free_list(plist);
    correct := 0;
    total := 0;
    splits := 10;
    num := (i+1) * N div splits-i * N div splits;
    args := Default(TLoadArgs);
    args.w := net.w;
    args.h := net.h;
    args.paths := plist;
    args.classes := 20;
    args.n := num;
    args.m := 0;
    args.d :=  @buffer;
    args.&type := dtCOMPARE_DATA;
    load_thread := load_data_in_thread(@args);
    for i := 1 to splits do
        begin
            time := clock();
            load_thread.WaitFor;//pthread_join(load_thread, 0);
            val := buffer;
            num := (i+1) * N div splits - i * N div splits;
            part := copy(plist,(i * N div splits));
            if i <> splits then
                begin
                    args.paths := part;
                    load_thread := load_data_in_thread(@args)
                end;
            writeln(format('Loaded: %d images in %f seconds', [val.X.rows, (clock()-time)/CLOCKS_PER_SEC]));
            time := clock();
            pred := network_predict_data(net, val);
            for j := 0 to val.y.rows -1 do
                for k := 0 to 20 -1 do
                    if val.y.vals[j][k * 2] <> val.y.vals[j][k * 2+1] then
                        begin
                            inc(total);
                            if (val.y.vals[j][k * 2] < val.y.vals[j][k * 2+1]) = (pred.vals[j][k * 2] < pred.vals[j][k * 2+1]) then
                                inc(correct)
                        end;
            free_matrix(pred);
            writeln(format('%d: Acc: %f, %f seconds, %d images',[ i, correct / total, (clock()-time)/CLOCKS_PER_SEC, val.X.rows]));
            free_data(val)
        end;
    free_network(net)
end;

var
  total_compares: longint=0;
  current_class: longint=0;
function elo_comparator(const a: TSortableBBox; const b: TSortableBBox): longint;
begin

    if a.elos[current_class] = b.elos[current_class] then
        exit(0);
    if a.elos[current_class] > b.elos[current_class] then
        exit(-1);
    exit(1)
end;

function bbox_comparator(const a: TSortableBBox; const b: TSortableBBox):longint;
var
    net: TNetwork;
    &class: longint;
    im1: TImageData;
    im2: TImageData;
    X, predictions : TSingles;
begin
    inc(total_compares);
    net := a.net;
    &class := a.&class;
    im1 := load_image_color(a.filename, net.w, net.h);
    im2 := load_image_color(b.filename, net.w, net.h);
    //setLength(X , net.w * net.h * net.c);
    X := TSingles.Create(net.w * net.h * net.c);
    //memcpy(X, im1.data   , im1.w * im1.h * im1.c * sizeof(float));
    move(im1.data[0], X[0], im1.w * im1.h * im1.c * sizeof(single));

    //memcpy(X+im1.w * im1.h * im1.c, im2.data, im2.w * im2.h * im2.c * sizeof(float));
    move(im2.data[0], X[im1.w * im1.h * im1.c] , im2.w * im2.h * im2.c * sizeof(single));

    predictions := network_predict(net, X);
    free_image(im1);
    free_image(im2);
    //free(X);
    X.Free;
    if predictions[&class * 2] > predictions[&class * 2+1] then
        exit(1);
    exit(-1)
end;

procedure bbox_update(var a, b: TSortableBBox; const &class: longint;
  const isResult: boolean);
var
    k: longint;
    EA, EB, SA, SB: single;
begin
    k := 32;
    EA := 1.0 / (1+power(10, (b.elos[&class] - a.elos[&class]) / 400));
    EB := 1.0 / (1+power(10, (a.elos[&class] - b.elos[&class]) / 400));
    if isResult then
        SA := 1
    else
        SA := 0;
    if isResult then
        SB := 0
    else
        SB := 1;
    a.elos[&class] :=  a.elos[&class] + (k * (SA-EA));
    b.elos[&class] :=  b.elos[&class] + (k * (SB-EB))
end;

procedure bbox_fight(var net: TNetwork; var a, b: TSortableBBox;
  const classes: longint; const &class: longint);
var
    im1, im2: TImageData;
    i: longint;
    isResult: boolean;
    X, predictions:TSingles;
begin
    im1 := load_image_color(a.filename, net.w, net.h);
    im2 := load_image_color(b.filename, net.w, net.h);
    //setLength(X, net.w * net.h * net.c );
    X := TSingles.Create(net.w * net.h * net.c);
    move(im1.data[0], X[0], im1.w * im1.h * im1.c * sizeof(single));
    //memcpy(X, im1.data, im1.w * im1.h * im1.c * sizeof(float));
    move(im2.data[0], X[im1.w * im1.h * im1.c], im2.w * im2.h * im2.c * sizeof(single));
    //memcpy(X+im1.w * im1.h * im1.c, im2.data, im2.w * im2.h * im2.c * sizeof(float));
    predictions := network_predict(net, X);
    inc(total_compares);
    for i := 0 to classes -1 do
        if (&class < 0) or (&class = i) then
            begin
                isResult := predictions[i * 2] > predictions[i * 2+1];
                bbox_update(a, b, i, isResult)
            end;
    //free_image(im1);
    //free_image(im2);
    //free(X)
end;

procedure SortMaster3000(const filename: string; const weightfile: string);
var
    i, N: longint;
    net: TNetwork;
    plist: TStringArray;
    time: clock_t;
    boxes :TArray<TSortableBBox> ;
begin
    i := 0;
    net := parse_network_cfg( filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    //srand(time(0));
    randomize;
    set_batch_network(@net, 1);
    plist := get_paths('data/compare.sort.list');
    //paths := char(list_to_array(plist));
    //int N = plist->size;
    N := length(plist);
    //free_list(plist);
    //sortable_bbox * boxes := calloc(N, sizeof(sortable_bbox));
    setLength(boxes, N);
    writeln(format('Sorting %d boxes...', [N]));
    for i := 0 to N -1 do
        begin
            boxes[i].filename := plist[i];
            boxes[i].net := net; // todo SortMaster3000 net could be a PNetwork instead or TNetwork
            boxes[i].&class := 7;
            boxes[i].elo := 1500
        end;
    time := clock();
    TTools<TSortableBBox>.QuickSort(@boxes[0],0, N-1,  bbox_comparator);
    for i := 0 to N -1 do
        writeln(format('%s', [boxes[i].filename]));
    writeln(format('Sorted in %d compares, %f secs', [total_compares, (clock()-time)/CLOCKS_PER_SEC]));
    free_network(net);
end;

procedure BattleRoyaleWithCheese(const filename: string;
  const weightfile: string);
var
    classes, i, j, N, total, round, &class: longint;
    net: TNetwork;
    plist: TArray<string>;
    time, round_time: clock_t;
    outfp:TextFile;
    buff: string;
    boxes : TArray<TSortableBBox>;
begin
    classes := 20;
    net := parse_network_cfg( filename);
    if weightfile<>'' then
        load_weights( @net, weightfile);
    //srand(time(0));
    randomize;
    set_batch_network( @net, 1);
    plist := get_paths('data/compare.sort.list');
    //paths := char(list_to_array(plist));
    N := length(plist);
    total := N;
    //free_list(plist);
    //sortable_bbox * boxes := calloc(N, sizeof(sortable_bbox));
    setLength(boxes, N);
    writeln(format('Battling %d boxes...', [N]));
    for i := 0 to N -1 do
        begin
            boxes[i].filename := plist[i];
            boxes[i].net := net;
            boxes[i].classes := classes;
            //boxes[i].elos := calloc(classes, sizeof(float));
            //setlength(boxes[i].elos, classes);
            boxes[i].elos := TSingles.Create(classes);
            for j := 0 to classes -1 do
                boxes[i].elos[j] := 1500
        end;
    time := clock();
    for round := 1 to 4 do
        begin
            round_time := clock();
            writeln(format('Round: %d', [round]));
            TTools<TSortableBBox>.shuffle(@boxes[0], N);
            for i := 0 to N div 2 -1 do
                bbox_fight(net, boxes[i * 2], boxes[i * 2+1], classes, -1);
            writeln(format('Round: %f secs, %d remaining',[ (clock()-round_time)/CLOCKS_PER_SEC, N]))
        end;
    for &class := 0 to classes -1 do
        begin
            N := total;
            current_class := &class;
            TTools<TSortableBBox>.QuickSort(@boxes[0], 0, N-1,  elo_comparator);
            N := N div 2;
            for round := 1 to 100 do
                begin
                    round_time := clock();
                    writeln(format('Round: %d', [round]));
                    TTools<TSortableBBox>.sorta_shuffle(@boxes[0], N, 10);
                    for i := 0 to N div 2 -1 do
                        bbox_fight(net, boxes[i * 2], boxes[i * 2+1], classes, &class);
                    TTools<TSortableBBox>.QuickSort(@boxes[0] , 0, N-1 , elo_comparator);
                    if round <= 20 then
                        N := (N * 9 div 10) div 2 * 2;
                    writeln(format('Round: %f secs, %d remaining', [(clock()-round_time)/CLOCKS_PER_SEC, N]))
                end;
            //char buff[256];
            buff:=format( 'results/battle_%d.log', [&class]);
            assignFile(outfp, buff);
            rewrite(outfp);
            for i := 0 to N -1 do
                writeln(outfp, format('%s %f', [boxes[i].filename, boxes[i].elos[&class]]));
            CloseFile(outfp)
        end;
    writeln(format('Tournament in %d compares, %f secs', [total_compares, (clock()-time)/CLOCKS_PER_SEC]));
    free_network(net)
end;

procedure run_compare(const argc: longint; const argv: TArray<string>);
var cfg, weights : string;
begin
    if (argc < 4) then
        begin
            writeln( format('usage: %s %s [train/test/valid] [cfg] [weights (optional)]', [argv[0], argv[1]] ));
            exit()
        end;
    cfg := argv[3];
    if (argc > 4) then
        weights := argv[4]
    else
        weights := '';
    if 0 = CompareStr(argv[2], 'train') then
        train_compare(cfg, weights)
    else
        if 0 = CompareStr(argv[2], 'valid') then
            validate_compare(cfg, weights)
    else
        if 0 = CompareStr(argv[2], 'sort') then
            SortMaster3000(cfg, weights)
    else
        if 0 = CompareStr(argv[2], 'battle') then
            BattleRoyaleWithCheese(cfg, weights)
end;


end.

