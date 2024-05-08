unit data;

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
  Classes, StrUtils, SysUtils, lightnet, imagedata, blas, matrix, utils, steroids;


{$ifndef FPC}
  {$ifdef MSWINDOWS}
  const DirectorySeparator = '\';
  {$else}
  const DirectorySeparator = '/';
  {$endif}

{$endif}



procedure free_data(var d:TData);
function get_sequential_paths(const paths: TArray<string>; n, m, mini_batch, augment_speed,contrastive: longint):TArray<string>;
function get_random_paths_custom(const paths:TArray<string>; const n, m, contrastive: longint):TArray<string>;
function get_random_paths(const paths:TStringArray;const n,m:longint):TStringArray;
function find_replace_paths(const paths:TStringArray;const n:longint;const find, replace:string):TStringArray;
function load_image_paths_gray(const paths:TStringArray; const n,w,h:longint):TMatrix;
function load_image_paths(const paths :TStringArray; const n,w,h:longint):TMatrix;
function load_image_augment_paths(const paths: TStringArray; const n, _min, _max, size: longint; const angle ,aspect ,hue, saturation, exposure: single; const center: boolean):TMatrix;                       overload;
function load_image_augment_paths(const paths: TStringArray; const n, _min, _max,w,h: longint; const angle ,aspect ,hue, saturation, exposure: single; const dontuse_opencv, constrative: longint):TMatrix;  overload;
function read_boxes(const filename: string;var count: Longint):TArray<TBoxLabel>;
procedure randomize_boxes(const b: TArray<TBoxLabel>; const n: longint);
procedure correct_boxes(const boxes: Tarray<TBoxlabel>; const n: longint; const dx, dy, sx, sy: single;const flip: boolean);
procedure fill_truth_swag(const path: string; const truth: PSingle; const classes: longint; const flip: boolean; const dx, dy, sx, sy: single);
procedure fill_truth_region(const path: string; const truth: PSingle; const classes, num_boxes: longint; const flip: boolean; const dx, dy, sx, sy: single);
procedure load_rle(const im: TImageData; const rle: TArray<longint>; const n: longint);
procedure or_image(const src, dest: TImageData; const c: longint);
procedure exclusive_image(const src: TImageData);
function bound_image(const im: TImageData):TBox;
procedure fill_truth_iseg(const path: string; const num_boxes: longint; const truth: PSingle; const classes, w, h: longint; const aug: TAugmentArgs; flip: boolean; const mw, mh: longint);
procedure fill_truth_mask(const path: string; const num_boxes: longint; const truth: PSingle; const classes, w, h: longint; const aug: TAugmentArgs; const flip: boolean; const mw, mh: longint);
procedure fill_truth_detection(const path: string; const num_boxes: longint; const truth: PSingle; const classes: longint; const flip: boolean; const dx, dy, sx, sy: single);

procedure print_letters(const pred: PSingle; const n: longint);
procedure fill_truth_captcha(const path: string; const n: longint; const truth: PSingle);
function load_data_captcha(paths: TStringArray; const n, m, k, w, h: longint):TData;
function load_data_captcha_encode(paths: TStringArray; const n, m, w, h: longint):TData;
procedure fill_truth(const path: string; const labels: TStringArray; const k: longint; const truth: PSingle);
procedure fill_truth_smooth(path: string; labels: TArray<string>; k: longint; truth: Psingle; label_smooth_eps: single);
procedure fill_hierarchy(const truth: PSingle; const k: longint; const hierarchy: TTree);
function load_regression_labels_paths(const paths: TStringArray; const n, k: longint):TMatrix;
function load_labels_paths(const paths: TStringArray; const n: longint; const labels: TStringArray; const k: longint; const hierarchy: TArray<TTree>):TMatrix;
function load_tags_paths(const paths: TStringArray; const n, k: longint):TMatrix;
function get_labels(const filename: string):TStringArray;
function get_labels_custom(filename: string; size: Plongint):TArray<string>;
function get_segmentation_image(const path: string; const w, h: longint; const classes: longint):TImageData;
function get_segmentation_image2(const path: string; const w, h, classes: longint):TImageData;
function load_data_seg(const n: longint; const paths: TStringArray; const m, w, h, classes, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single; const _div: longint):TData;
function load_data_iseg(const n: longint; const paths: TStringArray; m, w, h, classes, boxes, _div, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single):TData;
function load_data_mask(const n: longint; const paths: TStringArray; const m, w, h, classes, boxes, coords, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single):TData;
function load_data_region(const n: longint; const paths: TStringArray; const m, w, h, size, classes: longint; jitter: single; hue: single; const saturation, exposure: single):TData;
function load_data_compare(const n: longint; paths: TStringArray; const m, classes, w, h: longint):TData;
function load_data_swag(const paths: TStringArray; const n, classes: longint; const jitter: single):TData;
function load_data_detection(const n: longint; const paths: TStringArray; const m, w, h, boxes, classes: longint; const jitter, hue, saturation, exposure: single):TData;
procedure load_thread(ptr: pointer);
function load_data_in_thread(args: Pointer):TThread;
procedure load_threads(ptr: Pointer);
procedure load_data_blocking(const args: TLoadArgs);
function load_data(const args: TLoadArgs):TThread;
function load_data_writing(paths: TStringArray; const n, m, w, h, out_w, out_h: longint):TData;
function load_data_old(paths: TStringArray; const n, m: longint; const labels: TStringArray; const k, w, h: longint):TData;
function load_data_super(paths: TStringArray; const n, m, w, h, scale: longint):TData;
function load_data_regression(paths: TStringArray; const n, m, k, _min, _max, size: longint; const angle, aspect, hue, saturation, exposure: single):TData;
function select_data(const orig: PData; const inds: Plongint):TData;
function tile_data(const orig: TData; const divs, size: longint):TArray<TData>;
function resize_data(const orig: TData; const w, h: longint):TData;
function load_data_augment(paths: TStringArray; const n, m: longint; const labels: TStringArray; const k: longint; const hierarchy: TArray<TTree>; const _min, _max, size: longint; const angle, aspect, hue, saturation, exposure: single; const center: boolean):TData;
function load_data_tag(paths: TStringArray; const n, m, k, _min, _max, size: longint; const angle, aspect, hue, saturation, exposure: single):TData;
function concat_matrix(const m1, m2: TMatrix):TMatrix;
function concat_data(const d1, d2: TData):TData;
function concat_datas(const d: TArray<TData>; const n: longint):TData;
function load_categorical_data_csv(const filename: string; const target, k: longint):TData;
function load_cifar10_data(const filename: string):TData;
procedure get_random_batch(const d: TData; const n: longint; const X, y: PSingle);
procedure get_next_batch(const d: TData; const n, offset: longint; const X, y: PSingle);
procedure smooth_data(const d: TData);
function load_all_cifar10():TData;
function load_go(const filename: string):TData;
procedure randomize_data(const d: TData);
procedure scale_data_rows(const d: TData; const s: single);
procedure translate_data_rows(const d: TData; const s: single);
function copy_data(const d: TData):TData;
procedure normalize_data_rows(const d: TData);
function get_data_part(const d: TData; const part, total: longint):TData;
function get_random_data(const d: TData; const num: longint):TData;
function split_data(const d: TData; const part, total: longint):TArray<TData>;


implementation
{$ifndef fpc}
function RPos(const Substr: String; const Source: String): NativeInt;
var
  MaxLen,llen : NativeInt;
  c : char;
  pc,pc2 : pchar;
begin
  rPos:=0;
  llen:=Length(SubStr);
  maxlen:=length(source);
  if (llen>0) and (maxlen>0) and ( llen<=maxlen) then
   begin
 //    i:=maxlen;
     pc:=@source[maxlen];
     pc2:=@source[llen-1];
     c:=substr[llen];
     while pc>=pc2 do
      begin
        if (c=pc^) and
           CompareMem(@Substr[1],pchar(pc-llen+1), Length(SubStr)*sizeof(char)) then
         begin
           rPos:=pchar(pc-llen+1)-pchar(@source[1])+1;
           exit;
         end;
        dec(pc);
      end;
   end;
end;
{$endif}


//var mutex :TRTLCriticalSection;

function distance_from_edge(const x, _max: longint):single;inline;
var dx:longint;
begin
    dx := (_max div 2) - x;
    if (dx < 0) then dx := -dx;
    dx := (_max div 2) + 1 - dx;
    dx := dx * 2;
    result := dx / _max;
    if result > 1 then
        result := 1;
end;

function get_sequential_paths(const paths: TArray<string>; n, m, mini_batch, augment_speed,contrastive: longint):TArray<string>;
var
    speed, i, time_line_index, index: longint;
    start_time_indexes: TArray<longint>;
begin
    speed := rand_int(1, augment_speed);
    if (speed < 1) then
        speed := 1;
    setLength(result, n);
    //pthread_mutex_lock( and mutex);
    setLength(start_time_indexes, mini_batch);
    for i := 0 to mini_batch -1 do
        begin
            if (contrastive<>0) and ((i mod 2) = 1) then
                start_time_indexes[i] := start_time_indexes[i-1]
            else
                start_time_indexes[i] := trandom(m)
        end;
    for i := 0 to n -1 do
        repeat
            time_line_index := i mod mini_batch;
            index := start_time_indexes[time_line_index] mod m;
            start_time_indexes[time_line_index] := start_time_indexes[time_line_index] + speed;
            result[i] := paths[index];
            if length(result[i]) <= 4 then
                writeln(' Very small path to the image: %s ', result[i])
        until not(length(result[i]) = 0);
    //free(start_time_indexes);
    //pthread_mutex_unlock( and mutex);

end;

function get_random_paths_custom(const paths:TArray<string>; const n, m, contrastive: longint):TArray<string>;
var
    i, old_index, index: longint;
begin
    setLength(result, n);
    //pthread_mutex_lock( and mutex);
    old_index := 0;
    for i := 0 to n -1 do
        repeat
            index := trandom(m);//random_gen() mod m;
            if (contrastive<>0) and (i mod 2 = 1) then
                index := old_index
            else
                old_index := index;
            result[i] := paths[index];
            if length(result[i]) <= 4 then
                writeln(' Very small path to the image: ', result[i])
        until not(length(result[i]) = 0);
    //pthread_mutex_unlock( and mutex);
end;

//function load_data_in_thread(const arg: TLoadArgs): TThread;
//begin
//  // todo implement load_data_in_thread
//end;

function get_random_paths(const paths: TStringArray; const n, m: longint): TStringArray;
var
  i: Integer;
begin
  // todo get_random_paths check maybe n is not needed
  // note random is not thread-safe, using mutex/CriticalSections
  setLength(result,n);
  //EnterCriticalSection(mutex);
  for i:=0 to n-1 do
    result[i]:=paths[trandom(m)];
  //LeaveCriticalSection(mutex);
end;

function find_replace_paths(const paths: TStringArray; const n: longint;
  const find, replace: string): TStringArray;
var i:longint;
begin
  // todo find_replace_paths check maybe n is not needed
  setLength(result, n);
  for i:=0 to n-1 do
    result[i]:=StringReplace(paths[i],find, replace,[]);
end;

function load_image_paths_gray(const paths: TStringArray; const n, w, h: longint ): TMatrix;
var
  im, gray: TImageData;
  i:longint;
begin
  result.rows:=n;
  //result.vals := AllocMem(n * sizeof(TSingles));//TSingles2d.Create(n);
  setLength(result.vals,n);
  result.cols:=0;
  for i:=0 to n-1 do begin
      im := load_Image(paths[i],w,h,3);
      gray := grayscale_image(im);
      free_image(im);
      im:=gray;
      result.vals[i]:=im.data;
      result.cols :=im.w * im.h * im.c;
  end;
end;

function load_image_paths(const paths: TStringArray; const n, w, h: longint): TMatrix;
var i:longint;
  im:TImageData;
begin
  result.rows := n;
  //result.vals := AllocMem(n * sizeof(TSingles));//TSingles2d.Create(n);
  setLength(result.vals, n);
  result.cols := 0;
  for i:=0 to n-1 do begin
    im := load_image_color(paths[i],w, h);
    result.vals[i] := im.data;
    result.cols := im.w * im.h * im.c
  end;
end;

function load_image_augment_paths(const paths: TStringArray; const n, _min,
  _max, size: longint; const angle, aspect, hue, saturation, exposure: single;
  const center: boolean): TMatrix;
var
    i: longint;
    im: TImageData;
    crop, sized: TImageData;
    flip: boolean;
    img_index:longint;
begin
    result.rows := n;
    //result.vals := AllocMem(result.rows * sizeof(TSingles));//TSingles2d.Create(n);
    setLength(result.vals ,result.rows);
    result.cols := 0;
    // this procedure will mostly be called in a thread, entering critical section because random() is not a thread-safe
    //if not center then EnterCriticalSection(mutex);
    for i := 0 to n -1 do
        begin
            im := load_image_color(paths[i], 0, 0);
            if center then
                crop := center_crop_image(im, size, size)
            else
                crop := random_augment_image(im, angle, aspect, _min, _max, size, size);
            flip := random(2)=1;
            if flip then
                flip_image(crop);
            random_distort_image(crop, hue, saturation, exposure);
            free_image(im);
            result.vals[i] := crop.data;
            result.cols := crop.h * crop.w * crop.c
        end;
    //if not center then LeaveCriticalSection(mutex);
end;

function load_image_augment_paths(const paths: TStringArray; const n, _min,
  _max, w, h: longint; const angle, aspect, hue, saturation, exposure: single;
  const dontuse_opencv, constrative: longint): TMatrix;
var
    i, size: longint;
    im: TImageData;
    crop, sized: TImageData;
    flip: boolean;
    img_index:longint;
begin
    result.rows := n;
    //result.vals := AllocMem(result.rows * sizeof(TSingles));//TSingles2d.Create(n);
    setLength(result.vals ,result.rows);
    result.cols := 0;
    // this procedure will mostly be called in a thread, entering critical section because random() is not a thread-safe
    //if not center then EnterCriticalSection(mutex);
    for i := 0 to n -1 do
        begin
            if w>h then size := w else size := h;
            if constrative<>0 then img_index := i div 2 else img_index := i;
            //if dontuse_opencv then
                im := load_image_color(paths[i], 0, 0);
            //else
              //load image using opencv
            //
            crop := random_augment_image(im, angle, aspect, _min, _max, size, size);
            flip := random(2)=1;
            if flip then
                flip_image(crop);
            random_distort_image(crop, hue, saturation, exposure);
            sized := resize_image(crop, w, h);
            free_image(im);
            result.vals[i] := sized.data;
            result.cols := sized.h * sized.w * sized.c
        end;
    //if not center then LeaveCriticalSection(mutex);
end;


function read_boxes(const filename: string;var count: Longint):TArray<TBoxLabel>;
var
    f: TextFile;
    x, y, h, w: single;
    id: longint;
    img_hash  : longint;
const
    max_obj_img : longint = 4000;// 30000;
    //s:string;
begin
    result:=nil;
    if not FileExists(filename) then begin
        writeln(format('File [%s] not found!. (this is normal if using MSCOCO only)',[filename]));
        assignfile(f, 'bad.list');
        append(f);
        writeln(f,filename);
        closefile(f);
        result := [default(TBoxLabel)];
        exit()
    end;
    assign(f,filename);
    reset(f);
    img_hash := (custom_hash(filename) mod max_obj_img)*max_obj_img;
    count := 0;
    while not eof(f) do try
        read(f,id, x, y, w, h);
        setLength(result,length(result)+1);
        //result[count].
        result[count].id := id;

        result[count].x := x;
        result[count].y := y;
        result[count].h := h;
        result[count].w := w;
        result[count].left := x-w / 2;
        result[count].right := x+w / 2;
        result[count].top := y-h / 2;
        result[count].bottom := y+h / 2;
        inc(count)
    except
    end;
    close(f);
end;

procedure randomize_boxes(const b: TArray<TBoxLabel>; const n: longint);
var
    i: longint;
    swap: TBoxLabel;
    index: longint;
begin
    for i := 0 to n -1 do
        begin
            swap := b[i];
            index := random(n);
            b[i] := b[index];
            b[index] := swap
        end
end;

procedure correct_boxes(const boxes: Tarray<TBoxlabel>; const n: longint;
  const dx, dy, sx, sy: single; const flip: boolean);
var
    i: longint;
    swap: single;
begin
    for i := 0 to n -1 do
        begin
            if (boxes[i].x = 0) and (boxes[i].y = 0) then
                begin
                    boxes[i].x := 999999;
                    boxes[i].y := 999999;
                    boxes[i].w := 999999;
                    boxes[i].h := 999999;
                    continue
                end;
            if ((boxes[i].x + boxes[i].w / 2) < 0) or ((boxes[i].y + boxes[i].h / 2) < 0) or
                ((boxes[i].x - boxes[i].w / 2) > 1) or ((boxes[i].y - boxes[i].h / 2) > 1) then
            begin
                boxes[i].x := 999999;
                boxes[i].y := 999999;
                boxes[i].w := 999999;
                boxes[i].h := 999999;
                continue;
            end;
            boxes[i].left := boxes[i].left     * sx-dx;
            boxes[i].right := boxes[i].right   * sx-dx;
            boxes[i].top := boxes[i].top       * sy-dy;
            boxes[i].bottom := boxes[i].bottom * sy-dy;
            if flip then
                begin
                    swap := boxes[i].left;
                    boxes[i].left := 1.-boxes[i].right;
                    boxes[i].right := 1.-swap
                end;
            boxes[i].left := constrain(0, 1, boxes[i].left);
            boxes[i].right := constrain(0, 1, boxes[i].right);
            boxes[i].top := constrain(0, 1, boxes[i].top);
            boxes[i].bottom := constrain(0, 1, boxes[i].bottom);
            boxes[i].x := (boxes[i].left+boxes[i].right) / 2;
            boxes[i].y := (boxes[i].top+boxes[i].bottom) / 2;
            boxes[i].w := (boxes[i].right-boxes[i].left);
            boxes[i].h := (boxes[i].bottom-boxes[i].top);
            boxes[i].w := constrain(0, 1, boxes[i].w);
            boxes[i].h := constrain(0, 1, boxes[i].h)
        end
end;

procedure fill_truth_swag(const path: string; const truth: PSingle; const classes: longint; const flip: boolean; const dx, dy, sx, sy: single);
var
    labelpath: string;
    count, id, i, index: longint;
    boxes : TArray<TBoxLabel>;
    x, y, w, h: single;
begin
    replace_image_to_label(path, labelpath);
    count := 0;
    boxes := read_boxes(labelpath, count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    i := 0;
    while (i < count) and (i < 90) do begin
        x := boxes[i].x;
        y := boxes[i].y;
        w := boxes[i].w;
        h := boxes[i].h;
        id := boxes[i].id;
        if (w < 0.0) or (h < 0.0) then
            continue;
        index := (4+classes) * i;
        truth[_inc(index)] := x;
        truth[_inc(index)] := y;
        truth[_inc(index)] := w;
        truth[_inc(index)] := h;
        if id < classes then
            truth[index+id] := 1;
        inc(i)
    end;
    //free(boxes)
end;

procedure fill_truth_region(const path: string; const truth: PSingle; const classes, num_boxes: longint; const flip: boolean; const dx, dy, sx, sy: single);
var
    labelpath: string;
    boxes: TArray<TBoxLabel>;
    count: longint;
    x: single;
    y: single;
    w: single;
    h: single;
    id: longint;
    i: longint;
    col: longint;
    row: longint;
    index: longint;
begin
    replace_image_to_label(path, labelpath);
    count := 0;
    boxes := read_boxes(labelpath, count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    for i := 0 to count -1 do
        begin
            x := boxes[i].x;
            y := boxes[i].y;
            w := boxes[i].w;
            h := boxes[i].h;
            id := boxes[i].id;
            if (w < 0.001) or (h < 0.001) then
                continue;
            col := trunc((x * num_boxes));
            row := trunc((y * num_boxes));
            x := x * num_boxes-col;
            y := y * num_boxes-row;
            index := (col+row * num_boxes) * (5+classes);
            if truth[index]<>0 then
                continue;
            truth[_inc(index)] := 1;
            if id < classes then
                truth[index+id] := 1;
            index := index + classes;
            truth[_inc(index)] := x;
            truth[_inc(index)] := y;
            truth[_inc(index)] := w;
            truth[_inc(index)] := h
        end;
    //free(boxes)
end;

procedure load_rle(const im: TImageData; const rle: TArray<longint>;
  const n: longint);
var
    count: longint;
    curr: longint;
    i: longint;
    j: longint;
begin
    count := 0;
    curr := 0;
    for i := 0 to n -1 do
        begin
            for j := 0 to rle[i] -1 do
                im.data[_inc(count)] := curr;
            curr := 1-curr
        end;
    while count < im.h * im.w * im.c do begin
        im.data[count] := curr;
        inc(count)
    end
end;

procedure or_image(const src, dest: TImageData; const c: longint);
var
    i: longint;
begin
    for i := 0 to src.w * src.h -1 do
        if src.data[i]<>0 then
            dest.data[dest.w * dest.h * c+i] := 1
end;

procedure exclusive_image(const src: TImageData);
var
    k,j,i,s: longint;
begin
    s := src.w * src.h;
    for k := 0 to src.c-2 do
        for i := 0 to s -1 do
            if src.data[k * s+i]<>0 then
                for j := k+1 to src.c -1 do
                    src.data[j * s+i] := 0
end;

function bound_image(const im: TImageData):TBox;
var
    x,y,minx,miny,maxx,maxy: longint;
begin
    minx := im.w;
    miny := im.h;
    maxx := 0;
    maxy := 0;
    for y := 0 to im.h -1 do
        for x := 0 to im.w -1 do
            if im.data[y * im.w+x]<>0 then
                begin
                    if (x < minx) then
                        minx := x
                    else
                        minx := minx;
                    if (y < miny) then
                        miny := y
                    else
                        miny := miny;
                    if (x > maxx) then
                        maxx := x
                    else
                        maxx := maxx;
                    if (y > maxy) then
                        maxy := y
                    else
                        maxy := maxy
                end;
    result.x := minx;
    result.y := miny;
    result.w :=  maxx-minx+1;
    result.h :=  maxy-miny+1
end;

procedure fill_truth_iseg(const path: string; const num_boxes: longint; const truth: PSingle; const classes, w, h: longint; const aug: TAugmentArgs; flip: boolean; const mw, mh: longint);
var
    labelpath, buff: string;
    id, i, j, n: longint;
    part, sized, mask: TImageData;
    rle: TArray<longint>;
    f:TextFile;
begin

    labelPath := StringReplace(path, 'images', 'mask', []);
    labelPath := StringReplace(labelpath, 'JPEGImages', 'mask', []);
    labelPath := StringReplace(labelpath, '.jpg', '.txt', [rfIgnoreCase]);
    //labelPath := StringReplace((labelpath, '.JPG', '.txt', labelpath);
    labelPath := StringReplace(labelpath, '.jpeg', '.txt', [rfIgnoreCase]);
    if not FileExists(labelPath) then
        raise EFileNotFoundException.CreateFmt('Cannot find file[%s]',[labelPath]);
    AssignFile(f,labelpath);
    reset(f);
    i := 0;
    part := make_image(w, h, 1);
    while not EOF(f) and (i < num_boxes) do
      try
        read(f, id, buff); delete(buff,1,1);
        n := 0;
        rle := read_intlist(buff, n, 0);
        load_rle(part, rle, n);
        sized := rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if flip then
            flip_image(sized);
        mask := resize_image(sized, mw, mh);
        truth[i * (mw * mh+1)] := id;
        for j := 0 to mw * mh -1 do
            truth[i * (mw * mh+1)+1+j] := mask.data[j];
        inc(i);
        free_image(mask);
        free_image(sized);
        //free(rle)
      except
      end;
    if i < num_boxes then
        truth[i * (mw * mh+1)] := -1;
    close(f);
    free_image(part)
end;

procedure fill_truth_mask(const path: string; const num_boxes: longint; const truth: PSingle; const classes, w, h: longint; const aug: TAugmentArgs; const flip: boolean; const mw, mh: longint);
var
    labelpath, buff: string;
    id, i, j, n: longint;
    f:TextFile;
    part, sized, crop, mask: TImageData;
    rle:TArray<longint>;
    b: TBox;
begin
    labelPath:=StringReplace(path, 'images', 'mask', []);
    labelPath := StringReplace(labelpath, 'JPEGImages', 'mask', []);
    labelPath := StringReplace(labelpath, '.jpg', '.txt', [rfIgnoreCase]);
    labelPath := StringReplace(labelpath, '.jpeg', '.txt', [rfIgnoreCase]);
    if not FileExists(labelPath) then
        raise EFileNotFoundException.CreateFmt('Cannot find file[%s]',[labelPath]);
    assignFile(f,labelPath);
    reset(f);
    i := 0;
    part := make_image(w, h, 1);
    while not EOF(F) and (i< num_boxes) do
      try
        read(f, id, buff); delete(buff,1,1);
        n := 0;
        rle := read_intlist(buff,  n, 0);
        load_rle(part, rle, n);
        sized := rotate_crop_image(part, aug.rad, aug.scale, aug.w, aug.h, aug.dx, aug.dy, aug.aspect);
        if flip then
            flip_image(sized);
        b := bound_image(sized);
        if b.w > 0 then
            begin
                crop := crop_image(sized, round(b.x), round(b.y), round(b.w), round(b.h));
                mask := resize_image(crop, mw, mh);
                truth[i * (4+mw * mh+1)+0] := (b.x+b.w / 2.0) / sized.w;
                truth[i * (4+mw * mh+1)+1] := (b.y+b.h / 2.0) / sized.h;
                truth[i * (4+mw * mh+1)+2] := b.w / sized.w;
                truth[i * (4+mw * mh+1)+3] := b.h / sized.h;
                for j := 0 to mw * mh -1 do
                    truth[i * (4+mw * mh+1)+4+j] := mask.data[j];
                truth[i * (4+mw * mh+1)+4+mw * mh] := id;
                free_image(crop);
                free_image(mask);
                inc(i)
            end;
        free_image(sized);
        //free(rle)
      except
      end;
    close(f);
    free_image(part)
end;

procedure fill_truth_detection(const path: string; const num_boxes: longint; const truth: PSingle; const classes: longint; const flip: boolean; const dx, dy, sx, sy: single);
var
    labelpath: string;
    count, id, i, sub, track_id: longint;
    boxes :TArray<TBoxLabel>;
    x,y,w,h: single;
begin
    replace_image_to_label(path, labelpath);
    count := 0;
    boxes := read_boxes(labelpath,  count);
    randomize_boxes(boxes, count);
    correct_boxes(boxes, count, dx, dy, sx, sy, flip);
    if count > num_boxes then
        count := num_boxes;
    sub := 0;
    for i := 0 to count -1 do
        begin
            x := boxes[i].x;
            y := boxes[i].y;
            w := boxes[i].w;
            h := boxes[i].h;
            id := boxes[i].id;
            track_id := boxes[i].track_id;
            if ((w < 0.001) or (h < 0.001)) then
                begin
                    inc(sub);
                    continue
                end;
            truth[(i-sub) * 5+0] := x;
            truth[(i-sub) * 5+1] := y;
            truth[(i-sub) * 5+2] := w;
            truth[(i-sub) * 5+3] := h;
            truth[(i-sub) * 5+4] := id;
            truth[(i-sub) * 5+5] := track_id;
        end;
    //free(boxes)
end;

const NUMCHARS = 37;

procedure print_letters(const pred: PSingle; const n: longint);
var
    i: longint;
    index: longint;
    P:Pointer;
begin
    P:=@pred[i * NUMCHARS];
    for i := 0 to n -1 do
        begin
            index := max_index(p, NUMCHARS);
            write(int_to_alphanum(index))
        end;
    writeln('');
end;

procedure fill_truth_captcha(const path: string; const n: longint; const truth: PSingle);
var
    _begin: string;
    i: longint;
    index: longint;
begin
    _begin := copy(path,RPos(DirectorySeparator,path)+1);
    i := 0;
    while (i < length(_begin)) and (i < n) and (_begin[i] <> '.') do begin
        index := alphanum_to_int(_begin[i]);
        if index > 35 then
            writeln('Bad ', _begin[i]);
        truth[i * NUMCHARS+index] := 1;
        inc(i)
    end;
    while i < n do begin
        truth[i * NUMCHARS+NUMCHARS-1] := 1;
        inc(i)
    end
end;

function load_data_captcha(paths: TStringArray; const n, m, k, w, h: longint):TData;
var
    //d: TData;
    i: longint;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := Default(TData);
    result.shallow := false;
    result.X := load_image_paths(paths, n, w, h);
    result.y := make_matrix(n, k * NUMCHARS);
    for i := 0 to n -1 do
        fill_truth_captcha(paths[i], k, @result.y.vals[i][0]);
    //if m then
        //free(paths);
end;

function load_data_captcha_encode(paths: TStringArray; const n, m, w, h: longint):TData;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := Default(TData);
    result.shallow := false;
    result.X := load_image_paths(paths, n, w, h);
    result.X.cols := 17100;
    result.y := result.X;
    //if m then
    //    free(paths);
end;

procedure fill_truth(const path: string; const labels: TStringArray; const k: longint; const truth: PSingle);
var
    i, count: longint;
begin

    FillChar(truth[0], k * sizeof(single),0);
    count := 0;
    for i := 0 to k -1 do
        if pos(labels[i],path)>0 then
            begin
                truth[i] := 1;
                inc(count)
            end;
    if (count <> 1) and ((k <> 1) or (count <> 0)) then
        writeln(format('Too many or too few labels: %d, %s',[ count, path]))
end;

procedure fill_truth_smooth(path: string; labels: TArray<string>; k: longint; truth: Psingle; label_smooth_eps: single);
var
    i: longint;
    count: longint;
begin
    filldword(truth[0], k, 0);
    count := 0;
    for i := 0 to k -1 do
        begin
            if pos(labels[i], path) > 0 then
                begin
                    truth[i] := (1-label_smooth_eps);
                    inc(count)
                end
            else
                truth[i] := label_smooth_eps / (k-1)
        end;
    if count <> 1 then
        begin
            writeln('Too many or too few labels:', count,',  ', path);
            count := 0;
            for i := 0 to k -1 do
                if pos(labels[i], path) > 0 then
                    begin
                        writeln(#9' label ', count, ':  ', labels[i]);
                        inc(count)
                    end
        end
end;


procedure fill_hierarchy(const truth: PSingle; const k: longint;
  const hierarchy: TTree);
var
    j: longint;
    parent: longint;
    i: longint;
    count: longint;
    mask: boolean;
begin
    for j := 0 to k -1 do
        if truth[j]<>0 then
            begin
                parent := hierarchy.parent[j];
                while (parent >= 0) do
                    begin
                        truth[parent] := 1;
                        parent := hierarchy.parent[parent]
                    end
            end;
    count := 0;
    for j := 0 to hierarchy.groups -1 do
        begin
            mask := true;
            for i := 0 to hierarchy.group_size[j] -1 do
                if truth[count+i]<>0 then
                    begin
                        mask := false;
                        break
                    end;
            if mask then
                for i := 0 to hierarchy.group_size[j] -1 do
                    truth[count+i] := SECRET_NUM;
            count := count + hierarchy.group_size[j]
        end
end;

function load_regression_labels_paths(const paths: TStringArray; const n, k: longint):TMatrix;
var
    i, j: longint;
    labelpath: string;
    f:TextFile;
begin
    result:= make_matrix(n, k);
    for i := 0 to n -1 do
        begin
            labelPath:=StringReplace(paths[i], 'images', 'labels', []);
            labelPath:=StringReplace(labelpath, 'JPEGImages', 'labels', []);
            labelPath:=StringReplace(labelpath, '.bmp', '.txt', [rfIgnoreCase]);
            labelPath:=StringReplace(labelpath, '.jpeg', '.txt', [rfIgnoreCase]);
            labelPath:=StringReplace(labelpath, '.jpg', '.txt', [rfIgnoreCase]);
            labelPath:=StringReplace(labelpath, '.png', '.txt', [rfIgnoreCase]);
            labelPath:=StringReplace(labelpath, '.tif', '.txt', [rfIgnoreCase]);
            if not FileExists(labelPath) then
                continue;
            assignFile(f,labelPath);
            reset(f);
            for j := 0 to k -1 do
                read(f, result.vals[i][j]);
            close(f)
        end;
end;

function load_labels_paths(const paths: TStringArray; const n: longint; const labels: TStringArray; const k: longint; const hierarchy: TArray<TTree>):TMatrix;
var
    i: longint;
begin
    result := make_matrix(n, k);
    i := 0;
    while (i < n) and assigned(labels) do begin
        fill_truth(paths[i], labels, k, @result.vals[i][0]);
        if assigned(hierarchy) then
            fill_hierarchy(@result.vals[i][0], k, hierarchy[0]);
        inc(i)
    end;
end;

function load_tags_paths(const paths: TStringArray; const n, k: longint):TMatrix;
var
    i: longint;
    _label: string;
    f:TextFile;
    tag: longint;
begin
    result := make_matrix(n, k);
    for i := 0 to n -1 do
        begin
            _label := stringReplace(paths[i], 'images', 'labels', []);
            _label := stringReplace(_label, '.jpg', '.txt', [rfIgnoreCase]);
            if not fileExists(_label) then
                continue;//raise EFileNotFoundException.Create(_label+' : was not found');
            AssignFile(f,_label);
            Reset(f);
            while not EOF(f) do
                read(f, tag);
                begin
                    if tag < k then
                        result.vals[i][tag] := 1
                end;
            close(f)
        end;
end;

function get_labels_custom(filename: string; size: Plongint):TArray<string>;
begin
    result := get_paths(filename);
    size[0]:=length(result);
end;

function get_labels(const filename: string):TStringArray;
//var
//    labels: TStringArray;
begin
    result := get_paths(filename);
    //labels := char(list_to_array(plist));
    //free_list(plist);
    //exit(labels)
end;

procedure free_data(var d: TData);
begin
    //if not d.shallow then
    //    begin
    //        free_matrix(d.X);
    //        free_matrix(d.y)
    //    end
    //else
    //    begin
    //        d.x.vals:=nil;
    //        d.y.vals:=nil;
    //        //free(d.X.vals);
    //        //free(d.y.vals)
    //    end
end;

function get_segmentation_image(const path: string; const w, h: longint; const classes: longint):TImageData;
var
    labelpath: string;
    mask: TImageData;
    buff: string;
    id: longint;
    part: TImageData;
    f:TextFile;
    n: longint;
    rle: TArray<longint>;
begin
    labelpath := StringReplace(path, 'images', 'mask', []);
    labelpath := StringReplace(labelpath, 'JPEGImages', 'mask', []);
    labelpath := StringReplace(labelpath, '.jpg', '.txt', [rfIgnoreCase]);
    //labelpath := StringReplace(labelpath, '.JPG', '.txt', labelpath);
    labelpath := StringReplace(labelpath, '.jpeg', '.txt', [rfIgnoreCase]);
    result := make_image(w, h, classes);
    if not FileExists(labelPath) then
        raise EFileNotFoundException.Create(labelpath + ' : was not found');
    Assign(f, labelPath);
    reset(f);
    part := make_image(w, h, 1);
    while not EOF(F) do
        begin
            read(f,id, buff); delete(buff,1,1);// must delete first character because it is expected to be a space
            n := 0;
            rle := read_intlist(buff, n, 0);
            load_rle(part, rle, n);
            or_image(part, result, id);
            //free(rle)
        end;
    close(f);
    free_image(part)
end;

function get_segmentation_image2(const path: string; const w, h, classes: longint):TImageData;
var
    labelpath: string;
    i: longint;
    buff: string;
    id: longint;
    part: TImageData;
    n: longint;
    rle : TArray<longint>;
    f:TextFile;
begin
    labelPath := stringReplace(path, 'images', 'mask', []);
    labelPath := stringReplace(labelpath, 'JPEGImages', 'mask', []);
    labelPath := stringReplace(labelpath, '.jpg', '.txt', [rfIgnoreCase]);
    //labelPath := stringReplace(labelpath, '.JPG', '.txt', labelpath);
    labelPath := stringReplace(labelpath, '.jpeg', '.txt', [rfIgnoreCase]);
    result := make_image(w, h, classes+1);
    for i := 0 to w * h -1 do
        result.data[w * h * classes+i] := 1;
    if not FileExists(labelPath) then
      raise EFileNotFoundException.Create(labelPath+' : was not found');
    AssignFile(f,labelPath);
    reset(f);
    part := make_image(w, h, 1);
    while not EOF(f) do //(fscanf(file, '%d %s',  and id, buff) = 2) do
        begin
            read(f, id, buff); delete(buff,1,1);
            n := 0;
            rle := read_intlist(buff,  n, 0);
            load_rle(part, rle, n);
            or_image(part, result, id);
            for i := 0 to w * h -1 do
                if part.data[i]<>0 then
                    result.data[w * h * classes+i] := 0;
            //free(rle)
        end;
    close(f);
    free_image(part);
    //exit(result)
end;

function load_data_seg(const n: longint; const paths: TStringArray; const m, w, h, classes, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single; const _div: longint):TData;
var
    random_paths: TStringArray;
    i: longint;
    a: TAugmentArgs;
    sized, orig, mask, sized_m: TImageData;
    flip: boolean;
begin
    random_paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    result.y.rows := n;
    result.y.cols := h * w * classes div _div div _div;
    setLength(result.y.vals, result.y.rows);
    //result.y.vals := AllocMem(result.X.rows * sizeof(TSingles));

    for i := 0 to n -1 do
        begin
            orig := load_image_color(random_paths[i], 0, 0);
            a := random_augment_args(orig, angle, aspect, _min, _max, w, h);
            sized := rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
            flip := boolean(trandom(2));
            if flip then
                flip_image(sized);
            random_distort_image(sized, hue, saturation, exposure);
            result.X.vals[i] := sized.data;
            mask := get_segmentation_image(random_paths[i], orig.w, orig.h, classes);
            sized_m := rotate_crop_image(mask, a.rad, a.scale / _div, a.w div _div, a.h div _div, a.dx / _div, a.dy / _div, a.aspect);
            if flip then
                flip_image(sized_m);
            result.y.vals[i] := sized_m.data;
            free_image(orig);
            free_image(mask)
        end;
    //free(random_paths);
    //exit(result)
end;

function load_data_iseg(const n: longint; const paths: TStringArray; m, w, h, classes, boxes, _div, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single):TData;
var
    random_paths: TStringArray;
    i: longint;
    orig, sized: TImageData;
    a: TAugmentArgs;
    flip: boolean;
begin
    random_paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals, result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    result.y := make_matrix(n, (((w div _div) * (h div _div))+1) * boxes);
    for i := 0 to n -1 do
        begin
            orig := load_image_color(random_paths[i], 0, 0);
            a := random_augment_args(orig, angle, aspect, _min, _max, w, h);
            sized := rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
            flip := boolean(trandom(2));
            if flip then
                flip_image(sized);
            random_distort_image(sized, hue, saturation, exposure);
            result.X.vals[i] := sized.data;
            fill_truth_iseg(random_paths[i], boxes, @result.y.vals[i][0], classes, orig.w, orig.h, a, flip, w div _div, h div _div);
            free_image(orig)
        end;
    //free(random_paths);
    //exit(result)
end;

function load_data_mask(const n: longint; const paths: TStringArray; const m, w, h, classes, boxes, coords, _min, _max: longint; const angle, aspect, hue, saturation, exposure: single):TData;
var
    random_paths: TStringArray;
    i: longint;
    orig, sized: TImageData;
    a: TAugmentArgs;
    flip: boolean;
begin
    random_paths := get_random_paths(paths, n, m);
    result := Default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    result.y := make_matrix(n, (coords+1) * boxes);
    for i := 0 to n -1 do
        begin
            orig := load_image_color(random_paths[i], 0, 0);
            a := random_augment_args(orig, angle, aspect, _min, _max, w, h);
            sized := rotate_crop_image(orig, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
            flip := boolean(random(2));
            if flip then
                flip_image(sized);
            random_distort_image(sized, hue, saturation, exposure);
            result.X.vals[i] := sized.data;
            fill_truth_mask(random_paths[i], boxes, @result.y.vals[i][0], classes, orig.w, orig.h, a, flip, 14, 14);
            free_image(orig)
        end;
    //free(random_paths);
    //exit(result)
end;

function load_data_region(const n: longint; const paths: TStringArray; const m, w, h, size, classes: longint; jitter: single; hue: single; const saturation, exposure: single):TData;
var
    random_paths: TStringArray;
    i, k, oh, ow, dw, dh, pleft, pright, ptop, pbot, swidth,sheight: longint;
    d: TData;
    orig: TImageData;
    sx, sy, dx, dy: single;
    flip: boolean;
    cropped, sized: TImageData;
begin
    random_paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    k := size * size * (5+classes);
    result.y := make_matrix(n, k);
    for i := 0 to n -1 do
        begin
            orig := load_image_color(random_paths[i], 0, 0);
            oh := orig.h;
            ow := orig.w;
            dw := trunc(ow * jitter);
            dh := trunc(oh * jitter);
            pleft := trunc(rand_uniform(-dw, dw));
            pright := trunc(rand_uniform(-dw, dw));
            ptop := trunc(rand_uniform(-dh, dh));
            pbot := trunc(rand_uniform(-dh, dh));
            swidth := ow-pleft-pright;
            sheight := oh-ptop-pbot;
            sx := swidth / ow;
            sy := sheight / oh;
            flip := boolean(trandom(2));
            cropped := crop_image(orig, pleft, ptop, swidth, sheight);
            dx := (pleft / ow) / sx;
            dy := (ptop / oh) / sy;
            sized := resize_image(cropped, w, h);
            if flip then
                flip_image(sized);
            random_distort_image(sized, hue, saturation, exposure);
            result.X.vals[i] := sized.data;
            fill_truth_region(random_paths[i], @result.y.vals[i][0], classes, size, flip, dx, dy, 1. / sx, 1. / sy);
            free_image(orig);
            free_image(cropped)
        end;
    //free(random_paths);
    //exit(result)
end;

function load_data_compare(const n: longint; paths: TStringArray; const m, classes, w, h: longint):TData;
var
    i,j,k,id: longint;
    d: TData;
    im1, im2: TImageData;
    iou: single;
    fp1,fp2: TextFile;
    imlabel1, imlabel2: string;
begin
    if m<>0 then
        paths := get_random_paths(paths, 2 * n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals := AllocMem( result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 6;
    k := 2 * (classes);
    result.y := make_matrix(n, k);
    for i := 0 to n -1 do
        begin
            im1 := load_image_color(paths[i * 2], w, h);
            im2 := load_image_color(paths[i * 2+1], w, h);
            //result.X.vals[i] := TSingles.Create(result.X.cols);
            setLength(result.X.vals[i], result.X.cols);
            move(im1.data[0],result.X.vals[i][0],  h * w * 3 * sizeof(Single));
            move(im2.data[0],result.X.vals[i][h * w * 3], h * w * 3 * sizeof(Single));
            imlabel1 := StringReplace(paths[i * 2], 'imgs', 'labels', []);
            imlabel1 := StringReplace(imlabel1, 'jpg', 'txt', []);
            AssignFile(fp1, imlabel1);
            reset(fp1);
            while not EOF(fp1) do begin
              read(fp1, id,  iou);
              if result.y.vals[i][2 * id] < iou then
                  result.y.vals[i][2 * id] := iou
            end;
            imlabel2 := StringReplace(paths[i * 2+1], 'imgs', 'labels', []);
            imlabel2 := StringReplace(imlabel2, 'jpg', 'txt', []);
            AssignFile(fp2, imlabel2);
            reset(fp2);
            while not EOF(fp2) do
                begin
                    read(fp2, id, iou) ;
                    if result.y.vals[i][2 * id+1] < iou then
                        result.y.vals[i][2 * id+1] := iou
                end;
            for j := 0 to classes -1 do
                begin
                    if (result.y.vals[i][2 * j] > 0.5) and (result.y.vals[i][2 * j+1] < 0.5) then
                        begin
                            result.y.vals[i][2 * j] := 1;
                            result.y.vals[i][2 * j+1] := 0
                        end
                    else
                        if (result.y.vals[i][2 * j] < 0.5) and (result.y.vals[i][2 * j+1] > 0.5) then
                            begin
                                result.y.vals[i][2 * j] := 0;
                                result.y.vals[i][2 * j+1] := 1
                            end
                    else
                        begin
                            result.y.vals[i][2 * j] := SECRET_NUM;
                            result.y.vals[i][2 * j+1] := SECRET_NUM
                        end
                end;
            close(fp1);
            close(fp2);
            free_image(im1);
            free_image(im2)
        end;
    //if m then
    //    free(paths);
    //exit(result)
end;

function load_data_swag(const paths: TStringArray; const n, classes: longint; const jitter: single):TData;
var
    index, h, w, k, dw, dh, pleft, pright, ptop, pbot, swidth, sheight: longint;
    orig, cropped, sized: TImageData;
    sx, sy, dx, dy: single;
    flip: boolean;
    random_path : string;
begin
    index := trandom(n);
    random_path := paths[index];
    orig := load_image_color(random_path, 0, 0);
    h := orig.h;
    w := orig.w;
    result := default(TData);
    result.shallow := false;
    result.w := w;
    result.h := h;
    result.X.rows := 1;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals:=AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    k := (4+classes) * 90;
    result.y := make_matrix(1, k);
    dw := trunc(w * jitter);
    dh := trunc(h * jitter);
    pleft := trunc(rand_uniform(-dw, dw));
    pright := trunc(rand_uniform(-dw, dw));
    ptop := trunc(rand_uniform(-dh, dh));
    pbot := trunc(rand_uniform(-dh, dh));
    swidth := w-pleft-pright;
    sheight := h-ptop-pbot;
    sx := swidth / w;
    sy := sheight / h;
    flip := boolean(trandom(2));
    cropped := crop_image(orig, pleft, ptop, swidth, sheight);
    dx := (pleft / w) / sx;
    dy := (ptop / h) / sy;
    sized := resize_image(cropped, w, h);
    if flip then
        flip_image(sized);
    result.X.vals[0] := sized.data;
    fill_truth_swag(random_path, @result.y.vals[0][0], classes, flip, dx, dy, 1. / sx, 1. / sy);
    free_image(orig);
    free_image(cropped);
    //exit(result)
end;

function load_data_detection(const n: longint; const paths: TStringArray; const m, w, h, boxes, classes: longint; const jitter, hue, saturation, exposure: single):TData;
var
    random_paths: TStringArray;
    i: longint;
    orig, sized: TImageData;
    dw, dh, nw, nh, dx, dy, new_ar, scale: single;
    flip: boolean;
begin
    random_paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals ,result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.X.cols := h * w * 3;
    result.y := make_matrix(n, 5 * boxes);
    for i := 0 to n -1 do
        begin
            orig := load_image_color(random_paths[i], 0, 0);
            sized := make_image(w, h, orig.c);
            fill_image(sized, 0.5);
            dw := jitter * orig.w;
            dh := jitter * orig.h;
            new_ar := (orig.w+rand_uniform(-dw, dw)) / (orig.h+rand_uniform(-dh, dh));
            scale := 1;
            if new_ar < 1 then
                begin
                    nh := scale * h;
                    nw := nh * new_ar
                end
            else
                begin
                    nw := scale * w;
                    nh := nw / new_ar
                end;
            dx := rand_uniform(0, w-nw);
            dy := rand_uniform(0, h-nh);
            place_image(orig, trunc(nw), trunc(nh), trunc(dx), trunc(dy), sized);
            random_distort_image(sized, hue, saturation, exposure);
            flip := boolean(trandom(2));
            if flip then
                flip_image(sized);
            result.X.vals[i] := sized.data;
            fill_truth_detection(random_paths[i], boxes, @result.y.vals[i][0], classes, flip, -dx / w, -dy / h, nw / w, nh / h);
            free_image(orig)
        end;
    //free(random_paths);
    //exit(result)
end;

procedure load_thread(ptr: pointer);
var
    a: PLoadArgs absolute ptr;
begin
    //a :=  PLoadArgs(ptr)[0];
    if a.exposure = 0 then
        a.exposure := 1;
    if a.saturation = 0 then
        a.saturation := 1;
    if a.aspect = 0 then
        a.aspect := 1;
    if a.&type = dtOLD_CLASSIFICATION_DATA then
         a.d[0] := load_data_old(a.paths, a.n, a.m, a.labels, a.classes, a.w, a.h)
    else
        if a.&type = dtREGRESSION_DATA then
             a.d[0] := load_data_regression(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure)
    else
        if a.&type = dtCLASSIFICATION_DATA then
             a.d[0] := load_data_augment(a.paths, a.n, a.m, a.labels, a.classes, a.hierarchy, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.center)
    else
        if a.&type = dtSUPER_DATA then
             a.d[0] := load_data_super(a.paths, a.n, a.m, a.w, a.h, a.scale)
    else
        if a.&type = dtWRITING_DATA then
             a.d[0] := load_data_writing(a.paths, a.n, a.m, a.w, a.h, a.out_w, a.out_h)
    else
        if a.&type = dtISEG_DATA then
             a.d[0] := load_data_iseg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.scale, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure)
    else
        if a.&type = dtINSTANCE_DATA then
             a.d[0] := load_data_mask(a.n, a.paths, a.m, a.w, a.h, a.classes, a.num_boxes, a.coords, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure)
    else
        if a.&type = dtSEGMENTATION_DATA then
             a.d[0] := load_data_seg(a.n, a.paths, a.m, a.w, a.h, a.classes, a.min, a.max, a.angle, a.aspect, a.hue, a.saturation, a.exposure, a.scale)
    else
        if a.&type = dtREGION_DATA then
             a.d[0] := load_data_region(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure)
    else
        if a.&type = dtDETECTION_DATA then
             a.d[0] := load_data_detection(a.n, a.paths, a.m, a.w, a.h, a.num_boxes, a.classes, a.jitter, a.hue, a.saturation, a.exposure)
    else
        if a.&type = dtSWAG_DATA then
             a.d[0] := load_data_swag(a.paths, a.n, a.classes, a.jitter)
    else
        if a.&type = dtCOMPARE_DATA then
             a.d[0] := load_data_compare(a.n, a.paths, a.m, a.classes, a.w, a.h)
    else
        if a.&type = dtIMAGE_DATA then
            begin
                 a.im[0] := load_image_color(a.path, 0, 0);
                 a.resized[0] := resize_image( a.im[0], a.w, a.h)
            end
    else
        if a.&type = dtLETTERBOX_DATA then
            begin
                 a.im[0] := load_image_color(a.path, 0, 0);
                 a.resized[0] := letterbox_image( a.im[0], a.w, a.h)
            end
    else
        if a.&type = dtTAG_DATA then
             a.d[0] := load_data_tag(a.paths, a.n, a.m, a.classes, a.min, a.max, a.size, a.angle, a.aspect, a.hue, a.saturation, a.exposure);
    //free(ptr);
    //result:=nil
end;

function load_data_in_thread(args: Pointer):TThread;
begin
    {
        pthread_t thread;
        struct load_args *ptr = calloc(1, sizeof(struct load_args));
        *ptr = args;
        if(pthread_create(&thread, 0, load_thread, ptr)) error("Thread creation failed");
        return thread;
    }
  result := ExecuteInThread(load_thread,args)
end;

procedure load_threads(ptr: Pointer);
var
    i: longint;
    args: TArray<TLoadArgs>;
    buffers : TArray<TData>;
    threads : TArray<TThread>;
    _out: PData;
    total: longint;
    a:PLoadArgs absolute ptr;
begin
    //a :=  * load_args(ptr);
    if (a.threads = 0) then
        a.threads := 1;
    _out := a.d;
    total := a.n;
    //free(ptr);
    setLength(buffers,a.threads);
    setLength(args, a.threads);
    setLength(threads, a.threads);
    for i := 0 to a.threads -1 do
        begin
            args[i]:=a^;
            args[i].d := @buffers[i];
            args[i].n := (i+1) * total div a.threads - i * total div a.threads;
            threads[i] := load_data_in_thread(@args[i])
            //threads[i] := TThread.ExecuteInThread(load_thread,@args[i]);
        end;
    for i := 0 to a.threads -1 do
       threads[i].WaitFor;
     //*_out := concat_datas(buffers, a.threads);
    _out.shallow := false;
    for i := 0 to a.threads -1 do
        begin
            buffers[i].shallow := true;
            free_data(buffers[i])
        end;
    //free(buffers);
    //free(threads);
    //result:=nil
end;

procedure load_data_blocking(const args: TLoadArgs);
begin
    //ptr := calloc(1, sizeof());
    // * ptr := args;
    load_thread(@args)
end;

function load_data(const args: TLoadArgs):TThread;
begin
    //ptr := calloc(1, sizeof());
     //* ptr := args;
    result := ExecuteInThread(load_threads,@args);
    //if pthread_create( and thread, 0, load_threads, ptr) then
    //    error('Thread creation failed');
    //exit(thread)
end;

function load_data_writing(paths: TStringArray; const n, m, w, h, out_w, out_h: longint):TData;
var
    replace_paths: TStringArray;
    //d: TData;
    i: longint;
begin
    // todo check paths in load_data_* may be a local var when (m > 0) ? or maybe its not neccessary.
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    replace_paths := find_replace_paths(paths, n, '.png', '-label.png');
    result := default(TData);
    result.shallow := false;
    result.X := load_image_paths(paths, n, w, h);
    result.y := load_image_paths_gray(replace_paths, n, out_w, out_h);
    //if m then
    //    free(paths);
    //for i := 0 to n -1 do
        //free(replace_paths[i]);
    //free(replace_paths);
    //exit(result)
end;

function load_data_old(paths: TStringArray; const n, m: longint; const labels: TStringArray; const k, w, h: longint):TData;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X := load_image_paths(paths, n, w, h);
    result.y := load_labels_paths(paths, n, labels, k, nil);
    //if m<>0 then
    //    free(paths);
    //exit(result)
end;

function load_data_super(paths: TStringArray; const n, m, w, h, scale: longint):TData;
var
    d: TData;
    i: longint;
    im, crop, resize: TImageData;
    //crop: image;
    flip: boolean;
    //resize: image;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X.rows := n;
    setLength(result.X.vals,n);
    //result.X.vals := AllocMem(n * sizeof(TSingles));
    result.X.cols := w * h * 3;
    result.y.rows := n;
    setLength(result.y.vals, n);
    //result.y.vals := AllocMem(n * sizeof(TSingles));
    result.y.cols := w * scale * h * scale * 3;
    for i := 0 to n -1 do
        begin
            im := load_image_color(paths[i], 0, 0);
            crop := random_crop_image(im, w * scale, h * scale);
            flip := boolean(trandom(2));
            if flip then
                flip_image(crop);
            resize := resize_image(crop, w, h);
            result.X.vals[i] := resize.data;
            result.y.vals[i] := crop.data;
            free_image(im)
        end;
    //if m then
    //    free(paths);
    //exit(result)
end;

function load_data_regression(paths: TStringArray; const n, m, k, _min, _max, size: longint; const angle, aspect, hue, saturation, exposure: single):TData;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.X := load_image_augment_paths(paths, n, _min, _max, size, angle, aspect, hue, saturation, exposure, false);
    result.y := load_regression_labels_paths(paths, n, k);
    //if m<>0 then
    //    free(paths);
    //exit(result)
end;

function select_data(const orig: PData; const inds: Plongint):TData;
var
    i: longint;
begin
    result := default(TData);
    result.shallow := true;
    result.w := orig[0].w;
    result.h := orig[0].h;
    result.X.rows := orig[0].X.rows;
    result.y.rows := orig[0].X.rows;
    result.X.cols := orig[0].X.cols;
    result.y.cols := orig[0].y.cols;
    setLength(result.X.vals, orig[0].X.rows);
    //result.X.vals := AllocMem(orig[0].X.rows * sizeof(TSingles));
    setLength(result.y.vals, orig[0].y.rows);
    //result.y.vals := AllocMem(orig[0].y.rows * sizeof(TSingles));
    for i := 0 to result.X.rows -1 do
        begin
            result.X.vals[i] := orig[inds[i]].X.vals[i];
            result.y.vals[i] := orig[inds[i]].y.vals[i]
        end;
    //exit(d)
end;

function tile_data(const orig: TData; const divs, size: longint):TArray<TData>;
var
    //ds: TArray<TData>;
    i, j, x, y: longint;
    d: TData;
    im: TImageData;
begin
    setLength(result, divs * divs);
    // todo paralleliz for
    for i := 0 to divs * divs -1 do
        begin
            d.shallow := false;
            d.w := orig.w div divs * size;
            d.h := orig.h div divs * size;
            d.X.rows := orig.X.rows;
            d.X.cols := d.w * d.h * 3;
            setLength(d.X.vals, d.X.rows);
            //d.X.vals := AllocMem(d.X.rows * sizeof(TSingles));
            d.y := copy_matrix(orig.y);
            // todo paralleliz for
            for j := 0 to orig.X.rows -1 do
                begin
                    x := (i mod divs) * orig.w div divs-(d.w-orig.w div divs) div 2;
                    y := (i div divs) * orig.h div divs-(d.h-orig.h div divs) div 2;
                    im := float_to_image(orig.w, orig.h, 3, @orig.X.vals[j][0]);
                    d.X.vals[j] := crop_image(im, x, y, d.w, d.h).data
                end;
            result[i] := d
        end;
    //exit(result)
end;

function resize_data(const orig: TData; const w, h: longint):TData;
var
    i: longint;
    im: TImageData;
begin
    result := default(TData);
    result.shallow := false;
    result.w := w;
    result.h := h;
    result.X.rows := orig.X.rows;
    result.X.cols := w * h * 3;
    setLength(result.X.vals, result.X.rows);
    //result.X.vals := AllocMem(result.X.rows * sizeof(TSingles));
    result.y := copy_matrix(orig.y);
    for i := 0 to orig.X.rows -1 do
        begin
            im := float_to_image(orig.w, orig.h, 3, @orig.X.vals[i][0]);
            result.X.vals[i] := resize_image(im, w, h).data
        end;
    //exit(result)
end;

function load_data_augment(paths: TStringArray; const n, m: longint;
  const labels: TStringArray; const k: longint; const hierarchy: TArray<TTree>;
  const _min, _max, size: longint; const angle, aspect, hue, saturation,
  exposure: single; const center: boolean): TData;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.shallow := false;
    result.w := size;
    result.h := size;
    result.X := load_image_augment_paths(paths, n, _min, _max, size, angle, aspect, hue, saturation, exposure, center);
    result.y := load_labels_paths(paths, n, labels, k, hierarchy);
    //if m<>0 then
    //    free(paths);
    //exit(result)
end;

function load_data_tag(paths: TStringArray; const n, m, k, _min, _max, size: longint; const angle, aspect, hue, saturation, exposure: single):TData;
begin
    if m<>0 then
        paths := get_random_paths(paths, n, m);
    result := default(TData);
    result.w := size;
    result.h := size;
    result.shallow := false;
    result.X := load_image_augment_paths(paths, n, _min, _max, size, angle, aspect, hue, saturation, exposure, false);
    result.y := load_tags_paths(paths, n, k);
    //if m<>0 then
    //    free(paths);
    //exit(d)
end;

function concat_matrix(const m1, m2: TMatrix):TMatrix;
var
    i, count: longint;
    //m: TMatrix;
begin
    count := 0;
    result.cols := m1.cols;
    result.rows := m1.rows+m2.rows;
    setLength(result.vals, m1.rows+m2.rows );
    //result.vals := AllocMem((m1.rows+m2.rows)*sizeof(TSingles));
    for i := 0 to m1.rows -1 do
        result.vals[_inc(count)] := m1.vals[i];
    for i := 0 to m2.rows -1 do
        result.vals[_inc(count)] := m2.vals[i];
    //exit(result)
end;

function concat_data(const d1, d2: TData):TData;
begin
    result := default(TData);
    result.shallow := true;
    result.X := concat_matrix(d1.X, d2.X);
    result.y := concat_matrix(d1.y, d2.y);
    result.w := d1.w;
    result.h := d1.h;
    //exit(result)
end;

function concat_datas(const d: TArray<TData>; const n: longint):TData;
var
    i: longint;
    //out: TData;
    _new: TData;
begin
    result := default(TData);
    for i := 0 to n -1 do
        begin
            _new := concat_data(d[i], result);
            free_data(result);
            result := _new
        end;
    //exit(result)
end;

function load_categorical_data_csv(const filename: string; const target, k: longint):TData;
var
    //d: TData;
    X, y: TMatrix;
    truth: TArray<TArray<single>>;
    //truth :TSingles2d;
    truth_1d :TArray<Single>;
    //truth_1d :TSingles;

    //y: matrix;
begin
    result := default(TData);
    result.shallow := false;
    X := csv_to_matrix(filename);
    truth_1d := pop_column( X, target);
    //truth := one_hot_encode(truth_1d, X.rows, k);
    one_hot_encode(truth_1d, X.rows, k, truth);
    y.rows := X.rows;
    y.cols := k;
    y.vals := truth;
    result.X := X;
    result.y := y;
    //free(truth_1d);
    //truth_1d.free
    //exit(result)
end;

function load_cifar10_data(const filename: string):TData;
var
    d: TData;
    i, j : int64;
    X, y: TMatrix;
    bytes: array[0..3072] of byte;
    &class, n: longint;
    fp:File;
begin
    result := default(TData);
    result.shallow := false;
    X := make_matrix(10000, 3072);
    y := make_matrix(10000, 10);
    result.X := X;
    result.y := y;
    if not FileExists(filename) then
        raise EFileNotFoundException.create(filename+': was not found!');
    AssignFile(fp, filename);
    reset(fp, 1);
    for i := 0 to 10000 -1 do
        begin
            BlockRead(fp,bytes, 3073, n);
            &class := bytes[0];
            y.vals[i][&class] := 1;
            for j := 0 to X.cols -1 do
                X.vals[i][j] := {double(}bytes[j+1]{)}
        end;
    scale_data_rows(result, 1.0 / 255);
    CloseFile(fp);
    //exit(result)
end;

procedure get_random_batch(const d: TData; const n: longint; const X, y: PSingle);
var
    j: longint;
    index: longint;
begin
    for j := 0 to n -1 do
        begin
            index := random(d.X.rows);
            move(d.X.vals[index][0], X[j * d.X.cols], d.X.cols * sizeof(single));
            //memcpy(X+j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
            move(d.y.vals[index][0], y[j * d.y.cols], d.y.cols * sizeof(single));
            //memcpy(y+j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float))
        end
end;

procedure get_next_batch(const d: TData; const n, offset: longint; const X, y: PSingle);
var
    j: longint;
    index: longint;
begin
    for j := 0 to n -1 do
        begin
            index := offset+j;
            move(d.X.vals[index][0], X[j * d.X.cols], d.X.cols * sizeof(single));
            //memcpy(X+j * d.X.cols, d.X.vals[index], d.X.cols * sizeof(float));
            if assigned(y) then
                move(d.y.vals[index][0], y[j * d.y.cols], d.y.cols * sizeof(single));
                //memcpy(y+j * d.y.cols, d.y.vals[index], d.y.cols * sizeof(float))
        end
end;

procedure smooth_data(const d: TData);
var
    i, j: longint;
    scale, eps: single;
begin
    scale := 1.0 / d.y.cols;
    eps := 0.1;
    for i := 0 to d.y.rows -1 do
        for j := 0 to d.y.cols -1 do
            d.y.vals[i][j] := eps * scale+(1-eps) * d.y.vals[i][j]
end;

function load_all_cifar10():TData;
var
    i, j, b, n: longint;
    X, y: TMatrix;
    fp :file;
    buff: string;
    bytes: array [0..3072] of byte;
    &class: longint;
begin
    result := default(TData);
    result.shallow := false;
    X := make_matrix(50000, 3072);
    y := make_matrix(50000, 10);
    result.X := X;
    result.y := y;
    for b := 0 to 5 -1 do
        begin
            buff := format('data/cifar/cifar-10-batches-bin/data_batch_%result.bin', [b+1]);
            if not FileExists(buff) then
                raise EFileNotFoundException.Create(buff+ ': was not found!');
            AssignFile(fp,buff);
            reset(fp,1);
            for i := 0 to 10000 -1 do
                begin
                    BlockRead(fp, bytes, 3073, n);
                    &class := bytes[0];
                    y.vals[i+b * 10000][&class] := 1;
                    for j := 0 to X.cols -1 do
                        X.vals[i+b * 10000][j] := {double(}bytes[j+1]{)}
                end;
            closeFile(fp)
        end;
    scale_data_rows(result, 1. / 255);
    smooth_data(result);
    //exit(result)
end;

function load_go(const filename: string):TData;
var
    X, y: TMatrix;
    row, col, count, i, index: longint;
    _label: string;
    board: string;
    val: single;
    fp :TextFile;
    //d: TData;
begin
    if not FileExists(filename) then
        raise EFileNotFoundException.create(filename+': was not found!');
    AssignFile(fp,filename);
    reset(fp);
    X := make_matrix(3363059, 361);
    y := make_matrix(3363059, 361);
    //if not fp then
    //    file_error(filename);
    count := 0;
    while not eof(fp) do begin
    //repeat
        readln(fp,row,col);
        //((_label := fgetl(fp))) do
        if count = X.rows then
            begin
                X := resize_matrix(X, count * 2);
                y := resize_matrix(y, count * 2)
            end;
        //sscanf(_label, '%d %d',  and row,  and col);
        readln(fp,board);
        index := row * 19+col;
        y.vals[count][index] := 1;
        for i := 0 to 19 * 19 -1 do
            begin
                val := 0;
                if board[i] = '1' then
                    val := 1
                else if board[i] = '2' then
                    val := -1;
                X.vals[count][i] := val
            end;
        inc(count);
        //free(_label);
        //free(board)
    end;
    //until eof(fp);
    X := resize_matrix(X, count);
    y := resize_matrix(y, count);
    result := default(TData);
    result.shallow := false;
    result.X := X;
    result.y := y;
    close(fp);
    //exit(result)
end;

procedure randomize_data(const d: TData);
var
    i, index: longint;
    swap:TArray<single>;
    //swap : TSingles;
begin
    i := d.X.rows-1;
    while i > 0 do begin
        index := random(i);
        swap := d.X.vals[index];
        d.X.vals[index] := d.X.vals[i];
        d.X.vals[i] := swap;
        swap := d.y.vals[index];
        d.y.vals[index] := d.y.vals[i];
        d.y.vals[i] := swap;
        dec(i)
    end
end;

procedure scale_data_rows(const d: TData; const s: single);
var
    i: longint;
begin
    for i := 0 to d.X.rows -1 do
        scale_array(@d.X.vals[i][0], d.X.cols, s)
end;

procedure translate_data_rows(const d: TData; const s: single);
var
    i: longint;
begin
    for i := 0 to d.X.rows -1 do
        translate_array(@d.X.vals[i][0], d.X.cols, s)
end;

function copy_data(const d: TData):TData;
begin
    result := default(TData);
    result.w := d.w;
    result.h := d.h;
    result.shallow := false;
    result.num_boxes := d.num_boxes;
    result.boxes := d.boxes;
    result.X := copy_matrix(d.X);
    result.y := copy_matrix(d.y);
    //exit(result)
end;

procedure normalize_data_rows(const d: TData);
var
    i: longint;
begin
    for i := 0 to d.X.rows -1 do
        normalize_array(@d.X.vals[i][0], d.X.cols)
end;

function get_data_part(const d: TData; const part, total: longint):TData;
begin
    result := default(TData);
    result.shallow := true;
    result.X.rows := d.X.rows * (part+1) div total-d.X.rows * part div total;
    result.y.rows := d.y.rows * (part+1) div total-d.y.rows * part div total;
    result.X.cols := d.X.cols;
    result.y.cols := d.y.cols;
    result.X.vals := copy(d.X.vals, d.X.rows * part div total, result.X.rows);
    result.y.vals := copy(d.y.vals, d.y.rows * part div total, result.y.rows);
    //exit(result)
end;

function get_random_data(const d: TData; const num: longint):TData;
var
    r: TData;
    i: longint;
    index: longint;
begin
    result := default(TData);
    result.shallow := true;
    result.X.rows := num;
    result.y.rows := num;
    result.X.cols := d.X.cols;
    result.y.cols := d.y.cols;
    //result.X.vals := calloc(num, sizeof(float * ));
    setLength(result.X.vals, num);
    //result.X.vals := AllocMem(num * sizeof(TSingles));
    //result.y.vals := calloc(num, sizeof(float * ));
    setLength(result.Y.vals, num);
    //result.y.vals := AllocMem(num * sizeof(TSingles));
    for i := 0 to num -1 do
        begin
            index := random(d.X.rows);
            result.X.vals[i] := d.X.vals[index];
            result.y.vals[i] := d.y.vals[index]
        end;
    //exit(result)
end;

function split_data(const d: TData; const part, total: longint):TArray<TData>;
var
    //split: PData;
    i, start, &end: longint;
    train, test: TData;
begin
    setLength(result ,2);
    start := part * d.X.rows div total;
    &end := (part+1) * d.X.rows div total;
    train.shallow := true; test.shallow := true;
    test.y.rows := &end-start;
    test.X.rows := test.y.rows;
    train.y.rows := d.X.rows-(&end-start);
    train.X.rows := train.y.rows;
    test.X.cols := d.X.cols;
    train.X.cols := test.X.cols;
    test.y.cols := d.y.cols;
    train.y.cols := test.y.cols;
    //train.X.vals := calloc(train.X.rows, sizeof(float * ));
    setLength(train.X.vals, train.X.rows);// := AllocMem(train.X.rows * sizeof(TSingles));
    //test.X.vals := calloc(test.X.rows, sizeof(float * ));
    setLength(test.X.vals, test.X.rows);// := AllocMem(test.X.rows * sizeof(TSingles));
    //train.y.vals := calloc(train.y.rows, sizeof(float * ));
    setLength(train.y.vals, train.y.rows);// := AllocMem(train.y.rows * sizeof(TSingles));
    //test.y.vals := calloc(test.y.rows, sizeof(float * ));
    setLength(test.y.vals, test.y.rows);// := AllocMem(test.y.rows * sizeof(TSingles));
    for i := 0 to start -1 do
        begin
            train.X.vals[i] := d.X.vals[i];
            train.y.vals[i] := d.y.vals[i]
        end;
    for i := start to &end -1 do
        begin
            test.X.vals[i-start] := d.X.vals[i];
            test.y.vals[i-start] := d.y.vals[i]
        end;
    for i := &end to d.X.rows -1 do
        begin
            train.X.vals[i-(&end-start)] := d.X.vals[i];
            train.y.vals[i-(&end-start)] := d.y.vals[i]
        end;
    result[0] := train;
    result[1] := test;
    //exit(split)
end;

initialization
//  InitCriticalSection(mutex);

finalization
//  DoneCriticalSection(mutex);

end.

