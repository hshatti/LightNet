unit imagedata;

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
  Classes, SysUtils,
  {$ifdef STB_IMAGE}stb_image, {$else STB_IMAGE} {$if defined(FRAMEWORK_FMX)}FMX.Graphics, FMX.Types {$else}Graphics{$endif} ,{$endif}
  math, lightnet, blas;


function get_color(const c, x, _max: longint):single;
function mask_to_rgb(const mask: TImageData):TImageData;
function get_pixel(const m: TImageData; const x, y, c: longint):single;
function get_pixel_extend(const m: TImageData; const x, y, c: longint):single;
procedure set_pixel(const m: TImageData; const x, y, c: longint; const val: single);
procedure add_pixel(const m: TImageData; const x, y, c: longint; const val: single);
function bilinear_interpolate(const im: TImageData; x, y: single; const c: longint):single;
procedure composite_image(const source, dest: TImageData; const dx, dy: longint);
function border_image(const a: TImageData; const border: longint):TImageData;
function tile_images(const a, b: TImageData; const dx: longint):TImageData;
function get_label(const characters: TArray<TArray<TImageData>>; const str: string; size: longint):TImageData;
function get_label_v3(const characters: TArray<TArray<TImageData>>; const str: string; size: longint):TImageData;
procedure draw_label(const a: TImageData; r: longint; const c: longint; const _label: TImageData; const rgb: PSingle);
procedure draw_weighted_label(const a: TImageData; r: longint; const c: longint; &label: TImageData; const rgb: PSingle; const alpha: single);
procedure draw_box_bw(const a: TImageData; x1, y1, x2, y2: longint; const brightness: single);
procedure draw_box_width_bw(const a: TImageData; const x1, y1, x2, y2, w: longint; const brightness: single);
procedure draw_box(const a: TImageData; x1, y1, x2, y2: longint; const r, g, b: single);
procedure draw_box_width(const a: TImageData; const x1, y1, x2, y2, w: longint; const r, g, b: single);
procedure draw_bbox(const a: TImageData; const bbox: TBox; const w: longint; const r, g, b: single);
function load_alphabet():TArray<TArray<TImageData>>;
procedure draw_detections_v3(const im: TImageData; const dets: TArray<TDetection>; const num: longint; const thresh: single; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes, ext_output: longint);
procedure draw_detections(const im: TImageData; const dets: TArray<TDetection>; const num: longint; const thresh: single; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes: longint);                     overload;
procedure draw_detections(const im: TImageData; const num: longint; const thresh: single; const boxes: TArray<TBox>; probs: TArray<PSingle>; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes: longint);overload;
procedure transpose_image(const im: TImageData);
procedure rotate_image_cw(const im: TImageData; times: longint);
procedure flip_image(const a: TImageData);
function image_distance(const a, b: TImageData):TImageData;
procedure ghost_image(const source, dest: TImageData; const dx, dy: longint);
procedure blocky_image(const im: TImageData; const s: longint);
procedure censor_image(const im: TImageData; dx, dy:longint; const w, h: longint);
procedure embed_image(const source, dest: TImageData; const dx, dy: longint);
function collapse_image_layers(const source: TImageData; const border: longint):TImageData;
procedure constrain_image(const im: TImageData);
procedure normalize_image(const p: TImageData);
procedure normalize_image2(const p: TImageData);
procedure copy_image_into(const src, dest: TImageData);
function copy_image(const p: TImageData):TImageData;
procedure rgbgr_image(const im: TImageData);
function show_image(const p: TImageData; const name: string; const ms: longint):longint;
procedure save_image_options(const im: TImageData; const name: string; const f: TImType; const quality: longint);
procedure save_image(im: TImageData; const name: string);
procedure show_image_layers(const p: TImageData; const name: string);
procedure show_image_collapsed(const p: TImageData; const name: string);
function make_empty_image(const w, h, c: longint):TImageData;
function make_image(const w, h, c: longint):TImageData;
function make_random_image(const w, h, c: longint):TImageData;
function float_to_image_scaled(const w, h, c: longint; const data: Psingle):TImageData;
function float_to_image(const w, h, c: longint; const data: TSingles):TImageData;
procedure place_image(const im: TImageData; const w, h, dx, dy: longint; const canvas: TImageData);
function center_crop_image(const im: TImageData; const w, h: longint):TImageData;
function rotate_crop_image(const im: TImageData; const rad, s: single; const w, h: longint; const dx, dy, aspect: single):TImageData;
function rotate_image(const im: TImageData; const rad: single):TImageData;
procedure fill_image(const m: TImageData; const s: single);
procedure translate_image(const m: TImageData; const s: single);
procedure scale_image(const m: TImageData; const s: single);
function crop_image(const im: TImageData; const dx, dy, w, h: longint):TImageData;
function best_3d_shift_r(const a: TImageData; const b: TImageData; const _min, _max: longint):longint;
function best_3d_shift(const a, b: TImageData; const _min, _max: longint):longint;
procedure composite_3d(const f1, f2: string; _out: string; const delta: longint);
procedure letterbox_image_into(const im: TImageData; const w, h: longint; const boxed: TImageData);
function letterbox_image(const im: TImageData; const w, h: longint):TImageData;
function resize_max(const im: TImageData; const _max: longint):TImageData;
function resize_min(const im: TImageData; const _min: longint):TImageData;
function random_crop_image(const im: TImageData; const w, h: longint):TImageData;
function random_augment_args(const im: TImageData; const angle:single; aspect: single; const low, high, w, h: longint):TAugmentArgs;
function random_augment_image(const im: TImageData; const angle, aspect: single; const low, high, w, h: longint):TImageData;
function three_way_max(const a, b, c: single):single;
function three_way_min(const a, b, c: single):single;
procedure yuv_to_rgb(const im: TImageData);
procedure rgb_to_yuv(const im: TImageData);
procedure rgb_to_hsv(const im: TImageData);
procedure hsv_to_rgb(const im: TImageData);
procedure grayscale_image_3c(const im: TImageData);
function grayscale_image(const im: TImageData):TImageData;
function threshold_image(const im: TImageData; const thresh: single):TImageData;
function blend_image(const fore: TImageData; const back: TImageData; const alpha: single):TImageData;
procedure scale_image_channel(const im: TImageData; const c: longint; const v: single);
procedure translate_image_channel(const im: TImageData; const c: longint; const v: single);
function binarize_image(const im: TImageData):TImageData;
procedure saturate_image(const im: TImageData; const sat: single);
procedure hue_image(const im: TImageData; const hue: single);
procedure exposure_image(const im: TImageData; const sat: single);
procedure distort_image(const im: TImageData; const hue, sat, val: single);
procedure random_distort_image(const im: TImageData; const hue, saturation, exposure: single);
procedure saturate_exposure_image(const im: TImageData; const sat, exposure: single);
procedure quantize_image(const im: TImageData);
procedure make_image_red(const im: TImageData);
function make_attention_image(const img_size: longint;
  const original_delta_cpu, original_input_cpu: TArray<single>;
  w: longint; h: longint; c: longint; alpha: single): TImageData;function resize_image(const im: TImageData; const w, h: longint):TImageData;
procedure test_resize(const filename: string);
function load_image_stb(const filename: string; channels: longint):TImageData;
function load_image(const filename: string; const w, h, c: longint):TImageData;
function load_image_color(const filename: string; const w, h: longint):TImageData;
function get_image_layer(const m: TImageData; const l: longint):TImageData;
procedure print_image(const m: TImageData);
function collapse_images_vert(const ims: TArray<TImageData>; const n: longint):TImageData;
function collapse_images_horz(const ims: TArray<TImageData>; const n: longint):TImageData;

procedure show_image_normalized(const im: TImageData; const name: string);

procedure show_images(const ims: TArray<TImageData>; const n: longint; const window: string);
procedure free_image(const m: TImageData);
procedure copy_image_from_bytes(const im: TImageData; const pdata: PAnsiChar);
function bitmapToImage(const bmp:TBitmap):TImageData;
function imageToBitmap(const im:TImageData; bmp:TBitmap=nil):TBitmap;

implementation
uses box, utils;
const colors: array [0..5,0..2] of single = ( (1,0,1), (0,0,1),(0,1,1),(0,1,0),(1,1,0),(1,0,0) );

function get_color(const c, x, _max: longint):single;
var
    ratio: single;
    i,j: longint;
begin
    ratio := (x / _max) * 5;
    i := floor(ratio);
    j := ceil(ratio);
    ratio := ratio - i;
    result := (1-ratio) * colors[i][c]+ratio * colors[j][c];
end;

function mask_to_rgb(const mask: TImageData):TImageData;
var
    n, i, j, offset: longint;
    red, green, blue: single;
begin
    n := mask.c;
    result := make_image(mask.w, mask.h, 3);
    for j := 0 to n -1 do
        begin
            offset := j * 123457 mod n;
            red := get_color(2, offset, n);
            green := get_color(1, offset, n);
            blue := get_color(0, offset, n);
            for i := 0 to result.w * result.h -1 do
                begin
                    result.data[i+0 * result.w * result.h] := result.data[i+0 * result.w * result.h] + (mask.data[j * result.h * result.w+i] * red);
                    result.data[i+1 * result.w * result.h] := result.data[i+1 * result.w * result.h] + (mask.data[j * result.h * result.w+i] * green);
                    result.data[i+2 * result.w * result.h] := result.data[i+2 * result.w * result.h] + (mask.data[j * result.h * result.w+i] * blue)
                end
        end;
end;

function get_pixel(const m: TImageData; const x, y, c: longint):single;
begin
    assert((x < m.w) and (y < m.h) and (c < m.c));
    result := m.data[c * m.h * m.w+y * m.w+x]
end;

function get_pixel_extend(const m: TImageData; const x, y, c: longint):single;
begin
    if (x < 0) or (x >= m.w) or (y < 0) or (y >= m.h) then
        exit(0);
    if (c < 0) or (c >= m.c) then
        exit(0);
    result := get_pixel(m, x, y, c)
end;

procedure set_pixel(const m: TImageData; const x, y, c: longint; const val: single);
begin
    if (x < 0) or (y < 0) or (c < 0) or (x >= m.w) or (y >= m.h) or (c >= m.c) then
        exit();
    assert((x < m.w) and (y < m.h) and (c < m.c));
    m.data[c * m.h * m.w+y * m.w+x] := val
end;

procedure add_pixel(const m: TImageData; const x, y, c: longint; const val: single);
begin
    assert((x < m.w) and (y < m.h) and (c < m.c));
    m.data[c * m.h * m.w+y * m.w+x] := m.data[c * m.h * m.w+y * m.w+x] + val
end;

function bilinear_interpolate(const im: TImageData; x, y: single; const c: longint):single;
var
    ix, iy: longint;
    dx, dy: single;
begin
    ix := floor(x);
    iy := floor(y);
    dx := x-ix;
    dy := y-iy;
    result := (1-dy) * (1-dx) * get_pixel_extend(im, ix, iy, c)+dy * (1-dx) * get_pixel_extend(im, ix, iy+1, c)+(1-dy) * dx * get_pixel_extend(im, ix+1, iy, c)+dy * dx * get_pixel_extend(im, ix+1, iy+1, c);
end;

procedure composite_image(const source, dest: TImageData; const dx, dy: longint);
var
    x, y, k: longint;
    val, val2: single;
begin
    for k := 0 to source.c -1 do
        for y := 0 to source.h -1 do
            for x := 0 to source.w -1 do
                begin
                    val := get_pixel(source, x, y, k);
                    val2 := get_pixel_extend(dest, dx+x, dy+y, k);
                    set_pixel(dest, dx+x, dy+y, k, val * val2)
                end
end;

function border_image(const a: TImageData; const border: longint):TImageData;
var
    x, y, k: longint;
    val: single;
begin
    result := make_image(a.w+2 * border, a.h+2 * border, a.c);
    for k := 0 to result.c -1 do
        for y := 0 to result.h -1 do
            for x := 0 to result.w -1 do
                begin
                    val := get_pixel_extend(a, x-border, y-border, k);
                    if (x-border < 0) or (x-border >= a.w) or (y-border < 0) or (y-border >= a.h) then
                        val := 1;
                    set_pixel(result, x, y, k, val)
                end;
end;

function tile_images(const a, b: TImageData; const dx: longint):TImageData;
begin
    if a.w = 0 then
        exit(copy_image(b));
    result := make_image(a.w+b.w+dx, ifthen((a.h > b.h), a.h, b.h), ifthen((a.c > b.c), a.c, b.c));
    fill_cpu(result.w * result.h * result.c, 1, @result.data[0], 1);
    embed_image(a, result, 0, 0);
    composite_image(b, result, a.w+dx, 0);
end;

// note the following function will assemble an image with text [str] from images or charachters
function get_label(const characters: TArray<TArray<TImageData>>; const str: string;
  size: longint): TImageData;
var
    _label: TImageData;
    l, n: TImageData;
    i: longint;
begin
    //size := size div 10;
    if (size > 7) then
        size := 7;
    _label := make_empty_image(0, 0, 0);
    for i:=1 to length(str) do
        begin
            l := characters[size][longint(str[i])];
            n := tile_images(_label, l, -size-1+(size+1) div 2);
            free_image(_label);
            _label := n;
        end;
    result := border_image(_label,trunc( _label.h * 0.25));
    free_image(_label);
end;

function get_label_v3(const characters: TArray<TArray<TImageData>>; const str: string; size: longint):TImageData;
var
    &label, l, n: TImageData;
    i:longint;
begin
    size := size div 10;
    if (size > 7) then
        size := 7;
    &label := make_empty_image(0, 0, 0);
    for i:=1 to length(str) do
        begin
            l := characters[size][longint(str[i])];
            n := tile_images(&label, l, -size-1+(size+1) div 2);
            free_image(&label);
            &label := n;
        end;
    result := border_image(&label, trunc(&label.h * 0.05));
    free_image(&label);
end;

procedure draw_label(const a: TImageData; r: longint; const c: longint; const _label: TImageData; const rgb: PSingle);
var
    w, h, i, j, k: longint;
    val: single;
begin
    w := _label.w;
    h := _label.h;
    if r-h >= 0 then
        r := r-h;
    j := 0;
    while (j < h) and (j+r < a.h) do begin
        i := 0;
        while (i < w) and (i+c < a.w) do begin
            for k := 0 to _label.c -1 do
                begin
                    val := get_pixel(_label, i, j, k);
                    set_pixel(a, i+c, j+r, k, rgb[k] * val)
                end;
            inc(i)
        end;
        inc(j)
    end
end;

procedure draw_weighted_label(const a: TImageData; r: longint; const c: longint; &label: TImageData; const rgb: PSingle; const alpha: single);
var
    w, h, i, j, k: longint;
    val1, val2, val_dst: single;
begin
    w := &label.w;
    h := &label.h;
    if r-h >= 0 then
        r := r-h;
    j := 0;
    while (j < h) and (j+r < a.h) do begin
        i := 0;
        while (i < w) and (i+c < a.w) do begin
            for k := 0 to &label.c -1 do
                begin
                    val1 := get_pixel(&label, i, j, k);
                    val2 := get_pixel(a, i+c, j+r, k);
                    val_dst := val1 * rgb[k] * alpha+val2 * (1-alpha);
                    set_pixel(a, i+c, j+r, k, val_dst)
                end;
            inc(i)
        end;
        inc(j)
    end
end;

procedure draw_box_bw(const a: TImageData; x1, y1, x2, y2: longint; const brightness: single);
var
    i: longint;
begin
    if (x1 < 0) then
        x1 := 0;
    if (x1 >= a.w) then
        x1 := a.w-1;
    if x2 < 0 then
        x2 := 0;
    if x2 >= a.w then
        x2 := a.w-1;
    if y1 < 0 then
        y1 := 0;
    if y1 >= a.h then
        y1 := a.h-1;
    if y2 < 0 then
        y2 := 0;
    if y2 >= a.h then
        y2 := a.h-1;
    for i := x1 to x2 do
        begin
            a.data[i+y1 * a.w+0 * a.w * a.h] := brightness;
            a.data[i+y2 * a.w+0 * a.w * a.h] := brightness
        end;
    for i := y1 to y2 do
        begin
            a.data[x1+i * a.w+0 * a.w * a.h] := brightness;
            a.data[x2+i * a.w+0 * a.w * a.h] := brightness
        end
end;

procedure draw_box_width_bw(const a: TImageData; const x1, y1, x2, y2, w: longint; const brightness: single);
var
    i: longint;
    alternate_color: single;
begin
    for i := 0 to w -1 do
        begin
            if (w mod 2)<>0 then
                alternate_color := (brightness)
            else
                alternate_color := (1.0-brightness);
            draw_box_bw(a, x1+i, y1+i, x2-i, y2-i, alternate_color)
        end
end;


procedure draw_box(const a: TImageData; x1, y1, x2, y2: longint; const r, g, b: single);
var
    i: longint;
begin
    if (x1 < 0) then
        x1 := 0;
    if (x1 >= a.w) then
        x1 := a.w-1;
    if x2 < 0 then
        x2 := 0;
    if x2 >= a.w then
        x2 := a.w-1;
    if y1 < 0 then
        y1 := 0;
    if y1 >= a.h then
        y1 := a.h-1;
    if y2 < 0 then
        y2 := 0;
    if y2 >= a.h then
        y2 := a.h-1;
    for i := x1 to x2 do
        begin
            a.data[i+y1 * a.w+0 * a.w * a.h] := r;
            a.data[i+y2 * a.w+0 * a.w * a.h] := r;
            a.data[i+y1 * a.w+1 * a.w * a.h] := g;
            a.data[i+y2 * a.w+1 * a.w * a.h] := g;
            a.data[i+y1 * a.w+2 * a.w * a.h] := b;
            a.data[i+y2 * a.w+2 * a.w * a.h] := b
        end;
    for i := y1 to y2 do
        begin
            a.data[x1+i * a.w+0 * a.w * a.h] := r;
            a.data[x2+i * a.w+0 * a.w * a.h] := r;
            a.data[x1+i * a.w+1 * a.w * a.h] := g;
            a.data[x2+i * a.w+1 * a.w * a.h] := g;
            a.data[x1+i * a.w+2 * a.w * a.h] := b;
            a.data[x2+i * a.w+2 * a.w * a.h] := b
        end
end;

procedure draw_box_width(const a: TImageData; const x1, y1, x2, y2, w: longint; const r, g, b: single);
var
    i: longint;
begin
    for i := 0 to w -1 do
        draw_box(a, x1+i, y1+i, x2-i, y2-i, r, g, b)
end;

procedure draw_bbox(const a: TImageData; const bbox: TBox; const w: longint; const r, g, b: single);
var
    left, right, top, bot, i: longint;
begin
    left := trunc((bbox.x-bbox.w / 2) * a.w);
    right := trunc((bbox.x+bbox.w / 2) * a.w);
    top := trunc((bbox.y-bbox.h / 2) * a.h);
    bot := trunc((bbox.y+bbox.h / 2) * a.h);
    for i := 0 to w -1 do
        draw_box(a, left+i, top+i, right-i, bot-i, r, g, b)
end;

function load_alphabet():TArray<TArray<TImageData>>;
var
  i,j: longint;
  buff: string;
const
  nsize: longint = 8;
begin
    setLength(result, nsize);
    for j := 0 to nsize -1 do
        begin
            setLength(result[j], 128);
            for i := 32 to 127 -1 do
                begin
                    buff := format('data/labels/%d_%d.png', [i, j]);
                    result[j][i] := load_image_color(buff, 0, 0)
                end
        end
end;

function get_actual_detections(const dets: TArray<TDetection>; const dets_num: longint; const thresh: single; const selected_detections_num: PLongint; const names: TArray<string>):TArray<TDetectionWithClass>;
var
    selected_num: longint;
    i: longint;
    best_class: longint;
    best_class_prob: single;
    j: longint;
    show: boolean;
begin
    selected_num := 0;
    setLength(result, dets_num);
    for i := 0 to dets_num -1 do
        begin
            best_class := -1;
            best_class_prob := thresh;
            for j := 0 to dets[i].classes -1 do
                begin
                    show := names[j] <> 'dont_show';
                    if (dets[i].prob[j] > best_class_prob) and show then
                        begin
                            best_class := j;
                            best_class_prob := dets[i].prob[j]
                        end
                end;
            if best_class >= 0 then
                begin
                    result[selected_num].det := dets[i];
                    result[selected_num].best_class := best_class;
                    inc(selected_num)
                end
        end;
    if assigned(selected_detections_num) then
        selected_detections_num[0] := selected_num;
end;

function compare_by_lefts(const a, b: TDetectionWithClass):longint;
var delta: single;
begin
    delta := (a.det.bbox.x-a.det.bbox.w / 2)-(b.det.bbox.x-b.det.bbox.w / 2);
    exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

function compare_by_probs(const a, b: TDetectionWithClass):longint;
var
    delta: single;
begin
    delta := a.det.prob[a.best_class]-b.det.prob[b.best_class];
    exit(ifthen(delta < 0, -1, ifthen(delta > 0, 1, 0)))
end;

procedure draw_detections_v3(const im: TImageData; const dets: TArray<TDetection>; const num: longint; const thresh: single; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes, ext_output: longint);
var
    frame_id, selected_detections_num, i, best_class, j, width, offset: longint;
    red, green, blue: single;
    rgb : array[0..2] of single;
    b: TBox;
    left, right, top, bot: longint;
    labelstr, prob_str: string;
    &label, mask, resized_mask, tmask: TImageData;
    selected_detections : TArray<TDetectionWithClass>;
begin
    frame_id := 0;
    inc(frame_id);
    selected_detections := get_actual_detections(dets, num, thresh, @selected_detections_num, names);
    if selected_detections_num=0 then exit();
    TTools<TDetectionWithClass>.QuickSort(selected_detections, 0, selected_detections_num-1, compare_by_lefts);
    for i := 0 to selected_detections_num -1 do
        begin
            best_class := selected_detections[i].best_class;
            write(format('%s: %.0f%%', [names[best_class], selected_detections[i].det.prob[best_class] * 100]));
            if ext_output<>0 then
                writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)', [
                   (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
                   (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
                   selected_detections[i].det.bbox.w * im.w,
                   selected_detections[i].det.bbox.h * im.h]))
            else
                writeln('');
            for j := 0 to classes -1 do
                if (selected_detections[i].det.prob[j] > thresh) and (j <> best_class) then
                    begin
                        write(format('%s: %.0f%%', [names[j], selected_detections[i].det.prob[j] * 100]));
                        if ext_output<>0 then
                            writeln(format(#9'(left_x: %4.0f   top_y: %4.0f   width: %4.0f   height: %4.0f)',[
                                 (selected_detections[i].det.bbox.x-selected_detections[i].det.bbox.w / 2) * im.w,
                                 (selected_detections[i].det.bbox.y-selected_detections[i].det.bbox.h / 2) * im.h,
                                 selected_detections[i].det.bbox.w * im.w,
                                 selected_detections[i].det.bbox.h * im.h]))
                        else
                            writeln('')
                    end
        end;
    TTools<TDetectionWithClass>.QuickSort(selected_detections, 0, selected_detections_num-1, compare_by_probs);
    for i := 0 to selected_detections_num -1 do
        begin
            width := trunc(im.h * 0.002);
            if width < 1 then
                width := 1;
            offset := selected_detections[i].best_class * 123457 mod classes;
            red := get_color(2, offset, classes);
            green := get_color(1, offset, classes);
            blue := get_color(0, offset, classes);
            rgb[0] := red;
            rgb[1] := green;
            rgb[2] := blue;
            b := selected_detections[i].det.bbox;
            left  := trunc((b.x-b.w / 2) * im.w);
            right := trunc((b.x+b.w / 2) * im.w);
            top   := trunc((b.y-b.h / 2) * im.h);
            bot   := trunc((b.y+b.h / 2) * im.h);
            if left < 0 then
                left := 0;
            if right > im.w-1 then
                right := im.w-1;
            if top < 0 then
                top := 0;
            if bot > im.h-1 then
                bot := im.h-1;
            if im.c = 1 then
                draw_box_width_bw(im, left, top, right, bot, width, 0.8)
            else
                draw_box_width(im, left, top, right, bot, width, red, green, blue);
            if assigned(alphabet) then
                begin
                    labelstr:='';
                    labelstr:= labelstr + names[selected_detections[i].best_class];
                    prob_str := format(': %.2f', [selected_detections[i].det.prob[selected_detections[i].best_class]]);
                    labelstr := labelstr + prob_str;
                    for j := 0 to classes -1 do
                        if (selected_detections[i].det.prob[j] > thresh) and (j <> selected_detections[i].best_class) then
                                labelstr := labelstr + ', ' + names[j];
                    &label := get_label_v3(alphabet, labelstr, trunc(im.h * 0.02));
                    draw_weighted_label(im, top+width, left, &label, @rgb[0], 0.7);
                    free_image(&label)
                end;
            if assigned(selected_detections[i].det.mask) then
                begin
                    mask := float_to_image(14, 14, 1, @selected_detections[i].det.mask[0]);
                    resized_mask := resize_image(mask, trunc(b.w * im.w), trunc(b.h * im.h));
                    tmask := threshold_image(resized_mask, 0.5);
                    embed_image(tmask, im, left, top);
                    free_image(mask);
                    free_image(resized_mask);
                    free_image(tmask)
                end
        end;
    //free(selected_detections)
end;

procedure draw_detections(const im: TImageData; const num: longint; const thresh: single; const boxes: TArray<TBox>; probs: TArray<PSingle>; const names: TArray<string>; const alphabet: TArray<TArray<TImageData>>; const classes: longint);
var
    i, class_id: longint;
    prob: single;
    width, offset: longint;
    red, green, blue: single;
    rgb: array[0..2] of single;
    b: TBox;
    left, right, top, bot: longint;
    &label: TImageData;

begin
    for i := 0 to num -1 do
        begin
            class_id := max_index(Pointer(probs[i]), classes);
            prob := probs[i][class_id];
            if prob > thresh then
                begin
                    width := trunc(im.h * 0.012);
                    if false then
                        begin
                            width := trunc(power(prob, 1.0 / 2.0) * 10+1);
                            //alphabet := nil
                        end;
                    offset := class_id * 123457 mod classes;
                    red := get_color(2, offset, classes);
                    green := get_color(1, offset, classes);
                    blue := get_color(0, offset, classes);

                    rgb[0] := red;
                    rgb[1] := green;
                    rgb[2] := blue;
                    b := boxes[i];
                    left := trunc((b.x-b.w / 2) * im.w);
                    right := trunc((b.x+b.w / 2) * im.w);
                    top := trunc((b.y-b.h / 2) * im.h);
                    bot := trunc((b.y+b.h / 2) * im.h);
                    if left < 0 then
                        left := 0;
                    if right > im.w-1 then
                        right := im.w-1;
                    if top < 0 then
                        top := 0;
                    if bot > im.h-1 then
                        bot := im.h-1;
                    write(format('%s: %.0f%%', [names[class_id], prob * 100]));
                    writeln('');
                    draw_box_width(im, left, top, right, bot, width, red, green, blue);
                    if assigned(alphabet) then
                        begin
                            &label := get_label(alphabet, names[class_id], trunc((im.h * 0.03) / 10));
                            draw_label(im, top+width, left, &label, @rgb[0])
                        end
                end
        end
end;


procedure draw_detections(const im: TImageData; const dets: TArray<TDetection>;
  const num: longint; const thresh: single; const names: TArray<string>;
  const alphabet: TArray<TArray<TImageData>>; const classes: longint);
var
    i, j, &class, width, offset, left, right, top, bot: longint;
    labelstr: string;
    red, green, blue: single;
    rgb: array [0..2] of single;
    b: TBox;
    _label, mask, resized_mask, tmask: TImageData;
begin
    for i := 0 to num -1 do
        begin
            labelstr := '';
            &class := -1;
            for j := 0 to classes -1 do
                if dets[i].prob[j] > thresh then
                    begin
                        if &class < 0 then
                            begin
                                labelstr := labelstr + names[j];
                                &class := j
                            end
                        else
                            begin
                                labelstr := labelstr + ', ';
                                labelstr := labelstr + names[j]
                            end;
                        writeln(format('%s: %.0f%%', [names[j], dets[i].prob[j]*100 ]))
                    end;
            if &class >= 0 then
                begin
                    width := round(im.h * 0.006);
                    offset := &class * 123457 mod classes;
                    red := get_color(2, offset, classes);
                    green := get_color(1, offset, classes);
                    blue := get_color(0, offset, classes);

                    rgb[0] := red;
                    rgb[1] := green;
                    rgb[2] := blue;
                    b := dets[i].bbox;
                    left := round((b.x-b.w / 2) * im.w);
                    right := round((b.x+b.w / 2) * im.w);
                    top := round((b.y-b.h / 2) * im.h);
                    bot := round((b.y+b.h / 2) * im.h);
                    if left < 0 then
                        left := 0;
                    if right > im.w-1 then
                        right := im.w-1;
                    if top < 0 then
                        top := 0;
                    if bot > im.h-1 then
                        bot := im.h-1;
                    draw_box_width(im, left, top, right, bot, width, red, green, blue);
                    if assigned(alphabet) then
                        begin
                            _label := get_label(alphabet, labelstr, round(im.h * 0.03));
                            draw_label(im, top+width, left, _label, @rgb[0]);
                            free_image(_label)
                        end;
                    if assigned(dets[i].mask) then
                        begin
                            mask := float_to_image(14, 14, 1, @dets[i].mask[0]);
                            resized_mask := resize_image(mask, round(b.w * im.w), round(b.h * im.h));
                            tmask := threshold_image(resized_mask, 0.5);
                            embed_image(tmask, im, left, top);
                            free_image(mask);
                            free_image(resized_mask);
                            free_image(tmask)
                        end
                end
        end
end;

procedure transpose_image(const im: TImageData);
var
    n, m, c: longint;
    swap: single;
begin
    assert(im.w = im.h);
    for c := 0 to im.c -1 do
        for n := 0 to im.w-1 -1 do
            for m := n+1 to im.w -1 do
                begin
                    swap := im.data[m+im.w * (n+im.h * c)];
                    im.data[m+im.w * (n+im.h * c)] := im.data[n+im.w * (m+im.h * c)];
                    im.data[n+im.w * (m+im.h * c)] := swap
                end
end;

procedure rotate_image_cw(const im: TImageData; times: longint);
var
    i, x, y, c, n: longint;
    temp: single;
begin
    assert(im.w = im.h);
    times := (times+400) mod 4;
    n := im.w;
    for i := 0 to times -1 do
        for c := 0 to im.c -1 do
            for x := 0 to n div 2 -1 do
                for y := 0 to (n-1) div 2+1 -1 do
                    begin
                        temp := im.data[y+im.w * (x+im.h * c)];
                        im.data[y+im.w * (x+im.h * c)] := im.data[n-1-x+im.w * (y+im.h * c)];
                        im.data[n-1-x+im.w * (y+im.h * c)] := im.data[n-1-y+im.w * (n-1-x+im.h * c)];
                        im.data[n-1-y+im.w * (n-1-x+im.h * c)] := im.data[x+im.w * (n-1-y+im.h * c)];
                        im.data[x+im.w * (n-1-y+im.h * c)] := temp
                    end
end;

procedure flip_image(const a: TImageData);
var
    i, j, k, index, flip: longint;
    swap: single;
begin
    for k := 0 to a.c -1 do
        for i := 0 to a.h -1 do
            for j := 0 to a.w div 2 -1 do
                begin
                    index := j+a.w * (i+a.h * (k));
                    flip := (a.w-j-1)+a.w * (i+a.h * (k));
                    swap := a.data[flip];
                    a.data[flip] := a.data[index];
                    a.data[index] := swap
                end
end;

function image_distance(const a, b: TImageData):TImageData;
var
    i, j: longint;
    dist: TImageData;
begin
    result := make_image(a.w, a.h, 1);
    for i := 0 to a.c -1 do
        for j := 0 to a.h * a.w -1 do
            result.data[j] := result.data[j] + sqr(a.data[i * a.h * a.w+j]-b.data[i * a.h * a.w+j]{, 2});
    for j := 0 to a.h * a.w -1 do
        result.data[j] := sqrt(result.data[j]);
end;

procedure ghost_image(const source, dest: TImageData; const dx, dy: longint);
var
    x, y, k: longint;
    max_dist, dist, alpha, v1, v2, val: single;
begin
    max_dist := sqrt((-source.w / 2.0+0.5) * (-source.w / 2.0+0.5));
    for k := 0 to source.c -1 do
        for y := 0 to source.h -1 do
            for x := 0 to source.w -1 do
                begin
                    dist := sqrt((x-source.w / 2.0+0.5) * (x-source.w / 2.0+0.5)+(y-source.h / 2.0+0.5) * (y-source.h / 2.0+0.5));
                    alpha := (1-dist / max_dist);
                    if alpha < 0 then
                        alpha := 0;
                    v1 := get_pixel(source, x, y, k);
                    v2 := get_pixel(dest, dx+x, dy+y, k);
                    val := alpha * v1+(1-alpha) * v2;
                    set_pixel(dest, dx+x, dy+y, k, val)
                end
end;

procedure blocky_image(const im: TImageData; const s: longint);
var
    i, j, k: longint;
begin
    for k := 0 to im.c -1 do
        for j := 0 to im.h -1 do
            for i := 0 to im.w -1 do
                im.data[i+im.w * (j+im.h * k)] := im.data[i div s * s+im.w * (j div s * s+im.h * k)]
end;

procedure censor_image(const im: TImageData; dx, dy:longint; const w, h: longint);
var
    i,j,k, s: longint;
begin
    s := 32;
    if (dx < 0) then
        dx := 0;
    if dy < 0 then
        dy := 0;
    for k := 0 to im.c -1 do
        j := dy;
        while (j < dy+h) and (j < im.h) do begin
            i := dx;
            while (i < dx+w) and (i < im.w) do begin
                im.data[i+im.w * (j+im.h * k)] := im.data[i div s * s+im.w * (j div s * s+im.h * k)];
                inc(i)
            end;
            inc(j)
        end
end;

procedure embed_image(const source, dest: TImageData; const dx, dy: longint);
var
    x: longint;
    y: longint;
    k: longint;
    val: single;
begin
    for k := 0 to source.c -1 do
        for y := 0 to source.h -1 do
            for x := 0 to source.w -1 do
                begin
                    val := get_pixel(source, x, y, k);
                    set_pixel(dest, dx+x, dy+y, k, val)
                end
end;

function collapse_image_layers(const source: TImageData; const border: longint): TImageData;
var
    h, i, h_offset: longint;
    layer: TImageData;
begin
    h := source.h;
    h := (h+border) * source.c-border;
    result := make_image(source.w, h, 1);
    for i := 0 to source.c -1 do
        begin
            layer := get_image_layer(source, i);
            h_offset := i * (source.h+border);
            embed_image(layer, result, 0, h_offset);
            free_image(layer)
        end;
end;

procedure constrain_image(const im: TImageData);
var
    i: longint;
begin
    for i := 0 to im.w * im.h * im.c -1 do
        begin
            if im.data[i] < 0 then
                im.data[i] := 0;
            if im.data[i] > 1 then
                im.data[i] := 1
        end
end;

procedure normalize_image(const p: TImageData);
var
    i: longint;
    _min, _max, v: single;
begin
    _min := 9999999;
    _max := -999999;
    for i := 0 to p.h * p.w * p.c -1 do
        begin
            v := p.data[i];
            if v < _min then
                _min := v;
            if v > _max then
                _max := v
        end;
    if _max-_min < 0.000000001 then
        begin
            _min := 0;
            _max := 1
        end;
    for i := 0 to p.c * p.w * p.h -1 do
        p.data[i] := (p.data[i]-_min) / (_max-_min)
end;

procedure normalize_image2(const p: TImageData);
var
    _min, _max: TArray<single>;
    i, j: longint;
    v: single;
begin
    setLength(_min , p.c);
    setLength(_max , p.c);
    for i := 0 to p.c -1 do begin
        _max[i] := p.data[i * p.h * p.w];
        _min[i] := _max[i];
    end;
    for j := 0 to p.c -1 do
        for i := 0 to p.h * p.w -1 do
            begin
                v := p.data[i+j * p.h * p.w];
                if v < _min[j] then
                    _min[j] := v;
                if v > _max[j] then
                    _max[j] := v
            end;
    for i := 0 to p.c -1 do
        if _max[i]-_min[i] < 0.000000001 then
            begin
                _min[i] := 0;
                _max[i] := 1
            end;
    for j := 0 to p.c -1 do
        for i := 0 to p.w * p.h -1 do
            p.data[i+j * p.h * p.w] := (p.data[i+j * p.h * p.w]-_min[j]) / (_max[j]-_min[j]);
    //free(_min);
    //free(_max)
end;

procedure copy_image_into(const src, dest: TImageData);
begin
    move(src.data[0], dest.data[0], src.h * src.w * src.c * sizeof(single));
    //memcpy(dest.data, src.data, src.h * src.w * src.c * sizeof(float))
end;

function copy_image(const p: TImageData):TImageData;
begin
    result := p;
    result.data := copy(p.data);
    //setLength(result.data, p.h * p.w * p.c);//:= TSingles.Create(p.h * p.w * p.c);
    //move(p.data[0] ,result.data[0], p.h * p.w * p.c * sizeof(single));
end;

procedure rgbgr_image(const im: TImageData);
var
    i: longint;
    swap: single;
begin
    for i := 0 to im.w * im.h -1 do
        begin
            swap := im.data[i];
            im.data[i] := im.data[i+im.w * im.h * 2];
            im.data[i+im.w * im.h * 2] := swap
        end
end;

function show_image(const p: TImageData; const name: string; const ms: longint):longint;
var
    c: longint;
begin
  {$ifdef OPENCV}
    c := show_image_cv(p, name, ms);
    exit(c);
  {$else}
    writeln(ErrOutput, format('Not compiled with OpenCV, saving to %s.jpg instead', [name]));
    save_image(p, name);
    exit(-1)
  {$endif}
end;

procedure show_image_layers(const p: TImageData; const name: string);
var
    i: longint;
    buff: string;
    layer: TImageData;
begin
    for i := 0 to p.c -1 do
        begin
            buff := format( '%s - Layer %d', [name, i]);
            layer := get_image_layer(p, i);
            show_image(layer, buff, 1);
            free_image(layer)
        end
end;

procedure show_image_collapsed(const p: TImageData; const name: string);
var
    c: TImageData;
begin
    c := collapse_image_layers(p, 1);
    show_image(c, name, 1);
    free_image(c)
end;

function make_empty_image(const w, h, c: longint):TImageData;
begin
    result.data := nil;
    result.h := h;
    result.w := w;
    result.c := c;
end;

function make_image(const w, h, c: longint):TImageData;
begin
    result := make_empty_image(w, h, c);
    setLength(result.data , h * w * c);//:= TSingles.Create(h * w * c);
end;

function make_random_image(const w, h, c: longint):TImageData;
var
    i: longint;
begin
    result := make_empty_image(w, h, c);
    setLength(result.data, h * w * c);// := TSingles.Create(h * w * c);
    for i := 0 to w * h * c -1 do
        result.data[i] := (rand_normal() * 0.25)+0.5;
end;

function float_to_image_scaled(const w, h, c: longint; const data: Psingle):TImageData;
var
    &out: TImageData;
    abs_max: longint;
    i: longint;
begin
    &out := make_image(w, h, c);
    abs_max := 0;
    i := 0;
    for i := 0 to w * h * c -1 do
        if abs(data[i]) > abs_max then
            abs_max := trunc(abs(data[i]));
    for i := 0 to w * h * c -1 do
        &out.data[i] := data[i] / abs_max;
    exit(&out)
end;

function float_to_image(const w, h, c: longint; const data: TSingles):TImageData;
begin
    result := make_image(w, h, c);
    move(data[0],result.data[0], w* h* c * sizeof(single));
    //result.data := data;
end;

procedure place_image(const im: TImageData; const w, h, dx, dy: longint; const canvas: TImageData);
var
    x,y,c: longint;
    rx,ry,val: single;
begin
    for c := 0 to im.c -1 do
        for y := 0 to h -1 do
            for x := 0 to w -1 do
                begin
                    rx := (x / w) * im.w;
                    ry := (y / h) * im.h;
                    val := bilinear_interpolate(im, rx, ry, c);
                    set_pixel(canvas, x+dx, y+dy, c, val)
                end
end;

function center_crop_image(const im: TImageData; const w, h: longint):TImageData;
var
    m: longint;
    c: TImageData;
begin
    m := ifthen((im.w < im.h), im.w, im.h);
    c := crop_image(im, (im.w-m) div 2, (im.h-m) div 2, m, m);
    result := resize_image(c, w, h);
    free_image(c);
end;

function rotate_crop_image(const im: TImageData; const rad, s: single; const w, h: longint; const dx, dy, aspect: single):TImageData;
var
    x, y, c: longint;
    cx, cy: single;
    rx, ry, val: single;
begin
    cx := im.w / 2;
    cy := im.h / 2;
    result := make_image(w, h, im.c);
    for c := 0 to im.c -1 do
        for y := 0 to h -1 do
            for x := 0 to w -1 do
                begin
                    rx := Cos(rad) * ((x-w / 2.0) / s * aspect + dx / s * aspect) - sin(rad) * ((y-h / 2.0) / s+dy / s)+cx;
                    ry := Sin(rad) * ((x-w / 2.0) / s * aspect + dx / s * aspect) + cos(rad) * ((y-h / 2.0) / s+dy / s)+cy;
                    val := bilinear_interpolate(im, rx, ry, c);
                    set_pixel(result, x, y, c, val)
                end;
end;

function rotate_image(const im: TImageData; const rad: single):TImageData;
var
    x, y, c: longint;
    cx, cy: single;
    rx, ry, val: single;
begin
    cx := im.w / 2;
    cy := im.h / 2;
    result := make_image(im.w, im.h, im.c);
    for c := 0 to im.c -1 do
        for y := 0 to im.h -1 do
            for x := 0 to im.w -1 do
                begin
                    rx := cos(rad) * (x-cx)-sin(rad) * (y-cy)+cx;
                    ry := sin(rad) * (x-cx)+cos(rad) * (y-cy)+cy;
                    val := bilinear_interpolate(im, rx, ry, c);
                    set_pixel(result, x, y, c, val)
                end;
end;

procedure fill_image(const m: TImageData; const s: single);
var
    i: longint;
begin
    for i := 0 to m.h * m.w * m.c -1 do
        m.data[i] := s
end;

procedure translate_image(const m: TImageData; const s: single);
var
    i: longint;
begin
    for i := 0 to m.h * m.w * m.c -1 do
        m.data[i] := m.data[i] + s
end;

procedure scale_image(const m: TImageData; const s: single);
var
    i: longint;
begin
    for i := 0 to m.h * m.w * m.c -1 do
        m.data[i] := m.data[i] * s
end;

function crop_image(const im: TImageData; const dx, dy, w, h: longint):TImageData;
var
    i,j,k,r,c: longint;
    val: single;
begin
    result := make_image(w, h, im.c);
    for k := 0 to im.c -1 do
        for j := 0 to h -1 do
            for i := 0 to w -1 do
                begin
                    r := j+dy;
                    c := i+dx;
                    val := 0;
                    r := EnsureRange(r, 0, im.h-1);
                    c := EnsureRange(c, 0, im.w-1);
                    val := get_pixel(im, c, r, k);
                    set_pixel(result, i, j, k, val)
                end;
end;

function best_3d_shift_r(const a: TImageData; const b: TImageData; const _min, _max: longint):longint;
var
    mid: longint;
    c1,c2: TImageData;
    d1,d2: single;
begin
    if _min = _max then
        exit(_min);
    mid := floor((_min+_max) / 2.0);
    c1 := crop_image(b, 0, mid, b.w, b.h);
    c2 := crop_image(b, 0, mid+1, b.w, b.h);
    d1 := dist_array(@c1.data[0], @a.data[0], a.w * a.h * a.c, 10);
    d2 := dist_array(@c2.data[0], @a.data[0], a.w * a.h * a.c, 10);
    free_image(c1);
    free_image(c2);
    if d1 < d2 then
        exit(best_3d_shift_r(a, b, _min, mid))
    else
        exit(best_3d_shift_r(a, b, mid+1, _max))
end;

function best_3d_shift(const a, b: TImageData; const _min, _max: longint):longint;
var
    i: longint;
    best_distance, d: single;
    c: TImageData;
begin
    result := 0;
    best_distance := MaxSingle;//FLT_MAX;
    i := _min;
    while i <= _max do begin
        c := crop_image(b, 0, i, b.w, b.h);
        d := dist_array(@c.data[0], @a.data[0], a.w * a.h * a.c, 100);
        if d < best_distance then
            begin
                best_distance := d;
                result := i
            end;
        writeln(format('%d %f', [i, d]));
        free_image(c);
        inc(i,2)
    end;
end;

procedure composite_3d(const f1, f2:string; _out: string; const delta: longint);
var
    a, b, c1, c2, swap, c: TImageData;
    shift, i: longint;
    d1,d2: single;
begin
    if _out='' then
        _out := 'out';
    a := load_image(f1, 0, 0, 0);
    b := load_image(f2, 0, 0, 0);
    shift := best_3d_shift_r(a, b, -a.h div 100, a.h div 100);
    c1 := crop_image(b, 10, shift, b.w, b.h);
    d1 := dist_array(@c1.data[0], @a.data[0], a.w * a.h * a.c, 100);
    c2 := crop_image(b, -10, shift, b.w, b.h);
    d2 := dist_array(@c2.data[0], @a.data[0], a.w * a.h * a.c, 100);
    if (d2 < d1) and false then
        begin
            swap := a;
            a := b;
            b := swap;
            shift := -shift;
            writeln(format('swapped, %d', [shift]))
        end
    else
        writeln(shift);
    c := crop_image(b, delta, shift, a.w, a.h);
    for i := 0 to c.w * c.h -1 do
        c.data[i] := a.data[i];
    save_image(c, _out)
end;

procedure letterbox_image_into(const im: TImageData; const w, h: longint; const boxed: TImageData);
var
    new_w, new_h: longint;
    resized: TImageData;
begin
    new_w := im.w;
    new_h := im.h;
    if (w / im.w) < (h / im.h) then
        begin
            new_w := w;
            new_h := (im.h * w) div im.w
        end
    else
        begin
            new_h := h;
            new_w := (im.w * h) div im.h
        end;
    resized := resize_image(im, new_w, new_h);
    embed_image(resized, boxed, (w-new_w) div 2, (h-new_h) div 2);
    free_image(resized)
end;

function letterbox_image(const im: TImageData; const w, h: longint):TImageData;
var
    new_w, new_h: longint;
    resized: TImageData;
begin
    new_w := im.w;
    new_h := im.h;
    if (w / im.w) < (h / im.h) then
        begin
            new_w := w;
            new_h := (im.h * w) div im.w
        end
    else
        begin
            new_h := h;
            new_w := (im.w * h) div im.h
        end;
    resized := resize_image(im, new_w, new_h);
    result := make_image(w, h, im.c);
    fill_image(result, 0.5);
    embed_image(resized, result, (w-new_w) div 2, (h-new_h) div 2);
    free_image(resized);
end;

function resize_max(const im: TImageData; const _max: longint):TImageData;
var
    w, h: longint;
begin
    w := im.w;
    h := im.h;
    if w > h then
        begin
            h := (h * _max) div w;
            w := _max
        end
    else
        begin
            w := (w * _max) div h;
            h := _max
        end;
    if (w = im.w) and (h = im.h) then
        exit(im);
    result := resize_image(im, w, h);
end;

function resize_min(const im: TImageData; const _min: longint):TImageData;
var
    w, h: longint;
begin
    w := im.w;
    h := im.h;
    if w < h then
        begin
            h := (h * _min) div w;
            w := _min
        end
    else
        begin
            w := (w * _min) div h;
            h := _min
        end;
    if (w = im.w) and (h = im.h) then
        exit(im);
    result := resize_image(im, w, h);

end;

function random_crop_image(const im: TImageData; const w, h: longint):TImageData;
var
    dx,dy: longint;
begin
    dx := rand_int(0, im.w-w);
    dy := rand_int(0, im.h-h);
    result := crop_image(im, dx, dy, w, h);
end;

function random_augment_args(const im: TImageData; const angle:single; aspect: single; const low, high, w, h: longint):TAugmentArgs;
var
    r, _min: longint;
    scale, rad, dx, dy: single;
begin
    result := default(TAugmentArgs);
    aspect := rand_scale(aspect);
    r := rand_int(low, high);
    if (im.h < im.w * aspect) then
        _min := im.h
    else
        _min := round(im.w * aspect);
    scale := r / _min;
    rad := rand_uniform(-angle, angle) * TWO_PI / 360;
    dx := (im.w * scale / aspect-w) / 2;
    dy := (im.h * scale-w) / 2;
    dx := rand_uniform(-dx, dx);
    dy := rand_uniform(-dy, dy);
    result.rad := rad;
    result.scale := scale;
    result.w := w;
    result.h := h;
    result.dx := dx;
    result.dy := dy;
    result.aspect := aspect;
end;

function random_augment_image(const im: TImageData; const angle, aspect: single; const low, high, w, h: longint):TImageData;
var
    a: TAugmentArgs;
begin
    a := random_augment_args(im, angle, aspect, low, high, w, h);
    result := rotate_crop_image(im, a.rad, a.scale, a.w, a.h, a.dx, a.dy, a.aspect);
end;

function three_way_max(const a, b, c: single):single;
begin
    exit(ifthen((a > b), (ifthen((a > c), a, c)), (ifthen((b > c), b, c))))
end;

function three_way_min(const a, b, c: single):single;
begin
    exit(ifthen(((a < b)), (ifthen((a < c), a, c)), (ifthen(((b < c)), b, c))))
end;

procedure yuv_to_rgb(const im: TImageData);
var
    i, j: longint;
    r, g, b, y, u, v: single;
begin
    assert(im.c = 3);
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                y := get_pixel(im, i, j, 0);
                u := get_pixel(im, i, j, 1);
                v := get_pixel(im, i, j, 2);
                r := y+1.13983 * v;
                g := y+-0.39465 * u+-0.58060 * v;
                b := y+2.03211 * u;
                set_pixel(im, i, j, 0, r);
                set_pixel(im, i, j, 1, g);
                set_pixel(im, i, j, 2, b)
            end
end;

procedure rgb_to_yuv(const im: TImageData);
var
    i, j: longint;
    r, g, b, y, u, v: single;
begin
    assert(im.c = 3);
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                r := get_pixel(im, i, j, 0);
                g := get_pixel(im, i, j, 1);
                b := get_pixel(im, i, j, 2);
                y := 0.299 * r+0.587 * g+0.114 * b;
                u := -0.14713 * r+-0.28886 * g+0.436 * b;
                v := 0.615 * r+-0.51499 * g+-0.10001 * b;
                set_pixel(im, i, j, 0, y);
                set_pixel(im, i, j, 1, u);
                set_pixel(im, i, j, 2, v)
            end
end;

procedure rgb_to_hsv(const im: TImageData);
var
    i, j: longint;
    r, g, b, h, s, v, _max, _min, delta: single;
begin

    assert(im.c = 3);
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                r := get_pixel(im, i, j, 0);
                g := get_pixel(im, i, j, 1);
                b := get_pixel(im, i, j, 2);
                _max := three_way_max(r, g, b);
                _min := three_way_min(r, g, b);
                delta := _max-_min;
                v := _max;
                if _max = 0 then
                    begin
                        s := 0;
                        h := 0
                    end
                else
                    begin
                        s := delta / _max;
                        if r = _max then
                            h := (g-b) / delta
                        else
                            if g = _max then
                                h := 2+(b-r) / delta
                        else
                            h := 4+(r-g) / delta;
                        if h < 0 then
                            h := h + 6;
                        h := h / 6.
                    end;
                set_pixel(im, i, j, 0, h);
                set_pixel(im, i, j, 1, s);
                set_pixel(im, i, j, 2, v)
            end
end;

procedure hsv_to_rgb(const im: TImageData);
var
    i, j, index: longint;
    r, g, b, h, s, v, f, p, q, t: single;
begin
    assert(im.c = 3);
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                h := 6 * get_pixel(im, i, j, 0);
                s := get_pixel(im, i, j, 1);
                v := get_pixel(im, i, j, 2);
                if s = 0 then begin
                    r := v;
                    g := v;
                    b := v
                end
                else
                    begin
                        index := floor(h);
                        f := h-index;
                        p := v * (1-s);
                        q := v * (1-s * f);
                        t := v * (1-s * (1-f));
                        if index = 0 then
                            begin
                                r := v;
                                g := t;
                                b := p
                            end
                        else
                            if index = 1 then
                                begin
                                    r := q;
                                    g := v;
                                    b := p
                                end
                        else
                            if index = 2 then
                                begin
                                    r := p;
                                    g := v;
                                    b := t
                                end
                        else
                            if index = 3 then
                                begin
                                    r := p;
                                    g := q;
                                    b := v
                                end
                        else
                            if index = 4 then
                                begin
                                    r := t;
                                    g := p;
                                    b := v
                                end
                        else
                            begin
                                r := v;
                                g := p;
                                b := q
                            end
                    end;
                set_pixel(im, i, j, 0, r);
                set_pixel(im, i, j, 1, g);
                set_pixel(im, i, j, 2, b)
            end
end;

procedure grayscale_image_3c(const im: TImageData);
var
    i, j, k: longint;
    val: single;
const
    scale: array[0..2] of single =  (0.299, 0.587, 0.114);
begin
    assert(im.c = 3);
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                val := 0;
                for k := 0 to 3 -1 do
                    val := val + (scale[k] * get_pixel(im, i, j, k));
                im.data[0 * im.h * im.w+im.w * j+i] := val;
                im.data[1 * im.h * im.w+im.w * j+i] := val;
                im.data[2 * im.h * im.w+im.w * j+i] := val
            end
end;

function grayscale_image(const im: TImageData):TImageData;
var
    i, j, k: longint;
const
    scale: array[0..2] of single = (0.299, 0.587, 0.114);
begin
    assert(im.c = 3);
    result := make_image(im.w, im.h, 1);
    for k := 0 to im.c -1 do
        for j := 0 to im.h -1 do
            for i := 0 to im.w -1 do
                result.data[i+im.w * j] := result.data[i+im.w * j] + (scale[k] * get_pixel(im, i, j, k));
end;

function threshold_image(const im: TImageData; const thresh: single):TImageData;
var
    i: longint;
begin
    result := make_image(im.w, im.h, im.c);
    for i := 0 to im.w * im.h * im.c -1 do
        if im.data[i] > thresh then
            result.data[i] := 1
        else
            result.data[i] := 0;
end;

function blend_image(const fore: TImageData; const back: TImageData; const alpha: single):TImageData;
var
    i,j,k: longint;
    val: single;
begin
    assert((fore.w = back.w) and (fore.h = back.h) and (fore.c = back.c));
    result := make_image(fore.w, fore.h, fore.c);
    for k := 0 to fore.c -1 do
        for j := 0 to fore.h -1 do
            for i := 0 to fore.w -1 do
                begin
                    val := alpha * get_pixel(fore, i, j, k)+(1-alpha) * get_pixel(back, i, j, k);
                    set_pixel(result, i, j, k, val)
                end;
end;

procedure scale_image_channel(const im: TImageData; const c: longint; const v: single);
var
    i, j: longint;
    pix: single;
begin
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                pix := get_pixel(im, i, j, c);
                pix := pix * v;
                set_pixel(im, i, j, c, pix)
            end
end;

procedure translate_image_channel(const im: TImageData; const c: longint; const v: single);
var
    i, j: longint;
    pix: single;
begin
    for j := 0 to im.h -1 do
        for i := 0 to im.w -1 do
            begin
                pix := get_pixel(im, i, j, c);
                pix := pix+v;
                set_pixel(im, i, j, c, pix)
            end
end;

function binarize_image(const im: TImageData):TImageData;
var
    i: longint;
begin
    result := copy_image(im);
    for i := 0 to im.w * im.h * im.c -1 do
        begin
            if result.data[i] > 0.5 then
                result.data[i] := 1
            else
                result.data[i] := 0
        end;

end;

procedure saturate_image(const im: TImageData; const sat: single);
begin
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    hsv_to_rgb(im);
    constrain_image(im)
end;

procedure hue_image(const im: TImageData; const hue: single);
var
    i: longint;
begin
    rgb_to_hsv(im);
    for i := 0 to im.w * im.h -1 do
        begin
            im.data[i] := im.data[i]+hue;
            if im.data[i] > 1 then
                im.data[i] := im.data[i] - 1;
            if im.data[i] < 0 then
                im.data[i] := im.data[i] + 1
        end;
    hsv_to_rgb(im);
    constrain_image(im)
end;

procedure exposure_image(const im: TImageData; const sat: single);
begin
    rgb_to_hsv(im);
    scale_image_channel(im, 2, sat);
    hsv_to_rgb(im);
    constrain_image(im)
end;

procedure distort_image(const im: TImageData; const hue, sat, val: single);
var
    i: longint;
begin
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, val);
    for i := 0 to im.w * im.h -1 do
        begin
            im.data[i] := im.data[i]+hue;
            if im.data[i] > 1 then
                im.data[i] := im.data[i] - 1;
            if im.data[i] < 0 then
                im.data[i] := im.data[i] + 1
        end;
    hsv_to_rgb(im);
    constrain_image(im)
end;

procedure random_distort_image(const im: TImageData; const hue, saturation, exposure: single);
var
    dhue,dsat,dexp: single;
begin
    dhue := rand_uniform(-hue, hue);
    dsat := rand_scale(saturation);
    dexp := rand_scale(exposure);
    distort_image(im, dhue, dsat, dexp)
end;

procedure saturate_exposure_image(const im: TImageData; const sat, exposure: single);
begin
    rgb_to_hsv(im);
    scale_image_channel(im, 1, sat);
    scale_image_channel(im, 2, exposure);
    hsv_to_rgb(im);
    constrain_image(im)
end;

procedure quantize_image(const im: TImageData);
var
    size: longint;
    i: longint;
begin
    size := im.c * im.w * im.h;
    for i := 0 to size -1 do
        im.data[i] := trunc((im.data[i] * 255) / 255+(0.5 / 255))
end;

procedure make_image_red(const im: TImageData);
var
    r: longint;
    c: longint;
    k: longint;
    val: single;
begin
    for r := 0 to im.h -1 do
        for c := 0 to im.w -1 do
            begin
                val := 0;
                for k := 0 to im.c -1 do
                    begin
                        val := val + get_pixel(im, c, r, k);
                        set_pixel(im, c, r, k, 0)
                    end;
                for k := 0 to im.c -1 do
                    //set_pixel(im, c, r, k, val)
                    ;
                set_pixel(im, c, r, 0, val)
            end
end;

function make_attention_image(const img_size: longint;
  const original_delta_cpu, original_input_cpu: TArray<single>;
  w: longint; h: longint; c: longint; alpha: single): TImageData;
var
    attention_img, resized: TImageData;
    k: longint;
    min_val, mean_val, max_val, range, val: single;
begin
    attention_img.w := w;
    attention_img.h := h;
    attention_img.c := c;
    attention_img.data := original_delta_cpu;
    make_image_red(attention_img);
    min_val := 999999; mean_val := 0; max_val := -999999;
    for k := 0 to img_size -1 do
        begin
            if original_delta_cpu[k] < min_val then
                min_val := original_delta_cpu[k];
            if original_delta_cpu[k] > max_val then
                max_val := original_delta_cpu[k];
            mean_val := mean_val + original_delta_cpu[k]
        end;
    mean_val := mean_val / img_size;
    range := max_val-min_val;
    for k := 0 to img_size -1 do
        begin
            val := original_delta_cpu[k];
            val := abs(mean_val-val) / range;
            original_delta_cpu[k] := val * 4
        end;
    resized := resize_image(attention_img, w div 4, h div 4);
    attention_img := resize_image(resized, w, h);
    free_image(resized);
    for k := 0 to img_size -1 do
        attention_img.data[k] := attention_img.data[k] * alpha+(1-alpha) * original_input_cpu[k];
    exit(attention_img)
end;


function resize_image(const im: TImageData; const w, h: longint):TImageData;
var
    part: TImageData;
    r,c,k,ix,iy: longint;
    w_scale, h_scale, val, sx, dx, sy, dy: single;
begin
    result := make_image(w, h, im.c);
    part := make_image(w, im.h, im.c);
    w_scale := (im.w-1) / (w-1);
    h_scale := (im.h-1) / (h-1);
    for k := 0 to im.c -1 do
        for r := 0 to im.h -1 do
            for c := 0 to w -1 do
                begin
                    val := 0;
                    if (c = w-1) or (im.w = 1) then
                        val := get_pixel(im, im.w-1, r, k)
                    else
                        begin
                            sx := c * w_scale;
                            ix := trunc(sx);
                            dx := sx-ix;
                            val := (1-dx) * get_pixel(im, ix, r, k)+dx * get_pixel(im, ix+1, r, k)
                        end;
                    set_pixel(part, c, r, k, val)
                end;
    for k := 0 to im.c -1 do
        for r := 0 to h -1 do
            begin
                sy := r * h_scale;
                iy := trunc(sy);
                dy := sy-iy;
                for c := 0 to w -1 do
                    begin
                        val := (1-dy) * get_pixel(part, c, iy, k);
                        set_pixel(result, c, r, k, val)
                    end;
                if (r = h-1) or (im.h = 1) then
                    continue;
                for c := 0 to w -1 do
                    begin
                        val := dy * get_pixel(part, c, iy+1, k);
                        add_pixel(result, c, r, k, val)
                    end
            end;
    free_image(part);
end;

procedure test_resize(const filename: string);
var
    im, gray, c1, c2, c3, c4, aug, c: TImageData;
    mag, exposure, saturation, hue, dexp, dsat, dhue: single;
begin
    im := load_image(filename, 0, 0, 3);
    mag := mag_array(@im.data[0], im.w * im.h * im.c);
    writeln(format('L2 Norm: %f', [mag]));
    gray := grayscale_image(im);
    c1 := copy_image(im);
    c2 := copy_image(im);
    c3 := copy_image(im);
    c4 := copy_image(im);
    distort_image(c1, 0.1, 1.5, 1.5);
    distort_image(c2, -0.1, 0.66666, 0.66666);
    distort_image(c3, 0.1, 1.5, 0.66666);
    distort_image(c4, 0.1, 0.66666, 1.5);
    show_image(im, 'Original', 1);
    show_image(gray, 'Gray', 1);
    show_image(c1, 'C1', 1);
    show_image(c2, 'C2', 1);
    show_image(c3, 'C3', 1);
    show_image(c4, 'C4', 1);
{$ifdef OPEN_CV}
    while true do
        begin
            aug := random_augment_image(im, 0, 0.75, 320, 448, 320, 320);
            show_image(aug, 'aug', 1);
            free_image(aug);
            exposure := 1.15;
            saturation := 1.15;
            hue := 0.05;
            c := copy_image(im);
            dexp := rand_scale(exposure);
            dsat := rand_scale(saturation);
            dhue := rand_uniform(-hue, hue);
            distort_image(c, dhue, dsat, dexp);
            show_image(c, 'rand', 1);
            writeln(format(('%f %f %f', [dhue, dsat, dexp]));
            free_image(c)
        end
{$endif}
end;


const
    colorMap1:array[0..0] of byte = (0);
  {$if defined(MSWINDOWS)}
    colorMap3:array[0..2] of byte = (2,1,0);
    colorMap4:array[0..3] of byte = (2,1,0,3);
  {$elseif defined (MACOS) or defined (DARWIN)}
    colorMap3:array[0..2] of byte = (1,2,3);
    colorMap4:array[0..3] of byte = (1,2,3,0);
  {$else}
    colorMap3:array[0..2] of byte = (2,1,0);
    colorMap4:array[0..3] of byte = (0,1,2,3);
  {$endif}
procedure save_image_options(const im: TImageData; const name: string; const f: TImType; const quality: longint);
{$ifdef FRAMEWORK_FMX}
const pf :TPixelFormat = TPixelFormat.BGR;
var pic :TBitmap;
    data : TBitmapData;
    c, channels, x, y : longint;
    cm, d   : PByte;
    buff : string;
    success : boolean;
begin
  pic:=TBitmap.Create;
  if im.c=4 then
    pf := TPixelFormat.BGRA;
  cm:=@colormap4[0];

  pic.SetSize(im.w, im.h);
  pic.Map(TMapAccess.ReadWrite, data);
  channels := data.BytesPerPixel;
//  pic.PixelFormat := pf;
  if channels=3 then
    cm:=@colorMap3[0];
  success := false;
  case f of
    imtPNG :
      buff:=format( '%s.png', [name]);
    imtBMP :
      buff:=format( '%s.bmp', [name]);
    imtTGA :
      buff:=format( '%s.tga', [name]);
    imtJPG :
      buff:=format( '%s.jpg', [name]);
  else
      buff:=format( '%s.png', [name]);
  end;
  for y := 0 to im.h -1 do begin
    d:=data.GetScanline(y);
    for x := 0 to im.w-1 do begin
      d[x * channels + 3] := $ff;
      for c := 0 to im.c-1 do begin
        d[x * channels + cm[c]] := trunc(im.data[ c * im.h * im.w + y * im.w + x ] * $ff);
      end
    end
  end;

  pic.Unmap(data);
  pic.SaveToFile(buff);
  FreeAndNil(pic);
end;
{$else}
const
    pf= pf24bit;
var
    buff: string;
    data:TArray<Byte>;
    d,cm:PByte;
    i, j, k, c: longint;
    success: boolean;
{$ifndef STB_IMAGE}
    bmp:TBitmap;
    png:TPortableNetworkGraphic;
    tif:TTiffImage;
    jpg:TJpegImage;
    gif:TGIFImage;
{$endif}

begin
    success := false;
    case f of
      imtPNG :
        buff:=format( '%s.png', [name]);
      imtBMP :
        buff:=format( '%s.bmp', [name]);
      imtTGA :
        buff:=format( '%s.tga', [name]);
      imtJPG :
        buff:=format( '%s.jpg', [name]);
    else
        buff:=format( '%s.png', [name]);
    end;
    c:=4;
    cm:=@colorMap4[0];
    if pf=pf24bit then
        begin
            c:=3;
            cm:=@colorMap3[0];
        end;
{$ifdef STB_IMAGE}
    setLength(data , im.w * im.h * im.c);
    for k := 0 to im.c -1 do
        for i := 0 to im.w * im.h -1 do
            data[i * im.c+k] := round(255 * im.data[i+k * im.w * im.h]);
    if f = imtPNG then
        success := boolean(stbi_write_png(buff, im.w, im.h, im.c, @data[0], im.w * im.c))
    else
        if f = imtBMP then
            success := boolean(stbi_write_bmp(buff, im.w, im.h, im.c, @data[0]))
    else
        if f = imtTGA then
            success := boolean(stbi_write_tga(buff, im.w, im.h, im.c, @data[0]))
    else
        if f = imtJPG then
            success := boolean(stbi_write_jpg(buff, im.w, im.h, im.c, @data[0], quality));

{$else}
    case f of
      imtBMP:
        begin
          bmp:=TBitmap.Create;
          bmp.PixelFormat:=pf;
          bmp.SetSize(im.w, im.h);
          for k:=0 to im.c -1 do
            for j:=0 to im.h-1 do begin
              d:=bmp.ScanLine[j];
              for i:=0 to im.w-1 do
              {$ifdef MSWINDOWS}
                d[cm[k]+i*c]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$else}
                d[cm[k]+i*4]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$endif}
            end;
          bmp.SaveToFile(buff);
          freeAndNil(bmp)
        end;
      imtPNG:
        begin
          png:=TPortableNetworkGraphic.Create;
          png.PixelFormat:=pf;
          png.SetSize(im.w, im.h);
          for k:=0 to im.c -1 do
            for j:=0 to im.h-1 do begin
              d:=png.ScanLine[j];
              for i:=0 to im.w-1 do
              {$ifdef MSWINDOWS}
                  d[cm[k]+i*c]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$else}
                  d[cm[k]+i*4]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$endif}
            end;
          png.SaveToFile(buff);
          freeAndNil(png)
        end;
      imtJPG:
        begin
          jpg:=TJpegImage.Create;
          jpg.PixelFormat:=pf;
          jpg.SetSize(im.w, im.h);
          for k:=0 to im.c -1 do
            for j:=0 to im.h-1 do begin
              d:=jpg.ScanLine[j];
              for i:=0 to im.w-1 do
              {$ifdef MSWINDOWS}
                  d[cm[k]+i*c]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$else}
                  d[cm[k]+i*4]:=trunc(im.data[k*im.w*im.h+j*im.w+i]*$ff);
              {$endif}
            end;
          jpg.CompressionQuality:=quality;
          jpg.Compress;
          jpg.SaveToFile(buff);
          freeAndNil(jpg)
        end;
      imtTGA:
        begin
        end;
    else
    end;
    success := true;
{$endif}
    //free(data);
    if not success then
        writeln(ErrOutput, format('Failed to write TImageData %s', [buff]))
end;
{$endif}

procedure save_image(im: TImageData; const name: string);
begin
    save_image_options(im, name, imtJPG, 80)
end;

function load_image_stb(const filename: string; channels: longint):TImageData;
{$ifdef FRAMEWORK_FMX}
var pic :TBitmap;
    data : TBitmapData;
    c, x, y : longint;
    cm, d   : PByte;
begin
  pic:=TBitmap.Create;
  pic.LoadFromFile(filename);
  pic.Map(TMapAccess.Read, data);
  if channels=0 then
    channels := data.BytesPerPixel;
  case data.BytesPerPixel of
    4 :
      cm :=@colorMap4[0];
    3 :
      cm:= @colorMap3[0];
  end;

  result := make_image(pic.Width, pic.Height, channels);
  for y := 0 to pic.Height-1 do begin
    d := data.GetScanline(y);
    for x:=0 to pic.Width-1 do begin
      for c:=0 to channels-1 do
        result.data[c * pic.Height * pic.Width + y * pic.Width + x]:= d[x * data.BytesPerPixel + cm[c]]/ $ff
    end;
  end;
  pic.Unmap(data);
  freeAndNil(pic)
end;
{$else}
var
    w, h, c, x, y, k, dst_index, src_index: longint;
    data, cm : PByte; bpp:longint;
{$ifndef STB_IMAGE}
    pic:TPicture;
{$endif}
begin
{$ifdef STB_IMAGE}
    data := stbi_load(filename, @w, @h, @c, channels);
    if not assigned(data) then
        begin
            writeln(ErrOutput, format('Cannot load TImageData "%s"'#10'STB Reason: %s'), [filename, stbi_failure_reason()]);
            exit(nil)
        end;
    if channels<>0 then
        c := channels;
    result := make_image(w, h, c);
    for k := 0 to c -1 do
        for y := 0 to h -1 do
            for x := 0 to w -1 do
                begin
                    dst_index := x+w * y+w * h * k;
                    src_index := k+c * x+c * w * y;
                    result.data[dst_index] := data[src_index] / 255.
                end;
    free(data);
{$else}
   pic := TPicture.Create;
   pic.LoadFromFile(filename);
   cm:=@colorMap4[0];
   case pic.bitmap.PixelFormat of
     pf32bit : c := 4;
     pf24bit : c := 3;
     pf16bit, pf15bit : c := 2;
     pf8bit, pf4bit, pf1bit  : c := 1;
   else
       c:=3;
   end;
   if c=3 then
       cm:=@colorMap3[0];
   w:=pic.bitmap.Width;
   h:=pic.bitmap.Height;
   if channels=0 then channels := c;
   result := make_image(w, h, channels);
   {$ifdef fpc}
   bpp := pic.bitmap.RawImage.Description.BitsPerPixel div 8;
   {$endif}
   case pic.Bitmap.PixelFormat of
     pf32bit, pf24bit, pfDevice:
       for k:=0 to channels-1 do
           for y:=0 to pic.bitmap.height -1 do begin
              data := pic.Bitmap.ScanLine[y];
              for x:=0 to pic.bitmap.Width -1 do begin
                  dst_index := x + w*(y + h * k);
                  src_index := x;//+c * w * y;
                  {$ifdef fpc}
                  result.data[dst_index] := data[src_index*bpp+cm[k]]/$ff;
                  //if result.data[dst_index]>0 then
                      //sleep(1);
                  {$else}
                  result.data[dst_index] := data[src_index*c+cm[k]]/$ff;
                  {$endif}
              end;
           end;
     pf8bit:
       for k:=0 to channels-1 do
           for y:=0 to pic.bitmap.height -1 do begin
              data := pic.Bitmap.ScanLine[y];
              for x:=0 to pic.bitmap.Width -1 do begin
                  dst_index := x + w*(y + h * k);
                  src_index := x;//+c * w * y;
                  result.data[dst_index] := data[src_index]/$ff;
              end;
           end;
     pf1bit:
       for k:=0 to channels-1 do
           for y:=0 to pic.bitmap.height -1 do begin
              data := pic.Bitmap.ScanLine[y];
              for x:=0 to pic.bitmap.Width -1 do begin
                  dst_index := x + w*(y + h * k);
                  src_index := x;//+c * w * y;
                  result.data[dst_index] := byte(((1 shl (src_index mod 8)) and data[src_index div 8])>0);
              end;
           end;
   end;
   //print_Image(result);

   FreeAndNil(pic);
{$endif}
end;
{$endif}

function load_image(const filename: string; const w, h, c: longint):TImageData;
var
    resized: TImageData;
begin
{$ifdef OPEN-CV}
    result := load_image_cv(filename, c);
{$else}
    result := load_image_stb(filename, c);
{$endif}
    if (boolean(h) and boolean(w)) and ((h <> result.h) or (w <> result.w)) then
        begin
            resized := resize_image(result, w, h);
            free_image(result);
            result := resized
        end;
end;

function load_image_color(const filename: string; const w, h: longint):TImageData;
begin
    exit(load_image(filename, w, h, 3))
end;

function get_image_layer(const m: TImageData; const l: longint):TImageData;
var
    i: longint;
begin
    result := make_image(m.w, m.h, 1);
    move(m.data[l * m.h * m.w], result.data[0], m.w*m.h * sizeof(single));
    //for i := 0 to m.h * m.w -1 do
    //    result.data[i] := m.data[i+l * m.h * m.w];
end;

function bitmapToImage(const bmp:TBitmap):TImageData;
{$ifdef FRAMEWORK_FMX}
var
    data : TBitmapData;
    c, x, y : longint;
    cm, d   : PByte;
    channels:longint;
begin
  try
    bmp.Map(TMapAccess.Read, data);
    channels := data.BytesPerPixel;
    case data.BytesPerPixel of
      4 :
        cm :=@colorMap4[0];
      3 :
        cm:= @colorMap3[0];
    end;

    result := make_image(bmp.Width, bmp.Height, channels);
    for y := 0 to bmp.Height-1 do begin
      d := data.GetScanline(y);
      for x:=0 to bmp.Width-1 do begin
        for c:=0 to channels-1 do
          result.data[c * bmp.Height * bmp.Width + y * bmp.Width + x]:= d[x *channels + cm[c]]/ $ff
      end;
    end;
  finally
    bmp.Unmap(data);
  end;
end;
{$else}
var
    w, h, c, x, y, k, dst_index, src_index: longint;
    data : PByte;
  function readPixel(const x:longint; const p: byte):single;
  begin
    case bmp.PixelFormat of
        pf1bit : result :=  byte((($80 shr (x mod 8)) and data[x div 8])>0);
        pf4bit : ;
        pf8bit : result := data[x]/$ff;
        pf15bit: ;
        pf16bit: ;
        pf24bit: result := data[x*3+colorMap3[p]]/$ff;
    else
        result := data[x*4+colorMap4[p]]/$ff;
    end;
  end;

begin
   case bmp.PixelFormat of
     pf32bit : c := 4;
     pf24bit : c := 3;
     pf16bit, pf15bit : c := 2;
     pf8bit, pf4bit, pf1bit  : c := 1;
   else
       c:=3;
   end;
   w:=bmp.Width;
   h:=bmp.Height;
   result := make_image(w, h, c);
   for k:=0 to c-1 do
       for y:=0 to bmp.height -1 do begin
          data := bmp.ScanLine[y];
          for x:=0 to bmp.Width -1 do begin
              dst_index := x+w * y+w * h * k;
              src_index := x;//+c * w * y;
              result.data[dst_index] := readpixel(src_index, k);
          end;
       end;
   //print_Image(result);
end;
{$endif}


function imageToBitmap(const im: TImageData; bmp: TBitmap): TBitmap;
{$ifdef FRAMEWORK_FMX}
const pf :TPixelFormat = TPixelFormat.BGR;
var pic :TBitmap;
    data : TBitmapData;
    c, channels, x, y : longint;
    cm, d   : PByte;
begin
  if bmp<>nil then pic:=bmp else pic:=TBitmap.Create;
  if im.c=4 then
    pf := TPixelFormat.BGRA;
  cm:=@colormap4[0];

  pic.SetSize(im.w, im.h);
  pic.Map(TMapAccess.ReadWrite, data);
  channels := data.BytesPerPixel;
//  pic.PixelFormat := pf;
  if channels=3 then
    cm:=@colorMap3[0];
  for y := 0 to im.h -1 do begin
    d:=data.GetScanline(y);
    for x := 0 to im.w-1 do begin
      d[x * channels + 3] := $ff;
      for c := 0 to im.c-1 do begin
        d[x * channels + cm[c]] := trunc(im.data[ c * im.h * im.w + y * im.w + x ] * $ff);
      end
    end
  end;

  pic.Unmap(data);
  result := pic;

end;
{$else}
const
    pf :TPixelFormat = pf24bit;
var
    data:TArray<Byte>;
    d,cm:PByte; bpp:byte;
    x, y, c: longint;
begin
    case im.c of
      1: pf := pf8bit;
      3: pf := pf24bit;
      4: pf := pf32bit;
    end;
    cm:=@colorMap4[0];
    if pf=pf24bit then
            cm:=@colorMap3[0];
    if pf=pf8bit then
        cm:=@colorMap1[0];
    if bmp=nil then bmp:=TBitmap.Create;
    c:=bmp.Canvas.Pixels[1,1];
    bmp.PixelFormat:=pf;
    bmp.SetSize(im.w, im.h);
    bmp.BeginUpdate();
    {$ifdef fpc}
    bpp:=bmp.RawImage.Description.BitsPerPixel div 8;
    {$endif}
    for c:=0 to im.c -1 do
      for y:=0 to im.h-1 do begin
        d:=bmp.ScanLine[y];
        for x:=0 to im.w-1 do
          {$ifdef fpc}
            d[cm[c] + x*bpp]:=trunc(im.data[(c * im.h+y) * im.w + x]*$ff);
          {$else}
            d[cm[c] + x*im.c]:=trunc(im.data[(c * im.h+y) * im.w + x]*$ff);
          {$endif}
      end;
    bmp.EndUpdate();
    result := bmp;
    //free(data);
end;
{$endif}

procedure print_image(const m: TImageData);
var
    i,j,k: longint;
const shade : array[0..4] of WideChar = (' ',#$2591,#$2592,#$2593,#$2588);
begin
    for i := 0 to m.c -1 do
        begin
            for j := 0 to m.h -1 do
                begin
                    for k := 0 to m.w -1 do
                        begin
                            //write(format('%.2f, ', [m.data[i * m.h * m.w+j * m.w+k]]));
                            write(shade[trunc(4*m.data[i * m.h * m.w+j * m.w+k])]);
                            if k > 30 then
                                break
                        end;
                    writeln('');
                    if j > 30 then
                        break
                end;
            writeln('')
        end;
    writeln('')
end;

function collapse_images_vert(const ims: TArray<TImageData>; const n: longint): TImageData;
var
    color:boolean;
    border, h, w, c, i, j, h_offset, w_offset: longint;
    _copy, layer: TImageData;
begin
    color := true;
    border := 1;
    w := ims[0].w;
    h := (ims[0].h+border) * n-border;
    c := ims[0].c;
    if (c <> 3) or not color then
        begin
            w := (w+border) * c-border;
            c := 1
        end;
    result := make_image(w, h, c);
    for i := 0 to n -1 do
        begin
            h_offset := i * (ims[0].h+border);
            _copy := copy_image(ims[i]);
            if (c = 3) and color then
                embed_image(_copy, result, 0, h_offset)
            else
                for j := 0 to _copy.c -1 do
                    begin
                        w_offset := j * (ims[0].w+border);
                        layer := get_image_layer(_copy, j);
                        embed_image(layer, result, w_offset, h_offset);
                        free_image(layer)
                    end;
            free_image(_copy)
        end;
end;

function collapse_images_horz(const ims: TArray<TImageData>; const n: longint
  ): TImageData;
var
    color: boolean;
    border, h, w, c, size, i, j, w_offset, h_offset: longint;
    _copy, layer: TImageData;
begin
    color := true;
    border := 1;
    size := ims[0].h;
    h := size;
    w := (ims[0].w+border) * n-border;
    c := ims[0].c;
    if (c <> 3) or not color then
        begin
            h := (h+border) * c-border;
            c := 1
        end;
    result := make_image(w, h, c);
    for i := 0 to n -1 do
        begin
            w_offset := i * (size+border);
            _copy := copy_image(ims[i]);
            if (c = 3) and color then
                embed_image(_copy, result, w_offset, 0)
            else
                for j := 0 to _copy.c -1 do
                    begin
                        h_offset := j * (size+border);
                        layer := get_image_layer(_copy, j);
                        embed_image(layer, result, w_offset, h_offset);
                        free_image(layer)
                    end;
            free_image(_copy)
        end;
end;

procedure show_image_normalized(const im: TImageData; const name: string);
var
    c: TImageData;
begin
    c := copy_image(im);
    normalize_image(c);
    show_image(c, name, 1);
    free_image(c)
end;

procedure show_images(const ims: TArray<TImageData>; const n: longint;
  const window: string);
var
    m: TImageData;
begin
    m := collapse_images_vert(ims, n);
    normalize_image(m);
    {$ifdef OPENCV}
    save_image(m, window);
    {$endif}
    show_image(m, window, 1);
    free_image(m)
end;

procedure free_image(const m: TImageData);
begin
    //if assigned(m.data) then
        //m.data.free
end;

procedure copy_image_from_bytes(const im: TImageData; const pdata: PAnsiChar);
var
    data: PByte;
    i, k,j,w, h, c: longint;
    dst_index: longint;
    src_index: longint;
begin
    data := PByte(pdata);
    w := im.w;
    h := im.h;
    c := im.c;
    for k := 0 to c -1 do
        for j := 0 to h -1 do
            for i := 0 to w -1 do
                begin
                    dst_index := i+w * j+w * h * k;
                    src_index := k+c * i+c * w * j;
                    im.data[dst_index] := data[src_index] / 255
                end
end;


//var cfi: CONSOLE_FONT_INFOEX;
initialization
  //cfi.cbSize := sizeof(cfi);
  //cfi.nFont := 0;
  //cfi.dwFontSize.X := 24;                   // Width of each character in the font
  //cfi.dwFontSize.Y := 12;                  // Height
  //cfi.FontFamily := FF_DONTCARE;
  //cfi.FontWeight := FW_NORMAL;
  //cfi.FaceName := 'Consolas'; // Choose your font
  //if SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), false, @cfi) then
  //  writeln(SysErrorMessage(GetLastError));

end.

