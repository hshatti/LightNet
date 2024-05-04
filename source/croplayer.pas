unit CropLayer;

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
  SysUtils, darknet, image;

type
    TCropLayer = TLayer;

function get_crop_image(const l: TCropLayer):TImageData;


function make_crop_layer(const batch, h, w, c, crop_height, crop_width:longint;const flip: boolean; const angle, saturation, exposure: single):TCropLayer;

procedure resize_crop_layer(var l: TCropLayer; const w, h: longint);

procedure forward_crop_layer(var l: TCropLayer; const net: PNetworkState);

procedure backward_crop_layer(var l: TCropLayer; const net: PNetworkState);

{$ifdef GPU}
procedure backward_crop_layer_gpu(const l: TCropLayer; const net: PNetworkState);
{$endif}

implementation


function get_crop_image(const l: TCropLayer):TImageData;
var
    h,w,c: longint;
begin
    h := l.out_h;
    w := l.out_w;
    c := l.out_c;
    result:=float_to_image(w, h, c, l.output)
end;

procedure backward_crop_layer(var l: TCropLayer; const net: PNetworkState);
begin

end;

{$ifdef GPU}
procedure backward_crop_layer_gpu(const l: TCropLayer; const net: PNetworkState);
begin

end;
{$endif}

function make_crop_layer(const batch, h, w, c, crop_height, crop_width:longint;const flip: boolean; const angle, saturation, exposure: single):TCropLayer;
begin
    writeln(format( 'Crop Layer: %d x %d -> %d x %d x %d image', [h, w, crop_height, crop_width, c]));
    result := Default(TCropLayer);
    result.&type := ltCROP;
    result.batch := batch;
    result.h := h;
    result.w := w;
    result.c := c;
    result.scale := crop_height / h;
    result.flip := flip;
    result.angle := angle;
    result.saturation := saturation;
    result.exposure := exposure;
    result.out_w := crop_width;
    result.out_h := crop_height;
    result.out_c := c;
    result.inputs := result.w * result.h * result.c;
    result.outputs := result.out_w * result.out_h * result.out_c;
    //result.output := calloc(result.outputs * batch, sizeof(float));
    result.output := TSingles.Create(result.outputs * batch);
    result.forward := forward_crop_layer;
    result.backward := backward_crop_layer;
  {$ifdef GPU}
    result.forward_gpu := forward_crop_layer_gpu;
    result.backward_gpu := backward_crop_layer_gpu;
    result.output_gpu := cuda_make_array(result.output, result.outputs * batch);
    result.rand_gpu := cuda_make_array(0, result.batch * 8);
  {$endif}
end;

procedure resize_crop_layer(var l: TCropLayer; const w, h: longint);
begin
    l.w := w;
    l.h := h;
    l.out_w := trunc(l.scale * w);
    l.out_h := trunc(l.scale * h);
    l.inputs := l.w * l.h * l.c;
    l.outputs := l.out_h * l.out_w * l.out_c;
    l.output.reAllocate( l.batch * l.outputs );

    {$ifdef GPU}
    cuda_free(l.output_gpu);
    l.output_gpu := cuda_make_array(l.output, l.outputs * l.batch)
  {$endif}
end;

procedure forward_crop_layer(var l: TCropLayer; const net: PNetworkState);
var
    i, j, c, b, row, col, index, count, dh, dw: longint;
    flip : boolean;
    scale, trans: single;
begin
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.start(l.&type);
    {$endif}

    count := 0;
    flip := l.flip and boolean(random(2));
    dh := random(l.h-l.out_h);
    dw := random(l.w-l.out_w);
    scale := 2;
    trans := -1;
    if l.noadjust then
        begin
            scale := 1;
            trans := 0
        end;
    if not net.train then
        begin
            flip := false;
            dh := (l.h-l.out_h) div 2;
            dw := (l.w-l.out_w) div 2
        end;
    for b := 0 to l.batch -1 do
        for c := 0 to l.c -1 do
            for i := 0 to l.out_h -1 do
                for j := 0 to l.out_w -1 do
                    begin
                        if flip then
                            col := l.w-dw-j-1
                        else
                            col := j+dw;
                        row := i+dh;
                        index := col+l.w * (row+l.h * (c+l.c * b));
                        l.output[count] := net.input[index] * scale + trans;
                        inc(count)
                    end;
    {$ifdef USE_TELEMETRY}
    if benchmark then metrics.forward.finish(l.&type);
    {$endif}

end;



end.

