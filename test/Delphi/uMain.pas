unit uMain;
{$apptype console}
{$pointermath on}
{$writeableconst on}
{$excessprecision off}

interface

uses
  System.SysUtils, System.Types, System.UITypes, System.Classes, System.Variants,
  FMX.Types, FMX.Controls, FMX.Forms, FMX.Graphics, FMX.Dialogs,
  FMX.Controls.Presentation, FMX.StdCtrls,
  FMX.Memo.Types, FMX.Objects, FMX.ScrollBox, FMX.Memo, TypInfo
  , lightnet, image, cfg, box, nnetwork, parser, data
  , openclhelper ;

type
  TForm1 = class(TForm)
    Button1: TButton;
    Memo1: TMemo;
    Image1: TImage;
    Button3: TButton;
    Splitter1: TSplitter;
    procedure Button1Click(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure Button3Click(Sender: TObject);
  private
    { Private declarations }
  public
    { Public declarations }
  end;

var
  Form1: TForm1;
  ocl :TOpenCL;
implementation
uses   math, steroids
//, cfg, data, image, parser, box, network
//
//{$ifdef MSWINDOWS}
//  , windows
//{$endif}
//
;

{$R *.fmx}

const
  thresh = 0.25;
  hier_thresh=0.5;
  isCapturing :boolean =false;


procedure onNetForward(const idx:longint;const net:PNetwork);
var
    //s:UnicodeString;
  l:PLayer;
  p:TSingles;
  n:longint;
begin
  l:=@net.layers[idx];
  write(#13,'Detecting ... ',(100*idx/net.n):1:1,'% [',GetEnumName(TypeInfo(TLayerType), ord(net.layers[idx].&type)),']');
  Application.ProcessMessages
  //if net.n-1 = idx then begin
  //    writeln(format('%d x %d x %d , outputs = %d , classes = %d', [l.w, l.h, l.n, l.outputs, l.classes]));
      //writeln(format('detections (%.2f%% thresh) = %d', [_thresh,yolo_num_detections(l^,_thresh)]));
  //end;

end;

var isDetecting:boolean;

procedure test_detector(datacfg: string; cfgfile: string; weightfile: string; filenames: TArray<string>; thresh: single; hier_thresh: single; outfile: string; fullscreen: longint);
var
    options: TCFGSection;
    name_list: string;
    names: TArray<string>;
    alphabet: TArray<TArray<TImageData>>;
    time: int64;
    net :TNetwork;
    input: string;
    nms: single;
    im: TImageData;
    sized: TImageData;
    l: TLayer;
    X:TArray<single>;
    nboxes,i, labels_size: longint;
    dets : TArray<TDetection>;
begin
    if isDetecting then begin
       isDetecting:=False;
       exit
    end;


    options := read_data_cfg(datacfg);
    name_list := options.getStr('names', 'data/names.list');
    names := get_labels_custom(name_list, @labels_size);
    alphabet := load_alphabet();
    net := parse_network_cfg_custom(cfgfile,1,1);

    load_weights(@net, weightfile);
    benchmark:=true;
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    //set_batch_network(net, 1);

    //RandSeed:=(2222222);
    nms := 0.45;
    net.onForward:=onNetforward;

    l := net.layers[net.n-1];
    isDetecting:=True;
    i:=0;
    while isDetecting  do begin

      input:=filenames[i];
      im := load_image_color('data/'+input+'.jpg', 0, 0);
      Application.ProcessMessages;
      sized := letterbox_image(im, net.w, net.h);
      X := sized.data;
      time := clock();
      network_predict(net, X);
      writeln(format(#13#10#13#10'%s: Predicted in %.0f[ms]', [input, (clock()-time)/1000000]));
      nboxes := 0;
      dets := get_network_boxes(@net, im.w, im.h, thresh, hier_thresh, nil, true,  @nboxes,true);
      //writeln(format('output[%d][%dX%dX%d]',[l.outputs,l.n, l.h, l.w])+#13#10, l.output.toString(' ',min(200,l.outputs)));
      if (nms<>0) and assigned(dets) then
          do_nms_sort(dets, nboxes, l.classes, nms);

      writeln(format('  thersh[%.2f] Detections [%d]', [thresh, Length(dets)]));
      draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, 1);
        System.SysUtils.DeleteFile(string(input+'1.jpg'));
      save_image(im, input+'1');
      if FileExists(input+'1.jpg') then
          Form1.Image1.Bitmap.loadFromFile(input+'1.jpg');
      Form1.Image1.Repaint;
      application.ProcessMessages;
      inc(i);
      if i=length(filenames) then i:=0;
      writeln(sLineBreak, metrics.print);
      Application.ProcessMessages
    end;
    free_network(net);
//
    exit;
    if outfile<>'' then
        save_image(im, outfile)
    else
        begin
            save_image(im, 'predictions');
            //make_window('predictions', 512, 512, 0);
            //show_image(im, 'predictions', 0)
        end;



end;


{ TForm1 }

const filenames: TArray<string> = ['dog','kite','horses','person'];
procedure TForm1.Button1Click(Sender: TObject);
begin
  if isDetecting then
    Button1.Text := 'Start Detecting'
  else
    Button1.Text := 'Stop' ;

  test_detector('cfg/coco.data','cfg/yolov7.cfg','yolov7.weights',filenames,thresh,hier_thresh,'',0);
end;


function compares(const a,b:TArray<single>):IntPtr;
var
  i: IntPtr;
begin
  result:=-1;
  for i:=0 to length(a) -1 do
    if not SameValue(a[i],b[i],0) then exit(i)
end;

procedure xpays_pas(const x,y:PSingle; const a: single; const count:IntPtr);
var
  i: IntPtr;
begin
  for i:=0 to count-1 do
    x[i]:=x[i]+a*y[i]
end;


type

PPixel = ^ TPixel;
TPixel=record
  b,g,r:byte;
end;

//var
//  cap:PCvCapture;
//  im,im2:PIplImage;
//  isCapturing:boolean;

const max_iteration = 10000;
const lnxp1_max_iteration:single = 0;

function mapX(const x:single):single ;inline;
begin
  result:= x*3-2.1;
end;
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
function mapY(const y:single):single ;inline;
begin
  result:= y*3 - 1.5;
end;

procedure Mandel(  y: IntPtr;   p:Pointer);
const _max :single= 4.0;
var
  x, iteration:integer;
  //y : integer;
  c:byte;
  x0, xx, y0, yy, xtemp, diffTolast, diffToMax, coverageNum, currentAbs, oldAbs:single;
  d:PByte;
  bmp:^TBitmapData absolute p;
begin
    //y := f;
    //for y:=f to t do begin
        d:=bmp.GetScanLine(y);
        for x:=0 to bmp.width -1 do begin
            xx:=mapx(x/bmp.width);yy:=mapy(y/bmp.width);
            x0:=0.0;y0:=0.0;
            iteration:=0;
            oldabs:=0;
            coverageNum := max_iteration;
            while iteration < max_iteration do begin
                xtemp := x0*x0 - y0*y0;
                y0 := 2*x0*y0;
                x0 := xtemp;
                x0:=x0+xx;
                y0:=y0+yy;
                currentAbs:=x0*x0+y0*y0;
                if currentabs>4 then begin
                   difftoLast  := currentAbs - oldAbs;
                   diffToMax   :=       _max - oldAbs;
                   coverageNum := iteration + difftoMax/difftoLast;
                   break
                end;
                oldAbs:=currentAbs;
                inc(iteration);
            end;
            if iteration=max_iteration then begin
                PLongWord(@d[x*4])^ := $ff000000;
            end else
            begin
                c := trunc($ff * ln(1+coverageNum)/lnxp1_max_iteration);
                d[x*4+0] := c;
                d[x*4+1] := c;
                d[x*4+2] := c;
                d[x*4+3] := $ff
            end;
        end;
    //end;
end;

procedure TForm1.Button3Click(Sender: TObject);
var
  bmp:FMX.Graphics.TBitmap;
  data:TBitmapData;
  t : int64;
begin
  lnxp1_max_iteration := Ln(1+max_iteration);
  bmp := FMX.Graphics.TBitmap.Create;
  bmp.setSize(1600, 1600);
  bmp.map(TMapAccess.Write, data);
  t:=clock();
//  Mandel(0, data.Height-1 , @data);
  if True then begin
      ocl.SetParamElementSizes([bmp.height*bmp.Height*sizeof(longword), bmp.Width, bmp.Height]);
      ocl.SetGlobalWorkGroupSize(bmp.height, bmp.width);
      ocl.SetLocalWorkGroupSize(8, 8);
      ocl.CallKernel(1,data.GetScanline(0), bmp.width, bmp.Height);
  end
  else begin
      MP.&for(mandel,0, Data.Height, @data);
  end;
  writeln((clock()-t)/1000000:3:3,'MS');
  bmp.Unmap(Data);
  Image1.Bitmap:=bmp;
  bmp.Free
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  SetCurrentDir('..');
  Memo1.Lines.Add('Current directory : ' + GetCurrentDir)
end;

var i,j,k:IntPtr;
    a:array[0..255] of ansichar;
initialization

  ocl := TOpenCL.create(dtALL);
//  ocl.ActivePlatformId:=1;
  writeln('Platforms :');
  for i:=0 to ocl.PlatformCount-1 do
    writeln('  ',ocl.PlatformName(i));

//  ocl.ActivePlatformId:=1;

  writeln(#13'Devices:');
  for i:=0 to ocl.DeviceCount-1 do
    writeln('  ',ocl.DeviceName(i));
  writeln('');
  ocl.LoadFromFile(GetCurrentDir+'\source\cl_sgemm.c');
  writeln('Build :',ocl.Build);
  writeln(ocl.BuildLog);

  writeln(ocl.PlatformCount, ' | ',ocl.PlatformName(ocl.ActivePlatformId), ' | ',
                             ocl.DeviceName(ocl.ActiveDeviceId),' | ',ocl.CLDeviceDriver,' : ',
                             ocl.CLDeviceVersion, #13#10, ocl.DeviceBuiltInKernels,#13#10'  Units :',
                             ocl.MaxComputeUnits,' @ ',ocl.ProcessorsFrequency);
  for i:=0 to ocl.KernelCount-1 do begin
    writeln('  ', ansistring(ocl.KernelInfo(i).KernelName));
    for k:=0 to ocl.KernelInfo(i).KernelArgCount-1 do
      writeln('  ',ocl.KernelInfo(i).KernelArgs[k].ArgName + ' : ' +ocl.KernelInfo(i).KernelArgs[k].ArgType);
    writeln('');
  end;

finalization
//  ocl.free

end.
