unit uMain;
{$ifdef fpc}
   {$mode delphi}
   {$ModeSwitch nestedprocvars}
   {$asmmode intel}
{$endif}
{$apptype console}


interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls, math, StrUtils,
  lightnet, nnetwork, parser, cfg, data, image, box, steroids, OpenCLHelper
  {$ifdef MSWINDOWS}
  , opencv
  {$endif};

type

  { TForm1 }

  TForm1 = class(TForm)
    Button1: TButton;
    Button2: TButton;
    Button3: TButton;
    CheckBox1: TCheckBox;
    Image1: TImage;
    bmp:Graphics.TBitmap;
    Memo1: TMemo;
    Panel1: TPanel;
    Panel2: TPanel;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    procedure Button1Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure CheckBox1Change(Sender: TObject);
    procedure FormCreate(Sender: TObject);
  private
  public
  end;

var
  Form1 : TForm1;
  ocl : TOpenCL;

implementation

{$R *.lfm}

const
  thresh = 0.25;
  hier_thresh=0.5;
procedure OnForward(const idx:longint;const net:PNetwork);
var
    //s:UnicodeString;
  l:PLayer;
  p:TSingles;
  n:longint;
begin
  l:=@net.layers[idx];
  write(#13,idx:4,' Detecting ... ',(100*idx/net.n):1:1,'%  - ', get_layer_string(l.&type),'                     ');
  Application.ProcessMessages;
  //if net.n-1 = idx then begin
  //    writeln(format('%d x %d x %d , outputs = %d , classes = %d', [l.w, l.h, l.n, l.outputs, l.classes]));
      //writeln(format('detections (%.2f%% thresh) = %d', [_thresh,yolo_num_detections(l^,_thresh)]));
  //end;

end;
var isDetecting : boolean;

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
    //X: TSingles;
    X:TArray<single>;
    nboxes,i, labels_size: longint;
    dets : TArray<TDetection>;
begin
    if isDetecting then begin
      isDetecting:=false;
      exit()
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
    net.onForward:=OnForward;

    l := net.layers[net.n-1];

    i:=0;
    isDetecting := true;
    while isDetecting do begin
      input:=filenames[i];
      im := load_image_color('data/'+input+'.jpg', 0, 0);
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
      save_image(im, input+'1');
      if FileExists(input+'1.jpg') then
          Form1.Image1.picture.loadFromFile(input+'1.jpg');
      Form1.Image1.Repaint;
      application.ProcessMessages;
      inc(i);
      if i=length(filenames) then i:=0;

      writeln(sLineBreak+sLineBreak, metrics.print);

    end;
    free_network(net);

    //free_detections(dets, nboxes);
    //exit;
    //if outfile<>'' then
    //    save_image(im, outfile)
    //else
    //    begin
    //        save_image(im, 'predictions');
    //        //make_window('predictions', 512, 512, 0);
    //        //show_image(im, 'predictions', 0)
    //    end;



end;


{ TForm1 }

const filenames: TArray<string> = ['kite','dog','horses','person'];
procedure TForm1.Button1Click(Sender: TObject);
begin
  if isDetecting then begin
      button1.Caption:='Start Detecting Objects'
  end
  else
      button1.Caption:='Stop';
  test_detector('cfg/coco.data','cfg/yolov7.cfg','yolov7.weights',filenames,thresh,hier_thresh,'',0);
end;


function compares(const a,b:TArray<single>):PtrInt;
var
  i: PtrInt;
begin
  result:=-1;
  for i:=0 to length(a) -1 do
    if not SameValue(a[i],b[i],0) then exit(i)
end;

procedure xpays_pas(const x,y:PSingle; const a: single; const count:PtrInt);
var
  i: PtrInt;
begin
  for i:=0 to count-1 do
    x[i]:=x[i]+a*y[i]
end;


type

PPixel = ^ TPixel;
TPixel=record
  b,g,r:byte;
end;

var
  {$ifdef MSWINDOWS}
  cap:PCvCapture;
  im,im2:PIplImage;
  {$endif}


procedure TForm1.CheckBox1Change(Sender: TObject);

  var d:double;
  i:longint;
  bmp:Graphics.TBitmap;
begin
  {$ifdef MSWINDOWS}
  if not CheckBox1.Checked then exit;
  cap:=nil;
  im:=nil;
  bmp:=nil;
  //cvNamedWindow('koko',CV_GUI_NORMAL);
  cap := cvCreateCameraCapture(CV_CAP_ANY);

  if not assigned(cap) then
      raise exception.Create('ERROR : Cannot open device');
  cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH,640);
  cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT,480);

  im:=cvQueryFrame(cap);
  if not assigned(bmp) then
      begin
          bmp:=Graphics.TBitmap.create();
          bmp.PixelFormat:=([pf8bit, pf16bit, pf24bit, pf32bit])[im.nChannels-1];
          bmp.setSize(im.width,im.height);
      end;
  writeln(im.depth,':',im.imageSize,':',im.channelSeq,':',im.nChannels,' x ', im.height,' x ',im.width);
  while CheckBox1.Checked and not Application.Terminated do begin
    im:=cvQueryFrame(cap);
    bmp.BeginUpdate();
    for i:=0 to im.height-1 do
        move(im.imageData[i*im.width*im.nChannels], bmp.ScanLine[i]^,sizeof(TPixel)*im.width);
    bmp.EndUpdate();
    Image1.Picture.Graphic:=bmp;
    Application.ProcessMessages;
  end;
  bmp.free;
  cvReleaseCapture(@cap);
  {$endif}
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  SetCurrentDir('../');
  Memo1.Lines.Add('Current directory : ' + GetCurrentDir)
end;

const max_iteration = 10000;
const lnxp1_max_iteration:single = Ln(1+max_iteration);

function mapX(const x:single):single ;inline;
begin
  result:= x*3 - 2.1;
end;
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
function mapY(const y:single):single ;inline;
begin
  result:= y*3 - 1.5;
end;

{$ifdef fpc}
var semaphor :TRTLCRITICALSECTION;
{$else}
var semaphor :TCRITICALSECTION;
{$endif}


procedure Mandel(  y: PtrInt; ptr:Pointer);
const _max :single= 4.0;
var
  x, iteration:integer;
  //y : integer;
  c:byte;
  x0, xx, y0, yy, xtemp, diffTolast, diffToMax, coverageNum, currentAbs, oldAbs:single;
  d:PByte;
  bmp:^Graphics.TBitmap absolute ptr;
begin

        d:=bmp.ScanLine[y];
        for x:=0 to bmp.width -1 do begin
            xx:=mapx(x/bmp.width);yy:=mapy(y/bmp.Height);
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
                if currentabs>4.0 then begin
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
                d[x*4+1] := c;//trunc(c*1.2) and $ff;
                d[x*4+2] := c;//trunc(c*2.4) and $ff;
                d[x*4+3] := $ff
            end;
        end;
end;

procedure TForm1.Button3Click(Sender: TObject);
var
  t, i : int64;  b:PByte;
  p :TMPParams;
begin


  bmp := Graphics.TBitmap.Create;
  bmp.PixelFormat:=pf32bit;
  bmp.setSize(1600, 1600);


  b:=bmp.ScanLine[0];
  image1.Picture.Graphic:=bmp;
  bmp.BeginUpdate();
  t:=clock();
  if true then begin
    ocl.SetParamElementSizes([bmp.height*bmp.Height*sizeof(longword), bmp.Width, bmp.Height]);
    ocl.SetGlobalWorkGroupSize(bmp.height, bmp.width);
    ocl.SetLocalWorkGroupSize(2, 2);
    ocl.CallKernel(1,bmp.scanline[0], bmp.width, bmp.Height);
  end
  else begin
  //  Mandel(0, data.Height-1 , @data);
    MP.&for(mandel, 0, bmp.Height, @bmp);
  end;
  bmp.EndUpdate();
  writeln((clock()-t)/1000000:3:3,'MS');
  Image1.Picture.Graphic:=bmp;
  bmp.Free
end;
var i,j,k:IntPtr;
    a:array[0..255] of ansichar;
initialization
  InitCriticalSection(semaphor);
  ocl := TOpenCL.create(dtALL);
  ocl.ActivePlatformId:=0;
  writeln(ocl.DevicesTypeStr);

  writeln('Platforms :');
  for i:=0 to ocl.PlatformCount-1 do
    writeln(ifthen(i=ocl.ActivePlatformId,' *','  '),ocl.PlatformName(i));


  writeln(sLineBreak, sLineBreak,'Devices:');
  for i:=0 to ocl.DeviceCount-1 do
    writeln(ifthen(i=ocl.ActiveDeviceId,' *','  '),ocl.DeviceName(i),', ', ocl.CLDeviceDriver,' : ', ocl.CLDeviceVersion, ' Units :', ocl.ProcessorsCount,' @ ',ocl.ProcessorsFrequency,'Mhz ');
  writeln('');
  ocl.LoadFromFile(GetCurrentDir+'\source\cl_sgemm.c');
  writeln('Build :',ocl.Build);
  writeln(ocl.BuildLog, sLineBreak, sLineBreak, 'Kernels :', sLineBreak);

  for i:=0 to ocl.KernelCount-1 do begin
    writeln('  ',ocl.KernelInfo(i).KernelName);
    for k:=0 to ocl.KernelInfo(i).KernelArgCount-1 do
      writeln('  ',ocl.KernelInfo(i).KernelArgs[k].ArgName + ' : ' +ocl.KernelInfo(i).KernelArgs[k].ArgType);
    writeln('');
  end;


finalization
  DoneCriticalSection(semaphor);
  ocl.free

end.

