unit uMain;
{$ifdef fpc}
   {$mode delphi}
   {$ModeSwitch nestedprocvars}
   {$ifdef CPUX64}{$asmmode intel}{$endif}
{$endif}
{$apptype console}


interface

uses
  Classes, SysUtils, Forms, Controls, Graphics, Dialogs, StdCtrls, ExtCtrls,
  ComCtrls, math, TypInfo, StrUtils, lightnet, nnetwork, parser, cfg, data, imageData, uimgform,ConvolutionalLayer, box,
  steroids, OpenCLHelper
  {$ifdef USE_OPENCV}
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
    Label1: TLabel;
    Label2: TLabel;
    Label3: TLabel;
    Label4: TLabel;
    Memo1: TMemo;
    Panel1: TPanel;
    Panel2: TPanel;
    Splitter1: TSplitter;
    Splitter2: TSplitter;
    TrackBar1: TTrackBar;
    TrackBar2: TTrackBar;
    TrackBar3: TTrackBar;
    TrackBar4: TTrackBar;
    procedure Button1Click(Sender: TObject);
    procedure Button2Click(Sender: TObject);
    procedure Button3Click(Sender: TObject);
    procedure CheckBox1Change(Sender: TObject);
    procedure FormCreate(Sender: TObject);
    procedure FormDestroy(Sender: TObject);
    procedure FormShow(Sender: TObject);
    procedure TrackBar1Change(Sender: TObject);
  private
  public
  end;

var
  Form1 : TForm1;
  ocl : TOpenCL;
  img:TImageData;
  bmp:Graphics.TBitmap;

implementation

{$R *.lfm}

{$ifdef USE_OPENCV}
var
  cap:PCvCapture;
  im2:PIplImage;
{$endif}

const
  thresh = 0.5;
  hier_thresh=0.5;
procedure OnForward(const idx:longint;const net:PNetwork);
var
    //s:UnicodeString;
  l:PLayer;
  p:TSingles;
  n:longint;
  norm:TImageData;
begin
  l:=@net.layers[idx];
  write(#13,idx:4,' Detecting ... ',(100*idx/net.n):1:1,'%  - ', get_layer_string(l.&type),'                     ');
  //Form1.Memo1.Lines[Form1.Memo1.lines.Count-1]:=format('%4d Detecting ... %.1f%% - %s',[idx,(100*idx/net.n), get_layer_string(l.&type)]);
  ////if l.&type=ltCONVOLUTIONAL then begin
  ////  norm:=get_convolutional_image(l^);
  ////  normalize_image2(norm);
  ////  TImgForm.ShowImage( collapse_image_layers(norm,1),true);
  ////end;
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
    nboxes, c, y, xx, i, j, labels_size: longint;
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

    j:=0;
    isDetecting := true;
    while isDetecting do begin
      if Form1.CheckBox1.Checked  then begin
      {$ifdef USE_OPENCV}
        if not assigned(cap) then  begin
            cap := cvCreateCameraCapture(CV_CAP_ANY);
            if not assigned(cap) then
                raise exception.Create('ERROR : Cannot open device');
            cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH,1280);
            cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT,720);
            if assigned(bmp) then
              bmp.SetSize(640, 480);
            im.data:=nil;
        end;
        im2 := cvQueryFrame(cap);
        if not assigned(im.data) then
            im := make_image(im2.width, im2.height, im2.nChannels);
        for c:=0 to im.c-1 do
          for y:=0 to im.h-1 do
            for xx:= 0 to im.w-1 do
            im.data[(c*im.h + y)*im.w + xx] := im2.imageData[(y*im.w + xx)*im.c + 2-c]/$ff;
      {$endif}
      end
      else begin
        {$ifdef USE_OPENCV}
        if assigned(cap) then begin
            cvReleaseCapture(@cap);
            cap := nil
        end;
        {$endif}
        input:=filenames[j];
        im := load_image_color('data/'+input+'.jpg', 0, 0);
      end;
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
      draw_detections_v3(im, dets, nboxes, thresh, names, alphabet, l.classes, 0.5 ,true);
      Form1.Image1.Picture.Graphic:=imageToBitmap(im,bmp);
      application.ProcessMessages;
      inc(j);
      if j=length(filenames) then j:=0;

      writeln(sLineBreak+sLineBreak, metrics.print);

    end;
    free_network(net);
    {$ifdef USE_OPENCV}
    if assigned(cap) then
        cvReleaseCapture(@cap)
    {$endif}

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

procedure TForm1.Button2Click(Sender: TObject);
const fileN ='test.bmp';
var bmp: TBitmap;
    bmp2: TBitmap;
  im:TImageData;
begin
  if img.data=nil then
      img:=make_image(600,600,3);
  //bmp:=TBitmap.Create();
  //bmp.PixelFormat:=pf32bit;
  //bmp.SetSize(600, 600);
  //bmp.Canvas.Lock;
  //bmp.canvas.brush.Color:=clRed;
  //bmp.canvas.pen.Color:=clBlue;
  //bmp.canvas.pen.Width:=4;
  //bmp.canvas.Ellipse(200,200,400,400);
  //bmp.Canvas.Unlock;
  //bmp.SaveToFile(fileN);
  //bmp.free;
  //bmp2:=TBitmap.Create;
  //bmp2.LoadFromFile(fileN);
  //im:=load_image(fileN,0, 0, 0);

  bmp2:=imageToBitmap(img);
  //bmp2.canvas.brush.color:=clRed;
  //bmp2.canvas.Ellipse(200,200,400,400);
  image1.Picture.Graphic:=bmp2;
  image1.Picture.Bitmap.canvas.TextOut(250,250,  GetEnumName(TypeInfo(TPixelFormat), ord(bmp2.PixelFormat)));
  bmp2.free

end;

function compares(const a,b:TArray<single>):PtrInt;
var
  i: PtrInt;
begin
  result:=-1;
  for i:=0 to length(a) -1 do
    if not SameValue(a[i],b[i],0) then exit(i)
end;

procedure xpays_pas( x,y:PSingle; const a: single; const count:PtrInt);
asm
@while:
  vmovss      xmm0 , dword [x]
  //vmulss      xmm0 , a    , dword ptr [y]
  //vaddss      xmm0 , xmm0 , dword ptr [x]
  vfmadd231ss xmm0 , a , dword [y]
  vmovss      dword [x]  , xmm0
  add         x    , sizeof(single)
  add         y    , sizeof(single)
  dec         count
  jnz         @while
end;


type

PPixel = ^ TPixel;
TPixel=record
  b,g,r:byte;
end;


procedure TForm1.CheckBox1Change(Sender: TObject);

  var d:double;
  i:longint;
begin
  {$ifdef USE_OPENCV}
  if isDetecting then exit;
  if not CheckBox1.Checked then exit;
  cap:=nil;
  im2:=nil;
  //bmp:=nil;
  //cvNamedWindow('koko',CV_GUI_NORMAL);
  cap := cvCreateCameraCapture(CV_CAP_DSHOW);

  if not assigned(cap) then
      raise exception.Create('ERROR : Cannot open device');
  cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_WIDTH,1024);
  cvSetCaptureProperty(cap,CV_CAP_PROP_FRAME_HEIGHT,768);

  im2:=cvQueryFrame(cap);
  if assigned(bmp) then
      begin
          bmp.PixelFormat:=([pf8bit, pf16bit, pf24bit, pf32bit])[im2.nChannels-1];
          bmp.setSize(im2.width,im2.height);
      end;
  writeln(im2.depth,':',im2.imageSize,':',im2.channelSeq,':',im2.nChannels,' x ', im2.height,' x ',im2.width);
  while CheckBox1.Checked and not Application.Terminated do begin
    if isDetecting then exit;
    im2:=cvQueryFrame(cap);
    bmp.BeginUpdate();
    if assigned(im2) then
      for i:=0 to im2.height-1 do
        move(im2.imageData[i*im2.width*im2.nChannels], bmp.ScanLine[i]^,sizeof(TPixel)*im2.width);
    bmp.EndUpdate();
    Image1.Picture.Graphic:=bmp;
    Application.ProcessMessages;
  end;
  if assigned(cap) then
      cvReleaseCapture(@cap);
  cap:=nil
  {$endif}
end;

procedure TForm1.FormCreate(Sender: TObject);
begin
  Memo1.Lines.Add('Current directory : ' + GetCurrentDir) ;
  bmp := Graphics.TBitmap.Create;
end;

procedure TForm1.FormDestroy(Sender: TObject);
begin
  {$ifdef USE_OPENCV}
  if assigned(cap) then
    cvReleaseCapture(@cap);
  cap:=nil;
  {$endif}
  bmp.Free
end;

procedure TForm1.FormShow(Sender: TObject);
begin

end;

procedure TForm1.TrackBar1Change(Sender: TObject);
var c,y,x:longint; P:PSingle;
label render;
begin
  if (img.c=3) and (Sender=TrackBar4) then exit;
  if img.data<>nil then begin
      p:=@img.data[0];
      c:=0;
      for y:=0 to img.h-1 do
        for x:=0 to img.w-1 do begin
          p[(c * img.h + y)* img.w + x]:=TrackBar1.Position/TrackBar1.Max;
        end;
      if img.c=1 then goto render;
      c:=1;
      for y:=0 to img.h-1 do
        for x:=0 to img.w-1 do begin
          p[(c * img.h + y)* img.w + x]:=TrackBar2.Position/TrackBar2.Max;
        end;
      c:=2;
      for y:=0 to img.h-1 do
        for x:=0 to img.w-1 do begin
          p[(c * img.h + y)* img.w + x]:=TrackBar3.Position/TrackBar3.Max;
        end;
      if img.c=3 then goto render;
      c:=3;
      for y:=0 to img.h-1 do
        for x:=0 to img.w-1 do begin
          p[(c * img.h + y)* img.w + x]:=TrackBar4.Position/TrackBar4.Max;
        end;
  end;
render:
  Image1.Picture.Graphic:=imageToBitmap(img, TBitmap(Image1.Picture.Graphic));
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


procedure Mandel(y: PtrInt; ptr:Pointer);
const _max :single= 4.0;
var
  x, iteration:integer;
  //y : integer;
  c:byte;
  x0, xx, y0, yy, xtemp, diffTolast, diffToMax, coverageNum, currentAbs, oldAbs:single;
  d:PByte;
  bmp:^Graphics.TBitmap absolute ptr;
begin


    //for y:=f to t do
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
            {$ifdef MSWINDOWS}
                PLongWord(@d[x*4])^ := $ff000000;
            {$else}
                PLongWord(@d[x*4])^ := $000000ff;
            {$endif}
            end else
            begin
                c := trunc($ff * ln(1+coverageNum)/lnxp1_max_iteration);
            {$ifdef MSWINDOWS}
                d[x*4+0] := c;
                d[x*4+1] := c;//trunc(c*1.2) and $ff;
                d[x*4+2] := c;//trunc(c*2.4) and $ff;
                d[x*4+3] := $ff
            {$else}
                d[x*4+0] := $ff;
                d[x*4+1] := c;//trunc(c*1.2) and $ff;
                d[x*4+2] := c;//trunc(c*2.4) and $ff;
                d[x*4+3] := c
            {$endif}
            end;
        end;
      end;
end;



procedure baseline_mmul(const A, B, C:PSingle; N :size_t);
var row, col, idx :size_t;
begin
  // For each row...
  for row := 0 to N-1 do
    // For each col...
    for idx := 0 to N-1 do
      // For each element in the row/col pair...
      xpays_pas(@C[row * N], @B[idx * N], A[row * N + idx], N);
      //for col := 0 to N-1 do
      //  // Accumulate the partial results
      //  C[row * N + col] += A[row * N + idx] * B[idx * N + col];
end;

{$PointerMath on}
const TILE_S=16;
procedure blocked_mmul(const A, B, C:PSingle; N :size_t);
var row, block, chunk, sub_chunk, idx :size_t; CC, BB:PSingle; A_PART:single;
begin
  // For each row...
  for row := 0 to N -1 do begin
    // For each block in the row...
    // Solve for 16 elements at a time
    block:=0;
    while block < N do begin
      // For each chunk of A/B for this block
      chunk := 0;
      while chunk < N  do begin
        // For each row in the chunk

        for sub_chunk := 0 to TILE_S -1 do begin
          A_PART := A[row * N + chunk + sub_chunk];
          CC := @C[row * N + block];
          BB := @B[chunk * N + sub_chunk * N + block];
          // Go through all the elements in the sub chunk
          //xpays_pas(CC, BB, A_PART , TILE_S);
          for idx := 0 to TILE_S - 1 do begin
            CC[idx] +=
                A_PART *
                BB[idx];
          end;
        end;
        inc(chunk,TILE_S)
      end;
      inc(block,TILE_S)
    end;
  end;
end;

const TILE_R = 32;
procedure blocked_column_mmul(const A, B, C:PSingle; N :size_t);
var row, block, tile_row, col_chunk, tile, idx :size_t; CC, BB:PSingle; A_PART:single;
begin
  col_chunk := 0;
  while col_chunk < N do begin
    // For each row in that chunk of columns...
    for row := 0 to N-1 do begin
      // For each block of elements in this row of this column chunk
      // Solve for 16 elements at a time
      tile:=0;
      CC:=@C[row * N + col_chunk];
      prefetch(CC[0]);
      while tile < N do begin
        // For each row in the tile
        for tile_row := 0 to TILE_R -1 do begin
          BB:=@B[tile * N + tile_row * N + col_chunk];
          A_PART := A[row * N + tile + tile_row];
          // Solve for each element in this tile row
          for idx := 0 to TILE_R-1 do
            CC[idx] += A_PART * BB[idx];
        end;
        inc(tile , TILE_R)
      end;
    end;
    inc(col_chunk , TILE_R)
  end
end;

procedure TForm1.Button3Click(Sender: TObject);
const N = //384;
           //768;
           1152;
  uTickPerSec=1000000;
  EPS=0.000001;
var
  t, i : int64;  pb:PByte;
  p :TMPParams;
  //A,B,C,D:PSingle;
  //diff:single;
begin

  //A:=AllocMem((N*N+64) * sizeof(single));
  //A:=System.align(A,64);
  //B:=AllocMem((N*N+64) * sizeof(single));
  //B:=System.align(B,64);
  //C:=AllocMem((N*N+64) * sizeof(single));
  //C:=System.align(C,64);
  //D:=AllocMem((N*N+64) * sizeof(single));
  //D:=System.align(D,64);
  //
  //for i:=0 to N-1
  //  do A[i] := random()*20 - 10.0;
  //for i:=0 to N-1 do
  //  B[i] := random()*20 - 10.0;
  //
  //
  //t:=clock();
  //baseline_mmul(@A[0], @B[0], @C[0], N);
  //t:=clock()-t;
  //writeln('baseline gemm : ',t/uTickPerSec:3:3,'MS');
  //
  //t:=clock();
  //blocked_column_mmul(@A[0], @B[0], @D[0], N);
  //t:=clock()-t;
  //writeln('block gemm : ',t/uTickPerSec:3:3,'MS');
  //
  //for i:=0 to min(100,N-1) do begin
  //  diff:= 100*(D[i]-C[i])/D[i];
  //  if diff > eps then
  //      writeln(i,' ',C[i]:2:3, '  ', D[i]:2:3, '  ', diff:3:3,'%');
  //end;
  ////Freemem(A);
  ////Freemem(b);
  ////Freemem(C);
  ////Freemem(D);
  //exit;

  bmp.PixelFormat:=pf32bit;
  bmp.setSize(1600, 1600);


  pb:=bmp.ScanLine[0];
  image1.Picture.Graphic:=bmp;
  bmp.BeginUpdate();
  t:=clock();
  if true then begin
    ocl.SetParamElementSizes([bmp.height*bmp.Height*sizeof(longword), bmp.Width, bmp.Height]);
    ocl.SetGlobalWorkGroupSize(bmp.height, bmp.width);
    ocl.SetLocalWorkGroupSize(8, 8);
    ocl.CallKernel(0,bmp.scanline[0], bmp.width, bmp.Height);
  end
  else begin
    //for i:=0 to bmp.height-1 do Mandel(i , @bmp);
    MP.&for(mandel, 0, bmp.Height-1, @bmp);
  end;
  bmp.EndUpdate();
  writeln((clock()-t)/1000000:3:3,'MS');
  Image1.Picture.Graphic:=bmp;
end;

var i,j,k:IntPtr;
    a:array[0..255] of ansichar;
    wis:TWorkGroupSize;
initialization
  {$if defined(MACOS) or defined(DARWIN)}
  SetCurrentDir(ExtractFilePath(ParamStr(0))+'../../../');
  {$endif}
  InitCriticalSection(semaphor);
  if TOpenCL.Plaforms>0 then begin
    ocl := TOpenCL.create(dtALL);
    ocl.ActivePlatformId:=1;
    writeln(ocl.DevicesTypeStr);

    writeln('Platforms :');
    for i:=0 to ocl.PlatformCount-1 do
      writeln(ifthen(i=ocl.ActivePlatformId,' *','  '),ocl.PlatformName(i));

    ocl.ActiveDeviceId:=0;
    writeln(sLineBreak, sLineBreak,'Devices:');
    for i:=0 to ocl.DeviceCount-1 do
      writeln(ifthen(i=ocl.ActiveDeviceId,' *','  '),ocl.DeviceName(i),', ', ocl.CLDeviceDriver,' : ', ocl.CLDeviceVersion, ' Units :', ocl.ProcessorsCount,' @ ',ocl.ProcessorsFrequency,'Mhz ');
    writeln('');
    wis := ocl.MaxWorkItemSizes;
    writeln('MaxWorkItemSizes = ',wis[0],', ',wis[1],', ',wis[2],', ',#13#10'MaxWorkGroupSize = ', ocl.MaxWorkGroupSize);
    ocl.LoadFromFile(GetCurrentDir+'/source/cl_sgemm.c');
    writeln('Build :',ocl.Build());
    writeln(ocl.BuildLog, sLineBreak, sLineBreak, 'Kernels :', sLineBreak);

    for i:=0 to ocl.KernelCount-1 do begin
      writeln('  ',ocl.KernelInfo(i).KernelName);
      for k:=0 to ocl.KernelInfo(i).KernelArgCount-1 do
        writeln('  ',ocl.KernelInfo(i).KernelArgs[k].ArgName + ' : ' +ocl.KernelInfo(i).KernelArgs[k].ArgType);
      writeln('');
    end;
  end;


finalization
  DoneCriticalSection(semaphor);
  if assigned(ocl) then
    ocl.free

end.

