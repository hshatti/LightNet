unit col2im;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX86_64}
    {.$FPUType avx2}
  {$endif}
{$endif}
{$pointermath on}

interface
uses lightnet;

function im2col_get_pixel(const im: PSingle; const height, width, channels:longint; row, col:longint;const channel, pad: longint):single;overload;
procedure col2im_add_pixel(const im: PSingle; const height, width, channels: longint; row, col:longint;const channel, pad: longint; const val: single); overload;
procedure im2col_cpu(const data_im: PSingle; const channels, height, width, ksize, stride, pad: longint; const data_col: PSingle);  overload;
procedure col2im_cpu(const data_col: PSingle; const channels, height, width, ksize, stride, pad: longint; const data_im: PSingle);  overload;
procedure col2im_cpu_ext(data_col: PSingle; const channels: longint; const height: longint; const width: longint; const kernel_h: longint; const kernel_w: longint; const pad_h: longint; const pad_w: longint; const stride_h: longint; const stride_w: longint; const dilation_h: longint; const dilation_w: longint; data_im: Psingle); overload;
procedure im2col_cpu_ext(data_im: PSingle; const channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h, dilation_w: longint; data_col: PSingle);  overload;

implementation
uses steroids, blas;

function im2col_get_pixel(const im: PSingle; const height, width, channels:longint; row, col:longint;const channel, pad: longint):single;overload;
begin
    row := row - pad;
    col := col - pad;
    if (row < 0) or (col < 0) or (row >= height) or (col >= width) then
        exit(0);
    exit(im[col+width * (row+height * channel)])
end;

type
  Pi2cParams = ^Ti2cParams;
  Ti2cParams = record
     ksize, stride, pad, height_col, width_col, width, height :longint;
     data_im, data_col :PSingle;
  end;

  procedure i2c(const f,t:PtrInt;const ptr:pointer=nil);
  var
      c, y, x: longint;
      w_offset, h_offset: longint;
      c_im, im_row, im_col: longint;
      col_index: longint;
      a:Pi2cParams absolute ptr;
  begin
      for c := f to t do
          begin
              w_offset := c mod a.ksize;
              h_offset := (c div a.ksize) mod a.ksize;
              c_im := c div a.ksize div a.ksize;
              for y := 0 to a.height_col -1 do begin
                  im_row := h_offset+y * a.stride - a.pad;
                  for x := 0 to a.width_col -1 do
                      begin
                          im_col := w_offset+x * a.stride - a.pad;
                          col_index := (c * a.height_col+y) * a.width_col + x;
                          a.data_col[col_index]:=0;
                          if (im_row<0) or (im_row>=a.height) then continue;
                          if (im_col<0) or (im_col>=a.width)  then continue;
                          a.data_col[col_index] := a.data_im[im_col + a.width * (im_row + a.height * c_im)];
                          //data_col[col_index] := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad)
                      end
              end;
          end
  end;

procedure im2col_cpu(const data_im: PSingle; const channels, height, width,
  ksize, stride, pad: longint; const data_col: PSingle); overload;
var
    height_col: longint;
    width_col: longint;
    channels_col: longint;
    a:Ti2cParams;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2col);{$endif}

    height_col := (height+2 * pad-ksize) div stride+1;
    width_col := (width+2 * pad-ksize) div stride+1;
    channels_col := channels * ksize * ksize;

    a.ksize:=ksize;
    a.stride:=stride;
    a.pad:=pad;
    a.height_col:=height_col;
    a.width_col:=width_col;
    a.width:=width;
    a.height:=height;
    a.data_im:=data_im;
    a.data_col:=data_col;
    {$if defined(USE_MULTITHREADING)}
    mp.&for(i2c,0,channels_col-1,@a);
    {$else}
    i2c(0, channels_col-1, @a);
    {$endif}
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opIm2col);{$endif}
end;



procedure col2im_add_pixel(const im: PSingle; const height, width,
  channels: longint; row, col: longint; const channel, pad: longint;
  const val: single); overload;
var i:longint;
begin
    row := row - pad;
    col := col - pad;
    if (row < 0) or (col < 0) or (row >= height) or (col >= width) then
        exit();
    i:=col+width * (row+height * channel) ;
    im[i] := im[i] + val
end;

procedure col2im_cpu(const data_col: PSingle; const channels, height, width,
  ksize, stride, pad: longint; const data_im: PSingle);  overload;
var
    c, h, w, height_col, width_col, channels_col,w_offset,h_offset,c_im,im_row,im_col,col_index:longint;
    val: double;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opCol2im);{$endif}
    height_col := (height+2 * pad-ksize) div stride+1;
    width_col := (width+2 * pad-ksize) div stride+1;
    channels_col := channels * ksize * ksize;
    for c := 0 to channels_col -1 do
        begin
            w_offset := c mod ksize;
            h_offset := (c div ksize) mod ksize;
            c_im := c div ksize div ksize;
            for h := 0 to height_col -1 do
                for w := 0 to width_col -1 do
                    begin
                        im_row := h_offset+h * stride;
                        im_col := w_offset+w * stride;
                        col_index := (c * height_col+h) * width_col+w;
                        val := data_col[col_index];
                        col2im_add_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad, val)
                    end
        end ;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opCol2im);{$endif}
end;

procedure caffe_set(const N: longint; const alpha: single; Y: Psingle);
var
    i: longint;
begin
    //if alpha = 0 then
    //    begin
    // todo SIMDfy caffe_set
            filldword(Y[0], N, PLongWord(@alpha)^);
            exit()
        //end;
    //for i := 0 to N -1 do
    //    Y[i] := alpha
end;

function is_a_ge_zero_and_a_lt_b(a: longint; b: longint):boolean;
begin
    exit(longword(a) < longword(b))
end;

procedure col2im_cpu_ext(data_col: PSingle; const channels: longint; const height: longint; const width: longint; const kernel_h: longint; const kernel_w: longint; const pad_h: longint; const pad_w: longint; const stride_h: longint; const stride_w: longint; const dilation_h: longint; const dilation_w: longint; data_im: Psingle); overload;
var
    output_h, output_w, channel_size, channel, kernel_row, kernel_col, output_rows, output_col, input_row, input_col: longint;
begin
//    writeln('c2i');
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opCol2imExt);{$endif}
    caffe_set(height * width * channels, 0.0, data_im);
    output_h := (height+2 * pad_h-(dilation_h * (kernel_h-1)+1)) div stride_h+1;
    output_w := (width+2 * pad_w-(dilation_w * (kernel_w-1)+1)) div stride_w+1;
    channel_size := height * width;
    //channel := channels;
    //while boolean(channel) do begin
    for channel := channels downto 1 do begin
        for kernel_row := 0 to kernel_h -1 do
            for kernel_col := 0 to kernel_w -1 do
                begin
                    input_row := -pad_h+kernel_row * dilation_h;
                    //output_rows := output_h;
                    //while boolean(output_rows) do begin
                    for output_rows := output_h downto 1 do begin
                        if not is_a_ge_zero_and_a_lt_b(input_row, height) then
                            data_col := data_col + output_w
                        else
                            begin
                                input_col := -pad_w+kernel_col * dilation_w;
                                //output_col := output_w;
                                //while boolean(output_col) do begin
                                for output_col := output_w downto 1 do begin
                                    if is_a_ge_zero_and_a_lt_b(input_col, width) then
                                        data_im[input_row * width+input_col] := data_im[input_row * width+input_col] + data_col[0];
                                    inc(data_col);
                                    input_col := input_col + stride_w;
                                    //dec(output_col)
                                end
                            end;
                        input_row := input_row + stride_h;
                        //dec(output_rows)
                    end
                end;
        data_im := data_im + channel_size;
        //dec(channel)
    end ;
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opCol2imExt);{$endif}
end;

//procedure i2c_ext(idx:PtrInt; ptr:Pointer);
//var
//    channel_size, kernel_row, kernel_col, kernel_size, output_col, output_rows, out_channel_size, input_row, input_col: longint;
//    d_im,d_col:PSingle;
//    p : PMPParams absolute ptr;
//    output_w, output_h, kernel_w, kernel_h, pad_w, pad_h, stride_w, stride_h, dilation_w, dilation_h, width, height :longint;
//
//begin
//    d_im             := p.A;
//    d_col            := p.B;
//    output_w         := PLongint(p.C)^;
//    output_h         := PLongint(p.D)^;
//    kernel_w         := PLongint(p.E)^;
//    kernel_h         := PLongint(p.F)^;
//    pad_w            := PLongint(p.G)^;
//    pad_h            := PLongint(p.H)^;
//    stride_w         := PLongint(p.I)^;
//    stride_h         := PLongint(p.J)^;
//    dilation_w       := PLongint(p.K)^;
//    dilation_h       := PLongint(p.L)^;
//    width            := PLongint(p.M)^;
//    height           := PLongint(p.N)^;
//
//    channel_size     := height * width;
//    out_channel_size := output_w * output_h;
//    kernel_size      := kernel_w * kernel_h;
//    inc(d_im , channel_size * idx);
//    inc(d_col, kernel_size * out_channel_size * idx);
//
//    //fillchar(d_col[0], out_channel_size * kernel_size * sizeof(single), 0);
//    for kernel_row := 0 to kernel_h -1 do
//        for kernel_col := 0 to kernel_w -1 do
//            begin
//                input_row := -pad_h+kernel_row * dilation_h;
//                for output_rows := 0 to output_h-1 do begin
//                  //fillchar(d_col[0], output_w *sizeof(single), 0);
//                  if (input_row>=0) and (input_row < height) then begin
//                      input_col := -pad_w+kernel_col * dilation_w;
//                      for output_col := 0 to output_w-1 do begin
//                          if (input_col>=0) and (input_col < width) then
//                               d_col[output_col] := d_im[input_row * width+input_col]
//                          else
//                               d_col[output_col] := 0
//                               ;
//                          inc(input_col, stride_w);
//                      end;
//                  end
//                  else begin
//                      for output_col := 0 to output_w-1 do begin
//                          d_col[output_col] := 0;
//                      end;
//                  end;
//                  inc(d_col, output_w);
//                  inc(input_row, stride_h)
//                end
//            end;
//end;

procedure im2col_cpu_ext(data_im: PSingle; const channels, height, width,
  kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
  dilation_w: longint; data_col: PSingle);  overload;
var
  output_h, output_w, channel: longint;
  {$ifdef FPC}
  procedure i2c_ext(idx:PtrInt; ptr:Pointer);
  {$else}
  i2c_ext:TThreadProcNested;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2colExt);{$endif}
    i2c_ext := procedure (idx:PtrInt; ptr:Pointer)
    {$endif}
    var
        channel_size, kernel_row, kernel_col, kernel_size, output_col, output_rows, out_channel_size, input_row, input_col: longint;
        d_im, d_col: PSingle;
    begin
        channel_size     := height * width;
        out_channel_size := output_w * output_h;
        kernel_size      := kernel_w * kernel_h;
        d_im := data_im + channel_size * idx;
        d_col := data_col + kernel_size * out_channel_size * idx;

        //fillchar(d_col[0], out_channel_size * kernel_size * sizeof(single), 0);
        for kernel_row := 0 to kernel_h -1 do
            for kernel_col := 0 to kernel_w -1 do
                begin
                    input_row := -pad_h+kernel_row * dilation_h;
                    for output_rows := 0 to output_h-1 do begin
                      //fillchar(d_col[0], output_w *sizeof(single), 0);
                      if (input_row>=0) and (input_row < height) then begin
                          input_col := -pad_w+kernel_col * dilation_w;
                          for output_col := 0 to output_w-1 do begin
                              if (input_col>=0) and (input_col < width) then
                                   d_col[output_col] := d_im[input_row * width+input_col]
                              else
                                   d_col[output_col] := 0
                                   ;
                              inc(input_col, stride_w);
                          end;
                      end
                      else begin
                          for output_col := 0 to output_w-1 do begin
                              d_col[output_col] := 0;
                          end;
                      end;
                      inc(d_col, output_w);
                      inc(input_row, stride_h)
                    end
                end;
    end;

{$ifdef FPC}

    //p : TMPParams;
begin
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opIm2colExt);{$endif}
{$else}
{$endif}
    //if (kernel_h = kernel_w)
    //   and (dilation_h = dilation_w) and (dilation_h = 1)
    //   and (pad_h = pad_w) and (stride_h=stride_w) then
       //begin
       //     im2col_cpu(data_im, channels, height, width, kernel_h, stride_h, pad_h, data_col);
       //     exit
       //end;
//    writeln('i2c');

    output_w := (width+2 * pad_w-(dilation_w * (kernel_w-1)+1)) div stride_w+1;
    output_h := (height+2 * pad_h-(dilation_h * (kernel_h-1)+1)) div stride_h+1;
    //p.A  :=  data_im;
    //p.B  :=  data_col;
    //p.C  :=  @output_w;
    //p.D  :=  @output_h;
    //p.E  :=  @kernel_w;
    //p.F  :=  @kernel_h;
    //p.G  :=  @pad_w;
    //p.H  :=  @pad_h;
    //p.I  :=  @stride_w;
    //p.J  :=  @stride_h;
    //p.K  :=  @dilation_w;
    //p.L  :=  @dilation_h;
    //p.M  :=  @width;
    //p.N  :=  @height;
    {$ifdef USE_MULTITHREADING}
    mp2.&for(i2c_ext,0, channels{, @p});
    {$else}
    for channel:=0 to channels-1 do
        i2c_ext(channel,{@p}nil);
    {$endif}
    {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opIm2colExt);{$endif}
end;


initialization


end.

