unit ntensors;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX86_64}
     {$asmmode intel}
  {$endif}
  {$ifdef MSWINDOWS}{$FPUType AVX2}{$endif}
{$else}
{$excessprecision off}
{$endif}
{$pointermath on}
{$WRITEABLECONST ON}

interface
uses TypInfo;

type

  SizeInt = NativeInt;
  TSizes = TArray<SizeInt>;
  TMapFunc<T> = function(const a:T; const index:Sizeint):T;



  { TTensor }

  { TSingleTensor }

  TSingleTensor=record
  private 
    FShape:TSizes;
    FDimSizes:TSizes;
    FStrides: TSizes;  
    function GetDimensions: SizeInt;
    function GetValue(idx: TSizes): Single;
    procedure SetShape(AValue: TSizes);
    procedure SetStrides(AValue: TSizes);
    procedure SetValue(idx: TSizes; AValue: Single);
  
  public
    Data:PSingle;
    property Dimensions : SizeInt read GetDimensions;
    property Shape:TSizes read FShape write SetShape;
    property Strides:TSizes read FStrides write SetStrides;
    property Value[idx:TSizes]:Single read GetValue write SetValue;
    constructor Create(const newShape:TSizes);overload;
    procedure FreeData();
    //procedure convertTo<C>(var Trnsor:TTensor<C>);
    procedure Fill(const start:Single; const interval:Single=0; const stride:SizeInt=1);
    procedure FillRange(const start:Single; const Finish:Single);
    procedure setAll(const val:Single; const stride:SizeInt);
    procedure reShape(const newShape:TSizes);
    function transpose(const newArrange:TSizes; dstTensor:Pointer=nil):TSingleTensor;
    procedure CopyTo(const dest:PSingle; N:SizeInt; const dstStride:SizeInt=1; const srcStride:SizeInt=1);
    function getIndex(const idx:TSizes):SizeInt;inline;
    function Size(): SizeInt;
    function byteSize(): SizeInt;
    function ElementSize():SizeInt;
    procedure UnSqueeze(const newDim: TSizes);
    function toString(const separator:string=','):string;
    function fromString(const separator:string=','):string;
    
    procedure Add(const srcVector:PSingle;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Subtract(const srcVector:PSingle;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Multiply(const srcVector:PSingle;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Divide(const srcVector:PSingle;  N:SizeInt=-1; const dstStride:SizeInt=1;const srcStride:SizeInt=1); overload;

    procedure Add(const src:Single; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Subtract(const src:Single; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Multiply(const src:Single; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Divide(const src:Single; N:SizeInt=-1; const dstStride:SizeInt=1); overload;

    procedure axpy(const a:single; const y:PSingle; N:SizeInt=-1);
    function dot(const y:PSingle; N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1):single;
    

    procedure Normalize(const mean,stdDev:Single);
    function Sum(const stride:SizeInt=1):Single;
    function mean(const stride:SizeInt=1):Single;
    function Variance(const stride:SizeInt=1):Single;
    function stdDev(const stride:SizeInt=1):Single;
    function MSE(const vector: pointer; N:SizeInt):Single;
    function Max(const stride:SizeInt=1):Single;
    function Min(const stride:SizeInt=1):Single;
    function argMin(const stride:SizeInt=1):SizeInt;
    function argMax(const stride:SizeInt=1):SizeInt;
    class procedure map(func:TMapFunc<Single>; var dest:TSingleTensor);static ;

    procedure LerpValues(const _min,_max, _min2, _max2:Single);

    class procedure Concat(var dst:TSingleTensor; const tensor:TArray<TSingleTensor>);            overload;static;    
    class procedure Concat(var dst:TSingleTensor; const vector:TArray<Pointer>; const N:TSizes);  overload;static;

    class operator Implicit(arr:TArray<Single>):TSingleTensor;
    class operator Implicit(src:TSingleTensor):TArray<Single>;

  end;

  { TTensorView }

  TSingleTensorView = record
    type
      TViewRange=array of TSizes;
  private
    FTensor : TSingleTensor;
    FRange  : TViewRange;
  public
    function toString(const seperator:string=','):string;
    procedure CopyTo(var dest: TSingleTensor ;const range:TViewRange=nil);
  end;


implementation




{ TTensor }

function TSingleTensor.GetValue(idx: TSizes): Single;
begin
  result := Data[getIndex(idx)]
end;

function TSingleTensor.GetDimensions: SizeInt;
begin
  result := length(FShape)
end;

procedure TSingleTensor.SetShape(AValue: TSizes);
begin
  if FShape=AValue then Exit;
  FShape:=AValue;
end;

procedure TSingleTensor.SetStrides(AValue: TSizes);
begin
  if FStrides=AValue then Exit;
  FStrides:=AValue;
end;

procedure TSingleTensor.SetValue(idx: TSizes; AValue: Single);
begin
  data[getIndex(idx)] := AValue;
end;

constructor TSingleTensor.Create(const newShape: TSizes);
begin
  reshape(newShape);
  Data:=AllocMem(Size*Sizeof(Single))
end;

procedure TSingleTensor.FreeData();
var d:PSingle;
begin
  d:=Data;
  Data:=nil;
  Freemem(d);
end;

//procedure TSingleTensor.convertTo<C>(var Trnsor: TTensor<C>);
//begin
//
//end;

procedure TSingleTensor.Fill(const start: Single; const interval: Single;
  const stride: SizeInt);
var i:SizeInt;
begin
  assert(stride>0);
  i:=0;
  while i<Size() do begin
     Data[i]:=start + i* interval;
     inc(i,stride)
  end;
end;

procedure TSingleTensor.FillRange(const start:Single; const Finish:Single);
var i:SizeInt; interval:Double;
begin
  interval := (finish-start) / size();
  for i:=0 to Size()-1 do
     data[i]:=start + interval*i
end;

procedure TSingleTensor.setAll(const val: Single; const stride: SizeInt);
var i:SizeInt;
begin
  for i:=0 to Size()-1 do
    Data[i*stride]:=val
end;

procedure TSingleTensor.reShape(const newShape: TSizes);
var i, Dim:SizeInt;
begin
  Assert(Length(newShape)>0);
  Dim:=Length(FShape);
  FShape:= newShape;
  setLength(FStrides, Length(FShape));

  for i:=Dim to high(FStrides) do
    FStrides[i]:=1;
  if length(FShape)<2 then exit;
  setLength(FDimSizes, High(FShape));
  dim:=FShape[High(FShape)];
  FDimSizes[High(FDimSizes)]:=dim;
  for i:=high(FShape)-1 downto 1 do begin
    dim:=dim*FShape[i];
    FDimSizes[i-1]:=dim
  end;
end;

function TSingleTensor.transpose(const newArrange: TSizes; dstTensor: Pointer): TSingleTensor;
var j,y,x: SizeInt;
  newShape, newIndecies, indecies:TSizes;
  dst : ^TSingleTensor absolute dstTensor;

  procedure Permute(const lvl:SizeInt);
  var i:SizeInt;
  begin
    for i:=0 to FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        newIndecies[newArrange[lvl]] := i;
        if lvl<high(FShape) then
            Permute(lvl+1)
         else
            dst.Data[dst.getIndex(newIndecies)]:= Data[getIndex(indecies)]
    end;
  end;
begin
  setLength(newShape, length(newArrange));
  setLength(newIndecies, length(newArrange));
  setLength(indecies, length(newArrange));

  for j:=0 to High(newArrange) do
     newShape[newArrange[j]]:=FShape[j];

  if not assigned(dst) then begin
    result:=TSingleTensor.Create(newShape);
    dst:=@result;
  end
  else begin
    dst.reShape(newShape);
  end;
  permute(0);
  result := dst^
end;

procedure TSingleTensor.CopyTo(const dest: PSingle; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i: SizeInt;
begin
  if (dstStride=1) and (srcStride=1) then begin
    move(data^, dest^, N*sizeOf(Single));
    exit
  end;

  for i:=0 to N-1 do
    dest[i*dstStride] := data[i*srcStride]
end;

function TSingleTensor.getIndex(const idx: TSizes): SizeInt;
var i:SizeInt;
begin
  Assert(length(FShape)=Length(Idx), 'idx and Tensor shape must be identical.');
  result:=0;
  for i:=0 to high(FDimSizes) do
    inc(result, idx[i]*FDimSizes[i]);
  inc(result, idx[high(idx)])
end;

function TSingleTensor.Size(): SizeInt;
var i:SizeInt;
begin
  if not assigned(FShape) then exit(0);
  result:=FShape[0];
  for i:=1 to high(FShape) do
    result:=result * FShape[i];
end;

function TSingleTensor.byteSize(): SizeInt;
begin
  result := Sizeof(Single) * Size()
end;

function TSingleTensor.ElementSize(): SizeInt;
begin
  result:=SizeOf(Single)
end;

procedure TSingleTensor.UnSqueeze(const newDim: TSizes);
var s:TSizes;
begin
  Insert(newDim, FShape,0)
end;

function TSingleTensor.toString(const separator: string): string;
var indecies:TSizes;
  function subPrint(const lvl : SizeInt):string;
  var i:SizeInt; s:string;
  begin
    result :='';
    if lvl < High(FShape) then begin
      for i:=0 to FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        result:=result + ', '+subPrint(lvl+1);
      end
    end
    else begin
      for i:=0 to FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        str(data[getIndex(indecies)]:3:3,s);
        result := result +', '+s
      end;
    end;
    delete(result,1,1);
    result := '['+result +']'+sLineBreak
  end;
begin
  result := 'Empty Tensor []';
  if not Assigned(FShape) or not Assigned(Data) then exit();
  setLength(Indecies, length(FShape));
  result := subPrint(0)
end;

function TSingleTensor.fromString(const separator: string): string;
begin
  //todo fromString
end;

procedure TSingleTensor.Add(const srcVector: PSingle; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] + srcVector[i*srcStride]
end;

procedure TSingleTensor.Subtract(const srcVector: PSingle; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] - srcVector[i*srcStride]
end;

procedure TSingleTensor.Multiply(const srcVector: PSingle; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] * srcVector[i*srcStride]
end;

procedure TSingleTensor.Divide(const srcVector: PSingle; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] / srcVector[i*srcStride]
end;

procedure TSingleTensor.Add(const src: Single; N: SizeInt; const dstStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] + src
end;

procedure TSingleTensor.Subtract(const src: Single; N: SizeInt; const dstStride: SizeInt
  );
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] - src
end;

procedure TSingleTensor.Multiply(const src: Single; N: SizeInt; const dstStride: SizeInt
  );
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] * src
end;

procedure TSingleTensor.Divide(const src: Single; N: SizeInt; const dstStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  data[i*dstStride] / src
end;

function TSingleTensor.dot(const y: PSingle; N:SizeInt; const dstStride:SizeInt; const srcStride:SizeInt):single;
var i:SizeInt;
begin
  if N < 0 then
    N := Size();
  result := 0;
  for i := 0 to N-1 do
    Data[i*dstStride] := Data[i*dstStride] * y[i*srcStride]
   
end;

procedure TSingleTensor.Normalize(const mean, stdDev: Single);
var
  i:SizeInt;
begin
  for i:=0 to Size()-1 do
    data[i] :=  (data[i] - mean)/stdDev
end;

function TSingleTensor.Sum(const stride: SizeInt): Single;
var
  i:SizeInt;
begin
  result:=data[0];
  for i:=1 to Size()-1 do
    result := result + data[i*Stride]
end;

function TSingleTensor.mean(const stride: SizeInt): Single;
begin
  result := sum(stride)/Size()
end;

function TSingleTensor.Variance(const stride: SizeInt): Single;
var
  mea:Single;
  i:SizeInt;
begin
  mea:=Mean(stride);
  for i:=0 to Size()-1 do
     result := sqr(data[i*stride] - mea);
  result := result / Size()
end;

function TSingleTensor.stdDev(const stride: SizeInt): Single;
begin
  result := sqrt(variance)
end;

function TSingleTensor.MSE(const vector: pointer; N: SizeInt): Single;
var i:SizeInt;
  p:PSingle absolute vector;
  diff :Single;
begin
  diff := 0;
  for i:=0 to N-1 do
     diff := diff + sqr(Data[i]-p[i]);
  result :=diff / N
end;

function TSingleTensor.Max(const stride: SizeInt): Single;
var
  i: SizeInt;
begin
  result :=data[0];
  for i:=1 to Size()-1 do
     if data[i]>result then
         result := data[i]
end;

function TSingleTensor.Min(const stride: SizeInt): Single;
var
  i: SizeInt;
begin
  result :=data[0];
  for i:=1 to Size()-1 do
     if data[i]<result then
         result := data[i]
end;

function TSingleTensor.argMin(const stride: SizeInt): SizeInt;
var
  _max:Single;
  i: SizeInt;
begin
  _max :=data[0];
  for i:=1 to Size()-1 do
     if data[i]>result then begin
         _max := data[i];
         result :=i
     end;
end;

procedure TSingleTensor.axpy(const a: single; const y: PSingle; N:SizeInt);
var i:SizeInt;
begin
  if N<0 then 
    N:=Size();
  for i := 0 to N-1 do
    Data[i] := a * Data[i] + y[i]
end;

function TSingleTensor.argMax(const stride: SizeInt): SizeInt;
var
  _min :Single;
  i :SizeInt;
begin
  _min := data[0];
  for i:=1 to Size()-1 do
     if data[i]<result then begin
       _min := data[i];
       result :=i
   end;
end;

class procedure TSingleTensor.map(func: TMapFunc<Single>; var dest: TSingleTensor);
var
  i: SizeInt;
begin
  for i:=0 to dest.Size()-1 do
     dest.data[i]:=func(dest.data[i],i)
end;

procedure TSingleTensor.LerpValues(const _min, _max, _min2, _max2: Single);
var r:double;
  i:SizeInt;
begin
  r:=(_max2 - _min2)/(_max - _min);
  for i:=0 to Size()-1 do
     Data[i]:= _min2 + r*(data[i] - _min)
end;

class procedure TSingleTensor.Concat(var dst: TSingleTensor; const tensor: TArray<
  TSingleTensor>);
begin

end;

class procedure TSingleTensor.Concat(var dst: TSingleTensor; const vector: TArray<
  Pointer>; const N: TSizes);
begin

end;

class operator TSingleTensor.Implicit(arr: TArray<Single>): TSingleTensor;
begin
  result.reshape([length(arr)]);
  result.data := AllocMem(length(arr)*sizeof(Single));
  move(arr[0], result.data[0], length(arr)*sizeof(Single))
end;

class operator TSingleTensor.Implicit(src: TSingleTensor): TArray<Single>;
var i: SizeInt;
begin
  setLength(result, src.Size());
  move(src.data[0], result[0],src.size()*sizeof(Single))
end;

{ TTensorView }

// todo implement TTensorView

function TSingleTensorView.toString(const seperator: string):string;
begin                                             

  result := FTensor.toString(seperator)
  
end;

procedure TSingleTensorView.CopyTo(var dest: TSingleTensor; const range: TViewRange);
begin

end;

var ten, ten2:TSingleTensor  ;
initialization

 ten  := TSingleTensor.Create([3,3,3]);
 ten.fill(1,1);

 ten2 := ten.transpose([0,2,1]);
 //ten.reshape([5,4]);
 //ten.transpose([1,0],@ten2);
 writeln(ten.toString());
 //ten.transpose([1,0], @ten2);


 writeln(ten2.toString());
 if assigned(ten.data) then ten.FreeData();
 if assigned(ten2.data) then ten2.FreeData();
 //readln()


finalization

end.

