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
uses TypInfo, variants, Generics.Defaults;

type

  SizeInt = NativeInt;
  TSizes = TArray<SizeInt>;
  TMapFunc<T> = function(const a:T; const index:Sizeint):T;



  { TTensor }

  { TSingleTensor }

  TTensor<T>=record
  private

  Type
    PT = ^T;
    TUnaryFunc    = function (const b:T):T;
    TBinaryFunc   = function (const a,b:T):T;
    TBinaryOp     = procedure(var dst: T ; const src:T);
    TCastIOp      = function (const v:SizeInt):T;
    TTernaryOp    = procedure (var dst:T; const src1, src2:T);
    TUnaryVecFunc = function (const src:PT; const stride:SizeInt):T;
    TBinaryVecFunc= function (const src1:PT; const stride1:SizeInt; const src2:PT; const stride2:SizeInt):T;
    TUnaryVecOp   = procedure (var dst:PT; const dstStride; const src:PT; const srcStride:SizeInt; const ALPHA:T);
    TBinaryVecOp  = procedure (var dst:PT; dstStride: SizeInt; const src1:PT; const stride1:sizeInt; const src2:PT; stride2:SizeInt);
  var
    FShape:TSizes;
    FDimSizes:TSizes;
    FStrides: TSizes;
    function GetDimensions: SizeInt;
    function GetValue(idx: TSizes): T;
    procedure SetShape(AValue: TSizes);
    procedure SetStrides(AValue: TSizes);
    procedure SetValue(idx: TSizes; AValue: T);
    class procedure Permute(var dst: TTensor<T>; const src: TTensor<T>; const newShape,Indecies,newIndecies, newArrange: TSizes; const lvl:SizeInt); static;
    class function subPrint(const src:TTensor<T>; const Indecies:TSizes;const lvl:SizeInt):string; static;
    class function product(const e:TSizes):SizeInt;static;
  public
    Data:PT;
    property Dimensions : SizeInt read GetDimensions;
    property Shape:TSizes read FShape write SetShape;
    property Strides:TSizes read FStrides write SetStrides;
    property Value[idx:TSizes]:T read GetValue write SetValue;
    constructor Create(const newShape:TSizes);overload;
    procedure FreeData();
    //procedure convertTo<C>(var Trnsor:TTensor<C>);
    procedure Fill(const start:T; const interval:T; const stride:SizeInt=1);
    procedure FillGradient(const start:T; const Finish:T);
    procedure setAll(const val:T; const stride:SizeInt);
    procedure reShape(const newShape:TSizes);
    function Transpose(const newArrange:TSizes; dstTensor:Pointer=nil):TTensor<T>;
    procedure CopyTo(var dest:PT; N:SizeInt; const dstStride:SizeInt=1; const srcStride:SizeInt=1);
    function getIndex(const idx:TSizes):SizeInt;inline;
    function Size(): SizeInt;
    function byteSize(): SizeInt;
    function ElementSize():SizeInt;
    procedure UnSqueeze(const newDim: TSizes);
    function toString(const separator:string=','):string;
    function fromString(const separator:string=','):string;

    procedure Add(const srcVector:PT;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Subtract(const srcVector:PT;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Multiply(const srcVector:PT;  N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1); overload;
    procedure Divide(const srcVector:PT;  N:SizeInt=-1; const dstStride:SizeInt=1;const srcStride:SizeInt=1); overload;

    procedure Add(const src:T; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Subtract(const src:T; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Multiply(const src:T; N:SizeInt=-1; const dstStride:SizeInt=1); overload;
    procedure Divide(const src:T; N:SizeInt=-1; const dstStride:SizeInt=1); overload;

    procedure axpy(const a:T; const y:PT; N:SizeInt=-1);
    function dot(const y:PT; N:SizeInt=-1; const dstStride:SizeInt=1; const srcStride:SizeInt=1):T;


    procedure Normalize(const mean,stdDev:T);
    function Sum(const stride:SizeInt=1):T;
    function mean(const stride:SizeInt=1):T;
    function Variance(const stride:SizeInt=1):T;
    function stdDev(const stride:SizeInt=1):T;
    function MSE(const vector: pointer; N:SizeInt):T;
    function Max(const stride:SizeInt=1):T;
    function Min(const stride:SizeInt=1):T;
    function argMin(const stride:SizeInt=1):SizeInt;
    function argMax(const stride:SizeInt=1):SizeInt;
    class procedure map(func:TMapFunc<T>; const src:TTensor<T>; var dst:TTensor<T>);static ;

    procedure LerpValues(const _min,_max, _min2, _max2:T);

    class operator Implicit(arr:TArray<T>):TTensor<T>;
    class operator Implicit(src:TTensor<T>):TArray<T>;
    class var
      Plus, Minus, Times, Division    : TBinaryFunc;
      sqr, sqrt, exp, log : TUnaryFunc;
      CastI :TCastIOp;

      addvv, subvv, mulvv, divvv : TBinaryVecOp;
      addvs, subvs, mulvs, divvs : TUnaryVecOp;
      sumv:TUnaryVecFunc;
      dotv:TBinaryVecFunc;
      sqrv, sqrtv, expv, logv : TUnaryVecOp;

      toStr: function(const v:T):string;
      Compare: function(const a,b: T):SizeInt;
   end;


implementation

function _Plus(const a, b:single):single;overload;inline;
begin
  result:= a + b
end;

function _Plus(const a, b:Double):Double;overload;inline;
begin
  result:= a + b
end;

function _Plus(const a, b:int32):int32;overload;inline;
begin
  result:= a + b
end;

function _Plus(const a, b:int64):int64;overload;inline;
begin
  result:= a + b
end;

function _Minus(const a, b:single):single;overload;inline;
begin
  result:= a - b
end;

function _Minus(const a, b:Double):Double;overload;inline;
begin
  result:= a - b
end;

function _Minus(const a, b:int32):int32;overload;inline;
begin
  result:= a - b
end;

function _Minus(const a, b:int64):int64;overload;inline;
begin
  result:= a - b
end;

function _Times(const a, b:single):single;overload;inline;
begin
  result:= a * b
end;

function _Times(const a, b:Double):Double;overload;inline;
begin
  result:= a * b
end;

function _Times(const a, b:int32):int32;overload;inline;
begin
  result:= a * b
end;

function _Times(const a, b:int64):int64;overload;inline;
begin
  result:= a * b
end;

function _Division(const a, b:single):single;overload;inline;
begin
  result:= a / b
end;

function _Division(const a, b:Double):Double;overload;inline;
begin
  result:= a / b
end;

function _Division(const a, b:int32):int32;overload;inline;
begin
  result:= a div b
end;

function _Division(const a, b:int64):int64;overload;inline;
begin
  result:= a div b
end;

function _Sqr(const a:single):single;overload;inline;
begin
  result:= a * a
end;

function _Sqr(const a:Double):Double;overload;inline;
begin
  result:= a * a
end;

function _Sqr(const a:int32):int32;overload;inline;
begin
  result:= a * a
end;

function _Sqr(const a:int64):int64;overload;inline;
begin
  result:= a * a
end;


function _Sqrt(const a:single):single;overload;inline;
begin
  result:= sqrt(a)
end;

function _Sqrt(const a:Double):Double;overload;inline;
begin
  result:= sqrt(a)
end;

function Casts(const a:SizeInt):single;overload;inline;
begin
  result:= a
end;

function Castd(const a:SizeInt):Double;overload;inline;
begin
  result:= a
end;

function Casti32(const a:SizeInt):int32;overload;inline;
begin
  result:= a
end;

function Casti64(const a:SizeInt):int64;overload;inline;
begin
  result:= a
end;

function _toStr(const v:Single):string;overload;inline;
begin
  str(v:3:3,result)
end;

function _toStr(const v:Double):string;overload;inline;
begin
  str(v:3:3,result)
end;

function _toStr(const v:int32):string;overload;inline;
begin
  str(v:3,result)
end;

function _toStr(const v:int64):string;overload;inline;
begin
  str(v:3,result)
end;


{ TTensor }

function TTensor<T>.GetValue(idx: TSizes): T;
begin
  result := Data[getIndex(idx)]
end;

function TTensor<T>.GetDimensions: SizeInt;
begin
  result := length(FShape)
end;

procedure TTensor<T>.SetShape(AValue: TSizes);
begin
  if FShape=AValue then Exit;
  FShape:=AValue;
end;

procedure TTensor<T>.SetStrides(AValue: TSizes);
begin
  if FStrides=AValue then Exit;
  FStrides:=AValue;
end;

procedure TTensor<T>.SetValue(idx: TSizes; AValue: T);
begin
  data[getIndex(idx)] := AValue;
end;

constructor TTensor<T>.Create(const newShape: TSizes);
begin
  reshape(newShape);
  Data:=AllocMem(Size()*Sizeof(T))
end;

procedure TTensor<T>.FreeData();
var d:PT;
begin
  d:=Data;
  Data:=nil;
  Freemem(d);
end;

//procedure TTensor<T>.convertTo<C>(var Trnsor: TTensor<C>);
//begin
//
//end;

procedure TTensor<T>.Fill(const start: T; const interval: T;
  const stride: SizeInt);
var i:SizeInt;
begin
  assert(stride>0);
  i:=0;
  if Interval=Default(T) then
    for i:=0 to Size()-1 do Data[i]:=start
  else
    while i<Size() do begin
       Data[i]:=Plus(start , Times(CastI(i) , interval));
       inc(i,stride)
    end;
end;

procedure TTensor<T>.FillGradient(const start:T; const Finish:T);
var i:SizeInt; interval:T;
begin
  interval := Division(Minus(finish, start) , CastI(size()));
  for i:=0 to Size()-1 do
     data[i]:=Plus(start , Times(interval , CastI(i)))
end;

procedure TTensor<T>.setAll(const val: T; const stride: SizeInt);
var i:SizeInt;
begin
  for i:=0 to Size()-1 do
    Data[i*stride]:=val
end;

procedure TTensor<T>.reShape(const newShape: TSizes);
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

function TTensor<T>.Transpose(const newArrange: TSizes; dstTensor: Pointer): TTensor<T>;
var j,y,x: SizeInt;
  newShape, newIndecies, indecies:TSizes;
  dst : ^TTensor<T> absolute dstTensor;

begin
  setLength(newShape, length(newArrange));
  setLength(newIndecies, length(newArrange));
  setLength(indecies, length(newArrange));

  for j:=0 to High(newArrange) do
     newShape[newArrange[j]]:=FShape[j];

  if not assigned(dst) then begin
    result:=TTensor<T>.Create(newShape);
    dst:=@result;
  end
  else begin
    dst.reShape(newShape);
  end;
  permute(result,Self,newShape, Indecies, newIndecies, newArrange, 0);
  result := dst^
end;

procedure TTensor<T>.CopyTo(var dest: PT; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i: SizeInt;
begin
  if (dstStride=1) and (srcStride=1) then begin
    move(data^, dest^, N*sizeOf(T));
    exit
  end;

  for i:=0 to N-1 do
    dest[i*dstStride] := data[i*srcStride]
end;

function TTensor<T>.getIndex(const idx: TSizes): SizeInt;
var i:SizeInt;
begin
  Assert(length(FShape)=Length(Idx), 'idx and Tensor shape must be identical.');
  result:=0;
  for i:=0 to high(FDimSizes) do
    inc(result, idx[i]*FDimSizes[i]);
  inc(result, idx[high(idx)])
end;

function TTensor<T>.Size(): SizeInt;
var i:SizeInt;
begin
  if not assigned(FShape) then exit(0);
  result:=FShape[0];
  for i:=1 to high(FShape) do
    result:=result * FShape[i];
end;

function TTensor<T>.byteSize(): SizeInt;
begin
  result := Sizeof(T) * Size()
end;

function TTensor<T>.ElementSize(): SizeInt;
begin
  result:=SizeOf(T)
end;

procedure TTensor<T>.UnSqueeze(const newDim: TSizes);
var s:TSizes;
begin
  Insert(newDim, FShape,0);
  reAllocMem(Data, Size()*SizeOf(T));
end;

function TTensor<T>.toString(const separator: string): string;
var indecies:TSizes;
begin
  result := 'Empty Tensor []';
  if not Assigned(FShape) or not Assigned(Data) then exit();
  setLength(Indecies, length(FShape));
  result := subPrint(Self, Indecies,0)
end;

function TTensor<T>.fromString(const separator: string): string;
begin
  //todo fromString
end;

procedure TTensor<T>.Add(const srcVector: PT; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Plus(data[i*dstStride] , srcVector[i*srcStride])
end;

procedure TTensor<T>.Subtract(const srcVector: PT; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Minus(data[i*dstStride] , srcVector[i*srcStride])
end;

procedure TTensor<T>.Multiply(const srcVector: PT; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Times(data[i*dstStride] , srcVector[i*srcStride])
end;

procedure TTensor<T>.Divide(const srcVector: PT; N: SizeInt;
  const dstStride: SizeInt; const srcStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Division(data[i*dstStride] , srcVector[i*srcStride])
end;

procedure TTensor<T>.Add(const src: T; N: SizeInt; const dstStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Plus(data[i*dstStride] , src)
end;

procedure TTensor<T>.Subtract(const src: T; N: SizeInt; const dstStride: SizeInt
  );
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Minus(data[i*dstStride] , src)
end;

procedure TTensor<T>.Multiply(const src: T; N: SizeInt; const dstStride: SizeInt
  );
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Times(data[i*dstStride] , src)
end;

procedure TTensor<T>.Divide(const src: T; N: SizeInt; const dstStride: SizeInt);
var
  i:SizeInt;
begin
  if N<=0 then N:=Size();
  for i:=0 to N-1 do
    data[i*dstStride] :=  Division(data[i*dstStride] , src)
end;

function TTensor<T>.dot(const y: PT; N:SizeInt; const dstStride:SizeInt; const srcStride:SizeInt):T;
var i:SizeInt;
begin
  if N < 0 then
    N := Size();
  result := Default(T);
  for i := 0 to N-1 do
    Data[i*dstStride] := Times(Data[i*dstStride] , y[i*srcStride])

end;

procedure TTensor<T>.Normalize(const mean, stdDev: T);
var
  i:SizeInt;
begin
  for i:=0 to Size()-1 do
    data[i] :=  Division(Minus(data[i] , mean), stdDev)
end;

class procedure TTensor<T>.Permute(var dst: TTensor<T>; const src: TTensor<T>; const newShape,Indecies,newIndecies, newArrange: TSizes; const lvl:SizeInt);
var i:SizeInt;
begin
    for i:=0 to src.FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        newIndecies[newArrange[lvl]] := i;
        if lvl<high(src.FShape) then
            Permute(dst, src, newShape, Indecies, newIndecies, newArrange, lvl+1)
         else
            dst.Data[dst.getIndex(newIndecies)]:= src.Data[src.getIndex(indecies)]
    end;
end;

function TTensor<T>.Sum(const stride: SizeInt): T;
var
  i:SizeInt;
begin
  result:=data[0];
  for i:=1 to Size()-1 do
    result := Plus(result , data[i*Stride])
end;

function TTensor<T>.mean(const stride: SizeInt): T;
begin
  result := Division(sum(stride), CastI(Size()))
end;

function TTensor<T>.Variance(const stride: SizeInt): T;
var
  mea:T;
  i:SizeInt;
begin
  mea:=Mean(stride);
  for i:=0 to Size()-1 do
     result := sqr(Minus(data[i*stride] , mea));
  result := Division(result , CastI(Size()))
end;

function TTensor<T>.stdDev(const stride: SizeInt): T;
begin
  result := sqrt(variance)
end;

class function TTensor<T>.subPrint(const src:TTensor<T>; const Indecies: TSizes; const lvl: SizeInt): string;
var i:SizeInt;var s:string;
begin
    result :='';
    if lvl < High(src.FShape) then begin
      for i:=0 to src.FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        result:=result + ', '+subPrint(src, indecies, lvl+1);
      end
    end
    else begin
      for i:=0 to src.FShape[lvl]-1 do begin
        indecies[lvl]:=i;
        s:=toStr(src.data[src.getIndex(indecies)]);
        result := result +', '+s
      end;
    end;
    delete(result,1,1);
    result := '['+result +']'+sLineBreak
end;

class function TTensor<T>.product(const e: TSizes): SizeInt;
var i:SizeInt;
begin
  if e=nil then exit;
  e[0] := result;
  for i:=1 to High(e) do
     result:=result*e[i]
end;

function TTensor<T>.MSE(const vector: pointer; N: SizeInt): T;
var i:SizeInt;
  p:PT absolute vector;
  diff :T;
begin
  diff := Default(T);
  for i:=0 to N-1 do
     diff := Plus(diff , sqr(Minus(Data[i], p[i])));
  result :=Division(diff , CastI(N))
end;

function TTensor<T>.Max(const stride: SizeInt): T;
var
  i: SizeInt;
begin
  result :=data[0];
  for i:=1 to Size()-1 do
     if Compare(data[i], result)> 0 then
         result := data[i]
end;

function TTensor<T>.Min(const stride: SizeInt): T;
var
  i: SizeInt;
begin
  result :=data[0];
  for i:=1 to Size()-1 do
     if Compare(data[i], result)< 0 then
         result := data[i]
end;

function TTensor<T>.argMin(const stride: SizeInt): SizeInt;
var
  _min:T;
  i: SizeInt;
begin
  _min :=data[0];
  for i:=1 to Size()-1 do
     if Compare(data[i], _min)< 0 then begin
         _min := data[i];
         result :=i
     end;
end;

function TTensor<T>.argMax(const stride: SizeInt): SizeInt;
var
  _max :T;
  i :SizeInt;
begin
  _max := data[0];
  for i:=1 to Size()-1 do
     if Compare(data[i], _max)> 0 then begin
       _max := data[i];
       result :=i
   end;
end;

procedure TTensor<T>.axpy(const a: T; const y: PT; N:SizeInt);
var i:SizeInt;
begin
  if N<0 then
    N:=Size();
  for i := 0 to N-1 do
    Data[i] := Plus(Times(a , Data[i]) , y[i])
end;

class procedure TTensor<T>.map(func: TMapFunc<T>; const src: TTensor<T>;
  var dst: TTensor<T>);
var
  i: SizeInt;
begin
  for i:=0 to src.Size()-1 do
     dst.data[i]:=func(src.data[i],i)
end;

procedure TTensor<T>.LerpValues(const _min, _max, _min2, _max2: T);
var r:T;
  i:SizeInt;
begin
  r:=Division(Minus(_max2 , _min2), Minus(_max , _min));
  for i:=0 to Size()-1 do
     Data[i]:= Plus(_min2 , Times(r, Minus(data[i] , _min)))
end;


class operator TTensor<T>.Implicit(arr: TArray<T>): TTensor<T>;
begin
  result.reshape([length(arr)]);
  result.data := AllocMem(length(arr)*sizeof(T));
  move(arr[0], result.data[0], length(arr)*sizeof(T))
end;

class operator TTensor<T>.Implicit(src: TTensor<T>): TArray<T>;
var i: SizeInt;
begin
  setLength(result, src.Size());
  move(src.data[0], result[0],src.size()*sizeof(T))
end;


var ten, ten2:TTensor<Single>  ;
initialization

 TTensor<Single>.Plus:=_Plus;
 TTensor<Double>.Plus:=_Plus;
 TTensor<Int32>.Plus:=_Plus;
 TTensor<Int64>.Plus:=_Plus;

 TTensor<Single>.Minus:=_Minus;
 TTensor<Double>.Minus:=_Minus;
 TTensor<Int32>.Minus:=_Minus;
 TTensor<Int64>.Minus:=_Minus;

 TTensor<Single>.Times:=_Times;
 TTensor<Double>.Times:=_Times;
 TTensor<Int32>.Times:=_Times;
 TTensor<Int64>.Times:=_Times;

 TTensor<Single>.Division:=_Division;
 TTensor<Double>.Division:=_Division;
 TTensor<Int32>.Division:=_Division;
 TTensor<Int64>.Division:=_Division;

 TTensor<Single>.CastI:=Casts;
 TTensor<Double>.CastI:=Castd;
 TTensor<Int32>.CastI:=Casti32;
 TTensor<Int64>.CastI:=Casti64;

 TTensor<Single>.toStr:=_ToStr;
 TTensor<Double>.toStr:=_ToStr;
 TTensor<Int32>.toStr:=_ToStr;
 TTensor<Int64>.toStr:=_ToStr;

 ten  := TTensor<single>.Create([3,3,3]);
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

