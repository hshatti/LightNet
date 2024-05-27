
{ <Steroids : (Thread Pooling wrapper for Freepascal/Delphi) >
  Copyright (c) <2022> <Haitham Shatti  <haitham.shatti at gmail dot com> >
  Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
  to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
  and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
  The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
  DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
  USE OR OTHER DEALINGS IN THE SOFTWARE.
}

(* instead of the regular pascal for loop,
  Just prepare a prcedure of form :
  procedure loopedProcedure(i :IntPtr ; args:pointer);

  then call it using "MP.&For(loopedProcedure, forStart, forFinish)" where <i> in the loopedProcedure will loop from <forStart> to <forFinish>

example Fast Mandelbrot set creation on steroids (in FreePascal) :
*******************************************************************************************
// under implementation section
uses steroids;

// enable nested procedures
{$ModeSwitch nestedprocvars}

const max_iteration = 10000;
const lnxp1_max_iteration:single = Ln(1+max_iteration);

function mapX(const x:single):single ;inline;
begin
  result:= x*3 - 2.1;
end;

function mapY(const y:single):single ;inline;
begin
  result:= y*3 - 1.5;
end;


procedure Mandelbrot(const bmp:TBitmap);

  procedure Mandel(y: PtrInt; ptr:Pointer);
  const _max :single= 4.0;
  var
    x, iteration:integer;
    c:byte;
    x0, xx, y0, yy, xtemp, diffTolast, diffToMax, coverageNum, currentAbs, oldAbs:single;
    d:PByte;
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

begin
  bmp.BeginUpdate();
  MP.&for(mandel, 0, bmp.height-1)  // Now run Mandelbrot set on Steroids
  bmp.endUpdate();
end;

//PLEASE USE IN PEACE!!
********************************************************************************************************
*)
unit steroids;
{$ifdef FPC}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {$ifdef CPUX64}
    {$FPUType AVX2}
    {$asmmode intel}
  {$endif}
  //{$PackRecords C}
{$endif}
interface
uses Sysutils, Classes, math {$ifdef MSWINDOWS}, windows{$else}{$endif} {$ifndef FPC}, SyncObjs{$endif};

{$ifdef FPC}

{$else}

{$endif}
const DefaultStackSize = 4 * 1024 * 1024;
type
{$if not defined(PtrInt)}
  PPtrInt = ^PtrInt;
  Ptrint = IntPtr;
{$endif}
  PMPParams = ^ TMPParams;
  TMPParams = record
     A,B,C,D,E,F,G,H,I,J,K,L,M,N,O,P,Q:Pointer;
  end;
  {$if defined(MSWindows) or defined(linux)}
  TGroupPriority = TThreadPriority;
  {$else}
  TGroupPriority = TThreadPriority;
  {$endif}
  TGroupProc    = procedure(const _start,_end:IntPtr;const params:Pointer);
  {$ifdef fpc}  TGroupProcNested=procedure(const _start,_end:IntPtr; const params:Pointer)is nested;
  {$else}  TGroupProcNested= reference to procedure(const _start,_end:IntPtr; const params:Pointer); {$endif} // must be [register]?
  TGroupMethod  = procedure(const _start,_end:IntPtr;const params:Pointer) of object;
  TThreadProc   = procedure(arg :Pointer);
  TThreadProc2  = procedure( i:IntPtr ; arg:pointer);
  TThreadProcNested  = {$ifndef FPC} reference to {$endif}procedure( i:IntPtr ; arg:pointer){$ifdef FPC}is nested{$endif};

  TOPool = class;

{ TOThread }
TOThread=class(TThread)
  //TGroupProc=procedure(const params:PPointer);
private
  Fire:{$ifdef FPC}PRTLEvent{$else}{$ifdef MSWINDOWS}THandle{$else}TEvent {$endif}{$endif};
  FStart,FSpan,FEnd:IntPtr;
  FID:integer;
  FParams:Pointer;
  FSync:TThreadMethod;
  FProc:TGroupProc;
  FProcNested:TGroupProcNested;
  FMethod:TGroupMethod;
  FThreadProc  : TThreadProc;
  FThreadProc2 : TThreadProc2;
  FThreadProcNested:TThreadProcNested;
  FBusy:boolean;
  FPool:TOPool;
public
  procedure Execute; override;
  constructor Create(CreateSuspended: Boolean; const StackSize: NativeUInt= DefaultStackSize); overload;
  constructor Create(const Proc:TGroupProc;const Params:Pointer);                           overload;
  destructor Destroy; override;
  property Pool:TOpool read FPool;
  property Busy:boolean read FBusy;
  property Id:integer read FID;
  property From:IntPtr read FStart;
  property &To:IntPtr read FEnd;
end;


{ TOPool }
TOPool = class
type
  TCPUFreq= record
     base, max, bus, xtal : dword;
     ratio: double;
  end;
  TBuff = array[0..47] of ansichar;
private
  Pool : array of TOThread;
  OTCount:integer;
  {$ifdef fpc}
  PoolDone:PRTLEvent;
  {$else}
  PoolDone:TEvent;
  {$endif}
  procedure WaitForPool;
  function getWorkers:longint;
public
  constructor Create;
  class function CPUName():TBuff ;static;
  class function CPUFreqMhz():TCPUFreq; static;
  destructor Destroy; override;
  procedure setWorkers(Count:longint);
  procedure &For(const Proc: TGroupProc; const _from, _to: IntPtr; const Params: Pointer = nil); overload;

  procedure &For(const Proc:TGroupProcNested;const _from,_to:IntPtr;const Params:Pointer = nil); overload;

  procedure &For(const Proc:TGroupMethod;const _from,_to:IntPtr;const Params:Pointer = nil); overload;

  procedure &for(const proc:TThreadProc2; const start, count:IntPtr; const args:Pointer; const step:IntPtr=1; const sync:TThreadMethod=nil); overload;

  procedure &For(const proc: TThreadProcNested; const start, count: IntPtr; const args: Pointer=nil; const step: IntPtr=1; const sync: TThreadMethod=nil);overload;

  function isBusy:boolean;
  procedure setPriority(const priority:TGroupPriority);
  property Workers:longint read getWorkers write SetWorkers;
  property PendingWorkCount:integer read OTCount;
  //FCriticalSection:TRTLCriticalSection;
end;
var MP, MP2, MP3:TOPool;
function ExecuteInThread(const proc:TThreadProc; args:Pointer):TThread;
function GetSystemThreadCount: integer;
{$ifdef MSWINDOWS}
var
  SystemInfo: SYSTEM_INFO;
{$else}
{$ifdef fpc}
const _SC_NPROCESSORS_ONLN = 58;
function sysconf(cmd: Integer): longint;cdecl; external;
{$endif}
{$endif}

implementation

function ExecuteInThread(const proc:TThreadProc; args:Pointer):TThread;
begin
  result:=TOThread.Create(true);
  with TOThread(result) do begin
    FParams    := args;
    FThreadProc:= proc;
    Start
  end
end;

function GetSystemThreadCount: integer;
{$ifdef MSWINDOWS}
begin
    GetSystemInfo(SystemInfo);
    Result := SystemInfo.dwNumberOfProcessors;
end;

{$else}
begin
  {$ifdef fpc}
  result := Max(TThread.ProcessorCount,4);
  //result:= sysconf(_SC_NPROCESSORS_ONLN);
  {$else}
  result := Max(TThread.ProcessorCount,4);
  {$endif}
end;
{$endif}

{ TOPool }
constructor TOPool.Create;
var i:integer;
begin
  inherited;
  OTCount:=0;
  //InitCriticalSection(FCriticalSection);
  Setlength(Pool,ceil(GetSystemThreadCount*0.9));
  for i:=0 to High(Pool) do begin
    Pool[i]:=TOthread.Create(false);
    Pool[i].FID:=i;
    Pool[i].FPool:=Self
  end;
  {$ifdef FPC}
  PoolDone := RTLEventCreate;
  {$else}
  PoolDone := TEvent.Create();
  {$endif}
end;

class function TOPool.CPUName: TBuff;
{$if defined(CPUX64)}
asm
  push rbx
  mov  r8,    result

  mov eax   , $80000002
  cpuid
  mov dword ptr [r8]   , eax
  mov dword ptr [r8+ 4]  , ebx
  mov dword ptr [r8+ 8]  , ecx
  mov dword ptr [r8+12]  , edx

  mov eax   , $80000003
  cpuid
  mov dword ptr [r8+16]  , eax
  mov dword ptr [r8+20]  , ebx
  mov dword ptr [r8+24]  , ecx
  mov dword ptr [r8+28]  , edx

  mov eax   , $80000004
  cpuid
  mov dword ptr [r8+32]  , eax
  mov dword ptr [r8+36]  , ebx
  mov dword ptr [r8+40]  , ecx
  mov dword ptr [r8+44]  , edx
  pop rbx
end;

{$else}
begin
// assuming linux on ARM

end;
{$endif}

class function TOPool.CPUFreqMhz: TCPUFreq;
{$if defined(CPUX64)}
asm
  push rbx
  mov  r8  , result


  mov        eax, $15
  cpuid                                    // read leaf 15h
  cmp        ecx  , 0                      // got any frequency ?
  je         @fr                           // no?, skip to reading leaf 16h
  mov        TCPUFreq(r8).xtal  , ecx      // got crystal frequency in Hz

  cvtsi2sd   xmm0 , eax                    // CPU clock Ratio D
  cvtsi2sd   xmm1 , ebx                    // CPU clock ratio N
  divsd      xmm1 , xmm0
  movsd      TCPUFreq(r8).ratio , xmm1

  mov        r9  , rcx
  imul       r9  , rbx
  xchg       r9  , rax
  idiv       r9
  mov        r9  , rax

@fr:
  mov        eax, $16

  cpuid

  cmp        ebx     , 0
  je         @no
  mov        TCPUFreq(r8).base , eax
  mov        TCPUFreq(r8).max  , ebx
  mov        TCPUFreq(r8).bus  , ecx
  jmp        @done

@no:
  shr        r9d               , 20   // divid by 1024
  mov        TCPUFreq(r8).max  , r9d

@done:
  pop rbx
end;
{$else}
begin

end;
{$endif}

destructor TOPool.Destroy;
var i:integer;
begin
  for i:=0 to High(Pool) do begin
    Pool[i].Terminate;
    {$ifdef FPC}
    RTLEventSetEvent(Pool[i].Fire);
    {$else}
    {$ifdef MSWINDOWS}setEvent(Pool[i].Fire){$else}Pool[i].Fire.SetEvent{$endif};
    {$endif}
    //Pool[i].Free;
  end;
  inherited Destroy;
  {$ifdef FPC}
  RTLEventDestroy(PoolDone);
  {$else}
  FreeAndNil(PoolDone)
  {$endif}
end;

procedure TOPool.setWorkers(Count: longint);
var i:longint;
begin
    for i:=0 to High(Pool) do begin
      Pool[i].Terminate;
      {$ifdef FPC}
      RTLEventResetEvent(Pool[i].Fire);
      {$else}
      {$ifdef MSWINDOWS}ResetEvent(Pool[i].Fire){$else}Pool[i].Fire.ReSetEvent{$endif};
      {$endif}
    end;
    Setlength(Pool, Count);
    for i:=0 to High(Pool) do begin
      Pool[i]:=TOthread.Create(false);
      Pool[i].FID:=i;
      Pool[i].FPool:=Self
    end;
end;
//procedure lockInc(var int:integer);assembler;nostackframe;
//asm
//  lock inc [int]
//end;
//
//procedure lockDec(var int:integer);assembler;nostackframe;
//asm
//  lock Dec [int]
//end;
{ TOThread }

procedure TOThread.Execute;
begin
  if assigned(FThreadProc) then begin
    FThreadProc(FParams)
  end else
  while true do begin

    {$ifdef FPC}
    RTLEventWaitFor(Fire);
    {$else}
    {$ifdef MSWINDOWS}WaitForSingleObject(Fire,INFINITE){$else}Fire.WaitFor(){$endif};
    {$endif}
    if not Terminated then begin
      if Assigned(FProc) then begin
        FProc(FStart,FEnd,FParams);
        FProc:=nil;
      end;

      if Assigned(FProcNested) then begin
        FProcNested(FStart,FEnd,FParams);
        FProcNested:=nil;
      end;
      if Assigned(FMethod) then begin
        FMethod(FStart,FEnd,FParams);
        FMethod:=nil;
      end;

      if Assigned(FThreadProc2) then begin
        while FStart<FEnd do begin
           FThreadProc2(FStart,FParams);
           if assigned(FSync) then
                Queue(FSync);
           inc(FStart, FSpan);
        end;
        FThreadProc2:=nil;
      end;
      if Assigned(FThreadProcNested) then begin
        while FStart<FEnd do begin
           FThreadProcNested(FStart,FParams);
           if assigned(FSync) then
                Queue(FSync);
           inc(FStart, FSpan);
        end;
        FThreadProcNested:=nil;
      end;
      {$ifdef FPC}
      InterLockedDecrement(FPool.OTCount);
      {$else}
      TInterLocked.Decrement(FPool.OTCount);
      {$endif}
      {$ifndef FPC}
      {$ifdef MSWINDOWS}
//        ResetEvent(Fire);
      {$else}
        Fire.ResetEvent();
      {$endif}
      {$endif}
      FBusy:=false;
      {$ifdef FPC}
      RTLEventSetEvent(FPool.PoolDone);
      {$else}
      FPool.PoolDone.SetEvent;
      {$endif}
    end else
      exit;
  end
end;

constructor TOThread.Create(CreateSuspended: Boolean; const StackSize: NativeUInt);
begin
  inherited Create(CreateSuspended{$ifdef foc}, StackSize{$endif});
  FreeOnTerminate:=true;;
 {$ifdef FPC}
  Fire:=RTLEventCreate;
 {$else}
 {$ifdef MSWINDOWS}
 fire := CreateEvent(0,false,false,nil);
 {$else}
 Fire:=TEvent.Create();
 {$endif}
 {$endif}
//  Priority:=tpHighest;
  FBusy:=False;
end;

constructor TOThread.Create(const Proc: TGroupProc; const Params: Pointer);
begin
  inherited Create(true);
  FProc:=Proc;
  FParams:=Params
end;

destructor TOThread.Destroy;
begin
  {$ifdef FPC}
  RTLEventDestroy(Fire);
  {$else}
  {$ifdef MSWINDOWS}
  CloseHandle(Fire);
  {$else}
  Fire.Free;
  {$endif}
  {$endif}
  inherited Destroy;
end;

function TOPool.isBusy: boolean;
var i:integer;
begin
  result:=false;
  for i:=0 to length(Pool)-1 do
    if Pool[i].FBusy then
      exit(true)
end;

procedure TOPool.setPriority(const priority: TGroupPriority);
var i:integer;
begin
  for i:=0 to length(Pool)-1 do
    Pool[i].Priority:=priority;
end;

procedure TOPool.WaitForPool;
begin
  while isBusy() do
     {$ifdef fpc}
     RTLEventWaitFor(PoolDone);
     {$else}
     PoolDone.WaitFor()
     {$endif}
end;

function TOPool.getWorkers: longint;
begin
  result:=length(Pool)
end;

procedure TOPool.&For(const Proc: TGroupProc; const _from, _to: IntPtr;
  const Params: Pointer);
var i,N,group_m,group_t,ii:IntPtr;
begin
    if _to < _from then exit;
    while OTCount>0 do;// the pool is still hot! wait for it to cooldown before jumping in.
    N:=_to -_from;
    ii:=0;
    group_t:=ceil((N+1) / length(Pool));
    group_m:=N mod length(Pool);
    if group_t>0 then
      for i:=0 to High(Pool) do begin
          Pool[i].FStart:=ii;
          inc(ii,group_t - longint(i>group_m));
          Pool[i].FEnd:=ii-1;
          if ii<=Pool[i].FStart then break;
          Pool[i].FParams:=Params ;
          Pool[i].FProc:=Proc;
          Pool[i].FBusy:=true;
          //Pool[i].FSync:=sync;

          {$ifdef FPC}
          InterLockedIncrement(OTCount);
          RTLEventSetEvent(Pool[i].Fire);
          {$else}
          TInterLocked.Increment(OTCount);
          {$ifdef MSWINDOWS}
          SetEvent(Pool[i].Fire);
          {$else}
          Pool[i].Fire.SetEvent();
          {$endif}
          {$endif}
      end;
    waitForPool
end;

procedure TOPool.&For(const Proc: TGroupProcNested; const _from, _to: IntPtr;
  const Params: Pointer);
var i,N,group_m,group_t, ii:IntPtr;
begin
    if _to < _from then exit;
    while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
    N:=_to -_from;
    ii:=0;
    group_t:=ceil((N+1) / length(Pool));
    group_m:=N mod length(Pool);
    if group_t>0 then
      for i:=0 to High(Pool) do begin
          Pool[i].FStart:=ii;
          inc(ii,group_t - longint(i>group_m));
          Pool[i].FEnd:=ii-1;
          if ii<=Pool[i].FStart then break;
          Pool[i].FParams:=Params ;
          Pool[i].FProcNested:=Proc;
          Pool[i].FBusy:=true;
          //Pool[i].FSync:=sync;
          {$ifdef FPC}
          InterLockedIncrement(OTCount);
          RTLEventSetEvent(Pool[i].Fire);
          {$else}
          TInterLocked.Increment(OTCount);
          {$ifdef MSWINDOWS}
          SetEvent(Pool[i].Fire);
          {$else}
          Pool[i].Fire.SetEvent();
          {$endif}
          {$endif}
      end;
    waitForPool
end;

procedure TOPool.&For(const Proc: TGroupMethod; const _from, _to: IntPtr;
  const Params: Pointer);
var i,N,group_m,group_t,ii:IntPtr;
begin
    if _to < _from then exit;
    while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
    N:=_to -_from;
    ii:=0;
    group_t:=ceil((N+1) / length(Pool));
    group_m:=N mod length(Pool);
    if group_t>0 then
      for i:=0 to High(Pool) do begin
          Pool[i].FStart:=ii;
          inc(ii,group_t - longint(i>group_m));
          Pool[i].FEnd:=ii-1;
          if ii<=Pool[i].FStart then break;
          Pool[i].FParams:=Params ;
          Pool[i].FMethod:=Proc;
          Pool[i].FBusy:=true;
          //Pool[i].FSync:=sync;
          {$ifdef FPC}
          InterLockedIncrement(OTCount);
          RTLEventSetEvent(Pool[i].Fire);
          {$else}
          TInterLocked.Increment(OTCount);
          {$ifdef MSWINDOWS}
          SetEvent(Pool[i].Fire);
          {$else}
          Pool[i].Fire.SetEvent();
          {$endif}
          {$endif}
      end;
    waitForPool
end;

procedure TOPool.&for(const proc: TThreadProc2; const start, count: IntPtr;
  const args: Pointer; const step: IntPtr; const sync:TThreadMethod);
var
  span, i, ii:IntPtr;
begin
    if count < start then exit;
    while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
    span:=step*Length(Pool);
    ii:=start;
    for i:=0 to high(pool) do begin
        Pool[i].FStart:=ii;
        inc(ii,step);
        if ii>count then break;
        Pool[i].FSpan:=Span;
        Pool[i].FEnd:=count;
        Pool[i].FParams:=args ;
        Pool[i].FThreadProc2:=Proc;
        Pool[i].FBusy:=true;
        Pool[i].FSync:=sync;
        {$ifdef FPC}
        InterLockedIncrement(OTCount);
        RTLEventSetEvent(Pool[i].Fire);
        {$else}
        TInterLocked.Increment(OTCount);
          {$ifdef MSWINDOWS}
          SetEvent(Pool[i].Fire);
          {$else}
          Pool[i].Fire.SetEvent();
          {$endif}
        {$endif}
    end;
    waitForPool
end;

procedure TOPool.&For(const proc: TThreadProcNested; const start,
  count: IntPtr; const args: Pointer; const step: IntPtr;
  const sync: TThreadMethod);
var
  span, i, ii:IntPtr;
begin
    if count < start then exit;
    while OTCount>0 do;// the pool is still hot! waiting to cooldown before jumping in.
    span:=step*Length(Pool);
    ii:=start;
    for i:=0 to high(pool) do begin
        Pool[i].FStart:=ii;
        inc(ii,step);
        if ii>count then break;
        Pool[i].FSpan:=Span;
        Pool[i].FEnd:=count;
        Pool[i].FParams:=args ;
        Pool[i].FThreadProcNested:=Proc;
        Pool[i].FBusy:=true;
        Pool[i].FSync:=sync;
        {$ifdef FPC}
        InterLockedIncrement(OTCount);
        RTLEventSetEvent(Pool[i].Fire);
        {$else}
        TInterLocked.Increment(OTCount);
          {$ifdef MSWINDOWS}
          SetEvent(Pool[i].Fire);
          {$else}
          Pool[i].Fire.SetEvent();
          {$endif}
        {$endif}
    end;
    waitForPool
end;

var s:string;
initialization
  MP:=TOPool.Create;
  MP2:=TOPool.Create;
  MP3:=TOPool.Create;


finalization
  MP3.free;
  MP2.free;
  MP.Free;
end.
