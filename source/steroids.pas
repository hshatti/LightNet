
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
  Just prepare a TProcGroup (or TProcGroupNested or TMethodGroup) like prcedure
then call ir using "ParallelFor" you need to put in a loop
*******************************************************************************************
// example : Sine wave signal generator
program SineWaveGen;
{$mode objfpc}   // allow pointermath
uses {$ifdef unix} cthreads,{$endif} Classes, Types , Steroids, Crt;
const   N=1000000;  Freq=0.05;
var
  i,j,PortHeight:integer;
  a,b: array of Double;
  Amplitude: Double;
// prepare a parallel callback function
procedure _looper(const _start,_end:integer;params:PPointer);
var i:integer;
begin
  for i:=_start to _end do begin
    PDouble(params[0])[i]:=sin(i*freq); // array a is at parameter index 0
    PDouble(params[1])[i]:=cos(i*freq); // array b is at parameter index 1
  end;
end;
begin
  PortHeight:=ScreenHeight-2;
  Amplitude:=PortHeight*0.2;
  setLength(a,N);
  setLength(b,N);
// instead of the following loop :
{
for i:=0 to N-1 do begin
    a[i]:=sin(i*Freq);
    b[i]:=cos(i*Freq)
  end;
}
// put your loop on steroids!!, jump in the thread pool and use this :
  MP.&For(_looper),0,N-1,[ @a[0], @b[0] ]); // remove the "@" when using delphi mode!
  DirectVideo:=True;
  CheckBreak:=True;
  j:=0;
  while true do begin
    inc(j);
    if j>=N then j:=0 ;
    TextBackground(black);
    ClrScr;
    TextColor(Green);
    for i:=0 to ScreenWidth-1 do begin
      GotoXY(i,PortHeight div 4+round(a[j+i]*Amplitude));
      Write('-');
    end;
    TextColor(Red);
    for i:=0 to ScreenWidth-1 do begin
      GotoXY(i,PortHeight * 3 div 4+round(b[j*2+i]*Amplitude));
      Write('o');
    end;
    GotoXY(1,ScreenHeight);
    TextBackground(LightGray);
    TextColor(Black);
    write('Press ESC to exit') ;
    Delay(10);
    if KeyPressed then if ReadKey=#27 then exit;
  end;
//  TextAttr:=TextAttr and not blink;
end.
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
  {$FPUType AVX2}
  {$asmmode intel}
  //{$PackRecords C}
{$endif}
interface
uses Classes, math {$ifdef MSWINDOWS}, windows{$else}{$endif} {$ifndef FPC}, SyncObjs{$endif};

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
  TGroupPriority = integer;
  {$endif}
  TGroupProc    = procedure(const _start,_end:PtrInt;const params:Pointer);
  {$ifdef fpc}  TGroupProcNested=procedure(const _start,_end:PtrInt; const params:Pointer)is nested;register;{$endif} // must be [register]?
  TGroupMethod  = procedure(const _start,_end:PtrInt;const params:Pointer) of object;
  TThreadProc   = procedure(arg :Pointer);
  TThreadProc2  = procedure( i:IntPtr ; arg:pointer);
  TOPool = class;

{ TOThread }
TOThread=class(TThread)
  //TGroupProc=procedure(const params:PPointer);
private
  Fire:{$ifdef FPC}PRTLEvent{$else}{$ifdef MSWINDOWS}THandle{$else}TEvent {$endif}{$endif};
  FStart,FSpan,FEnd:PtrInt;
  FID:integer;
  FParams:Pointer;
  FSync:TThreadMethod;
  FProc:TGroupProc;
{$ifdef FPC}
  FProcNested:TGroupProcNested;
{$endif}
  FMethod:TGroupMethod;
  FThreadProc  : TThreadProc;
  FThreadProc2 : TThreadProc2;
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
  property From:PtrInt read FStart;
  property &To:PtrInt read FEnd;
end;
{ TOPool }
TOPool = class
private
  Pool : array of TOThread;
  OTCount:integer;
  procedure WaitForPool;
  function getWorkers:longint;
public
  constructor Create;
  destructor Destroy; override;
  procedure setWorkers(Count:longint);
  procedure &For(const Proc: TGroupProc; const _from, _to: PtrInt; const Params: Pointer = nil); overload;
{$ifdef FPC}
  procedure &For(const Proc:TGroupProcNested;const _from,_to:PtrInt;const Params:Pointer = nil); overload;
{$endif}
  procedure &For(const Proc:TGroupMethod;const _from,_to:PtrInt;const Params:Pointer = nil); overload;

  procedure &for(const proc:TThreadProc2; const start, count:PtrInt; const args:Pointer; const step:PtrInt=1; const sync:TThreadMethod=nil); overload;
  function isBusy:boolean;
  procedure setPriority(const priority:TGroupPriority);
  property Workers:longint read getWorkers write SetWorkers;
  property PendingWorkCount:integer read OTCount;
  //PoolDone:PRTLEvent;
  //FCriticalSection:TRTLCriticalSection;
end;
var MP, MP2, MP3:TOPool;
function ExecuteInThread(const proc:TThreadProc; args:Pointer):TThread;
function GetSystemThreadCount: integer;
{$ifdef MSWINDOWS}
{$else}
{$ifdef fpc}
const _SC_NPROCESSORS_ONLN = 58;
function sysconf(cmd: Integer): longint;winapi; external;
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
var
  SystemInfo: SYSTEM_INFO;
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
  Setlength(Pool,GetSystemThreadCount);
  for i:=0 to High(Pool) do begin
    Pool[i]:=TOthread.Create(false);
    Pool[i].FID:=i;
    Pool[i].FPool:=Self
  end;
  //PoolDone:=RTLEventCreate;
end;

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
    //RTLEventDestroy(PoolDone);
  //DoneCriticalSection(FCriticalSection);
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
    //RTLEventDestroy(PoolDone);
    Setlength(Pool, Count);
    for i:=0 to High(Pool) do begin
      Pool[i]:=TOthread.Create(false);
      Pool[i].FID:=i;
      Pool[i].FPool:=Self
    end;
    //PoolDone:=RTLEventCreate;
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
{$ifdef FPC}
      if Assigned(FProcNested) then begin
        FProcNested(FStart,FEnd,FParams);
        FProcNested:=nil;
      end;
{$endif}
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
      //EnterCriticalSection(FPool.FCriticalSection);
      {$ifdef FPC}
      InterLockedDecrement(FPool.OTCount);
      {$else}
      TInterLocked.Decrement(FPool.OTCount);
      {$endif}
      //if FPool.OTCount<=0 then
      //  RTLEventSetEvent(FPool.PoolDone);  // jump out of the pool
      //LeaveCriticalSection(FPool.FCriticalSection);
      {$ifdef FPC}
      //RTLEventResetEvent(Fire)
      {$else}
      {$ifdef MSWINDOWS}
//        ResetEvent(Fire);
      {$else}
        Fire.ResetEvent();
      {$endif}
      {$endif}
      FBusy:=false
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
var dummy:int64 ;
begin
  while isBusy() do

//    InterLockedExchange64(dummy,not dummy);
//  asm
//    pause
//  end
end;

function TOPool.getWorkers: longint;
begin
  result:=length(Pool)
end;

procedure TOPool.&For(const Proc: TGroupProc; const _from, _to: PtrInt;
  const Params: Pointer);
var i,N,group_m,group_t,ii:PtrInt;
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
    //RTLEventWaitFor(PoolDone);
    waitForPool
end;
{$ifdef FPC}

procedure TOPool.&For(const Proc: TGroupProcNested; const _from, _to: PtrInt;
  const Params: Pointer);
var i,N,group_m,group_t, ii:PtrInt;
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
          InterLockedIncrement(OTCount);
          RTLEventSetEvent(Pool[i].Fire)
      end;
    //RTLEventWaitFor(PoolDone);
    waitForPool
end;
{$endif}

procedure TOPool.&For(const Proc: TGroupMethod; const _from, _to: PtrInt;
  const Params: Pointer);
var i,N,group_m,group_t,ii:PtrInt;
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
    //RTLEventWaitFor(PoolDone);
    waitForPool
end;

procedure TOPool.&for(const proc: TThreadProc2; const start, count: PtrInt;
  const args: Pointer; const step: PtrInt; const sync:TThreadMethod);
var
  span, i, ii:PtrInt;
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
    //RTLEventWaitFor(PoolDone);
    waitForPool
end;

initialization
  MP:=TOPool.Create;
  MP2:=TOPool.Create;
  MP3:=TOPool.Create;

finalization
  MP3.free;
  MP2.free;
  MP.Free;
end.
