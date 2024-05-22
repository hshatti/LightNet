unit OpenCLHelper;
{$H+}
{$ifdef FPC}
  {$mode delphi}
{$endif}

{$IFDEF MSWINDOWS}

{$endif}
{.$define debug}
interface

uses
  Classes, SysUtils, {$ifdef DARWIN}CL {$else} OpenCL{$endif} ;

const
  cInfoSize=$7fff;

type
{$if not declared(size_t)}
  psize_t = ^size_t;
  size_t  = NativeUInt;
{$endif}

  TCLDeviceType=(
    dtNone = 0 ,
    dtDefault = CL_DEVICE_TYPE_DEFAULT,
    dtCPU = CL_DEVICE_TYPE_CPU,
    dtGPU = CL_DEVICE_TYPE_GPU,
    dtACCELERATOR = CL_DEVICE_TYPE_ACCELERATOR,
    dtALL = CL_DEVICE_TYPE_ALL
  );

  PWorkGroupSize=^TWorkGroupSize;
  TWorkGroupSize=array [0..2] of size_t;

  TCLKernelArgInfo=record
    ArgName:array[0..cInfoSize-1] of ansichar;
    ArgType:array[0..cInfoSize-1] of ansichar;
    ArgAccess:cl_uint;
    ArgAddress:cl_uint;
    ArgTypeQualifier:cl_bitfield;
  end;

  PComplexF=^TComplexF;
  TComplexF=record
    re,im:single
  end;

  PComplexD=^TComplexD;
  TComplexD=record
    re,im:Double
  end;

  TCLKernelInfo=record
    KernelName:array[0..cInfoSize] of ansichar;
    KernelGlobalWorkSize:array[0..2] of size_t;
    KernelWorkGroupSize:size_t;
    KernelLocalMemSize:cl_ulong;
    KernelPrivateMemSize:cl_ulong;
    KernelArgCount:cl_uint;
    KernelArgs:array of TCLKernelArgInfo;
  end;

  { TOpenCl }

  TOpenCL=class
  type
    TDeviceStr =array[0..cInfoSize-1] of ansichar;
  private
    FSrc:TStringList;
    FActivePlatformId: integer;
    FActiveDeviceId: integer;
    FActiveKernelId: integer;
    FActivePlatform: cl_platform_id;
    FActiveDevice: cl_device_id;
    FActiveKernel: cl_kernel ;
    FActiveKernelInfo: TCLKernelInfo;
    FPlatformCount:cl_uint;
    FDeviceCount:cl_uint;
    FKernelCount:cl_uint;
    FPlatforms:array of cl_platform_id;
    FDevices:array of cl_device_id;  // array of pointer to opaque record
    FDevicesType:array of TCLDeviceType;
    FContext:cl_context;
    FKernels:array of cl_kernel;
    FDeviceType: TCLDeviceType;
    FProgramSource: ansistring;
    FQueue:cl_command_queue;
    FErr:cl_int;
    cinfo:TDeviceStr;
    FWorkItemDimensions: integer;
    N:size_t;
    FGlobalOffsets:TWorkGroupSize;
    FGlobalMemSize:size_t;
    FGlobalWorkGroupSizes:TWorkGroupSize;
    FLocalWorkGroupSizes:TWorkGroupSize;

    FLocalMemSize:cl_ulong;
    FExecCaps:cl_device_exec_capabilities;
    FMaxWorkItemDimensions:cl_uint;
    FMaxWorkGroupSize:size_t;
    FMaxWorkItemSizes:TWorkGroupSize;
    FMaxComputeUnits:cl_uint;
    FMaxMemAllocSize:cl_ulong;
    FMaxFrequency:cl_uint;

    FDeviceBuiltInKernels:TDeviceStr;
    FIsBuilt:boolean;
    FProgram:cl_program;
    FBuildStatus:cl_build_status ;
    FBuildLog:ansistring;
    FCallParams:array[0..$ff] of cl_mem;
    FParamSizes:array[0..$ff] of size_t;

    FDevsTypeStr:ansistring;
    FSharedMemory:boolean;
    function GetDevices(index: cl_uint): cl_device_id;
    procedure SetActiveDeviceId(AValue: integer);
    procedure SetActiveKernelId(AValue: integer);
    procedure SetActivePlatformId(AValue: integer);
    procedure SetDeviceType(AValue: TCLDeviceType);
    procedure CheckError;
    function getCL_Device_Type(const dt:TClDeviceType):cl_uint;
    procedure SetGlobalWorkGroupSize(AValue: TWorkGroupSize);overload;
    procedure SetProgramSource(AValue: ansistring);
    procedure SetWorkItemDimensions(AValue: integer);
  public
    CLDeviceVersion:TDeviceStr;
    CLDeviceDriver:TDeviceStr;
    constructor Create(deviceType:TCLDeviceType=dtGPU);
    destructor Destroy;override;
    procedure SetGlobalWorkGroupSize(const x: size_t; y: size_t=0; z: size_t=0); overload;
    procedure SetLocalWorkGroupSize(const x: size_t; y: size_t=0; z: size_t=0);
    procedure SetParamElementSizes(paramSizes: array of size_t);
    function DevicesTypeStr:ansistring;
    procedure SetGlobalOffsets(const x: size_t; y: size_t=0; z: size_t=0);
    function CleanUp(const keepContext: boolean=false): boolean;
    function ProcessorsCount:integer;
    function ProcessorsFrequency:integer;
    property DeviceType:TCLDeviceType read FDeviceType write SetDeviceType;
    property Devices[index:cl_uint]:cl_device_id read GetDevices;
    property ActivePlatformId:Integer read FActivePlatformId write SetActivePlatformId;
    property ActiveDeviceId:Integer read FActiveDeviceId write SetActiveDeviceId;
    property ProgramSource:ansistring read FProgramSource write SetProgramSource;

    property LocalMemSize:cl_ulong                 read FLocalMemSize ;
    property ExecCaps:cl_device_exec_capabilities  read FExecCaps;
    property MaxWorkItemDimensions:cl_uint         read FMaxWorkItemDimensions;
    property MaxWorkGroupSize:size_t               read FMaxWorkGroupSize;
    property MaxWorkItemSizes:TWorkGroupSize       read FMaxWorkItemSizes;
    property MaxComputeUnits:cl_uint               read FMaxComputeUnits;
    property MaxMemAllocSize:cl_ulong              read FMaxMemAllocSize;
    property MaxFrequency:cl_uint                  read FMaxFrequency;
    property ActivePlatform : cl_platform_id read FActivePlatform;
    property ActiveDevice : cl_device_id read FActiveDevice;
    property ActiveContext : cl_context read FContext;
    property ActiveQueue : cl_command_queue read FQueue;
    property ActiveKernel : cl_kernel read FActiveKernel;
    property ActiveKernerInfo : TCLKernelInfo read FActiveKernelInfo;
    property ExecCapabilities : cl_device_exec_capabilities read FExecCaps;

    function PlatformName(Index: integer): ansistring;
    function DeviceName(Index: integer): ansistring;
    function PlatformCount:integer;
    function DeviceCount:integer;
    function Build(const params:ansistring=''):boolean;
    property BuildLog:ansistring read FBuildLog;
    function KernelCount:integer;
    function KernelInfo(index:integer):TCLKernelInfo;
    property GlobalWorkGroupSize:TWorkGroupSize read FGlobalWorkGroupSizes;
    property LocalWorkGroupSize:TWorkGroupSize read FLocalWorkGroupSizes;
    function CanExecuteNative:boolean;
    procedure LoadFromFile(FileName:ansistring);
    //function KernelArgs(index:integer):TCLKernelArgInfo;
    property DeviceBuiltInKernels : TDeviceStr read FDeviceBuiltInKernels;
    property ActiveKernelId:Integer read FActiveKernelId write SetActiveKernelId;
    property WorkItemDimensions:integer read FWorkItemDimensions write SetWorkItemDimensions;
(*    procedure CallKernel(const Index: integer; const dst: PLongWord;const c: integer);inline;  *)
    procedure CallKernel(const Index: integer; const dst, a, b: PSingle; const bias:single;const c: integer); overload;
    procedure CallKernel(const Index: integer; const params: TArray<Pointer>);    overload;
    procedure CallKernel(const Index: integer; const dst: PLongWord; const a, b: integer);  overload;
    class function Plaforms:cl_uint;static;

  end;

{$if not declared(clGetKernelArgInfo)}
    cl_kernel_arg_info                         = cl_uint;

  const
    CL_KERNEL_ARG_ADDRESS_QUALIFIER = $1196;
    CL_KERNEL_ARG_ACCESS_QUALIFIER  = $1197;
    CL_KERNEL_ARG_TYPE_NAME         = $1198;
    CL_KERNEL_ARG_TYPE_QUALIFIER    = $1199;
    CL_KERNEL_ARG_NAME              = $119A;
    CL_DEVICE_BUILT_IN_KERNELS      = $103f;
    CL_DEVICE_HOST_UNIFIED_MEMORY   = $1035;

  function clGetKernelArgInfo (kernel:cl_kernel;
                     arg_indx:cl_uint;
                     param_name:cl_kernel_arg_info;
                     param_value_size:size_t;
                     param_value:pointer;
                     param_value_size_ret:psize_t):cl_int;winapi;external;
{$endif}

implementation

(*
// CONSTANTS
// The source code of the kernel is represented as a ansistring
// located inside file: "fft1D_1024_kernel_src.cl". For the details see the next listing.

// Looking up the available GPUs
case ComboBox1.ItemIndex of
  0:deviceType:=CL_DEVICE_TYPE_GPU;
  1:deviceType:=CL_DEVICE_TYPE_CPU;
  2:deviceType:=CL_DEVICE_TYPE_DEFAULT;
end;


ret:=clGetDeviceIDs(nil, deviceType, 0, nil, @num);
if ret<>CL_SUCCESS then raise Exception.create('Cannot list Processors');

setLength(devices,num);
       //cl_device_id devices[1];
ret:=clGetDeviceIDs(nil, deviceType, num, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get ALL Device id ');
ListBox1.Items.Clear;
for i:=0 to num -1 do begin
  clGetDeviceInfo(devices[i],CL_DEVICE_NAME,256,@deviceInfo[0],retSize);
  ListBox1.Items.add(deviceInfo);
end;

// create a compute context with GPU device
context := clCreateContextFromType(nil, deviceType, nil, nil, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Create context from GPU Type');

// create a command queue
ret:=clGetDeviceIDs(nil, deviceType, 1, @devices[0], nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot get Default Device');

ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_COMPUTE_UNITS,256,@deviceInfo[0],retSize);
ListBox1.Items.Add(IntToStr(PLongWord(@deviceInfo)^)+' Units');
ret:=clGetDeviceInfo(devices[0],CL_DEVICE_MAX_CLOCK_FREQUENCY,256,@deviceInfo[0],retSize);
ListBox1.Items[ListBox1.Items.count-1]:=ListBox1.Items[ListBox1.Items.count-1]+'@'+IntToStr(PLongWord(@deviceInfo)^)+'Mhz';



queue := clCreateCommandQueue(context, devices[0], 0{props}, ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create command queue');

t:=GetTickCount64;
bmp.BeginUpdate();

// allocate the buffer memory objects
 memobjs[0]:=  clCreateBuffer(context, CL_MEM_WRITE_ONLY , 4*w*h, nil, ret);
 if ret<>CL_SUCCESS then raise Exception.create('Cannot create ReadMem');
 //memobjs[1]:=  clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(single) * 2 * NUM_ENTRIES, nil, ret);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot create WriteMem');
// cl_mem memobjs[0] = // FIXED, SEE ABOVE
// cl_mem memobjs[1] = // FIXED, SEE ABOVE

// create the compute program
// const ansichar* fft1D_1024_kernel_src[1] = {  };
prog := clCreateProgramWithSource(context, 1, PPAnsiCHAR(@src), nil, ret);

if ret<>CL_SUCCESS then raise Exception.create('Cannot create ProgramWithSource');

// build the compute program executable
ret:=clBuildProgram(prog, 0, nil, nil, nil, nil);

if ret<>CL_BUILD_SUCCESS then begin
  clGetProgramBuildInfo(prog,devices[0],CL_PROGRAM_BUILD_LOG,256,@buildLog[0],retSize);
  raise Exception.CreateFmt('Cannot Build executable message:'#13#10'[%s]',[buildLog]);
end;
// create the compute kernel


kernel := clCreateKernel(prog, 'render', ret);
if ret<>CL_SUCCESS then raise Exception.create('Cannot create Kernel');
// set the args values

//size_t local_work_size[1] = { 256 };

ret:=clSetKernelArg(kernel, 0, sizeof(cl_mem), @memobjs[0]);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[0]');
ret:=clSetKernelArg(kernel, 1, sizeof(max_iteration), @max_iteration);
if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 1, sizeof(cl_mem), @memobjs[1]);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[1]');
//ret:=clSetKernelArg(kernel, 2, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[2]');
//ret:=clSetKernelArg(kernel, 3, sizeof(single)*(local_work_size[0] + 1) * 16, nil);
//       if ret<>CL_SUCCESS then raise Exception.create('Cannot set argument[3]');
//
// create N-D range object with work-item dimensions and execute kernel
//size_t global_work_size[1] = { 256 };

//global_work_size[0] := NUM_ENTRIES;
//local_work_size[0] := 64; //Nvidia: 192 or 256

ret:=clEnqueueNDRangeKernel(queue, kernel, 2, global_work_offset, global_work_size, nil, 0, nil, nil);
if ret<>CL_SUCCESS then raise Exception.create('Cannot Enqueue ND Range kernel');

clEnqueueReadBuffer(queue,memobjs[0],cl_false,{offset in byte }0,w*h*4{size in byte},bmp.ScanLine[0],0,nil,nil);
//clFlush(queue);
clFinish(queue);
ListBox1.Items.Add(format(' -Rendering took %d MilliSeconds',[GetTickCount64-t]));
bmp.EndUpdate();
Image1.picture.Graphic:=bmp ;

clReleaseMemObject(memobjs[0]);
clReleaseCommandQueue(queue);
clReleaseContext(context);
clReleaseKernel(kernel);
clReleaseProgram(prog);
*)

{ TOpenCl }

procedure TOpenCL.SetDeviceType(AValue: TCLDeviceType);
var wasBuilt:boolean;
begin
  if FDeviceType=AValue then Exit;
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  FDeviceType:=AValue;
  FActiveDeviceId:=-1;
  wasBuilt:=FIsBuilt;
  SetActivePlatformId(FActivePlatformId);
  if wasBuilt then
    Build
end;

procedure TOpenCL.CheckError;
begin
  if FErr<>CL_SUCCESS then
    writeln(clErrorText(FErr));
    //raise Exception.Create(clErrorText(FErr));
end;

function TOpenCL.getCL_Device_Type(const dt: TClDeviceType): cl_uint;
begin
  case dt of
    dtDefault :result:= CL_DEVICE_TYPE_DEFAULT;
    dtCPU :result:=CL_DEVICE_TYPE_CPU;
    dtGPU :result:=CL_DEVICE_TYPE_GPU;
    dtACCELERATOR :result:=CL_DEVICE_TYPE_ACCELERATOR;
    dtALL :result:=CL_DEVICE_TYPE_ALL
  end;
end;

procedure TOpenCL.SetGlobalWorkGroupSize(AValue: TWorkGroupSize);
begin
//  if FGlobalWorkGroupSize=AValue then Exit;
  FGlobalWorkGroupSizes:=AValue;
end;

procedure TOpenCL.SetProgramSource(AValue: ansistring);
var i:integer;
begin
  if FProgramSource=AValue then Exit;
  if Assigned(FKernels) then
    for i:=0 to FKernelCount-1 do
      clReleaseKernel(FKernels[i]);
  setLength(FKernels,0);
  if Assigned(FProgram) then clReleaseProgram(FProgram);
  FIsBuilt:=false;
  FProgramSource:=AValue;
end;

procedure TOpenCL.SetWorkItemDimensions(AValue: integer);
begin
  if FWorkItemDimensions=AValue then Exit;
  FWorkItemDimensions:=AValue;
end;

constructor TOpenCL.Create(deviceType: TCLDeviceType);
var i:integer;
begin
  FDeviceType:=deviceType;
  FPlatformCount:=0;
  FillChar(FParamSizes,sizeof(FParamSizes),0);
  for i:=0 to high(FGlobalOffsets) do FGlobalOffsets[i]:=0;
  N:=Length(cInfo);
  FErr:=clGetPlatformIDs(0,nil,@FPlatformCount);
  if FErr=CL_SUCCESS then
    if FPlatformCount>0 then begin
      SetLength(FPlatforms,FPlatformCount);
      FErr:=clGetPlatformIDs(FPlatformCount,@FPlatforms[0],nil);CheckError;
      FSrc:=TStringList.Create;
      FActivePlatformId:=$7fffffff;
      FActiveDeviceId:=$7fffffff;
      FWorkItemDimensions:=1;
      SetActivePlatformId(0);
    end;
end;

destructor TOpenCL.Destroy;
begin
  CleanUp(false);
  FSrc.Free;
  inherited Destroy;
end;

procedure TOpenCL.SetGlobalWorkGroupSize(const x: size_t; y: size_t; z: size_t);
begin
  FGlobalWorkGroupSizes[0]:=x;
  FGlobalWorkGroupSizes[1]:=y;
  FGlobalWorkGroupSizes[2]:=z;
  FillChar(FGlobalOffsets,sizeof(FGlobalOffsets),0);
  if z>0 then FWorkItemDimensions:=3
  else if y>0 then FWorkItemDimensions:=2
  else FWorkItemDimensions:=1;
end;

procedure TOpenCL.SetLocalWorkGroupSize(const x: size_t; y: size_t; z: size_t);
begin
  FLocalWorkGroupSizes[0]:=x;
  FLocalWorkGroupSizes[1]:=y;
  FLocalWorkGroupSizes[2]:=z;
end;

procedure TOpenCL.SetParamElementSizes(paramSizes: array of size_t);
var i:integer;
begin
  for i:=0 to High(paramSizes) do
    FParamSizes[i]:=paramSizes[i]
end;

function TOpenCL.DevicesTypeStr: ansistring;
begin
  result:=FDevsTypeStr;
end;

procedure TOpenCL.SetGlobalOffsets(const x: size_t; y: size_t; z: size_t);
begin
  FGlobalOffsets[0]:=x;
  FGlobalOffsets[1]:=y;
  FGlobalOffsets[2]:=z;
end;

function TOpenCL.CleanUp(const keepContext:boolean): boolean;
var i:integer;
begin
  try
    for i:=0 to High(FKernels) do begin
      clReleaseKernel(FKernels[i]);  CheckError;
      FKernels[i] := nil
    end;

    if FProgram<>nil then begin
      clReleaseProgram(FProgram);CheckError;
      FProgram := nil
    end;

    if FQueue<>nil then begin
      clReleaseCommandQueue(FQueue);CheckError;
      FQueue := nil
    end;

    if not keepContext then if
      FContext<>nil then begin
        clReleaseContext(FContext);CheckError;
        FContext:=nil;
      end;

    FIsBuilt:=false;
    result:=true
  except on E:Exception do
    begin
      result:=false
    end;

  end;
end;

function TOpenCL.ProcessorsCount: integer;
begin
  result:=FMaxComputeUnits;
end;

function TOpenCL.ProcessorsFrequency: integer;
begin
  result:=FMaxFrequency;
end;

function TOpenCL.PlatformName(Index: integer): ansistring;
begin

  clGetPlatformInfo(FPlatforms[Index],CL_PLATFORM_NAME,cInfoSize,@cinfo[0],N);
  result:=cinfo;
end;

function TOpenCL.DeviceName(Index: integer): ansistring;
begin
  N:=255;
  clGetDeviceInfo(FDevices[Index],CL_DEVICE_NAME,cInfoSize,@cinfo[0],N);
  result:=cinfo;
end;

function TOpenCL.PlatformCount: integer;
begin
  result:=FPlatformCount
end;

function TOpenCL.DeviceCount: integer;
begin
  result:=FDeviceCount
end;

function TOpenCL.Build(const params: ansistring): boolean;
var src,par:PAnsiChar; sz:cl_uint;
begin
  result:=False;
  src:=PAnsiChar(FProgramSource);
  par:=PAnsiCHar('-cl-kernel-arg-info -cl-fast-relaxed-math -cl-mad-enable '+params);
  FProgram:=clCreateProgramWithSource(FContext,1,@src,nil,FErr);CheckError;
  FErr:=clBuildProgram(Fprogram,FDeviceCount,@FDevices[0],par,nil,nil);
  FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_STATUS,cInfoSize,@FBuildStatus,N);CheckError;
  if FBuildStatus<> CL_BUILD_SUCCESS then begin
    FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0],N);CheckError;
    FBuildLog:=copy(cinfo,0,N);
  end else begin
    FErr:=clCreateKernelsInProgram(FProgram,0,nil,FKernelCount);CheckError;
    setLength(FKernels,FKernelCount);
    FErr:=clCreateKernelsInProgram(FProgram,FKernelCount,@FKernels[0],sz);CheckError;
    FErr:=clGetProgramBuildInfo(FProgram,FActiveDevice,CL_PROGRAM_BUILD_LOG,cInfoSize,@cinfo[0],N);CheckError;
    if N>1 then FBuildLog:=cinfo;
    FActiveKernelId:=-1;
    SetActiveKernelId(0);
    FIsBuilt:=True;
    Result:=True
  end;
//  if cinfo='' then cinfo:='Success';
end;

function TOpenCL.KernelCount: integer;
begin
  result:=FKernelCount;
end;

function TOpenCL.KernelInfo(index: integer): TCLKernelInfo;
var sz:size_t;i:integer;
begin
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_FUNCTION_NAME,cInfoSize,@result.KernelName[0],N);                                      CheckError;
    FErr:=clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS,cInfoSize,@result.KernelArgCount,N);                                          CheckError;
//    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_GLOBAL_WORK_SIZE,cInfoSize,@result.KernelGlobalWorkSize[0],@N);  CheckError;
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_WORK_GROUP_SIZE,cInfoSize,@result.KernelWorkGroupSize,@N);     CheckError;
    FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_LOCAL_MEM_SIZE,cInfoSize,@result.KernelLocalMemSize,@N);       CheckError;
    //FErr:=clGetKernelWorkGroupInfo(FKernels[Index],FActiveDevice,CL_KERNEL_PRIVATE_MEM_SIZE,cInfoSize,@result.KernelPrivateMemSize,@N);     CheckError;
    setLength(result.KernelArgs,result.KernelArgCount);
    for i:=0 to result.KernelArgCount-1 do begin
        N:=$ff;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_NAME,cInfoSize,@result.KernelArgs[i].ArgName[0],@N);                         CheckError;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_TYPE_NAME,cInfoSize,@result.KernelArgs[i].ArgType[0],@N);                    CheckError;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_TYPE_QUALIFIER,cInfoSize,@result.KernelArgs[i].ArgTypeQualifier,@N);         CheckError;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_ACCESS_QUALIFIER,cInfoSize,@result.KernelArgs[i].ArgAccess,@N);              CheckError;
        FErr:=clGetKernelArgInfo(FKernels[Index],i,CL_KERNEL_ARG_ADDRESS_QUALIFIER,cInfoSize,@result.KernelArgs[i].ArgAddress,@N);            CheckError;
    end;

end;

function TOpenCL.CanExecuteNative: boolean;
begin
  result:=FExecCaps and CL_EXEC_NATIVE_KERNEL>0;
end;

procedure TOpenCL.LoadFromFile(FileName: ansistring);
begin
  FSrc.LoadFromFile(FileName);
  ProgramSource:=FSrc.Text;
end;

(*
procedure TOpenCL.CallKernel(const Index: integer; const dst:PLongWord;const c: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];
  FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE,sz*SizeOf(LongWord),nil,FErr);CheckError;
  //FCallParams[1]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*4,nil,FErr);
  //FCallParams[2]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY, c*8,nil,FErr);
  FErr:=clSetKernelArg(FActiveKernel,0,sizeOf(@FCallParams[0]),@FCallParams[0]);CheckError;
  //FErr:=clSetKernelArg(FActiveKernel,1,sizeOf(cl_mem),FCallParams[1]);
  //FErr:=clSetKernelArg(FActiveKernel,2,sizeOf(cl_mem),FCallParams[2]);
  FErr:=clSetKernelArg(FActiveKernel,1,SizeOf(c),@c);CheckError;
  FErr:=clEnqueueNDRangeKernel(FQueue,FActiveKernel,FWorkItemDimensions,FGlobalOffsets,FGlobalWorkGroupSizes,FLocalWorkGroupSizesPtr,0,nil,nil);CheckError;
  FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_True,0,sz*SizeOf(LongWord),dst,0,nil,nil);CheckError;
  //FErr:=clFlush(FQueue);
  //FErr:=clFinish(FQueue);

end;
*)
procedure TOpenCL.CallKernel(const Index: integer; const dst, a, b: PSingle;
  const bias: single; const c: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];

    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_WRITE_ONLY {or CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR}, {FParamSizes[0]*}sz*SizeOf(dst^),nil,FErr);CheckError;
    FCallParams[1]:=clCreateBuffer(FContext,CL_MEM_USE_HOST_PTR or CL_MEM_READ_ONLY,{FParamSizes[1]*}sz*SizeOf(a^),    a,FErr);CheckError;
    FCallParams[2]:=clCreateBuffer(FContext,CL_MEM_USE_HOST_PTR or CL_MEM_READ_ONLY,{FParamSizes[2]*}c*c*SizeOf(b^),    b,FErr);CheckError;
    FErr:=clSetKernelArg(FKernels[Index],0,sizeof(@FCallParams[0]),@FCallParams[0]);CheckError;
    FErr:=clSetKernelArg(FKernels[Index],1,sizeOf(@FCallParams[1]),@FCallParams[1]);CheckError;
    FErr:=clSetKernelArg(FKernels[Index],2,sizeOf(@FCallParams[2]),@FCallParams[2]);CheckError;
    FErr:=clSetKernelArg(FKernels[Index],3,SizeOf(bias),@bias);CheckError;
    FErr:=clSetKernelArg(FKernels[Index],4,SizeOf(c),@c);CheckError;
    FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index], FWorkItemDimensions, @FGlobalOffsets[0], @FGlobalWorkGroupSizes[0], @FLocalWorkGroupSizes[0], 0, nil, nil); CheckError;
//  if not FSharedMemory then
    FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_FALSE,0,{FParamSizes[0]*}sz*SizeOf(dst^),dst,0,nil,nil);CheckError;
  //FErr:=clFlush(FQueue);
  FErr:=clFinish(FQueue);

end;

procedure TOpenCL.CallKernel(const Index: integer; const params: TArray<Pointer>
  );
var
  ki:TCLKernelInfo;
  //sz:size_t;
  i,j:integer; s:ansistring;
begin
  if FSharedMemory then
    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE or CL_MEM_USE_HOST_PTR {or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],Params[0],FErr)
  else
    FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_READ_WRITE {or CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],nil      ,FErr);CheckError;

  for i:=0 to KernelInfo(Index).KernelArgCount-1 do begin
    j:=KernelInfo(Index).KernelArgs[i].ArgAccess;
    if Pos('*',KernelInfo(Index).KernelArgs[i].ArgType)>0 then
      FCallParams[i]:=clCreateBuffer(FContext,CL_MEM_READ_ONLY or CL_MEM_USE_HOST_PTR ,sizeof(cl_mem),params[i],FErr);CheckError;
  end;

  for i:=0 to KernelInfo(Index).KernelArgCount-1 do begin
    if Pos('*',KernelInfo(Index).KernelArgs[i].ArgType)>0 then
      FErr:=clSetKernelArg(FKernels[Index],i,sizeof(@FCallParams[i]),@FCallParams[i])
    else
      FErr:=clSetKernelArg(FKernels[Index],i,FParamSizes[i],params[i]);CheckError;
  end;
  FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index],FWorkItemDimensions, @FGlobalOffsets[0], @FGlobalWorkGroupSizes[0], @FLocalWorkGroupSizes[0] ,0,nil,nil); CheckError;
  if not FSharedMemory then begin
    FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_FALSE,0,FParamSizes[0],params[0],0,nil,nil);CheckError;
    FErr:=clFinish(FQueue);
  end;

end;

procedure TOpenCL.CallKernel(const Index: integer; const dst: PLongWord;
  const a, b: integer);
var ki:TCLKernelInfo;sz:size_t;i:integer;
begin
  if Index > high(FKernels) then
    raise ERangeError.CreateFmt('Kernel [%d] out of Bounds : Number of Kernels = %d',[Index, length(FKernels)]);
  sz:=FGlobalWorkGroupSizes[0];
  for i:=1 to FWorkItemDimensions-1 do
    sz:=sz*FGlobalWorkGroupSizes[i];

  FCallParams[0]:=clCreateBuffer(FContext,CL_MEM_WRITE_ONLY {or CL_MEM_USE_HOST_PTR or CL_MEM_ALLOC_HOST_PTR}, FParamSizes[0],nil,FErr);CheckError;

  FErr:=clSetKernelArg(FKernels[Index],0,sizeof(FCallParams[0]),@FCallParams[0]);CheckError;
  FErr:=clSetKernelArg(FKernels[Index],1,SizeOf(a),@a);CheckError;
  FErr:=clSetKernelArg(FKernels[Index],2,SizeOf(b),@b);CheckError;
  FErr:=clEnqueueNDRangeKernel(FQueue,FKernels[Index] ,FWorkItemDimensions ,@FGlobalOffsets[0] , @FGlobalWorkGroupSizes[0] ,@FLocalWorkGroupSizes[0] ,0 ,nil ,nil ); CheckError;
  FErr:=clEnqueueReadBuffer(FQueue,FCallParams[0],CL_TRUE,0,{FParamSizes[0]*}sz*SizeOf(dst^),dst,0,nil,nil);CheckError;
  //FErr:=clFinish(FQueue);
  FErr:=clReleaseMemObject(FCallParams[0]);CheckError;

end;

class function TOpenCL.Plaforms: cl_uint;

begin
    clGetPlatformIDs(0,nil,@result);
end;

//function TOpenCL.KernelArgs(index: integer): TCLKernelArgInfo;
//begin
//  clGetKernelInfo(FKernels[Index],CL_KERNEL_NUM_ARGS,SizeOf(Result),@result,N);
//end;

function TOpenCL.GetDevices(index: cl_uint): cl_device_id;
begin
  result:=FDevices[index];
end;

procedure TOpenCL.SetActiveDeviceId(AValue: integer);
var wasBuilt:boolean; isShared:cl_bool;
begin
  if FActiveDevice=FDevices[AValue] then Exit;
  if AValue>High(FDevices) then
    raise Exception.Create('Device index out of bounds!');
  wasBuilt:=FIsBuilt;
  CleanUp(true);
  FQueue:=clCreateCommandQueue(FContext,FDevices[AValue],0, 0 (* QWord(@FErr) *) );
  CheckError;
  FActiveDevice:=FDevices[AValue];
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_EXECUTION_CAPABILITIES,SizeOf(cl_device_exec_capabilities),@FExecCaps,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_GROUP_SIZE,SizeOf(FMaxWorkGroupSize),@FMaxWorkGroupSize,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,SizeOf(FMaxWorkItemDimensions),@FMaxWorkItemDimensions,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_MEM_ALLOC_SIZE,SizeOf(FMaxMemAllocSize),@FMaxMemAllocSize,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_WORK_ITEM_SIZES,SizeOf(size_t)*3,@FMaxWorkItemSizes[0],N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_COMPUTE_UNITS,SizeOf(FMaxComputeUnits),@FMaxComputeUnits,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_MAX_CLOCK_FREQUENCY,SizeOf(FMaxFrequency),@FMaxFrequency,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_GLOBAL_MEM_SIZE,SizeOf(FGlobalMemSize),@FGlobalMemSize,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_LOCAL_MEM_SIZE,SizeOf(FLocalMemSize),@FLocalMemSize,N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_BUILT_IN_KERNELS,cInfoSize,@FDeviceBuiltInKernels[0],N);CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_OPENCL_C_VERSION,cInfoSize,@CLDeviceVersion[0],N); CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_VENDOR,cInfoSize,@CLDeviceDriver,N);            CheckError;
  FErr:=clGetDeviceInfo(FActiveDevice,CL_DEVICE_HOST_UNIFIED_MEMORY,SizeOf(isShared),@isShared,N);CheckError;
  FSharedMemory:=isShared=CL_TRUE;
  if wasBuilt then
    Build;
  FActiveDeviceId:=AValue;
end;

procedure TOpenCL.SetActiveKernelId(AValue: integer);
begin
  if FActiveKernelId=AValue then exit;
  FActiveKernel:=FKernels[AValue];
  FActiveKernelId:=AValue;
  FActiveKernelInfo := KernelInfo(AValue)
end;

procedure TOpenCL.SetActivePlatformId(AValue: integer);
var i:integer; dt:cl_device_type;
begin
  if (FActivePlatform=FPlatforms[AValue]) or (AValue>High(FPlatforms)) then Exit;
  if AValue>High(FPlatforms) then raise Exception.Create('Platform index out of bounds!');
  FActivePlatform:=FPlatforms[AValue];
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),0,nil,@FDeviceCount);  CheckError;
  if FDeviceCount=0 then raise Exception.Create('No Devices found!');
  setLength(FDevices,FDeviceCount);
  setLength(FDevicesType,FDeviceCount);
  FErr:=clGetDeviceIDs(FActivePlatform,getCL_Device_Type(FDeviceType),FDeviceCount,@FDevices[0],nil);  CheckError;
  FDevsTypeStr:='';
  for i:=0 to FDeviceCount-1 do
    begin
      FErr:=clGetDeviceInfo(FDevices[i],CL_DEVICE_TYPE_INFO,SizeOf(size_t),@dt,N);  CheckError;
      Case dt of
        CL_DEVICE_TYPE_DEFAULT:begin FDevicesType[i]:=dtDefault;FDevsTypeStr:=FDevsTypeStr+#13#10'DEFAULT' end;
        CL_DEVICE_TYPE_CPU:begin FDevicesType[i]:=dtCPU;FDevsTypeStr:=FDevsTypeStr+#13#10'CPU' end;
        CL_DEVICE_TYPE_GPU:begin FDevicesType[i]:=dtGPU;FDevsTypeStr:=FDevsTypeStr+#13#10'GPU' end;
        CL_DEVICE_TYPE_ACCELERATOR:begin FDevicesType[i]:=dtACCELERATOR;FDevsTypeStr:=FDevsTypeStr+#13#10'ACCELERATOR' end;
      end;
    end;
  delete(FDevsTypeStr,1,2);
  if FContext<>nil then begin
    clReleaseContext(FContext);CheckError;
    FContext:=nil
  end;
  FContext:=clCreateContext(nil,FDeviceCount,@FDevices[0],nil,nil,FErr);CheckError;
  FActiveDeviceId:=-1;
  SetActiveDeviceId(0);
  FActivePlatformId:=AValue;
end;

initialization

end.

