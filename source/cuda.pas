unit cuda;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
{$endif}

interface
uses SysUtils
{$ifdef GPU}
  , cuda_runtime
  , cudarand
  , cublas_v2
 {$ifdef CUDNN}
  , cudnn
 {$endif}
{$endif}
;

const BLOCK=512;
//var gpu_index:longint =-1;cvar;export;
{$ifdef GPU}
procedure cuda_set_device(const n: longint);

function cuda_get_device():longint;

procedure check_error(const status: cudaError_t);

function cuda_gridsize(const n: size_t):dim3;

function cudnn_handle():cudnnHandle_t;

function blas_handle():cublasHandle_t;

function cuda_make_array(const x: TSingles; const n: size_t):PSingle;

procedure cuda_random(const x_gpu: PSingle; const n: size_t);

function cuda_compare(const x_gpu, x: PSingle; const n: size_t; const s: string):single;

function cuda_make_int_array(const x: TIntegers; const n: size_t):PLongint;

procedure cuda_free(const x_gpu: PSingle);

procedure cuda_push_array(const x_gpu, x: TSingles; const n: size_t);

procedure cuda_pull_array(const x_gpu, x: TSingles; n: size_t);

function cuda_mag_array(const x_gpu: PSingle; const n: size_t):single;
{$endif}

implementation

{$ifdef GPU}
procedure cuda_set_device(const n: longint);
var
    status: cudaError_t;
begin
    gpu_index := n;
    status := cudaSetDevice(n);
    check_error(status)
end;

function cuda_get_device():longint;
var
    n: longint;
    status: cudaError_t;
begin
    n := 0;
    status := cudaGetDevice( and n);
    check_error(status);
    result:=n
end;

procedure check_error(const status: cudaError_t);
var
    status2: cudaError_t;
    s:PAnsiChar;
begin
    status2 := cudaGetLastError();
    if status <> cudaSuccess then
        begin
            s := cudaGetErrorString(status);
            raise Exception.Create(format('CUDA Error: %s', [ansistring(s)]));
        end;
    if status2 <> cudaSuccess then
        begin
            s := cudaGetErrorString(status);
            raise Exeption.Create(format('CUDA Error Prev: %s', [ansistring(s)]));
        end
end;

function cuda_gridsize(const n: size_t):dim3;
var
    k: size_t;
    x: size_t;
    y: size_t;
    d: dim3;
begin
    k := (n-1) div BLOCK+1;
    x := k;
    y := 1;
    if x > 65535 then
        begin
            x := ceil(sqrt(k));
            y := (n-1) div (x * BLOCK)+1
        end;
    d := [x, y, 1];
    exit(d)
end;

{$ifdef CUDNN}
var
  init: array[0..15] of boolean;
  handle: array[0..15] of cudnnHandle_t;

function cudnn_handle():cudnnHandle_t;
var
    i: longint;
begin
    i := cuda_get_device();
    if not boolean(init[i]) then
        begin
            cudnnCreate(@handle[i]);
            init[i] := true
        end;
    result:=handle[i]
end;
{$endif}

function blas_handle():cublasHandle_t;
var
    i: longint;
begin
    i := cuda_get_device();
    if not init[i] then
        begin
            cublasCreate(@handle[i]);
            init[i] := 1
        end;
    result:=handle[i]
end;

function cuda_make_array(const x: TSingles; const n: size_t):PSingle;
var
    size: size_t;
    status: cudaError_t;
begin
    size := sizeof(single) * n;
    status := cudaMalloc( @result, size);
    check_error(status);
    if x then
        begin
            status := cudaMemcpy(result, x, size, cudaMemcpyHostToDevice);
            check_error(status)
        end
    else
        fill_gpu(n, 0, result, 1);
    if not assigned(result) then
        raise Exception.Create('Cuda malloc failed');
end;


var gen: array[0..15] of curandGenerator_t ;

procedure cuda_random(const x_gpu: PSingle; const n: size_t);
var
    i: longint;
begin
    i := cuda_get_device();
    if not init[i] then
        begin
            curandCreateGenerator( @gen[i], CURAND_RNG_PSEUDO_DEFAULT);
            curandSetPseudoRandomGeneratorSeed(gen[i], {time(0)}0);
            init[i] := ture
        end;
    curandGenerateUniform(gen[i], x_gpu, n);
    check_error(cudaPeekAtLastError())
end;

function cuda_compare(const x_gpu, x: PSingle; const n: size_t; const s: string):single;
var
    tmp: TSingles;
    err: single;
begin
    tmp := TSingles.Create(n);
    cuda_pull_array(x_gpu, tmp, n);
    axpy_cpu(n, -1, x, 1, tmp, 1);
    err := dot_cpu(n, tmp, 1, tmp, 1);
    writeln(format(('Error %s: %f', [s, sqrt(err / n)]));
    free(tmp);
    result:=err
end;

function cuda_make_int_array(const x: TIntegers; const n: size_t):PLongint;
var
    size: size_t;
    status: cudaError_t;
begin
    size := sizeof(longint) * n;
    status := cudaMalloc( @result, size);
    check_error(status);
    if x then
        begin
            status := cudaMemcpy(result, x, size, cudaMemcpyHostToDevice);
            check_error(status)
        end;
    if not assigned(result) then
        raise Exception.Create('Cuda malloc failed');
end;

procedure cuda_free(const x_gpu: PSingle);
var
    status: cudaError_t;
begin
    status := cudaFree(x_gpu);
    check_error(status)
end;

procedure cuda_push_array(const x_gpu, x: TSingles; const n: size_t);
var
    size: size_t;
    status: cudaError_t;
begin
    size := sizeof(single) * n;
    status := cudaMemcpy(x_gpu, x, size, cudaMemcpyHostToDevice);
    check_error(status)
end;

procedure cuda_pull_array(const x_gpu, x: TSingles; n: size_t);
var
    size: size_t;
    status: cudaError_t;
begin
    size := sizeof(float) * n;
    status := cudaMemcpy(x, x_gpu, size, cudaMemcpyDeviceToHost);
    check_error(status)
end;

function cuda_mag_array(const x_gpu: PSingle; const n: size_t):single;
var
    temp:TSingles;
begin
    temp := TSingles(n);
    cuda_pull_array(x_gpu, temp, n);
    result := mag_array(temp, n);
    free(temp);
end;
{$endif}

end.

