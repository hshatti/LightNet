unit gemm;

{$ifdef fpc}
  {$mode delphi}
  {$ModeSwitch cvar}
  {$ModeSwitch typehelpers}
  {$ModeSwitch nestedprocvars}
  {$ModeSwitch advancedrecords}
  {.$FPUType AVX2}
  {$asmmode intel}
{$else}
  {$excessprecision off}
{$endif}
{$pointermath on}

interface
uses classes, sysutils, col2im, darknet
  , steroids
{$if defined(CLBLAST)}
    , cl
{$endif}
{$if defined(OPENBLAS)}
  , openblas
{$elseif defined(MKL)}
  , mkl_cblas
{$else}
  {$define GEMM}
{$endif}
;

(*
Keywords :
  Internal Covariate Shift :
    a problem accures in deep neural networks when too many multiplications after forward make the output numbers large with big varriance, the solution is to add an "Auxilary Head Classifier" and to normalize the output vectors

  Vanished Gradients :
    a problem accures when networks go deeper the numbers become  smaller and next to nil which will vanish at the end of the deep network, one of the solutions is to use skip connections (shortcut layer, vector addition) such as ResNet

  Network scaling :
    a term used when increasing the size of : layers or featchers called (width or channels also) or image resolution  (Compound scaling Cofficients)

  EfficientNet :

  Compound Scaling :
   a method to calculate the optimum Depth/Channels/Resolution for a neural network using a special neural network (NAS) to desine it the result is called EfficientNet
   network scaling factor = Depth(named as A) * features(named as B)^F * resolution(named as C)^F    , F is a constant coefficient to be calculated later
   the creators of the EfficientNet model came out with these optimum results for A, B and C constants which is 1.2 , 1.1, 1.15 for F=1   (changable after performing a grid search) (?)

  Neural Architectur Search (NAS) :
    a neural network for designing an efficientNet achitect given the Resolution of the image for the Compound Scaling calculation?

  Extended Efficient Layer aggregation:
    ----


*)

{$ifdef CLBLAST}
const clblast= 'clblast.dll';
type
  TCLBlastStatusCode =(
    CLBlastSuccess                   =   0, // CL_SUCCESS
    CLBlastOpenCLCompilerNotAvailable=  -3, // CL_COMPILER_NOT_AVAILABLE
    CLBlastTempBufferAllocFailure    =  -4, // CL_MEM_OBJECT_ALLOCATION_FAILURE
    CLBlastOpenCLOutOfResources      =  -5, // CL_OUT_OF_RESOURCES
    CLBlastOpenCLOutOfHostMemory     =  -6, // CL_OUT_OF_HOST_MEMORY
    CLBlastOpenCLBuildProgramFailure = -11, // CL_BUILD_PROGRAM_FAILURE: OpenCL compilation error
    CLBlastInvalidValue              = -30, // CL_INVALID_VALUE
    CLBlastInvalidCommandQueue       = -36, // CL_INVALID_COMMAND_QUEUE
    CLBlastInvalidMemObject          = -38, // CL_INVALID_MEM_OBJECT
    CLBlastInvalidBinary             = -42, // CL_INVALID_BINARY
    CLBlastInvalidBuildOptions       = -43, // CL_INVALID_BUILD_OPTIONS
    CLBlastInvalidProgram            = -44, // CL_INVALID_PROGRAM
    CLBlastInvalidProgramExecutable  = -45, // CL_INVALID_PROGRAM_EXECUTABLE
    CLBlastInvalidKernelName         = -46, // CL_INVALID_KERNEL_NAME
    CLBlastInvalidKernelDefinition   = -47, // CL_INVALID_KERNEL_DEFINITION
    CLBlastInvalidKernel             = -48, // CL_INVALID_KERNEL
    CLBlastInvalidArgIndex           = -49, // CL_INVALID_ARG_INDEX
    CLBlastInvalidArgValue           = -50, // CL_INVALID_ARG_VALUE
    CLBlastInvalidArgSize            = -51, // CL_INVALID_ARG_SIZE
    CLBlastInvalidKernelArgs         = -52, // CL_INVALID_KERNEL_ARGS
    CLBlastInvalidLocalNumDimensions = -53, // CL_INVALID_WORK_DIMENSION: Too many thread dimensions
    CLBlastInvalidLocalThreadsTotal  = -54, // CL_INVALID_WORK_GROUP_SIZE: Too many threads in total
    CLBlastInvalidLocalThreadsDim    = -55, // CL_INVALID_WORK_ITEM_SIZE: ... or for a specific dimension
    CLBlastInvalidGlobalOffset       = -56, // CL_INVALID_GLOBAL_OFFSET
    CLBlastInvalidEventWaitList      = -57, // CL_INVALID_EVENT_WAIT_LIST
    CLBlastInvalidEvent              = -58, // CL_INVALID_EVENT
    CLBlastInvalidOperation          = -59, // CL_INVALID_OPERATION
    CLBlastInvalidBufferSize         = -61, // CL_INVALID_BUFFER_SIZE
    CLBlastInvalidGlobalWorkSize     = -63, // CL_INVALID_GLOBAL_WORK_SIZE

    // Status codes in common with the clBLAS library
    CLBlastNotImplemented            = -1024, // Routine or functionality not implemented yet
    CLBlastInvalidMatrixA            = -1022, // Matrix A is not a valid OpenCL buffer
    CLBlastInvalidMatrixB            = -1021, // Matrix B is not a valid OpenCL buffer
    CLBlastInvalidMatrixC            = -1020, // Matrix C is not a valid OpenCL buffer
    CLBlastInvalidVectorX            = -1019, // Vector X is not a valid OpenCL buffer
    CLBlastInvalidVectorY            = -1018, // Vector Y is not a valid OpenCL buffer
    CLBlastInvalidDimension          = -1017, // Dimensions M, N, and K have to be larger than zero
    CLBlastInvalidLeadDimA           = -1016, // LD of A is smaller than the matrix's first dimension
    CLBlastInvalidLeadDimB           = -1015, // LD of B is smaller than the matrix's first dimension
    CLBlastInvalidLeadDimC           = -1014, // LD of C is smaller than the matrix's first dimension
    CLBlastInvalidIncrementX         = -1013, // Increment of vector X cannot be zero
    CLBlastInvalidIncrementY         = -1012, // Increment of vector Y cannot be zero
    CLBlastInsufficientMemoryA       = -1011, // Matrix A's OpenCL buffer is too small
    CLBlastInsufficientMemoryB       = -1010, // Matrix B's OpenCL buffer is too small
    CLBlastInsufficientMemoryC       = -1009, // Matrix C's OpenCL buffer is too small
    CLBlastInsufficientMemoryX       = -1008, // Vector X's OpenCL buffer is too small
    CLBlastInsufficientMemoryY       = -1007, // Vector Y's OpenCL buffer is too small

    // Custom additional status codes for CLBlast
    CLBlastInsufficientMemoryTemp    = -2050, // Temporary buffer provided to GEMM routine is too small
    CLBlastInvalidBatchCount         = -2049, // The batch count needs to be positive
    CLBlastInvalidOverrideKernel     = -2048, // Trying to override parameters for an invalid kernel
    CLBlastMissingOverrideParameter  = -2047, // Missing override parameter(s) for the target kernel
    CLBlastInvalidLocalMemUsage      = -2046, // Not enough local memory available on this device
    CLBlastNoHalfPrecision           = -2045, // Half precision (16-bits) not supported by the device
    CLBlastNoDoublePrecision         = -2044, // Double precision (64-bits) not supported by the device
    CLBlastInvalidVectorScalar       = -2043, // The unit-sized vector is not a valid OpenCL buffer
    CLBlastInsufficientMemoryScalar  = -2042, // The unit-sized vector's OpenCL buffer is too small
    CLBlastDatabaseError             = -2041, // Entry for the device was not found in the database
    CLBlastUnknownError              = -2040, // A catch-all error code representing an unspecified error
    CLBlastUnexpectedError           = -2039  // A catch-all error code representing an unexpected exception
  );

  TCLBlastLayout = ( CLBlastLayoutRowMajor = 101,
                   CLBlastLayoutColMajor = 102 );
  TCLBlastTranspose = ( CLBlastTransposeNo = 111, CLBlastTransposeYes = 112,
                      CLBlastTransposeConjugate = 113 );
  TCLBlastTriangle = ( CLBlastTriangleUpper = 121,
                     CLBlastTriangleLower = 122 );
  TCLBlastDiagonal = ( CLBlastDiagonalNonUnit = 131,
                     CLBlastDiagonalUnit = 132 );
  TCLBlastSide =  ( CLBlastSideLeft = 141, CLBlastSideRight = 142 );
  TCLBlastKernelMode = ( CLBlastKernelModeCrossCorrelation = 151, CLBlastKernelModeConvolution = 152 );

  // Precision enum (values in bits)
  TCLBlastPrecision = ( CLBlastPrecisionHalf = 16, CLBlastPrecisionSingle = 32,
                                   CLBlastPrecisionDouble = 64, CLBlastPrecisionComplexSingle = 3232,
                                   CLBlastPrecisionComplexDouble = 6464 );

  const CblasRowMajor = CLBlastLayoutRowMajor;
        CblasColMajor = CLBlastLayoutColMajor;
        CblasTrans    = CLBlastTransposeYes;
        CblasNoTrans    = CLBlastTransposeNo;

function CLBlastSgemm(const layout :TCLBlastLayout ; const a_transpose, b_transpose :TCLBlastTranspose ;
                                            const  m, n, k: IntPtr;
                                            const  alpha: single;
                                            const a_buffer :cl_mem; const a_offset, a_ld: IntPtr;
                                            const b_buffer :cl_mem; const b_offset, b_ld: IntPtr;
                                            const beta :single;
                                            c_buffer :cl_mem ; const c_offset, c_ld: IntPtr;
                                            queue :Pcl_command_queue ; event : Pcl_event):TCLBlastStatusCode ;external clblast;

{$endif}

procedure set_bit(const src :PByte; const index:IntPtr);
function get_bit(const src :PByte; const index:IntPtr):boolean;
procedure float_to_bit(const src:PSingle; const dst: PByte;const size:IntPtr);
procedure transpose_bin(A: PUInt32; B: PUInt32; const n: longint; const m: longint; const lda: longint; const ldb: longint; const block_size: longint);
procedure repack_input(const input, re_packed_input: Psingle; const w, h, c: longint);
procedure im2col_cpu_custom(data_im: Psingle; channels: longint; height: longint; width: longint; ksize: longint; stride: longint; pad: longint; data_col: Psingle);
procedure transpose_uint32(src, dst: Puint32; src_h: longint; src_w: longint; src_align: longint; dst_align: longint);
procedure gemm_nn_custom_bin_mean_transposed(const M, N, K: longint; const ALPHA_UNUSED: single; const A: PByte; const lda: longint; const B: PByte; const ldb: longint; const C: Psingle; const ldc: longint; const mean_arr: Psingle);
procedure im2col_cpu_custom_bin(const data_im: Psingle; const channels, height, width, ksize, stride, pad: longint; const data_col: Psingle; const bit_align: longint);
procedure forward_maxpool_layer_avx(const src,dst: Psingle; const indexes: Plongint; const size, w, h, out_w, out_h, c, pad, stride, batch: longint);

procedure gemm_bin(const M, N, K:PtrInt; const ALPHA:single; const A:PAnsiChar;
            const lda:PtrInt; const B:PSingle; const ldb:PtrInt; const C:PSingle; const ldc:PtrInt);

//procedure gemm(const TA, TB, M, N, K:PtrInt;
//            const ALPHA:single; const A:PSingle; const lda:PtrInt; const B:PSingle; const ldb:PtrInt;
//            const BETA:single; const C:PSingle; const ldc:PtrInt);

procedure sgemm_cpu( const TA, TB, M, N, K:PtrInt;
             const ALPHA:single;  const A:PSingle;  const lda:PtrInt;  const B:PSingle;  const ldb:PtrInt;
             const BETA:single;  const C:PSingle;  const ldc:PtrInt);

procedure sgemm(const TA, TB, M, N, K:PtrInt;
             const ALPHA:single; const A:PSingle; const lda:PtrInt; const B:PSingle; const ldb:PtrInt;
             const BETA:single; const C:PSingle; const ldc:PtrInt);

{$ifdef GEMM}
{$if defined(CPUX64) and defined(AVX2)}
procedure smulvs(const x: PSingle; const a:single; const INCX, count:PtrInt);
{$endif}
{$endif}
{$ifdef GPU}
procedure gemm_gpu(const TA, TB, M, N, K:longint;
            const ALPHA:single; const A_gpu:PSingle; const lda:longint; const B_gpu:PSingle; const ldb:longint;
            const BETA:single; const C_gpu:PSingle; const ldc:longint);
{$endif}

//procedure xpays( const x, y: PSingle;  const a: Single;  const count: PtrInt);

implementation
uses
 math;

const
  regs = 8 ;
  {$ifdef FPC}
  shft  = BsfQWord(regs);
  {$else}
  {$if regs = 8}shft = 3{$else} shft = 2{$endif};
  {$endif}
  off = regs * sizeof(single);


{$ifdef GEMM}


{$if defined(CPUX64) and defined(FPUAVX2)}
procedure smulvs(const x: PSingle; const a:single; const INCX, count:PtrInt);assembler;{$ifdef FPC}nostackframe; {$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
{$if regs = 4}
  vbroadcastss xmm1      , a
//  mov          r11       , count
//  shr          r11       , (shft + 2)    // div by 16 (4*4) = turns * regs
//  jz           @rem1
//
//@while:
//
//  vmulps       xmm0      , xmm1   ,  [x]
//  vmulps       xmm2      , xmm1   ,  [x+off]
//  vmulps       xmm3      , xmm1   ,  [x+off*2]
//  vmulps       xmm4      , xmm1   ,  [x+off*3]
//
//  vmovups      [x]       , xmm0
//  vmovups      [x+off]   , xmm2
//  vmovups      [x+off*2] , xmm3
//  vmovups      [x+off*3] , xmm4
//
//  add          x         , 4 * off   // turns * offset
//
//  dec          r11
//  jnz          @while
//
//@rem1:
  mov          r11       , count
  //and          r11       , (off-1)       // mod 16  ( turns * regs)
  shr          r11       , shft             // div regs
  jz           @rem

@while1:

  vmulps       xmm0      , xmm1   ,  [x]
  vmovups      [x]       , xmm0

  add          x         , off
  dec          r11
  jnz          @while1

@rem:
  mov          r11       , count
  and          r11       , (regs -1)       // mod regs
  jz           @done

@while2:
  vmulss       xmm0      , xmm1   ,  dword ptr [x]
  vmovss       [x]       , xmm0
  add          x         , 4

  dec          r11
  jnz          @while2

  {$elseif regs=8}
  vbroadcastss ymm1      , a
//  mov          r11       , count
//  shr          r11       , (shft + 2)    // div by 16 (4*4) = turns * regs
//  jz           @rem1
//
//@while:
//
//  vmulps       ymm0      , ymm1   ,  [x]
//  vmulps       ymm2      , ymm1   ,  [x+off]
//  vmulps       ymm3      , ymm1   ,  [x+off*2]
//  vmulps       ymm4      , ymm1   ,  [x+off*3]
//
//  vmovups      [x]       , ymm0
//  vmovups      [x+off]   , ymm2
//  vmovups      [x+off*2] , ymm3
//  vmovups      [x+off*3] , ymm4
//
//  add          x         , 4 * off   // turns * offset
//
//  dec          r11
//  jnz          @while
//
//@rem1:
  mov          r11       , count
  //and          r11       , (off-1)       // mod 16  ( turns * regs)
  shr          r11       , shft             // div regs
  jz           @rem

@while1:

  vmulps       ymm0      , ymm1   ,  [x]
  vmovups      [x]       , ymm0

  add          x         , off
  dec          r11
  jnz          @while1

@rem:
  mov          r11       , count
  and          r11       , (regs -1)       // mod regs
  jz           @done

@while2:
  vmulss       xmm0      , xmm1   ,  dword ptr [x]
  vmovss       [x]       , xmm0
  add          x         , 4

  dec          r11
  jnz          @while2
{$endif}

@done:


end;

{$else}
procedure smulvs(const x: PSingle; const a:single; const INCX, count:PtrInt);inline;local;
var i:PtrInt;
begin
  for i:=0 to count-1 do
      x[i+INCX]:=x[i+INCX]*a
end;

{$endif}


{$if defined(CPUX64) and defined(AVX2)}
procedure sxpay(const x, y: PSingle; const a: Single; const count: PtrInt);assembler;{$ifdef FPC}nostackframe;{$endif}
asm
  //push         r11
  //push         count
  {$ifndef FPC}
  .NOFRAME
  {$endif}
{$if regs = 4}
  vzeroupper
//  movss         xmm2   , a
  vbroadcastss xmm2   , a
  mov          r11    , count
  shr          r11    , (shft + 2)    // div by 16 (4*4) = turns * regs
  jz           @rem1

@while:
  vmovups      xmm0   , [x]
  vmovups      xmm1   , [x+off]
  vmovups      xmm3   , [x+off*2]
  vmovups      xmm4   , [x+off*3]

  vfmadd231ps  xmm0   , xmm2       , [y]      //xmm0
  vfmadd231ps  xmm1   , xmm2       , [y+off]  //xmm1
  vfmadd231ps  xmm3   , xmm2       , [y+off*2]//xmm8
  vfmadd231ps  xmm4   , xmm2       , [y+off*3]//xmm3

  vmovups      [x]        , xmm0
  vmovups      [x+off]    , xmm1
  vmovups      [x+off*2]  , xmm3
  vmovups      [x+off*3]  , xmm4

  add          x      , 4 * off   // turns * offset
  add          y      , 4 * off
  dec          r11
  jnz          @while

@rem1:
  mov          r11    , count
  and          r11    , (off-1)       // mod 16  ( turns * regs)
  shr          r11    , shft             // div regs
  jz           @rem

@while1:
  vmovups      xmm0   , [x]

  vfmadd231ps  xmm0   , xmm2       , [y]//xmm5
  vmovups      [x]    , xmm0
  add          x      , off
  add          y      , off
  dec          r11
  jnz          @while1

@rem:
  mov          r11    , count
  and          r11    , (regs -1)       // mod regs
  jz           @done

@while2:
  vmovss       xmm0   , [x]
  vfmadd231ss  xmm0   , xmm2, [y]
  vmovss       [x]    , xmm0
  add          x      , 4
  add          y      , 4
  dec          r11
  jnz          @while2
{$elseif regs=8}
//  movss         xmm2   , a
  vbroadcastss ymm2   , a
  mov          r11    , count
  shr          r11    , (shft + 2)    // div by 16 (4*4) = turns * regs
  jz           @rem1

@while:
  vmovups      ymm0   , [x]
  vmovups      ymm1   , yword [x+off]
  vmovups      ymm3   , yword [x+off*2]
  vmovups      ymm4   , yword [x+off*3]

  vfmadd231ps  ymm0   , ymm2       , [y]      //xmm0
  vfmadd231ps  ymm1   , ymm2       , yword [y+off]  //xmm1
  vfmadd231ps  ymm3   , ymm2       , yword [y+off*2]//xmm8
  vfmadd231ps  ymm4   , ymm2       , yword [y+off*3]//xmm3

  vmovups      yword [x]        , ymm0
  vmovups      yword [x+off]    , ymm1
  vmovups      yword [x+off*2]  , ymm3
  vmovups      yword [x+off*3]  , ymm4

  add          x      , 4 * off   // turns * offset
  add          y      , 4 * off
  dec          r11
  jnz          @while

@rem1:
  mov          r11    , count
  and          r11    , (off-1)       // mod 16  ( turns * regs)
  shr          r11    , shft             // div regs
  jz           @rem

@while1:
  vmovups      ymm0   , [x]

  vfmadd231ps  ymm0   , ymm2       , [y]//xmm5
  vmovups      [x]    , ymm0
  add          x      , off
  add          y      , off
  dec          r11
  jnz          @while1

@rem:
  mov          r11    , count
  and          r11    , (regs -1)       // mod regs
  jz           @done

@while2:
  vmovss       xmm0   , [x]
  vfmadd231ss  xmm0   , xmm2, [y]
  vmovss       [x]    , xmm0
  add          x      , 4
  add          y      , 4
  dec          r11
  jnz          @while2
{$endif}

@done:
  //pop          r11
  //vzeroupper
end;
{$else}
procedure sxpay(const x,y:PSingle; const a:single; const count:PtrInt);inline;local;
var i:PtrInt;
begin
  for i:=0 to count-1 do
      x[i]:=x[i] + a*y[i]
end;
{$endif}


const TILE_M = 4; // four operations
const TILE_N = 16 ;  // AVX 2 operations * 8 (8 singles);
const TILE_K = 16 ;  // loops

{$if defined(CPUX64) and defined(FPUAVX2)}
procedure nn_fast(const A, B, C:PSingle; const ALPHA:single; const lda, ldb, ldc, i, j, k:PtrInt);assembler;
asm
// save non-volatile registers to stack
  push               r12
  push               r13
  push               r14
  push               r15
{$ifdef MSWINDOWS}
  sub                  rsp      , 16*10                     // making stackspace to save xmm6-15
  vmovdqu              [rsp+$00], xmm6
  vmovdqu              [rsp+$10], xmm7
  vmovdqu              [rsp+$20], xmm8
  vmovdqu              [rsp+$30], xmm9
  vmovdqu              [rsp+$40], xmm10
  vmovdqu              [rsp+$50], xmm11
  vmovdqu              [rsp+$60], xmm12
  vmovdqu              [rsp+$70], xmm13
  vmovdqu              [rsp+$80], xmm14
  vmovdqu              [rsp+$90], xmm15
{$endif}

  mov                r11      , i
  imul               r11      , ldc
  add                r11      , j                         // (i*0)*ldc + j
  mov                r12      , r11
  add                r11      , ldc                       // (1+i)*ldc + j
  mov                r13      , r11
  add                r11      , ldc                       // (2+i)*ldc + j
  mov                r14      , r11
  add                r11      , ldc                       // (3+i)*ldc + j
  mov                r15      , r11

  vmovups            ymm8     , yword [C + 4 * r12]       // C[i*ldc + j]
  vmovups            ymm10    , yword [C + 4 * r12 + 32]  // C[i*ldc + j+8]

  vmovups            ymm9     , yword [C + 4 * r13]       // (1+i)*ldc + j
  vmovups            ymm11    , yword [C + 4 * r13 + 32]  // (1+i)*ldc + j+8

  vmovups            ymm12    , yword [C + 4 * r14]       // C[(2+i)*ldc + j]
  vmovups            ymm14    , yword [C + 4 * r14 + 32]  // C[(2+i)*ldc + j+8]

  vmovups            ymm13    , yword [C + 4 * r15]       // C[(3+i)*ldc + j]
  vmovups            ymm15    , yword [C + 4 * r15 + 32]  // C[(3+i)*ldc + j+8]

  mov                r11      , TILE_K

@while:

  mov                rax      , i
  imul               rax      , lda                           //  i * lda
  add                rax      , k                             //  i * lda + k

  vmulss             xmm0     , ALPHA  ,  dword [A + 4 * rax] // A[i * lda + k] * ALPHA
  vbroadcastss       ymm0     , xmm0

  add                rax      , lda                           //   (i+1)*lda + k
  vmulss             xmm1     , ALPHA  ,  dword [A + 4 * rax] // A[(i+1)*lda + k] * ALPHA
  vbroadcastss       ymm1     , xmm1

  add                rax      , lda                           //   (i+2)*lda + k
  vmulss             xmm2     , ALPHA  ,  dword [A + 4 * rax] // A[(i+2)*lda + k] * ALPHA
  vbroadcastss       ymm2     , xmm2

  add                rax      , lda                           //   (i+3)*lda + k
  vmulss             xmm4     , ALPHA  ,  dword [A + 4 * rax] // A[(i+3)*lda + k] * ALPHA
  vbroadcastss       ymm4     , xmm4

  mov                rax      , k
  imul               rax      , ldb                       // k * ldb
  add                rax      , j                         // k * ldb + j
  vmovups            ymm6     , yword [B + 4 * rax]       // B[k * ldb + j]
  vmovups            ymm7     , yword [B + 4 * rax + 32]  // B[k * ldb + j+8]

  vfmadd231ps        ymm8     , ymm0    , ymm6
  //vmulps             ymm5     , ymm0    , ymm6
  //vaddps             ymm8     , ymm8    , ymm5

  vfmadd231ps        ymm10    , ymm0    , ymm7
  //vmulps             ymm5     , ymm0    , ymm7
  //vaddps             ymm10    , ymm10   , ymm5

  vfmadd231ps        ymm9     , ymm1    , ymm6
  //vmulps             ymm5     , ymm1    , ymm6
  //vaddps             ymm9     , ymm9    , ymm5

  vfmadd231ps        ymm11    , ymm1    , ymm7
  //vmulps             ymm5     , ymm1    , ymm7
  //vaddps             ymm11    , ymm11   , ymm5

  vfmadd231ps        ymm12     , ymm2    , ymm6
  //vmulps             ymm5     , ymm2    , ymm6
  //vaddps             ymm12    , ymm12   , ymm5

  vfmadd231ps        ymm14     , ymm2    , ymm7
  //vmulps             ymm5     , ymm2    , ymm7
  //vaddps             ymm14    , ymm14   , ymm5


  vfmadd231ps        ymm13     , ymm4    , ymm6
  //vmulps             ymm5     , ymm4    , ymm6
  //vaddps             ymm13    , ymm13   , ymm5

  vfmadd231ps        ymm15     , ymm4    , ymm7
  //vmulps             ymm5     , ymm4    , ymm7
  //vaddps             ymm15    , ymm15   , ymm5

  add                A        , 4       // sizeof(single)

  mov                rax      , ldb
  imul               rax      , 4       // sizeof(single)
  add                B        , rax

  dec                r11
  jnz                @while

  vmovups            yword [C + 4 * r12]      , ymm8   // C[(0+i)*ldc + j]
  vmovups            yword [C + 4 * r12 + 32] , ymm10  // C[(0+i)*ldc + j+8]
  vmovups            yword [C + 4 * r13]      , ymm9   // C[(1+i)*ldc + j]
  vmovups            yword [C + 4 * r13 + 32] , ymm11  // C[(1+i)*ldc + j+8]
  vmovups            yword [C + 4 * r14]      , ymm12  // C[(2+i)*ldc + j]
  vmovups            yword [C + 4 * r14 + 32] , ymm14  // C[(2+i)*ldc + j+8]
  vmovups            yword [C + 4 * r15]      , ymm13  // C[(3+i)*ldc + j]
  vmovups            yword [C + 4 * r15 + 32] , ymm15  // C[(3+i)*ldc + j+8]

//restore non-volatile registers
{$ifdef MSWINDOWS}
  vmovdqu              xmm6   , [rsp+$00]
  vmovdqu              xmm7   , [rsp+$10]
  vmovdqu              xmm8   , [rsp+$20]
  vmovdqu              xmm9   , [rsp+$30]
  vmovdqu              xmm10  , [rsp+$40]
  vmovdqu              xmm11  , [rsp+$50]
  vmovdqu              xmm12  , [rsp+$60]
  vmovdqu              xmm13  , [rsp+$70]
  vmovdqu              xmm14  , [rsp+$80]
  vmovdqu              xmm15  , [rsp+$90]
  add                  rsp     , 16*10
{$endif}
  pop r15
  pop r14
  pop r13
  pop r12
end;

procedure nn_fastMP( idx:PtrInt; ptr:pointer);
var
  A_PART:Single;
  j
  , kk
  , i_d, k_d:PtrInt;
  p:PMPParams absolute ptr;
  A, B, C:PSingle;
  lda, ldb, ldc,
  RN
  ,RK, RM
  ,N
  ,K
   :PtrInt;
  ALPHA:single;
begin
  A     := p.A;
  B     := p.B;
  C     := p.C;
  lda   := PPtrInt(p.d)^;
  ldb   := PPtrInt(p.e)^;
  ldc   := PPtrInt(p.f)^;
  ALPHA := PSingle(p.g)^;
  RK    := PPtrInt(p.h)^;
  RN    := PPtrInt(p.i)^;
  N     := PPtrInt(p.j)^;
  K     := PPtrInt(p.k)^;

  kk    :=0;
  while kk < RK do begin
      j:=0;
      while j < RN do begin
          nn_fast(A, B, C, ALPHA, lda, ldb, ldc,idx, j, kk);
          inc(j, TILE_N)
      end;

      for i_d:=idx to idx+TILE_M -1 do
          for k_d:=kk to kk + TILE_K-1 do begin
              A_PART := ALPHA*A[i_d*lda + k_d];
              sxpay(@C[i_d*ldc + RN], @B[k_d*ldb + RN], A_PART, N-RN);
//              for j:= (N div TILE_N)*TILE_N to N-1 do
//                  C[i_d*ldc + j] := C[i_d*ldc + j] + A_PART*B[k_d*ldb + j];
          end;
      inc(kk, TILE_K)
  end;

  for kk := RK to K-1 do
      for i_d:=idx to idx+TILE_M -1 do begin
          A_PART:= ALPHA*A[i_d*lda + kk];
          sxpay(@C[i_d*ldc], @B[kk*ldb], A_PART, N);
//          for j:=0 to N-1 do
//              C[i_d*ldc + j] := C[i_d*ldc + j] + A_PART*B[kk*ldb + j]
      end;
end;

procedure gemm_nn_fast(const M, N, K:PtrInt; const ALPHA:single;
            const A: PSingle; const lda:PtrInt;
            const B: PSingle; const ldb:PtrInt;
            const C: PSingle; const ldc:PtrInt);inline;
var
  i, kk, RN, RM, RK:PtrInt;
  A_PART:Single;
  j, i_d, k_d: PtrInt;
  P :TMPParams;
begin
  RK := (K div TILE_K)*TILE_K;
  RM := (M div TILE_M)*TILE_M;
  RN := (N div TILE_N)*TILE_N;

  p.A :=  A     ;
  p.B :=  B     ;
  p.C :=  C     ;
  p.d :=  @lda   ;
  p.e :=  @ldb   ;
  p.f :=  @ldc   ;
  p.g :=  @ALPHA ;
  p.h :=  @RK    ;
  p.i :=  @RN    ;
  p.j :=  @N     ;
  p.k :=  @K     ;

//  i := 0;
//  while i < RM do begin
//      kk:=0;
//      while kk < RK do begin
//          j:=0;
//          while j < RN do begin
//              nn_fast(A, B, C, ALPHA, lda, ldb, ldc,i, j, kk);
//              inc(j, TILE_N)
//          end;
//
//          for i_d:=i to i+TILE_M -1 do
//              for k_d:=kk to kk + TILE_K-1 do begin
//                  A_PART := ALPHA*A[i_d*lda + k_d];
//                  sxpay(@C[i_d*ldc + RN], @B[k_d*ldb + RN], A_PART, N-RN);
//                  //for j:= (N div TILE_N)*TILE_N to N-1 do
//                  //    C[i_d*ldc + j] := C[i_d*ldc + j] + A_PART*B[k_d*ldb + j];
//              end;
//          inc(kk, TILE_K)
//      end;
//
//      for kk := RK to K-1 do
//          for i_d:=i to i+TILE_M -1 do begin
//              A_PART:= ALPHA*A[i_d*lda + kk];
//              sxpay(@C[i_d*ldc], @B[kk*ldb], A_PART, N);
////              for j:=0 to N-1 do
////                  C[i_d*ldc + j] := C[i_d*ldc + j] + A_PART*B[kk*ldb + j]
//          end;
//      inc(i,TILE_M)
//  end;

  MP.&for(nn_fastMP, 0, RM, @p, TILE_M);
  //i:=0;
  //while i< RM do begin
  //   nn_fastMP(i, @p);
  //   inc(i, TILE_M)
  //end;

  for i := RM to M-1 do
      for kk := 0 to K-1 do begin
          A_PART := ALPHA*A[i*lda + kk];
          sxpay(@C[i*ldc], @B[kk*ldb], A_PART, N);
          //for j := 0 to N-1 do
          //    C[i*ldc + j] := C[i*ldc + j] + A_PART*B[kk*ldb + j];
      end
end;
{$endif}


procedure nn(const f,t:PtrInt;  const ptr:pointer);
var
  i,j,kk, K, N, lda, ldb, ldc: PtrInt;
  A_PART, ALPHA: single;
  p:PMPParams absolute ptr;
  A,B,C,CC :PSingle;
begin
    ALPHA:=PSingle(p.D)^;
    A:=p.A;
    B:=p.B;
    C:=p.C;
    lda:=PPtrInt(p.E)^;
    ldb:=PPtrInt(p.F)^;
    ldc:=PPtrInt(p.G)^;
    K  :=PPtrInt(p.K)^;
    N  :=PPtrInt(p.N)^;
    for i := f to t do
      for kk := 0 to K -1 do begin
          A_PART := ALPHA * A[i * lda + kk];
          sxpay(@C[i*ldc], @B[kk*ldb], A_PART, N);
          //for j:=0 to N-1 do
          //    C[i*ldc + j]:=C[i*ldc + j] + A_PART*B[kk*ldb + j]
      end;
end;

procedure gemm_nn(const M, N, K: PtrInt; const ALPHA: single; const A: PSingle; const lda: PtrInt; const B: PSingle; const ldb: PtrInt; const C: PSingle; const ldc: PtrInt);inline;
var p:TMPParams;
begin
  p.D:=@ALPHA;
  p.A:=A;
  P.B:=B;
  P.C:=C;
  p.E:=@lda;
  P.F:=@ldb;
  p.G:=@ldc;
  p.K:=@K;
  p.N:=@N;
  {$if defined(USE_MULTITHREADING)}
  mp.&for(nn,0,M-1,@p) ;
  {$else}
  nn(0, M-1, @p)
  {$endif}
end;

const ymmd: array [0..7] of single =(0,1,2,3,4,5,6,7);

{$if defined(CPUX64) and defined(AVX2)}
function dot1(const A,B:PSingle; const N:PtrInt):single;assembler;{$ifdef FPC}nostackframe;{$endif}
asm
  {$ifndef FPC}
  .NOFRAME
  {$endif}
{$if regs = 4}

{$elseif regs=8}

   mov              r11     ,     N
   shr              r11     ,    shft
   pxor             xmm0    ,    xmm0
   jz               @rem
   vpxor            ymm0    ,    ymm0   ,   ymm0
@while:
   vmovups          ymm1    ,    [A]
   vfmadd231ps      ymm0    ,    ymm1   ,   [B]
   add              A       ,    off
   add              B       ,    off
   dec              r11
   jnz              @while
@rem:

   mov              r8      ,    N
   vmovd            xmm3    ,    r8d
   vpxor            ymm1    ,    ymm1    , ymm1
   vpxor            ymm2    ,    ymm2    , ymm2
   vpbroadcastd     ymm3    ,    xmm3
   vpcmpgtd         ymm3    ,    ymm3    , [rip+ymmd]
   vmaskmovps       ymm1    ,    ymm3    , [a]
   vmaskmovps       ymm2    ,    ymm3    , [b]
   vfmadd231ps      ymm0    ,    ymm1    , ymm2



   vextractf128     xmm1    ,    ymm0   ,   $1
   vaddps           xmm0    ,    xmm0   ,   xmm1
   vhaddps          xmm0    ,    xmm0   ,   xmm0
   vhaddps          xmm0    ,    xmm0   ,   xmm0
{$endif}
@done:

end;
{$else}
function dot1(const A,B:PSingle; const N:PtrInt):single;local;
var i :PtrInt;
begin
  result :=0;
  for i:=0 to N-1 do
    result := result + a[i]*b[i]
end;

{$endif}

function dot2(const A,B:PSingle; const N:PtrInt):single;
var i:PtrInt;
begin
  result:=0;
  for i:=0 to N-1 do
     result := result + a[i]*b[i]
end;


procedure nt(const f,t:PtrInt;const params:Pointer);
var
    i, j, kk, K,N,lda, ldb, ldc: PtrInt;
    A, B, C :PSingle;
    ALPHA, sum: single;
    p :PMPParams absolute params;
begin
    ALPHA:=PSingle(p.D)^;
    A:=p.A;
    B:=p.B;
    C:=p.C;
    lda:=PPtrInt(p.E)^;
    ldb:=PPtrInt(p.F)^;
    ldc:=PPtrInt(p.G)^;
    K  :=PPtrInt(p.K)^;
    N  :=PPtrInt(p.N)^;
    
    for i := f to t do
        for j := 0 to N -1 do
            begin    // todo optimize nt
                //sum := 0;
                //for kk := 0 to K -1 do
                //    sum := sum + ALPHA * A[i * lda+kk] * B[j * ldb+kk];
                sum := ALPHA * dot1(@A[i * lda], @B[j * ldb], N);
                C[i * ldc+j] := C[i * ldc+j] + sum
            end
end;

procedure gemm_nt(const M, N, K: PtrInt; const ALPHA: single; const A: PSingle; const lda: PtrInt; const B: PSingle; const ldb: PtrInt; const C: PSingle; const ldc: PtrInt);local;
var p:TMPParams;
begin
  p.D:=@ALPHA;
  p.A:=A;
  P.B:=B;
  P.C:=C;
  p.E:=@lda;
  P.F:=@ldb;
  p.G:=@ldc;
  p.K:=@K;
  p.N:=@N;
  {$if defined(USE_MULTITHREADING)}
  mp.&For(nt, 0, M-1,@p);
  {$else}
  nt(0, M-1, @p)
  {$endif}
end;

procedure tn(const f,t:PtrInt; const params:Pointer);
var
    i, j, kk, K, N, lda, ldb, ldc: longint;
    A_PART, ALPHA: single;
    A, B, C :PSingle;
    p:PMPParams absolute params;
begin
    ALPHA:=PSingle(p.D)^;
    A:=p.A;
    B:=p.B;
    C:=p.C;
    lda:=PPtrInt(p.E)^;
    ldb:=PPtrInt(p.F)^;
    ldc:=PPtrInt(p.G)^;
    K  :=PPtrInt(p.K)^;
    N  :=PPtrInt(p.N)^;

  for i := f to t do
    for kk := 0 to K -1 do
      begin        // optimize tn
          A_PART := ALPHA * A[kk * lda+i];
          sxpay(@C[i*ldc], @B[kk*ldb], A_PART, N);
          //for j := 0 to N -1 do
          //    C[i * ldc+j] := C[i * ldc+j] + A_PART * B[kk * ldb+j]
      end
end;

procedure gemm_tn(const M, N, K: PtrInt; const ALPHA: single; const A: PSingle; const lda: PtrInt; const B: PSingle; const ldb: PtrInt; const C: PSingle; const ldc: PtrInt);local;
var p:TMPParams;
begin
  p.D:=@ALPHA;
  p.A:=A;
  P.B:=B;
  P.C:=C;
  p.E:=@lda;
  P.F:=@ldb;
  p.G:=@ldc;
  p.K:=@K;
  p.N:=@N;
  {$if defined(USE_MULTITHREADING)}
  mp.&For(tn, 0, M-1, @p);
  {$else}
  tn(0, M-1, @p);
  {$endif}
end;


procedure tt(const f,t:PtrInt;const params:Pointer);
var
    i, j, kk, K, N, lda, ldb, ldc: longint;
    sum, ALPHA: single;
    A,B,C :PSingle;
    p : PMPParams absolute params;
begin
    ALPHA:=PSingle(p.D)^;
    A:=p.A;
    B:=p.B;
    C:=p.C;
    lda:=PPtrInt(p.E)^;
    ldb:=PPtrInt(p.F)^;
    ldc:=PPtrInt(p.G)^;
    K  :=PPtrInt(p.K)^;
    N  :=PPtrInt(p.N)^;

    for i := f to t do
        for j := 0 to N -1 do
            begin           // todo optimize tt
                sum := 0;
                for kk := 0 to K -1 do
                    sum := sum + ALPHA * A[i+kk * lda] * B[kk+j * ldb];
                C[i * ldc+j] := C[i * ldc+j] + sum
            end
end;

procedure gemm_tt(const M, N, K: PtrInt; const ALPHA: single; const A: PSingle; const lda: PtrInt; const B: PSingle; const ldb: PtrInt; const C: PSingle; const ldc: PtrInt);local;

var p:TMPParams;
begin
  p.D:=@ALPHA;
  p.A:=A;
  P.B:=B;
  P.C:=C;
  p.E:=@lda;
  P.F:=@ldb;
  p.G:=@ldc;
  p.K:=@K;
  p.N:=@N;
  {$if defined(USE_MULTITHREADING)}
  mp.&For(tt, 0, M-1,@p);
  {$else}
  tt(0, M-2, @p);
  {$endif}
end;

{$endif}

procedure sgemm_cpu(const TA, TB, M, N, K: PtrInt; const ALPHA: single;
  const A: PSingle; const lda: PtrInt; const B: PSingle; const ldb: PtrInt;
  const BETA: single; const C: PSingle; const ldc: PtrInt);
var
    i, j: longint;
begin
    //todo TParallel.for and SIMDfy
{$if defined(OPENBLAS) or defined(MKL) or defined(CLBLAS)}
    if not boolean(TA) and not boolean(TB) then
        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasNoTrans,
            M, N, K, ALPHA, A, lda, B, ldb, BETA, C,ldc)
    else
        if boolean(TA) and not boolean(TB) then
            cblas_sgemm(
                CblasRowMajor,
                CblasTrans, CblasNoTrans,
                M, N, K, ALPHA, A, lda, B, ldb, BETA, C,ldc)
    else
        if not boolean(TA) and boolean(TB) then
            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K, ALPHA, A, lda, B, ldb, BETA, C,ldc)
    else
        cblas_sgemm(
            CblasRowMajor,
            CblasTrans, CblasTrans,
            M, N, K, ALPHA, A, lda, B, ldb, BETA, C,ldc)
{$else}
    if BETA<>1 then
      if ldc =1 then
        smulvs(C, BETA, ldc ,M*N)
      else
          for i := 0 to M -1 do
            smulvs(@C[i*ldc], BETA, ldc ,N);
            //for j := 0 to N -1 do
            //  C[i * ldc+j] := C[i * ldc+j] * BETA;
    if not boolean(TA) and not boolean(TB) then
        {$if defined(FPUAVX2)}
        gemm_nn_fast(M, N, K, ALPHA, A, lda, B, ldb, C, ldc)
        {$else}
        gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc)
        {$endif}
    else  if boolean(TA) and not boolean(TB) then
        gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc)
    else if not boolean(TA) and boolean(TB) then
        gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc)
    else
        gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc)
{$endif}
end;


procedure bin(const f,t:PtrInt; const params:Pointer=nil);
var
    i, j, kk, K, N , lda, ldb, ldc: longint;
    A:PAnsiChar;
    B, C:PSingle;
    A_PART: ansichar;
    p:PMPParams absolute params;
begin
    A:=p.A;
    B:=p.B;
    C:=p.C;
    lda:=PPtrInt(p.E)^;
    ldb:=PPtrInt(p.F)^;
    ldc:=PPtrInt(p.G)^;
    K  :=PPtrInt(p.K)^;
    N  :=PPtrInt(p.N)^;

    for i := f to t do
        for kk := 0 to K -1 do
            begin        // todo optimize gemm bin
                A_PART := A[i * lda+kk];
                if boolean(A_PART) then
                    for j := 0 to N -1 do
                        C[i * ldc+j] := C[i * ldc+j] + B[kk * ldb+j]
                else
                    for j := 0 to N -1 do
                        C[i * ldc+j] := C[i * ldc+j] - B[kk * ldb+j]
            end
end;

procedure gemm_bin(const M, N, K: PtrInt; const ALPHA: single; const A: PAnsiChar;
  const lda: PtrInt; const B: PSingle; const ldb: PtrInt; const C: PSingle;
  const ldc: PtrInt);
var p:TMPParams;
begin
  p.A:=A;
  P.B:=B;
  P.C:=C;
  p.E:=@lda;
  P.F:=@ldb;
  p.G:=@ldc;
  p.K:=@K;
  p.N:=@N;
  {$if defined(USE_MULTITHREADING)}
  mp.&For(bin, 0, M-1,@p);
  {$else}
  bin( 0, M-1,@p);
  {$endif}
end;


  procedure maxpoolMP(const f,t:PtrInt; const p:pointer=nil);
  var
    k, i, j, m, n, out_index, max_i, cur_h, cur_w, index
    , out_w, out_h, c, b, size, stride, w_offset, h_offset, w, h :longint;
    src,dst:PSingle;
    indexes:PLongint;
    max, val: single;
    a:PMPParams absolute p;
  begin
    out_w     := PLongint(a.A)^;
    out_h     := PLongint(a.B)^;
    c         := PLongint(a.C)^;
    b         := PLongint(a.D)^;
    size      := PLongint(a.E)^;
    stride    := PLongint(a.F)^;
    w_offset  := PLongint(a.G)^;
    h_offset  := PLongint(a.H)^;
    w         := PLongint(a.I)^;
    h         := PLongint(a.J)^;
    src       := a.K;
    dst       :=a.L;
    indexes  := a.M;

    for k := f to t do
        begin
            for i := 0 to out_h -1 do
                for j := 0 to out_w -1 do
                    begin
                        out_index := j+out_w * (i+out_h * (k+c * b));
                        max := -MaxSingle;
                        max_i := -1;
                        for n := 0 to size -1 do
                            for m := 0 to size -1 do
                                begin
                                    cur_h := h_offset+i * stride+n;
                                    cur_w := w_offset+j * stride+m;
                                    index := cur_w + w * (cur_h + h * (k+b * c));
                                    if (cur_h >= 0) and (cur_h < h) and (cur_w >= 0) and (cur_w < w) then
                                        val := src[index]
                                    else
                                        val := -MaxSingle;
                                    if (val > max) then
                                        max_i := index
                                    else
                                        max_i := max_i;
                                    if (val > max) then
                                        max := val
                                    else
                                        max := max
                                end;
                        dst[out_index] := max;
                        if assigned(indexes) then
                            indexes[out_index] := max_i
                    end
        end
  end;

procedure forward_maxpool_layer_avx(const src,dst: Psingle; const indexes: Plongint; const size, w, h, out_w, out_h, c, pad, stride, batch: longint);
var
  b, w_offset, h_offset:longint;
  a:TMPParams;
begin
    w_offset := -pad div 2;
    h_offset := -pad div 2;
    a.A := @out_w      ;
    a.B := @out_h      ;
    a.C := @c          ;
    a.D := @b          ;
    a.E := @size       ;
    a.F := @stride     ;
    a.G := @w_offset   ;
    a.H := @h_offset   ;
    a.I := @w          ;
    a.J := @h          ;
    a.K := src         ;
    a.L := dst         ;
    a.M := indexes     ;
    for b := 0 to batch -1 do begin
    {$if defined(USE_MULTITHREADING)}
        mp.&for(maxPoolMP, 0, c-1,@a)
    {$else}
        maxPoolMP(0, C-1, @a)
    {$endif}
    end;
end;

procedure sgemm(const TA, TB, M, N, K:PtrInt;
            const ALPHA:single; const A:PSingle; const lda:PtrInt; const B:PSingle; const ldb:PtrInt;
            const BETA:single; const C:PSingle; const ldc:PtrInt);
begin
//  writeln('sgemm');
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.start(opGemm);{$endif}
  sgemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
  {$ifdef USE_TELEMETRY} if benchmark then metrics.ops.finish(opGemm);{$endif}
end;

procedure set_bit(const src: PByte; const index: IntPtr);
var
    p:PByte;
begin
  p:=@src[index div 8];
  p[0]:=P[0] or (1 shl (index mod 8))
end;

function get_bit(const src: PByte; const index: IntPtr): boolean;
var
  p: PByte;
begin
  p:=@src[index div 8];
  result:=P^ and (1 shl (index mod 8))>0
end;

procedure float_to_bit(const src:PSingle; const dst: PByte;const size:IntPtr);
var
  dst_size, i:IntPtr;
  dst_tmp : byte;
  byte_arr:TArray<byte>;
begin
    dst_size := size div 8 + 1;
    fillchar(dst[0], dst_size,0);

    setLength(byte_arr, size);
    for i := 0 to size-1 do
        if src[i] > 0 then byte_arr[i] := 1;

    //for (i = 0; i < size; ++i) {
    //    dst[i / 8] |= byte_arr[i] << (i % 8);
    //}

    for i := 0 to size-1 do begin
        dst_tmp := 0;
        dst_tmp := dst_tmp + byte_arr[i + 0] shl 0;
        dst_tmp := dst_tmp + byte_arr[i + 1] shl 1;
        dst_tmp := dst_tmp + byte_arr[i + 2] shl 2;
        dst_tmp := dst_tmp + byte_arr[i + 3] shl 3;
        dst_tmp := dst_tmp + byte_arr[i + 4] shl 4;
        dst_tmp := dst_tmp + byte_arr[i + 5] shl 5;
        dst_tmp := dst_tmp + byte_arr[i + 6] shl 6;
        dst_tmp := dst_tmp + byte_arr[i + 7] shl 7;
        dst[i div 8] := dst_tmp;
    end;
    //free(byte_arr);
end;

function reverse_8_bit(const a: uint8):uint8;inline;
begin
    exit(((a * $0802 and $22110) or (a * $8020 and $88440)) * $10101 shr 16)
end;

function reverse_32_bit(const a: uint32):uint32;inline;
begin
    exit(
         (reverse_8_bit(a shr 24) shl 0) or
         (reverse_8_bit(a shr 16) shl 8) or
         (reverse_8_bit(a shr 8) shl 16) or
         (reverse_8_bit(a shr 0) shl 24)
        )
end;

type TLongWordx32 = array[0..31] of uint32;
procedure transpose32_optimized(A: TLongWordx32);inline;
var
    j, k: longint;
    m, t: uint32;
    tmp: uint32;
begin
    j := 16;
    m := $0000FFFF;
    k := 0;
    while k < 32 do begin
          t := (A[k] xor (A[k+j] shr j)) and m;
          A[k] := A[k] xor t;
          A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 8;
    m := $00ff00ff;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 4;
    m := $0f0f0f0f;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 2;
    m := $33333333;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    j := 1;
    m := $55555555;
    k := 0;
    while k < 32 do begin
        t := (A[k] xor (A[k+j] shr j)) and m;
        A[k] := A[k] xor t;
        A[k+j] := A[k+j] xor (t shl j);
        k := (k+j+1) and not j
    end;
    for j := 0 to 16 -1 do
        begin
            tmp := A[j];
            A[j] := reverse_32_bit(A[31-j]);
            A[31-j] := reverse_32_bit(tmp)
        end
end;

procedure transpose_32x32_bits_reversed_diagonale(const A: Puint32; const B: Puint32; const m, n: longint); inline;
var
    //A_tmp: array[0..31] of uint32;
    A_tmp : TLongWordx32;
    i: longint;
begin
    // todo unroll optimiztion
    for i := 0 to 32 -1 do
        A_tmp[i] := A[i * m];
    transpose32_optimized(A_tmp);
    //todo unroll optimization
    for i := 0 to 32 -1 do
        B[i * n] := A_tmp[i]
end;

procedure transpose_bin(A: PUInt32; B: PUInt32; const n: longint; const m: longint; const lda: longint; const ldb: longint; const block_size: longint);
var
    i, j, a_index, b_index: longint;
begin
    i := 0;
    // todo SIMDIfy
    while i < n do begin
        j := 0;
        while j < m do begin
            a_index := i * lda+j;
            b_index := j * ldb+i;
            transpose_32x32_bits_reversed_diagonale( @A[a_index div 32],  @B[b_index div 32], lda div 32, ldb div 32);
            j := j + 32
        end;
        while j < m do begin
            if get_bit(PByte(A), i * lda+j) then
                set_bit(PByte(B), j * ldb+i);
            inc(j)
        end;
        i := i + 32
    end
end;

procedure repack_input(const input, re_packed_input: Psingle; const w, h, c: longint);
var
    items_per_channel, chan, i, c_pack: longint;
    src: single;
begin
    items_per_channel := w * h;
    chan := 0;
    while chan < c do begin
        for i := 0 to items_per_channel -1 do
            begin
                for c_pack := 0 to 32 -1 do
                    begin
                        src := input[(chan+c_pack) * items_per_channel+i];
                        re_packed_input[chan * items_per_channel+i * 32+c_pack] := src
                    end
            end;
        chan := chan + 32
    end
end;


procedure im2col_cpu_custom_bin(const data_im: Psingle; const channels, height,
  width, ksize, stride, pad: longint; const data_col: Psingle;
  const bit_align: longint);
var
    c, height_col, width_col, channels_col, new_ldb, h, w, w_offset, h_offset, c_im, im_row, im_col, col_index: longint;
    val: single;
begin
    height_col := (height+2 * pad-ksize) div stride+1;
    width_col := (width+2 * pad-ksize) div stride+1;
    channels_col := channels * ksize * ksize;
    if (height_col = height) and (width_col = width) and (stride = 1) and (pad = 1) then
        begin
            new_ldb := bit_align;
            for c := 0 to channels_col -1 do
                begin
                    w_offset := c mod ksize;
                    h_offset := (c div ksize) mod ksize;
                    c_im := c div ksize div ksize;
                    for h := pad to height_col-pad -1 do
                        begin
                            w := pad;
                            while w < width_col-pad-8 do begin
                                im_row := h_offset+h-pad;
                                im_col := w_offset+w-pad;
                                col_index := c * new_ldb+h * width_col+w;
                                val := data_im[im_col+width * (im_row+height * c_im)];
                                if val > 0 then
                                    set_bit(PByte(data_col), col_index);
                                w := w + 1
                            end;
                            while w < width_col-pad do begin
                                im_row := h_offset+h-pad;
                                im_col := w_offset+w-pad;
                                col_index := c * new_ldb+h * width_col+w;
                                val := data_im[im_col+width * (im_row+height * c_im)];
                                if val > 0 then
                                    set_bit(PByte(data_col), col_index);
                                inc(w)
                            end
                        end;
                    w := 0;
                    for h := 0 to height_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    w := width_col-1;
                    for h := 0 to height_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    h := 0;
                    for w := 0 to width_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end;
                    h := height_col-1;
                    for w := 0 to width_col -1 do
                        begin
                            im_row := h_offset+h;
                            im_col := w_offset+w;
                            col_index := c * new_ldb+h * width_col+w;
                            val := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad);
                            if val > 0 then
                                set_bit(PByte(data_col), col_index)
                        end
                end
        end
    else
        writeln(#10' Error: is no non-optimized version '#10)
end;

procedure im2col_cpu_custom(data_im: Psingle; channels: longint; height: longint; width: longint; ksize: longint; stride: longint; pad: longint; data_col: Psingle);
var
    c,height_col,width_col,channels_col, h, w, w_offset, h_offset, c_im, im_row, im_col, col_index: longint;
begin
    im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col);
    exit();
    height_col := (height+2 * pad-ksize) div stride+1;
    width_col := (width+2 * pad-ksize) div stride+1;
    channels_col := channels * ksize * ksize;
    if (height_col = height) and (width_col = width) and (stride = 1) and (pad = 1) then
        for c := 0 to channels_col -1 do
            begin
                w_offset := c mod ksize;
                h_offset := (c div ksize) mod ksize;
                c_im := c div ksize div ksize;
                for h := pad to height_col-pad -1 do
                    begin
                        for w := pad to width_col-pad -1 do
                            begin
                                im_row := h_offset+h-pad;
                                im_col := w_offset+w-pad;
                                col_index := (c * height_col+h) * width_col+w;
                                data_col[col_index] := data_im[im_col+width * (im_row+height * c_im)]
                            end;
                        while w < width_col-pad do begin
                            im_row := h_offset+h-pad;
                            im_col := w_offset+w-pad;
                            col_index := (c * height_col+h) * width_col+w;
                            data_col[col_index] := data_im[im_col+width * (im_row+height * c_im)];
                            inc(w)
                        end
                    end;
                w := 0;
                for h := 0 to height_col -1 do
                    begin
                        im_row := h_offset+h;
                        im_col := w_offset+w;
                        col_index := (c * height_col+h) * width_col+w;
                        data_col[col_index] := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad)
                    end;
                w := width_col-1;
                for h := 0 to height_col -1 do
                    begin
                        im_row := h_offset+h;
                        im_col := w_offset+w;
                        col_index := (c * height_col+h) * width_col+w;
                        data_col[col_index] := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad)
                    end;
                h := 0;
                for w := 0 to width_col -1 do
                    begin
                        im_row := h_offset+h;
                        im_col := w_offset+w;
                        col_index := (c * height_col+h) * width_col+w;
                        data_col[col_index] := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad)
                    end;
                h := height_col-1;
                for w := 0 to width_col -1 do
                    begin
                        im_row := h_offset+h;
                        im_col := w_offset+w;
                        col_index := (c * height_col+h) * width_col+w;
                        data_col[col_index] := im2col_get_pixel(data_im, height, width, channels, im_row, im_col, c_im, pad)
                    end
            end
    else
        im2col_cpu(data_im, channels, height, width, ksize, stride, pad, data_col)
end;

procedure transpose_uint32(src, dst: Puint32; src_h: longint; src_w: longint; src_align: longint; dst_align: longint);
var
    i,j: longint;
begin
    i := 0;
    // todo simdfy
    while i < src_h do begin
        j := 0;
        while j < src_w do begin
            dst[j * dst_align div 32+i] := src[i * src_align+j];
            j := j + 1
        end;
        inc(i)
    end
end;

function xnor_int64(const a,b:uint64):uint64;inline;
begin
    result := not(a xor b)
end;

{$ifndef FPC}
function popcnt(v:uint32):uint32;inline;
begin
  v := v - ((v shr 1) and $55555555);
  v := (v and $33333333) + ((v shr 2) and $33333333);
  result := ((v + (v shr 4) and $F0F0F0F) * $1010101) shr 24;
end;
{$endif}


procedure gemm_nn_custom_bin_mean_transposed(const M, N, K: longint; const ALPHA_UNUSED: single; const A: PByte; const lda: longint; const B: PByte; const ldb: longint; const C: Psingle; const ldc: longint; const mean_arr: Psingle);
var
    i, j, kk, count, tmp_count: longint;
    mean_val: single;
    a_bit64, b_bit64, c_bit64: uint64;
begin
    for i := 0 to M -1 do
        begin
            mean_val := mean_arr[i];
            for j := 0 to N -1 do
                begin
                    count := 0;
                    kk := 0;
                    while kk < K do begin
                        a_bit64 :=  PUint64((A+(i * lda+kk) div 8))^;
                        b_bit64 :=  PUint64((B+(j * ldb+kk) div 8))^;
                        c_bit64 := xnor_int64(a_bit64, b_bit64);
                        tmp_count := POPCNT(c_bit64);
                        if K-kk < 64 then
                            tmp_count := tmp_count-(64-(K-kk));
                        count := count + tmp_count;
                        kk := kk + 64
                    end;
                    C[i * ldc+j] := (2 * count-K) * mean_val
                end
        end
end;



{$ifdef GPU}
procedure gemm_gpu(const TA, TB, M, N, K:longint;
            const ALPHA:single; const A_gpu:PSingle; const lda:longint; const B_gpu:PSingle; const ldb:longint;
            const BETA:single; const C_gpu:PSingle; const ldc:longint);
var
  handle : cublasHandle_t;
  startus : cudaError_t;
begin
  handle := blas_handle();
  status := cublasSgemm(handle, ifthen(TB<>0 , CUBLAS_OP_T , CUBLAS_OP_N),
          ifthen(TA<>0 , CUBLAS_OP_T , CUBLAS_OP_N), N, M, K, @ALPHA, B_gpu, ldb, A_gpu, lda, @BETA, C_gpu, ldc);
  check_error(status);

end;
{$endif}

{$ifdef OPENBLAS}
//const
  //OMP_STR: array[0..2] of string = ('Sequential','MultiThreading','OpenMP');
  //i:longint=0;
{$endif}

//
//var a, b:TArray<single>;
//    i: integer;
//    c1,c2:single;
initialization

//   setLength(a, 1000007);
//   setLength(b, length(a));
//   for i:=0 to high (a) do begin
//     a[i]:=10*random();
//     b[i]:=10*random()
//
//   end;
//
//   c1:=dot1(Psingle(a), PSingle(b), length(a));
//   c2:=dot2(Psingle(a), PSingle(b), length(a));
//
//   writeln(' dot 1', c1);
//   writeln(' dot 2', c2);
//   writeln('is same value? :', SameValue(c1,c2));


{$ifdef OPENBLAS}
   //if IsConsole then begin
   //    i := openblas_get_parallel;
   //    writeln('OpenBLAS [',OMP_STR[i],']', #13#10,'Core Name: ',openblas_get_corename,#13#10,'Config: ', openblas_get_config,#13#10,'Threads: ',openblas_get_num_threads);
   //end;
{$endif}
finalization


end.

