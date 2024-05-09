unit openblas;
{$Z4}
interface

uses types;
{.$define static}
{$ifdef static}
{$linklib openblas}
{$else}
  {$if defined(MSWINDOWS)}
    const libopenblas = 'libopenblas.dll' ;
  {$elseif defined(DARWIN)}
    {$define static}
    {$LinkFramework Accelerate}
    //const libopenblas='libopenblas.dylib';
  {$else}
    const libopenblas='openblas.so.0' ;
  {$endif}
{$endif}

  Type
  Pbfloat16  = ^bfloat16;
  bfloat16   = word;
  size_t = NativeUInt;
  {$ifdef LINUX}
  Pcpu_set_t  = ^cpu_set_t;
  cpu_set_t = longint;
  {$endif}

{$if defined(WINDOWS) and defined(CPUX86_64)}
  BLASLONG = int64;
  BLASULONG = qword;
{$else}
  BLASLONG = int32;
  BLASULONG = dword;
{$endif}

{$ifdef OPENBLAS_USE64BITINT}
  blasint = BLASLONG;
{$else}
  blasint = integer;
{$endif}


{$IFDEF FPC}
{$PACKRECORDS C}
{$ENDIF}

  type
    openblas_complex_float = record
        real : single;
        imag : single;
      end;

    openblas_complex_double = record
        real : double;
        imag : double;
      end;

  {Set the number of threads on runtime. }

  procedure openblas_set_num_threads(num_threads:longint); winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_set_num_threads';

  procedure goto_set_num_threads(num_threads:longint); winapi ;external {$ifndef static}libopenblas{$endif} name 'goto_set_num_threads';

  {Get the number of threads on runtime. }
  function openblas_get_num_threads:longint; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_get_num_threads';

  {Get the number of physical processors (cores). }
  function openblas_get_num_procs:longint; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_get_num_procs';

  {Get the build configure on runtime. }
  function openblas_get_config:pchar; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_get_config';

  {Get the CPU corename on runtime. }
  function openblas_get_corename:pchar; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_get_corename';

{$ifdef LINUX}
  { Sets thread affinity for OpenBLAS threads. `thread_idx` is in [0, openblas_get_num_threads()-1].  }
  function openblas_setaffinity(thread_idx:longint; cpusetsize:size_t; cpu_set:Pcpu_set_t):longint; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_setaffinity';

{$endif}
  { Get the parallelization type which is used by OpenBLAS  }

  function openblas_get_parallel:longint; winapi ;external {$ifndef static}libopenblas{$endif} name 'openblas_get_parallel';

  { OpenBLAS is compiled for sequential use   }
  const
    OPENBLAS_SEQUENTIAL = 0;    
  { OpenBLAS is compiled using normal threading model  }
    OPENBLAS_THREAD = 1;    
  { OpenBLAS is compiled using OpenMP threading model  }
    OPENBLAS_OPENMP = 2;    
  {
   * Since all of GotoBlas was written without const,
   * we disable it at build time.
    }
  { #ifndef OPENBLAS_CONST
  # define OPENBLAS_CONST const
  #endif
   }


  type
    CBLAS_INDEX = size_t;    
    CBLAS_ORDER = (CblasRowMajor = 101,CblasColMajor = 102 );

    CBLAS_TRANSPOSE = (CblasNoTrans = 111,CblasTrans = 112,CblasConjTrans = 113, CblasConjNoTrans = 114);

    CBLAS_UPLO = (CblasUpper = 121,CblasLower = 122);

    CBLAS_DIAG = (CblasNonUnit = 131,CblasUnit = 132);

    CBLAS_SIDE = (CblasLeft = 141,CblasRight = 142);

    CBLAS_LAYOUT = CBLAS_ORDER;

  function cblas_sdsdot(n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; 
             incy:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sdsdot';

  function cblas_dsdot(n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsdot';

  function cblas_sdot(n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sdot';

  function cblas_ddot(n:blasint; x:Pdouble; incx:blasint; y:Pdouble; incy:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ddot';

  function cblas_cdotu(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint):openblas_complex_float; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cdotu';

  function cblas_cdotc(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint):openblas_complex_float; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cdotc';

  function cblas_zdotu(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint):openblas_complex_double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdotu';

  function cblas_zdotc(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint):openblas_complex_double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdotc';

  procedure cblas_cdotu_sub(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint; 
              ret:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cdotu_sub';

  procedure cblas_cdotc_sub(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint; 
              ret:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cdotc_sub';

  procedure cblas_zdotu_sub(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint; 
              ret:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdotu_sub';

  procedure cblas_zdotc_sub(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint; 
              ret:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdotc_sub';

  function cblas_sasum(n:blasint; x:Psingle; incx:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sasum';

  function cblas_dasum(n:blasint; x:Pdouble; incx:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dasum';

  function cblas_scasum(n:blasint; x:pointer; incx:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_scasum';

  function cblas_dzasum(n:blasint; x:pointer; incx:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dzasum';

  function cblas_ssum(n:blasint; x:Psingle; incx:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssum';

  function cblas_dsum(n:blasint; x:Pdouble; incx:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsum';

  function cblas_scsum(n:blasint; x:pointer; incx:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_scsum';

  function cblas_dzsum(n:blasint; x:pointer; incx:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dzsum';

  function cblas_snrm2(N:blasint; X:Psingle; incX:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_snrm2';

  function cblas_dnrm2(N:blasint; X:Pdouble; incX:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dnrm2';

  function cblas_scnrm2(N:blasint; X:pointer; incX:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_scnrm2';

  function cblas_dznrm2(N:blasint; X:pointer; incX:blasint):double; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dznrm2';

  function cblas_isamax(n:blasint; x:Psingle; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_isamax';

  function cblas_idamax(n:blasint; x:Pdouble; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_idamax';

  function cblas_icamax(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_icamax';

  function cblas_izamax(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_izamax';

  function cblas_isamin(n:blasint; x:Psingle; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_isamin';

  function cblas_idamin(n:blasint; x:Pdouble; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_idamin';

  function cblas_icamin(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_icamin';

  function cblas_izamin(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_izamin';

  function cblas_ismax(n:blasint; x:Psingle; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ismax';

  function cblas_idmax(n:blasint; x:Pdouble; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_idmax';

  function cblas_icmax(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_icmax';

  function cblas_izmax(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_izmax';

  function cblas_ismin(n:blasint; x:Psingle; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ismin';

  function cblas_idmin(n:blasint; x:Pdouble; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_idmin';

  function cblas_icmin(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_icmin';

  function cblas_izmin(n:blasint; x:pointer; incx:blasint):CBLAS_INDEX; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_izmin';

  procedure cblas_saxpy(n:blasint; alpha:single; x:Psingle; incx:blasint; y:Psingle; 
              incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_saxpy';

  procedure cblas_daxpy(n:blasint; alpha:double; x:Pdouble; incx:blasint; y:Pdouble; 
              incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_daxpy';

  procedure cblas_caxpy(n:blasint; alpha:pointer; x:pointer; incx:blasint; y:pointer; 
              incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_caxpy';

  procedure cblas_zaxpy(n:blasint; alpha:pointer; x:pointer; incx:blasint; y:pointer; 
              incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zaxpy';

  procedure cblas_scopy(n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_scopy';

  procedure cblas_dcopy(n:blasint; x:Pdouble; incx:blasint; y:Pdouble; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dcopy';

  procedure cblas_ccopy(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ccopy';

  procedure cblas_zcopy(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zcopy';

  procedure cblas_sswap(n:blasint; x:Psingle; incx:blasint; y:Psingle; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sswap';

  procedure cblas_dswap(n:blasint; x:Pdouble; incx:blasint; y:Pdouble; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dswap';

  procedure cblas_cswap(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cswap';

  procedure cblas_zswap(n:blasint; x:pointer; incx:blasint; y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zswap';

  procedure cblas_srot(N:blasint; X:Psingle; incX:blasint; Y:Psingle; incY:blasint; 
              c:single; s:single); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_srot';

  procedure cblas_drot(N:blasint; X:Pdouble; incX:blasint; Y:Pdouble; incY:blasint; 
              c:double; s:double); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_drot';

  procedure cblas_csrot(n:blasint; x:pointer; incx:blasint; y:pointer; incY:blasint; 
              c:single; s:single); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_csrot';

  procedure cblas_zdrot(n:blasint; x:pointer; incx:blasint; y:pointer; incY:blasint; 
              c:double; s:double); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdrot';

  procedure cblas_srotg(a:Psingle; b:Psingle; c:Psingle; s:Psingle); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_srotg';

  procedure cblas_drotg(a:Pdouble; b:Pdouble; c:Pdouble; s:Pdouble); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_drotg';

  procedure cblas_crotg(a:pointer; b:pointer; c:Psingle; s:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_crotg';

  procedure cblas_zrotg(a:pointer; b:pointer; c:Pdouble; s:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zrotg';

  procedure cblas_srotm(N:blasint; X:Psingle; incX:blasint; Y:Psingle; incY:blasint; 
              P:Psingle); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_srotm';

  procedure cblas_drotm(N:blasint; X:Pdouble; incX:blasint; Y:Pdouble; incY:blasint; 
              P:Pdouble); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_drotm';

  procedure cblas_srotmg(d1:Psingle; d2:Psingle; b1:Psingle; b2:single; P:Psingle); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_srotmg';

  procedure cblas_drotmg(d1:Pdouble; d2:Pdouble; b1:Pdouble; b2:double; P:Pdouble); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_drotmg';

  procedure cblas_sscal(N:blasint; alpha:single; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sscal';

  procedure cblas_dscal(N:blasint; alpha:double; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dscal';

  procedure cblas_cscal(N:blasint; alpha:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cscal';

  procedure cblas_zscal(N:blasint; alpha:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zscal';

  procedure cblas_csscal(N:blasint; alpha:single; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_csscal';

  procedure cblas_zdscal(N:blasint; alpha:double; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zdscal';

  procedure cblas_sgemv(order:CBLAS_ORDER; trans:CBLAS_TRANSPOSE; m:blasint; n:blasint; alpha:single; 
              a:Psingle; lda:blasint; x:Psingle; incx:blasint; beta:single; 
              y:Psingle; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sgemv';

  procedure cblas_dgemv(order:CBLAS_ORDER; trans:CBLAS_TRANSPOSE; m:blasint; n:blasint; alpha:double; 
              a:Pdouble; lda:blasint; x:Pdouble; incx:blasint; beta:double; 
              y:Pdouble; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dgemv';

  procedure cblas_cgemv(order:CBLAS_ORDER; trans:CBLAS_TRANSPOSE; m:blasint; n:blasint; alpha:pointer; 
              a:pointer; lda:blasint; x:pointer; incx:blasint; beta:pointer; 
              y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgemv';

  procedure cblas_zgemv(order:CBLAS_ORDER; trans:CBLAS_TRANSPOSE; m:blasint; n:blasint; alpha:pointer; 
              a:pointer; lda:blasint; x:pointer; incx:blasint; beta:pointer; 
              y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgemv';

  procedure cblas_sger(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:single; X:Psingle; 
              incX:blasint; Y:Psingle; incY:blasint; A:Psingle; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sger';

  procedure cblas_dger(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:double; X:Pdouble; 
              incX:blasint; Y:Pdouble; incY:blasint; A:Pdouble; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dger';

  procedure cblas_cgeru(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgeru';

  procedure cblas_cgerc(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgerc';

  procedure cblas_zgeru(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgeru';

  procedure cblas_zgerc(order:CBLAS_ORDER; M:blasint; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgerc';

  procedure cblas_strsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:Psingle; lda:blasint; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_strsv';

  procedure cblas_dtrsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:Pdouble; lda:blasint; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtrsv';

  procedure cblas_ctrsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctrsv';

  procedure cblas_ztrsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztrsv';

  procedure cblas_strmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:Psingle; lda:blasint; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_strmv';

  procedure cblas_dtrmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:Pdouble; lda:blasint; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtrmv';

  procedure cblas_ctrmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctrmv';

  procedure cblas_ztrmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztrmv';

  procedure cblas_ssyr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:Psingle; 
              incX:blasint; A:Psingle; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssyr';

  procedure cblas_dsyr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:Pdouble; 
              incX:blasint; A:Pdouble; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsyr';

  procedure cblas_cher(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:pointer; 
              incX:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cher';

  procedure cblas_zher(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:pointer; 
              incX:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zher';

  procedure cblas_ssyr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:Psingle; 
              incX:blasint; Y:Psingle; incY:blasint; A:Psingle; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssyr2';

  procedure cblas_dsyr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:Pdouble; 
              incX:blasint; Y:Pdouble; incY:blasint; A:Pdouble; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsyr2';

  procedure cblas_cher2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cher2';

  procedure cblas_zher2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; A:pointer; lda:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zher2';

  procedure cblas_sgbmv(order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; M:blasint; N:blasint; KL:blasint; 
              KU:blasint; alpha:single; A:Psingle; lda:blasint; X:Psingle; 
              incX:blasint; beta:single; Y:Psingle; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sgbmv';

  procedure cblas_dgbmv(order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; M:blasint; N:blasint; KL:blasint; 
              KU:blasint; alpha:double; A:Pdouble; lda:blasint; X:Pdouble; 
              incX:blasint; beta:double; Y:Pdouble; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dgbmv';

  procedure cblas_cgbmv(order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; M:blasint; N:blasint; KL:blasint; 
              KU:blasint; alpha:pointer; A:pointer; lda:blasint; X:pointer; 
              incX:blasint; beta:pointer; Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgbmv';

  procedure cblas_zgbmv(order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; M:blasint; N:blasint; KL:blasint; 
              KU:blasint; alpha:pointer; A:pointer; lda:blasint; X:pointer; 
              incX:blasint; beta:pointer; Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgbmv';

  procedure cblas_ssbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; K:blasint; alpha:single; 
              A:Psingle; lda:blasint; X:Psingle; incX:blasint; beta:single; 
              Y:Psingle; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssbmv';

  procedure cblas_dsbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; K:blasint; alpha:double; 
              A:Pdouble; lda:blasint; X:Pdouble; incX:blasint; beta:double; 
              Y:Pdouble; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsbmv';

  procedure cblas_stbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:Psingle; lda:blasint; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_stbmv';

  procedure cblas_dtbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:Pdouble; lda:blasint; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtbmv';

  procedure cblas_ctbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctbmv';

  procedure cblas_ztbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztbmv';

  procedure cblas_stbsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:Psingle; lda:blasint; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_stbsv';

  procedure cblas_dtbsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:Pdouble; lda:blasint; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtbsv';

  procedure cblas_ctbsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctbsv';

  procedure cblas_ztbsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              K:blasint; A:pointer; lda:blasint; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztbsv';

  procedure cblas_stpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:Psingle; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_stpmv';

  procedure cblas_dtpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:Pdouble; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtpmv';

  procedure cblas_ctpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctpmv';

  procedure cblas_ztpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztpmv';

  procedure cblas_stpsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:Psingle; X:Psingle; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_stpsv';

  procedure cblas_dtpsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:Pdouble; X:Pdouble; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtpsv';

  procedure cblas_ctpsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctpsv';

  procedure cblas_ztpsv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; N:blasint; 
              Ap:pointer; X:pointer; incX:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztpsv';

  procedure cblas_ssymv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; A:Psingle; 
              lda:blasint; X:Psingle; incX:blasint; beta:single; Y:Psingle; 
              incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssymv';

  procedure cblas_dsymv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; A:Pdouble; 
              lda:blasint; X:Pdouble; incX:blasint; beta:double; Y:Pdouble; 
              incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsymv';

  procedure cblas_chemv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; A:pointer; 
              lda:blasint; X:pointer; incX:blasint; beta:pointer; Y:pointer; 
              incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chemv';

  procedure cblas_zhemv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; A:pointer; 
              lda:blasint; X:pointer; incX:blasint; beta:pointer; Y:pointer; 
              incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhemv';

  procedure cblas_sspmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; Ap:Psingle; 
              X:Psingle; incX:blasint; beta:single; Y:Psingle; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sspmv';

  procedure cblas_dspmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; Ap:Pdouble; 
              X:Pdouble; incX:blasint; beta:double; Y:Pdouble; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dspmv';

  procedure cblas_sspr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:Psingle; 
              incX:blasint; Ap:Psingle); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sspr';

  procedure cblas_dspr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:Pdouble; 
              incX:blasint; Ap:Pdouble); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dspr';

  procedure cblas_chpr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:pointer; 
              incX:blasint; A:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chpr';

  procedure cblas_zhpr(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:pointer; 
              incX:blasint; A:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhpr';

  procedure cblas_sspr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:single; X:Psingle; 
              incX:blasint; Y:Psingle; incY:blasint; A:Psingle); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sspr2';

  procedure cblas_dspr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:double; X:Pdouble; 
              incX:blasint; Y:Pdouble; incY:blasint; A:Pdouble); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dspr2';

  procedure cblas_chpr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; Ap:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chpr2';

  procedure cblas_zhpr2(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; X:pointer; 
              incX:blasint; Y:pointer; incY:blasint; Ap:pointer); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhpr2';

  procedure cblas_chbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; K:blasint; alpha:pointer; 
              A:pointer; lda:blasint; X:pointer; incX:blasint; beta:pointer; 
              Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chbmv';

  procedure cblas_zhbmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; K:blasint; alpha:pointer; 
              A:pointer; lda:blasint; X:pointer; incX:blasint; beta:pointer; 
              Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhbmv';

  procedure cblas_chpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; Ap:pointer; 
              X:pointer; incX:blasint; beta:pointer; Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chpmv';

  procedure cblas_zhpmv(order:CBLAS_ORDER; Uplo:CBLAS_UPLO; N:blasint; alpha:pointer; Ap:pointer; 
              X:pointer; incX:blasint; beta:pointer; Y:pointer; incY:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhpmv';

  procedure cblas_sgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:single; A:Psingle; lda:blasint; B:Psingle; 
              ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sgemm';

  procedure cblas_dgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:double; A:Pdouble; lda:blasint; B:Pdouble; 
              ldb:blasint; beta:double; C:Pdouble; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dgemm';

  procedure cblas_cgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:pointer; A:pointer; lda:blasint; B:pointer; 
              ldb:blasint; beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgemm';

  procedure cblas_cgemm3m(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:pointer; A:pointer; lda:blasint; B:pointer; 
              ldb:blasint; beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgemm3m';

  procedure cblas_zgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:pointer; A:pointer; lda:blasint; B:pointer; 
              ldb:blasint; beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgemm';

  procedure cblas_zgemm3m(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:pointer; A:pointer; lda:blasint; B:pointer; 
              ldb:blasint; beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgemm3m';

  procedure cblas_ssymm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; 
              beta:single; C:Psingle; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssymm';

  procedure cblas_dsymm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:double; A:Pdouble; lda:blasint; B:Pdouble; ldb:blasint; 
              beta:double; C:Pdouble; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsymm';

  procedure cblas_csymm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_csymm';

  procedure cblas_zsymm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zsymm';

  procedure cblas_ssyrk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:single; A:Psingle; lda:blasint; beta:single; C:Psingle; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssyrk';

  procedure cblas_dsyrk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:double; A:Pdouble; lda:blasint; beta:double; C:Pdouble; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsyrk';

  procedure cblas_csyrk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; beta:pointer; C:pointer; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_csyrk';

  procedure cblas_zsyrk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; beta:pointer; C:pointer; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zsyrk';

  procedure cblas_ssyr2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:single; A:Psingle; lda:blasint; B:Psingle; ldb:blasint; 
              beta:single; C:Psingle; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ssyr2k';

  procedure cblas_dsyr2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:double; A:Pdouble; lda:blasint; B:Pdouble; ldb:blasint; 
              beta:double; C:Pdouble; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dsyr2k';

  procedure cblas_csyr2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_csyr2k';

  procedure cblas_zsyr2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zsyr2k';

  procedure cblas_strmm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:single; A:Psingle; lda:blasint; 
              B:Psingle; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_strmm';

  procedure cblas_dtrmm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:double; A:Pdouble; lda:blasint; 
              B:Pdouble; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtrmm';

  procedure cblas_ctrmm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:pointer; A:pointer; lda:blasint; 
              B:pointer; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctrmm';

  procedure cblas_ztrmm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:pointer; A:pointer; lda:blasint; 
              B:pointer; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztrmm';

  procedure cblas_strsm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:single; A:Psingle; lda:blasint; 
              B:Psingle; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_strsm';

  procedure cblas_dtrsm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:double; A:Pdouble; lda:blasint; 
              B:Pdouble; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dtrsm';

  procedure cblas_ctrsm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:pointer; A:pointer; lda:blasint; 
              B:pointer; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ctrsm';

  procedure cblas_ztrsm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; TransA:CBLAS_TRANSPOSE; Diag:CBLAS_DIAG; 
              M:blasint; N:blasint; alpha:pointer; A:pointer; lda:blasint; 
              B:pointer; ldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_ztrsm';

  procedure cblas_chemm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_chemm';

  procedure cblas_zhemm(Order:CBLAS_ORDER; Side:CBLAS_SIDE; Uplo:CBLAS_UPLO; M:blasint; N:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:pointer; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zhemm';

  procedure cblas_cherk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:single; A:pointer; lda:blasint; beta:single; C:pointer; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cherk';

  procedure cblas_zherk(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:double; A:pointer; lda:blasint; beta:double; C:pointer; 
              ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zherk';

  procedure cblas_cher2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:single; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cher2k';

  procedure cblas_zher2k(Order:CBLAS_ORDER; Uplo:CBLAS_UPLO; Trans:CBLAS_TRANSPOSE; N:blasint; K:blasint; 
              alpha:pointer; A:pointer; lda:blasint; B:pointer; ldb:blasint; 
              beta:double; C:pointer; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zher2k';

  procedure cblas_xerbla(p:blasint; rout:Pchar; form:Pchar);varargs;overload; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_xerbla';

//  procedure cblas_xerbla(p:blasint; rout:Pchar; form:Pchar); winapi ;overload;external {$ifndef static}libopenblas{$endif} name 'cblas_xerbla';

  {** BLAS extensions ** }
  procedure cblas_saxpby(n:blasint; alpha:single; x:Psingle; incx:blasint; beta:single; 
              y:Psingle; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_saxpby';

  procedure cblas_daxpby(n:blasint; alpha:double; x:Pdouble; incx:blasint; beta:double; 
              y:Pdouble; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_daxpby';

  procedure cblas_caxpby(n:blasint; alpha:pointer; x:pointer; incx:blasint; beta:pointer; 
              y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_caxpby';

  procedure cblas_zaxpby(n:blasint; alpha:pointer; x:pointer; incx:blasint; beta:pointer; 
              y:pointer; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zaxpby';

  procedure cblas_somatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:single; 
              a:Psingle; clda:blasint; b:Psingle; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_somatcopy';

  procedure cblas_domatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:double; 
              a:Pdouble; clda:blasint; b:Pdouble; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_domatcopy';

  procedure cblas_comatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:Psingle; 
              a:Psingle; clda:blasint; b:Psingle; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_comatcopy';

  procedure cblas_zomatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:Pdouble; 
              a:Pdouble; clda:blasint; b:Pdouble; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zomatcopy';

  procedure cblas_simatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:single; 
              a:Psingle; clda:blasint; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_simatcopy';

  procedure cblas_dimatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:double; 
              a:Pdouble; clda:blasint; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dimatcopy';

  procedure cblas_cimatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:Psingle; 
              a:Psingle; clda:blasint; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cimatcopy';

  procedure cblas_zimatcopy(CORDER:CBLAS_ORDER; CTRANS:CBLAS_TRANSPOSE; crows:blasint; ccols:blasint; calpha:Pdouble; 
              a:Pdouble; clda:blasint; cldb:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zimatcopy';

  procedure cblas_sgeadd(CORDER:CBLAS_ORDER; crows:blasint; ccols:blasint; calpha:single; a:Psingle; 
              clda:blasint; cbeta:single; c:Psingle; cldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sgeadd';

  procedure cblas_dgeadd(CORDER:CBLAS_ORDER; crows:blasint; ccols:blasint; calpha:double; a:Pdouble; 
              clda:blasint; cbeta:double; c:Pdouble; cldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dgeadd';

  procedure cblas_cgeadd(CORDER:CBLAS_ORDER; crows:blasint; ccols:blasint; calpha:Psingle; a:Psingle; 
              clda:blasint; cbeta:Psingle; c:Psingle; cldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_cgeadd';

  procedure cblas_zgeadd(CORDER:CBLAS_ORDER; crows:blasint; ccols:blasint; calpha:Pdouble; a:Pdouble; 
              clda:blasint; cbeta:Pdouble; c:Pdouble; cldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_zgeadd';

  {** BFLOAT16 and INT8 extensions ** }
  { convert float array to BFLOAT16 array by rounding  }
  procedure cblas_sbstobf16(n:blasint; _in:Psingle; incin:blasint; _out:Pbfloat16; incout:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbstobf16';

  { convert double array to BFLOAT16 array by rounding  }
  procedure cblas_sbdtobf16(n:blasint; _in:Pdouble; incin:blasint; _out:Pbfloat16; incout:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbdtobf16';

  { convert BFLOAT16 array to float array  }
  procedure cblas_sbf16tos(n:blasint; _in:Pbfloat16; incin:blasint; _out:Psingle; incout:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbf16tos';

  { convert BFLOAT16 array to double array  }
  procedure cblas_dbf16tod(n:blasint; _in:Pbfloat16; incin:blasint; _out:Pdouble; incout:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_dbf16tod';

  { dot production of BFLOAT16 input arrays, and output as float  }
  function cblas_sbdot(n:blasint; x:Pbfloat16; incx:blasint; y:Pbfloat16; incy:blasint):single; winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbdot';

  procedure cblas_sbgemv(order:CBLAS_ORDER; trans:CBLAS_TRANSPOSE; m:blasint; n:blasint; alpha:single; 
              a:Pbfloat16; lda:blasint; x:Pbfloat16; incx:blasint; beta:single; 
              y:Psingle; incy:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbgemv';

  procedure cblas_sbgemm(Order:CBLAS_ORDER; TransA:CBLAS_TRANSPOSE; TransB:CBLAS_TRANSPOSE; M:blasint; N:blasint; 
              K:blasint; alpha:single; A:Pbfloat16; lda:blasint; B:Pbfloat16; 
              ldb:blasint; beta:single; C:Psingle; ldc:blasint); winapi ;external {$ifndef static}libopenblas{$endif} name 'cblas_sbgemm';


implementation


end.
