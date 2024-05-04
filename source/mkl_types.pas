unit mkl_types;
{$Z4}

interface

uses types;

{$IFDEF FPC}
  {$mode delphi}
  {$PACKRECORDS C}
{$ENDIF}


  {******************************************************************************
  * Copyright 1999-2021 Intel Corporation.
  *
  * This software and the related documents are Intel copyrighted  materials,  and
  * your use of  them is  governed by the  express license  under which  they were
  * provided to you (License).  Unless the License provides otherwise, you may not
  * use, modify, copy, publish, distribute,  disclose or transmit this software or
  * the related documents without Intel's prior written permission.
  *
  * This software and the related documents  are provided as  is,  with no express
  * or implied  warranties,  other  than those  that are  expressly stated  in the
  * License.
  ****************************************************************************** }
  {
  ! Content:
  !      Intel(R) oneAPI Math Kernel Library (oneMKL) types definition
  !*************************************************************************** }
  { oneMKL Complex type for single precision  }
{$ifndef MKL_Complex8}

  type
    {$ifndef WINDOWS} size_t=NativeUInt;{$endif}

    MKL_Complex8 = record
        real : single;
        imag : single;
      end;

    PPMKL_Complex8 = ^PMKL_Complex8;
    PMKL_Complex8 = ^TMKL_Complex8;
    TMKL_Complex8 = MKL_Complex8;
{$endif}
  { oneMKL Complex type for double precision  }
{$ifndef MKL_Complex16}

  type
    MKL_Complex16 = record
        real : double;
        imag : double;
      end;

    PPMKL_Complex16 = ^PMKL_Complex16;
    PMKL_Complex16 = ^TMKL_Complex16;
    TMKL_Complex16 = MKL_Complex16;
{$endif}
  { oneMKL Version type  }

  type
    PPSingle = ^PSingle;
    PPDouble = ^PDouble;

    PMKLVersion = ^TMKLVersion;
    TMKLVersion = record
        MajorVersion : longint;
        MinorVersion : longint;
        UpdateVersion : longint;
        ProductStatus : Pchar;
        Build : Pchar;
        Processor : Pchar;
        Platform : Pchar;
      end;
  { oneMKL integer types for LP64 and ILP64  }
    PMKL_INT = ^MKL_INT        ;
    PMKL_UINT = ^MKL_UINT      ;
    PMKL_LONG = ^MKL_LONG      ;

    PMKL_INT64 = ^MKL_INT64;    

    MKL_INT64 = int64;

    PMKL_UINT64 = ^MKL_UINT64;
    MKL_UINT64 = UINT64;

{$ifdef MKL_ILP64}

  { oneMKL ILP64 integer types  }
  {$ifndef MKL_INT}
    MKL_INT = MKL_INT64;    
  {$endif}
  {$ifndef MKL_UINT}
    MKL_UINT = MKL_UINT64;    
  {$endif}
    MKL_LONG = MKL_INT64;    

{$else}

  { oneMKL LP64 integer types  }
  {$ifndef MKL_INT}
    MKL_INT = integer;
  {$endif}
  {$ifndef MKL_UINT}
    MKL_UINT = dword;    
  {$endif}
    MKL_LONG = longint;    
{$endif}


  { oneMKL integer types  }


{$ifndef MKL_UINT8}
    PMKL_UINT8 = ^MKL_UINT8;    
    MKL_UINT8 = byte;    
{$endif}
{$ifndef MKL_INT8}
    PMKL_INT8 = ^MKL_INT8;    
    MKL_INT8 = char;    
{$endif}
{$ifndef MKL_INT16}
    PMKL_INT16 = ^MKL_INT16;    
    MKL_INT16 = smallint;    
{$endif}
{$ifndef MKL_BF16}
    PMKL_BF16 = ^MKL_BF16;    
    MKL_BF16 = word;    
{$endif}
{$ifndef MKL_INT32}
    PMKL_INT32 = ^MKL_INT32;
    MKL_INT32 = longint;    
{$endif}
{$ifndef MKL_F16}
    PMKL_F16 = ^MKL_F16;    
    MKL_F16 = word;    
{$endif}
    TMKL_F16 = MKL_F16    ;
    TMKL_BF16 = MKL_BF16  ;
  { oneMKL domain names  }

  const
    MKL_DOMAIN_ALL = 0;    
    MKL_DOMAIN_BLAS = 1;    
    MKL_DOMAIN_FFT = 2;    
    MKL_DOMAIN_VML = 3;    
    MKL_DOMAIN_PARDISO = 4;    
    MKL_DOMAIN_LAPACK = 5;    
  { oneMKL CBWR  }
  { mkl_cbwr_get options  }
    MKL_CBWR_BRANCH = 1;    
    MKL_CBWR_ALL =  not (0);    
  { flag specific values  }
    MKL_CBWR_STRICT = $10000;    
  { branch specific values  }
    MKL_CBWR_OFF = 0;    
    MKL_CBWR_UNSET_ALL = MKL_CBWR_OFF;    
    MKL_CBWR_BRANCH_OFF = 1;    
    MKL_CBWR_AUTO = 2;    
    MKL_CBWR_COMPATIBLE = 3;    
    MKL_CBWR_SSE2 = 4;    
    MKL_CBWR_SSSE3 = 6;    
    MKL_CBWR_SSE4_1 = 7;    
    MKL_CBWR_SSE4_2 = 8;    
    MKL_CBWR_AVX = 9;    
    MKL_CBWR_AVX2 = 10;    
    MKL_CBWR_AVX512_MIC = 11;    
    MKL_CBWR_AVX512 = 12;    
    MKL_CBWR_AVX512_MIC_E1 = 13;    
    MKL_CBWR_AVX512_E1 = 14;    
  { error codes  }
    MKL_CBWR_SUCCESS = 0;    
    MKL_CBWR_ERR_INVALID_SETTINGS = -(1);    
    MKL_CBWR_ERR_INVALID_INPUT = -(2);    
    MKL_CBWR_ERR_UNSUPPORTED_BRANCH = -(3);    
    MKL_CBWR_ERR_UNKNOWN_BRANCH = -(4);    
    MKL_CBWR_ERR_MODE_CHANGE_FAILURE = -(8);    
  { obsolete  }
    MKL_CBWR_SSE3 = 5;    

  type
    PMKL_LAYOUT = ^TMKL_LAYOUT;
    TMKL_LAYOUT = (MKL_ROW_MAJOR = 101,MKL_COL_MAJOR = 102
      );

    PMKL_TRANSPOSE = ^TMKL_TRANSPOSE;
    TMKL_TRANSPOSE = (MKL_NOTRANS = 111,MKL_TRANS = 112,MKL_CONJTRANS = 113, MKL_CONJ = 114);

    PMKL_UPLO = ^TMKL_UPLO;
    TMKL_UPLO = (MKL_UPPER = 121,MKL_LOWER = 122);

    PMKL_DIAG = ^TMKL_DIAG;
    TMKL_DIAG = (MKL_NONUNIT = 131,MKL_UNIT = 132);

    PMKL_SIDE = ^TMKL_SIDE;
    TMKL_SIDE = (MKL_LEFT = 141,MKL_RIGHT = 142);

    PMKL_COMPACT_PACK = ^TMKL_COMPACT_PACK;
    TMKL_COMPACT_PACK = (MKL_COMPACT_SSE = 181,MKL_COMPACT_AVX = 182,
      MKL_COMPACT_AVX512 = 183);

    Tsgemm_jit_kernel_t = procedure (_para1:pointer; _para2:Psingle; _para3:Psingle; _para4:Psingle);cdecl;

    Tdgemm_jit_kernel_t = procedure (_para1:pointer; _para2:Pdouble; _para3:Pdouble; _para4:Pdouble);cdecl;

    Tcgemm_jit_kernel_t = procedure (_para1:pointer; _para2:PMKL_Complex8; _para3:PMKL_Complex8; _para4:PMKL_Complex8);cdecl;

    Tzgemm_jit_kernel_t = procedure (_para1:pointer; _para2:PMKL_Complex16; _para3:PMKL_Complex16; _para4:PMKL_Complex16);cdecl;

    Pmkl_jit_status_t = ^Tmkl_jit_status_t;
    Tmkl_jit_status_t = (MKL_JIT_SUCCESS = 0,MKL_NO_JIT = 1, MKL_JIT_ERROR = 2);

implementation


end.
