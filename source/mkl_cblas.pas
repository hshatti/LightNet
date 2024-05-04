unit mkl_cblas;

interface

uses types, mkl_types;

{$include mkl.inc}

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
  !  Content:
  !      Intel(R) oneAPI Math Kernel Library (oneMKL) CBLAS interface
  !***************************************************************************** }
  {
   * Enumerated and derived types
    }
  { this may vary between platforms  }

  type
    CBLAS_INDEX = size_t;    
    TCBLAS_INDEX = size_t;    
  type
    TCBLAS_LAYOUT = (CblasRowMajor = 101,CblasColMajor = 102);
     CBLAS_LAYOUT = TCBLAS_LAYOUT;
    TCBLAS_TRANSPOSE = (CblasNoTrans = 111,CblasTrans = 112,CblasConjTrans = 113);
     CBLAS_TRANSPOSE = TCBLAS_TRANSPOSE;
    TCBLAS_UPLO = (CblasUpper = 121,CblasLower = 122);
     CBLAS_UPLO = TCBLAS_UPLO;
    TCBLAS_DIAG = (CblasNonUnit = 131,CblasUnit = 132);
     CBLAS_DIAG = TCBLAS_DIAG;
    TCBLAS_SIDE = (CblasLeft = 141,CblasRight = 142);
     CBLAS_SIDE = TCBLAS_SIDE;
    TCBLAS_STORAGE = (CblasPacked = 151);
     CBLAS_STORAGE = TCBLAS_STORAGE;
    TCBLAS_IDENTIFIER = (CblasAMatrix = 161,CblasBMatrix = 162);
     CBLAS_IDENTIFIER = TCBLAS_IDENTIFIER;
    TCBLAS_OFFSET = (CblasRowOffset = 171,CblasColOffset = 172, CblasFixOffset = 173);
     CBLAS_OFFSET = TCBLAS_OFFSET;
    PCBLAS_LAYOUT = ^TCBLAS_LAYOUT;
    PCBLAS_TRANSPOSE = ^TCBLAS_TRANSPOSE;
    PCBLAS_UPLO = ^TCBLAS_UPLO;
    PCBLAS_DIAG = ^TCBLAS_DIAG;
    PCBLAS_SIDE = ^TCBLAS_SIDE;
    PCBLAS_STORAGE = ^TCBLAS_STORAGE;
    PCBLAS_IDENTIFIER = ^TCBLAS_IDENTIFIER;
    PCBLAS_OFFSET = ^TCBLAS_OFFSET;
  { this for backward compatibility with CBLAS_ORDER  }
  {
   * ===========================================================================
   * Prototypes for level 1 BLAS functions (complex are recast as routines)
   * ===========================================================================
    }

function cblas_dcabs1(const z:pointer):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_scabs1(const c:pointer):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_sdot(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_sdoti(const N:MKL_INT; const X:Psingle; const indx:PMKL_INT; const Y:Psingle):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_ddot(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_ddoti(const N:MKL_INT; const X:Pdouble; const indx:PMKL_INT; const Y:Pdouble):double;winapi external {$ifdef libmkl} libmkl{$endif};
function cblas_dsdot(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_sdsdot(const N:MKL_INT; const sb:single; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const
           incY:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

{
 * Functions having prefixes Z and C only
  }

procedure cblas_cdotu_sub(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            dotu:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cdotui_sub(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer; const dotui:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cdotc_sub(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            dotc:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cdotci_sub(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer; const dotui:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdotu_sub(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            dotu:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdotui_sub(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer; const dotui:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdotc_sub(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            dotc:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdotci_sub(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer; const dotui:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

{
 * Functions having prefixes S D SC DZ
  }
function cblas_snrm2(const N:MKL_INT; const X:Psingle; const incX:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_sasum(const N:MKL_INT; const X:Psingle; const incX:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_dnrm2(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_dasum(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_scnrm2(const N:MKL_INT; const X:pointer; const incX:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_scasum(const N:MKL_INT; const X:pointer; const incX:MKL_INT):single;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_dznrm2(const N:MKL_INT; const X:pointer; const incX:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_dzasum(const N:MKL_INT; const X:pointer; const incX:MKL_INT):double;winapi external {$ifdef libmkl} libmkl{$endif};

{
 * Functions having standard 4 prefixes (const S D C Z)
  }

function cblas_isamax(const N:MKL_INT; const X:Psingle; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_idamax(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_icamax(const N:MKL_INT; const X:pointer; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_izamax(const N:MKL_INT; const X:pointer; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_isamin(const N:MKL_INT; const X:Psingle; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_idamin(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_icamin(const N:MKL_INT; const X:pointer; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_izamin(const N:MKL_INT; const X:pointer; const incX:MKL_INT):TCBLAS_INDEX;winapi external {$ifdef libmkl} libmkl{$endif};

{
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
  }
{
 * Routines with standard 4 prefixes (const s, d, c, z)
    }

procedure cblas_sswap(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_scopy(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_saxpy(const N:MKL_INT; const alpha:single; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_saxpby(const N:MKL_INT; const alpha:single; const X:Psingle; const incX:MKL_INT; const beta:single; const
            Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_saxpyi(const N:MKL_INT; const alpha:single; const X:Psingle; const indx:PMKL_INT; const Y:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgthr(const N:MKL_INT; const Y:Psingle; const X:Psingle; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgthrz(const N:MKL_INT; const Y:Psingle; const X:Psingle; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssctr(const N:MKL_INT; const X:Psingle; const indx:PMKL_INT; const Y:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_srotg(const a:Psingle; const b:Psingle; const c:Psingle; const s:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dswap(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dcopy(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_daxpy(const N:MKL_INT; const alpha:double; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_daxpby(const N:MKL_INT; const alpha:double; const X:Pdouble; const incX:MKL_INT; const beta:double; const
            Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_daxpyi(const N:MKL_INT; const alpha:double; const X:Pdouble; const indx:PMKL_INT; const Y:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgthr(const N:MKL_INT; const Y:Pdouble; const X:Pdouble; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgthrz(const N:MKL_INT; const Y:Pdouble; const X:Pdouble; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsctr(const N:MKL_INT; const X:Pdouble; const indx:PMKL_INT; const Y:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_drotg(const a:Pdouble; const b:Pdouble; const c:Pdouble; const s:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cswap(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ccopy(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_caxpy(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const Y:pointer; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_caxpby(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_caxpyi(const N:MKL_INT; const alpha:pointer; const X:pointer; const indx:PMKL_INT; const Y:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgthr(const N:MKL_INT; const Y:pointer; const X:pointer; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgthrz(const N:MKL_INT; const Y:pointer; const X:pointer; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csctr(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_crotg(const a:pointer; const b:pointer; const c:Psingle; const s:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zswap(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zcopy(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zaxpy(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const Y:pointer; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zaxpby(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zaxpyi(const N:MKL_INT; const alpha:pointer; const X:pointer; const indx:PMKL_INT; const Y:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgthr(const N:MKL_INT; const Y:pointer; const X:pointer; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgthrz(const N:MKL_INT; const Y:pointer; const X:pointer; const indx:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsctr(const N:MKL_INT; const X:pointer; const indx:PMKL_INT; const Y:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zrotg(const a:pointer; const b:pointer; const c:Pdouble; const s:pointer);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with S and D prefix only
  }
procedure cblas_srotmg(const d1:Psingle; const d2:Psingle; const b1:Psingle; const b2:single; const P:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_srot(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT; const
            c:single; const s:single);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sroti(const N:MKL_INT; const X:Psingle; const indx:PMKL_INT; const Y:Psingle; const c:single; const
            s:single);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_srotm(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const Y:Psingle; const incY:MKL_INT; const
            P:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_drotmg(const d1:Pdouble; const d2:Pdouble; const b1:Pdouble; const b2:double; const P:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_drot(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT; const
            c:double; const s:double);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_drotm(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT; const
            P:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_droti(const N:MKL_INT; const X:Pdouble; const indx:PMKL_INT; const Y:Pdouble; const c:double; const
            s:double);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with CS and ZD prefix only
  }

procedure cblas_csrot(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            c:single; const s:single);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdrot(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const
            c:double; const s:double);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with S D C Z CS and ZD prefixes
  }
procedure cblas_sscal(const N:MKL_INT; const alpha:single; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dscal(const N:MKL_INT; const alpha:double; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cscal(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zscal(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csscal(const N:MKL_INT; const alpha:single; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdscal(const N:MKL_INT; const alpha:double; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * ===========================================================================
 * Prototypes for level 2 BLAS
 * ===========================================================================
  }
{
 * Routines with standard 4 prefixes (const S, D, C, Z)
  }

procedure cblas_sgemv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:single; const
            A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT; const beta:single; const
            Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgbmv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const KL:MKL_INT; const
            KU:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const X:Psingle; const
            incX:MKL_INT; const beta:single; const Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_stbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_stpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:Psingle; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_stbsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_stpsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:Psingle; const X:Psingle; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:double; const
            A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT; const beta:double; const
            Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgbmv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const KL:MKL_INT; const
            KU:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const X:Pdouble; const
            incX:MKL_INT; const beta:double; const Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:Pdouble; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtbsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtpsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:Pdouble; const X:Pdouble; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgbmv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const KL:MKL_INT; const
            KU:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const X:pointer; const
            incX:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctbsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctpsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgbmv(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const KL:MKL_INT; const
            KU:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const X:pointer; const
            incX:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztbsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            K:MKL_INT; const A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztpsv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const N:MKL_INT; const
            Ap:pointer; const X:pointer; const incX:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with S and D prefixes only
  }
procedure cblas_ssymv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const A:Psingle; const
            lda:MKL_INT; const X:Psingle; const incX:MKL_INT; const beta:single; const Y:Psingle; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const K:MKL_INT; const alpha:single; const
            A:Psingle; const lda:MKL_INT; const X:Psingle; const incX:MKL_INT; const beta:single; const
            Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sspmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const Ap:Psingle; const
            X:Psingle; const incX:MKL_INT; const beta:single; const Y:Psingle; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sger(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:single; const X:Psingle; const
            incX:MKL_INT; const Y:Psingle; const incY:MKL_INT; const A:Psingle; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:Psingle; const
            incX:MKL_INT; const A:Psingle; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sspr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:Psingle; const
            incX:MKL_INT; const Ap:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:Psingle; const
            incX:MKL_INT; const Y:Psingle; const incY:MKL_INT; const A:Psingle; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sspr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:Psingle; const
            incX:MKL_INT; const Y:Psingle; const incY:MKL_INT; const A:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsymv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const A:Pdouble; const
            lda:MKL_INT; const X:Pdouble; const incX:MKL_INT; const beta:double; const Y:Pdouble; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const K:MKL_INT; const alpha:double; const
            A:Pdouble; const lda:MKL_INT; const X:Pdouble; const incX:MKL_INT; const beta:double; const
            Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dspmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const Ap:Pdouble; const
            X:Pdouble; const incX:MKL_INT; const beta:double; const Y:Pdouble; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dger(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:double; const X:Pdouble; const
            incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT; const A:Pdouble; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:Pdouble; const
            incX:MKL_INT; const A:Pdouble; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dspr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:Pdouble; const
            incX:MKL_INT; const Ap:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:Pdouble; const
            incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT; const A:Pdouble; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dspr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:Pdouble; const
            incX:MKL_INT; const Y:Pdouble; const incY:MKL_INT; const A:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with C and Z prefixes only
  }
procedure cblas_chemv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const A:pointer; const
            lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const Y:pointer; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_chbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const K:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_chpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const Ap:pointer; const
            X:pointer; const incX:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgeru(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgerc(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cher(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:pointer; const
            incX:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_chpr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:single; const X:pointer; const
            incX:MKL_INT; const A:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cher2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_chpr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const Ap:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhemv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const A:pointer; const
            lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const Y:pointer; const
            incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhbmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const K:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const X:pointer; const incX:MKL_INT; const beta:pointer; const
            Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhpmv(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const Ap:pointer; const
            X:pointer; const incX:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgeru(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgerc(const Layout:TCBLAS_LAYOUT; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zher(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:pointer; const
            incX:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhpr(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:double; const X:pointer; const
            incX:MKL_INT; const A:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zher2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const A:pointer; const lda:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhpr2(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const N:MKL_INT; const alpha:pointer; const X:pointer; const
            incX:MKL_INT; const Y:pointer; const incY:MKL_INT; const Ap:pointer);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * ===========================================================================
 * Prototypes for level 3 BLAS
 * ===========================================================================
  }
{
 * Routines with standard 4 prefixes (const S, D, C, Z)
  }
procedure cblas_sgemm(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const B:Psingle; const
            ldb:MKL_INT; const beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemm_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:Psingle; const A_Array:PPsingle; const lda_Array:PMKL_INT; const B_Array:PPsingle; const
            ldb_Array:PMKL_INT; const beta_Array:Psingle; const C_Array:PPsingle; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemm_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const stridea:MKL_INT; const
            B:Psingle; const ldb:MKL_INT; const strideb:MKL_INT; const beta:single; const C:Psingle; const
            ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemmt(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const B:Psingle; const
            ldb:MKL_INT; const beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssymm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:single; const A:Psingle; const lda:MKL_INT; const B:Psingle; const ldb:MKL_INT; const
            beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyrk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:single; const A:Psingle; const lda:MKL_INT; const beta:single; const C:Psingle; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyrk_batch_strided(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:single; const A:Psingle; const lda:MKL_INT; const stridea:MKL_INT; const beta:single; const
            C:Psingle; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyrk_batch(const Layout:TCBLAS_LAYOUT; const Uplo_array:PCBLAS_UPLO; const Trans_array:PCBLAS_TRANSPOSE; const N_array:PMKL_INT; const K_array:PMKL_INT; const
            alpha_array:Psingle; const A_array:PPsingle; const lda_array:PMKL_INT; const beta_array:Psingle; const C_array:PPsingle; const
            ldc_array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ssyr2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:single; const A:Psingle; const lda:MKL_INT; const B:Psingle; const ldb:MKL_INT; const
            beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strmm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const
            B:Psingle; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strsm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const
            B:Psingle; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strsm_batch(const Layout:TCBLAS_LAYOUT; const Side_Array:PCBLAS_SIDE; const Uplo_Array:PCBLAS_UPLO; const TransA_Array:PCBLAS_TRANSPOSE; const Diag_Array:PCBLAS_DIAG; const
            M_Array:PMKL_INT; const N_Array:PMKL_INT; const alpha_Array:Psingle; const A_Array:PPsingle; const lda_Array:PMKL_INT; const
            B_Array:PPsingle; const ldb_Array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_strsm_batch_strided(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:single; const A:Psingle; const lda:MKL_INT; const
            stridea:MKL_INT; const B:Psingle; const ldb:MKL_INT; const strideb:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemm(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const B:Pdouble; const
            ldb:MKL_INT; const beta:double; const C:Pdouble; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemm_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:Pdouble; const A_Array:PPdouble; const lda_Array:PMKL_INT; const B_Array:PPdouble; const
            ldb_Array:PMKL_INT; const beta_Array:Pdouble; const C_Array:PPdouble; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemm_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const stridea:MKL_INT; const
            B:Pdouble; const ldb:MKL_INT; const strideb:MKL_INT; const beta:double; const C:Pdouble; const
            ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemmt(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const N:MKL_INT; const
            K:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const B:Pdouble; const
            ldb:MKL_INT; const beta:double; const C:Pdouble; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsymm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:double; const A:Pdouble; const lda:MKL_INT; const B:Pdouble; const ldb:MKL_INT; const
            beta:double; const C:Pdouble; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyrk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:double; const A:Pdouble; const lda:MKL_INT; const beta:double; const C:Pdouble; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyrk_batch(const Layout:TCBLAS_LAYOUT; const Uplo_array:PCBLAS_UPLO; const Trans_array:PCBLAS_TRANSPOSE; const N_array:PMKL_INT; const K_array:PMKL_INT; const
            alpha_array:Pdouble; const A_array:PPdouble; const lda_array:PMKL_INT; const beta_array:Pdouble; const C_array:PPdouble; const
            ldc_array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyrk_batch_strided(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:double; const A:Pdouble; const lda:MKL_INT; const stridea:MKL_INT; const beta:double; const
            C:Pdouble; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dsyr2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:double; const A:Pdouble; const lda:MKL_INT; const B:Pdouble; const ldb:MKL_INT; const
            beta:double; const C:Pdouble; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrmm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const
            B:Pdouble; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrsm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const
            B:Pdouble; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrsm_batch(const Layout:TCBLAS_LAYOUT; const Side_Array:PCBLAS_SIDE; const Uplo_Array:PCBLAS_UPLO; const Transa_Array:PCBLAS_TRANSPOSE; const Diag_Array:PCBLAS_DIAG; const
            M_Array:PMKL_INT; const N_Array:PMKL_INT; const alpha_Array:Pdouble; const A_Array:PPdouble; const lda_Array:PMKL_INT; const
            B_Array:PPdouble; const ldb_Array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dtrsm_batch_strided(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:double; const A:Pdouble; const lda:MKL_INT; const
            stridea:MKL_INT; const B:Pdouble; const ldb:MKL_INT; const strideb:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemm(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemm3m(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemm_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const B_Array:Ppointer; const
            ldb_Array:PMKL_INT; const beta_Array:pointer; const C_Array:Ppointer; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemm_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const
            B:pointer; const ldb:MKL_INT; const strideb:MKL_INT; const beta:pointer; const C:pointer; const
            ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemm3m_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const B_Array:Ppointer; const
            ldb_Array:PMKL_INT; const beta_Array:pointer; const C_Array:Ppointer; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemmt(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csymm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csyrk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const beta:pointer; const C:pointer; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csyrk_batch(const Layout:TCBLAS_LAYOUT; const Uplo_array:PCBLAS_UPLO; const Trans_array:PCBLAS_TRANSPOSE; const N_array:PMKL_INT; const K_array:PMKL_INT; const
            alpha_array:pointer; const A_array:Ppointer; const lda_array:PMKL_INT; const beta_array:pointer; const C_array:Ppointer; const
            ldc_array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csyrk_batch_strided(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const beta:pointer; const
            C:pointer; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_csyr2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrmm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            B:pointer; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrsm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            B:pointer; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrsm_batch(const Layout:TCBLAS_LAYOUT; const Side_Array:PCBLAS_SIDE; const Uplo_Array:PCBLAS_UPLO; const Transa_Array:PCBLAS_TRANSPOSE; const Diag_Array:PCBLAS_DIAG; const
            M_Array:PMKL_INT; const N_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const
            B_Array:Ppointer; const ldb_Array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ctrsm_batch_strided(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            stridea:MKL_INT; const B:pointer; const ldb:MKL_INT; const strideb:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemm(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemm3m(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemm_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const B_Array:Ppointer; const
            ldb_Array:PMKL_INT; const beta_Array:pointer; const C_Array:Ppointer; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemm_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const
            B:pointer; const ldb:MKL_INT; const strideb:MKL_INT; const beta:pointer; const C:pointer; const
            ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemm3m_batch(const Layout:TCBLAS_LAYOUT; const TransA_Array:PCBLAS_TRANSPOSE; const TransB_Array:PCBLAS_TRANSPOSE; const M_Array:PMKL_INT; const N_Array:PMKL_INT; const
            K_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const B_Array:Ppointer; const
            ldb_Array:PMKL_INT; const beta_Array:pointer; const C_Array:Ppointer; const ldc_Array:PMKL_INT; const group_count:MKL_INT; const
            group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemmt(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const N:MKL_INT; const
            K:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const
            ldb:MKL_INT; const beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsymm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsyrk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const beta:pointer; const C:pointer; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsyrk_batch(const Layout:TCBLAS_LAYOUT; const Uplo_array:PCBLAS_UPLO; const Trans_array:PCBLAS_TRANSPOSE; const N_array:PMKL_INT; const K_array:PMKL_INT; const
            alpha_array:pointer; const A_array:Ppointer; const lda_array:PMKL_INT; const beta_array:pointer; const C_array:Ppointer; const
            ldc_array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsyrk_batch_strided(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const beta:pointer; const
            C:pointer; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zsyr2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrmm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            B:pointer; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrsm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            B:pointer; const ldb:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrsm_batch(const Layout:TCBLAS_LAYOUT; const Side_Array:PCBLAS_SIDE; const Uplo_Array:PCBLAS_UPLO; const Transa_Array:PCBLAS_TRANSPOSE; const Diag_Array:PCBLAS_DIAG; const
            M_Array:PMKL_INT; const N_Array:PMKL_INT; const alpha_Array:pointer; const A_Array:Ppointer; const lda_Array:PMKL_INT; const
            B_Array:Ppointer; const ldb_Array:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ztrsm_batch_strided(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const TransA:TCBLAS_TRANSPOSE; const Diag:TCBLAS_DIAG; const
            M:MKL_INT; const N:MKL_INT; const alpha:pointer; const A:pointer; const lda:MKL_INT; const
            stridea:MKL_INT; const B:pointer; const ldb:MKL_INT; const strideb:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with prefixes C and Z only
  }
procedure cblas_chemm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cherk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:single; const A:pointer; const lda:MKL_INT; const beta:single; const C:pointer; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cher2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:single; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zhemm(const Layout:TCBLAS_LAYOUT; const Side:TCBLAS_SIDE; const Uplo:TCBLAS_UPLO; const M:MKL_INT; const N:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:pointer; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zherk(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:double; const A:pointer; const lda:MKL_INT; const beta:double; const C:pointer; const
            ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zher2k(const Layout:TCBLAS_LAYOUT; const Uplo:TCBLAS_UPLO; const Trans:TCBLAS_TRANSPOSE; const N:MKL_INT; const K:MKL_INT; const
            alpha:pointer; const A:pointer; const lda:MKL_INT; const B:pointer; const ldb:MKL_INT; const
            beta:double; const C:pointer; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Routines with prefixes S and D only
  }
function cblas_sgemm_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemm_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const src:Psingle; const ld:MKL_INT; const dest:Psingle);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemm_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const A:Psingle; const lda:MKL_INT; const B:Psingle; const ldb:MKL_INT; const
            beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_dgemm_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemm_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:double; const src:Pdouble; const ld:MKL_INT; const dest:Pdouble);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemm_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const A:Pdouble; const lda:MKL_INT; const B:Pdouble; const ldb:MKL_INT; const
            beta:double; const C:Pdouble; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_hgemm(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:TMKL_F16; const A:PMKL_F16; const lda:MKL_INT; const B:PMKL_F16; const
            ldb:MKL_INT; const beta:TMKL_F16; const C:PMKL_F16; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_hgemm_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_hgemm_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:TMKL_F16; const src:PMKL_F16; const ld:MKL_INT; const dest:PMKL_F16);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_hgemm_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const A:PMKL_F16; const lda:MKL_INT; const B:PMKL_F16; const ldb:MKL_INT; const
            beta:TMKL_F16; const C:PMKL_F16; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Integer Routines
  }

procedure cblas_gemm_s16s16s32(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const OffsetC:TCBLAS_OFFSET; const M:MKL_INT; const
            N:MKL_INT; const K:MKL_INT; const alpha:single; const A:PMKL_INT16; const lda:MKL_INT; const
            ao:MKL_INT16; const B:PMKL_INT16; const ldb:MKL_INT; const bo:MKL_INT16; const beta:single; const
            C:PMKL_INT32; const ldc:MKL_INT; const cb:PMKL_INT32);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_s8u8s32(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const OffsetC:TCBLAS_OFFSET; const M:MKL_INT; const
            N:MKL_INT; const K:MKL_INT; const alpha:single; const A:pointer; const lda:MKL_INT; const
            ao:MKL_INT8; const B:pointer; const ldb:MKL_INT; const bo:MKL_INT8; const beta:single; const
            C:PMKL_INT32; const ldc:MKL_INT; const cb:PMKL_INT32);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_bf16bf16f32(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const TransB:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const A:PMKL_BF16; const lda:MKL_INT; const B:PMKL_BF16; const
            ldb:MKL_INT; const beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_gemm_s8u8s32_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_gemm_s16s16s32_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

function cblas_gemm_bf16bf16f32_pack_get_size(const identifier:TCBLAS_IDENTIFIER; const M:MKL_INT; const N:MKL_INT; const K:MKL_INT):size_t;winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_s8u8s32_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const src:pointer; const ld:MKL_INT; const dest:pointer);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_s16s16s32_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const src:PMKL_INT16; const ld:MKL_INT; const dest:PMKL_INT16);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_bf16bf16f32_pack(const Layout:TCBLAS_LAYOUT; const identifier:TCBLAS_IDENTIFIER; const Trans:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const src:PMKL_BF16; const ld:MKL_INT; const dest:PMKL_BF16);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_s8u8s32_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const offsetc:TCBLAS_OFFSET; const M:MKL_INT; const
            N:MKL_INT; const K:MKL_INT; const alpha:single; const A:pointer; const lda:MKL_INT; const
            ao:MKL_INT8; const B:pointer; const ldb:MKL_INT; const bo:MKL_INT8; const beta:single; const
            C:PMKL_INT32; const ldc:MKL_INT; const co:PMKL_INT32);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_s16s16s32_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const offsetc:TCBLAS_OFFSET; const M:MKL_INT; const
            N:MKL_INT; const K:MKL_INT; const alpha:single; const A:PMKL_INT16; const lda:MKL_INT; const
            ao:MKL_INT16; const B:PMKL_INT16; const ldb:MKL_INT; const bo:MKL_INT16; const beta:single; const
            C:PMKL_INT32; const ldc:MKL_INT; const co:PMKL_INT32);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_gemm_bf16bf16f32_compute(const Layout:TCBLAS_LAYOUT; const TransA:MKL_INT; const TransB:MKL_INT; const M:MKL_INT; const N:MKL_INT; const
            K:MKL_INT; const alpha:single; const A:PMKL_BF16; const lda:MKL_INT; const B:PMKL_BF16; const
            ldb:MKL_INT; const beta:single; const C:Psingle; const ldc:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{
 * Jit routines
  }
{$ifndef mkl_jit_create_dgemm}
  {$define mkl_jit_create_dgemm := mkl_cblas_jit_create_dgemm}    
{$endif}
function mkl_cblas_jit_create_dgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const
           n:MKL_INT; const k:MKL_INT; const alpha:double; const lda:MKL_INT; const ldb:MKL_INT; const
           beta:double; const ldc:MKL_INT):Tmkl_jit_status_t;winapi external {$ifdef libmkl} libmkl{$endif};
{$ifndef mkl_jit_create_sgemm}
  {$define mkl_jit_create_sgemm := mkl_cblas_jit_create_sgemm}    
{$endif}
function mkl_cblas_jit_create_sgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const
           n:MKL_INT; const k:MKL_INT; const alpha:single; const lda:MKL_INT; const ldb:MKL_INT; const
           beta:single; const ldc:MKL_INT):Tmkl_jit_status_t;winapi external {$ifdef libmkl} libmkl{$endif};
{$ifndef mkl_jit_create_cgemm}
  {$define mkl_jit_create_cgemm := mkl_cblas_jit_create_cgemm}    
{$endif}
function mkl_cblas_jit_create_cgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const
           n:MKL_INT; const k:MKL_INT; const alpha:pointer; const lda:MKL_INT; const ldb:MKL_INT; const
           beta:pointer; const ldc:MKL_INT):Tmkl_jit_status_t;winapi external {$ifdef libmkl} libmkl{$endif};
{$ifndef mkl_jit_create_zgemm}
   {$define mkl_jit_create_zgemm := mkl_cblas_jit_create_zgemm}    
{$endif}
function mkl_cblas_jit_create_zgemm(const jitter:Ppointer; const layout:TMKL_LAYOUT; const transa:TMKL_TRANSPOSE; const transb:TMKL_TRANSPOSE; const m:MKL_INT; const
           n:MKL_INT; const k:MKL_INT; const alpha:pointer; const lda:MKL_INT; const ldb:MKL_INT; const
           beta:pointer; const ldc:MKL_INT):Tmkl_jit_status_t;winapi external {$ifdef libmkl} libmkl{$endif};

function mkl_jit_get_dgemm_ptr(const jitter:pointer):Tdgemm_jit_kernel_t;winapi external {$ifdef libmkl} libmkl{$endif};

function mkl_jit_get_sgemm_ptr(const jitter:pointer):Tsgemm_jit_kernel_t;winapi external {$ifdef libmkl} libmkl{$endif};

function mkl_jit_get_cgemm_ptr(const jitter:pointer):Tcgemm_jit_kernel_t;winapi external {$ifdef libmkl} libmkl{$endif};

function mkl_jit_get_zgemm_ptr(const jitter:pointer):Tzgemm_jit_kernel_t;winapi external {$ifdef libmkl} libmkl{$endif};
function mkl_jit_destroy(const jitter:pointer):Tmkl_jit_status_t;winapi external {$ifdef libmkl} libmkl{$endif};
{ Level1 BLAS batch API  }
procedure cblas_saxpy_batch(const n:PMKL_INT; const alpha:Psingle; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const
            incy:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_daxpy_batch(const n:PMKL_INT; const alpha:Pdouble; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const
            incy:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_caxpy_batch(const n:PMKL_INT; const alpha:pointer; const x:Ppointer; const incx:PMKL_INT; const y:Ppointer; const
            incy:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zaxpy_batch(const n:PMKL_INT; const alpha:pointer; const x:Ppointer; const incx:PMKL_INT; const y:Ppointer; const
            incy:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_saxpy_batch_strided(const N:MKL_INT; const alpha:single; const X:Psingle; const incX:MKL_INT; const stridex:MKL_INT; const
            Y:Psingle; const incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_daxpy_batch_strided(const N:MKL_INT; const alpha:double; const X:Pdouble; const incX:MKL_INT; const stridex:MKL_INT; const
            Y:Pdouble; const incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_caxpy_batch_strided(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const stridex:MKL_INT; const
            Y:pointer; const incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zaxpy_batch_strided(const N:MKL_INT; const alpha:pointer; const X:pointer; const incX:MKL_INT; const stridex:MKL_INT; const
            Y:pointer; const incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_scopy_batch(const n:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const y:PPsingle; const incy:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dcopy_batch(const n:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const y:PPdouble; const incy:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ccopy_batch(const n:PMKL_INT; const x:Ppointer; const incx:PMKL_INT; const y:Ppointer; const incy:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zcopy_batch(const n:PMKL_INT; const x:Ppointer; const incx:PMKL_INT; const y:Ppointer; const incy:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_scopy_batch_strided(const N:MKL_INT; const X:Psingle; const incX:MKL_INT; const stridex:MKL_INT; const Y:Psingle; const
            incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dcopy_batch_strided(const N:MKL_INT; const X:Pdouble; const incX:MKL_INT; const stridex:MKL_INT; const Y:Pdouble; const
            incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ccopy_batch_strided(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const stridex:MKL_INT; const Y:pointer; const
            incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zcopy_batch_strided(const N:MKL_INT; const X:pointer; const incX:MKL_INT; const stridex:MKL_INT; const Y:pointer; const
            incY:MKL_INT; const stridey:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};
{ Level2 BLAS batch API  }
procedure cblas_sgemv_batch(const Layout:TCBLAS_LAYOUT; const TransA:PCBLAS_TRANSPOSE; const M:PMKL_INT; const N:PMKL_INT; const alpha:Psingle; const
            A:PPsingle; const lda:PMKL_INT; const X:PPsingle; const incX:PMKL_INT; const beta:Psingle; const
            Y:PPsingle; const incY:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sgemv_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:single; const
            A:Psingle; const lda:MKL_INT; const stridea:MKL_INT; const X:Psingle; const incX:MKL_INT; const
            stridex:MKL_INT; const beta:single; const Y:Psingle; const incY:MKL_INT; const stridey:MKL_INT; const
            batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemv_batch(const Layout:TCBLAS_LAYOUT; const TransA:PCBLAS_TRANSPOSE; const M:PMKL_INT; const N:PMKL_INT; const alpha:Pdouble; const
            A:PPdouble; const lda:PMKL_INT; const X:PPdouble; const incX:PMKL_INT; const beta:Pdouble; const
            Y:PPdouble; const incY:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_dgemv_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:double; const
            A:Pdouble; const lda:MKL_INT; const stridea:MKL_INT; const X:Pdouble; const incX:MKL_INT; const
            stridex:MKL_INT; const beta:double; const Y:Pdouble; const incY:MKL_INT; const stridey:MKL_INT; const
            batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemv_batch(const Layout:TCBLAS_LAYOUT; const TransA:PCBLAS_TRANSPOSE; const M:PMKL_INT; const N:PMKL_INT; const alpha:pointer; const
            A:Ppointer; const lda:PMKL_INT; const X:Ppointer; const incX:PMKL_INT; const beta:pointer; const
            Y:Ppointer; const incY:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cgemv_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const X:pointer; const incX:MKL_INT; const
            stridex:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT; const stridey:MKL_INT; const
            batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemv_batch(const Layout:TCBLAS_LAYOUT; const TransA:PCBLAS_TRANSPOSE; const M:PMKL_INT; const N:PMKL_INT; const alpha:pointer; const
            A:Ppointer; const lda:PMKL_INT; const X:Ppointer; const incX:PMKL_INT; const beta:pointer; const
            Y:Ppointer; const incY:PMKL_INT; const group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zgemv_batch_strided(const Layout:TCBLAS_LAYOUT; const TransA:TCBLAS_TRANSPOSE; const M:MKL_INT; const N:MKL_INT; const alpha:pointer; const
            A:pointer; const lda:MKL_INT; const stridea:MKL_INT; const X:pointer; const incX:MKL_INT; const
            stridex:MKL_INT; const beta:pointer; const Y:pointer; const incY:MKL_INT; const stridey:MKL_INT; const
            batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sdgmm_batch(const layout:TCBLAS_LAYOUT; const side:PCBLAS_SIDE; const m:PMKL_INT; const n:PMKL_INT; const a:PPsingle; const
            lda:PMKL_INT; const x:PPsingle; const incx:PMKL_INT; const c:PPsingle; const ldc:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_sdgmm_batch_strided(const layout:TCBLAS_LAYOUT; const side:TCBLAS_SIDE; const m:MKL_INT; const n:MKL_INT; const a:Psingle; const
            lda:MKL_INT; const stridea:MKL_INT; const x:Psingle; const incx:MKL_INT; const stridex:MKL_INT; const
            c:Psingle; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ddgmm_batch(const layout:TCBLAS_LAYOUT; const side:PCBLAS_SIDE; const m:PMKL_INT; const n:PMKL_INT; const a:PPdouble; const
            lda:PMKL_INT; const x:PPdouble; const incx:PMKL_INT; const c:PPdouble; const ldc:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_ddgmm_batch_strided(const layout:TCBLAS_LAYOUT; const side:TCBLAS_SIDE; const m:MKL_INT; const n:MKL_INT; const a:Pdouble; const
            lda:MKL_INT; const stridea:MKL_INT; const x:Pdouble; const incx:MKL_INT; const stridex:MKL_INT; const
            c:Pdouble; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cdgmm_batch(const layout:TCBLAS_LAYOUT; const side:PCBLAS_SIDE; const m:PMKL_INT; const n:PMKL_INT; const a:Ppointer; const
            lda:PMKL_INT; const x:Ppointer; const incx:PMKL_INT; const c:Ppointer; const ldc:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_cdgmm_batch_strided(const layout:TCBLAS_LAYOUT; const side:TCBLAS_SIDE; const m:MKL_INT; const n:MKL_INT; const a:pointer; const
            lda:MKL_INT; const stridea:MKL_INT; const x:pointer; const incx:MKL_INT; const stridex:MKL_INT; const
            c:pointer; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdgmm_batch(const layout:TCBLAS_LAYOUT; const side:PCBLAS_SIDE; const m:PMKL_INT; const n:PMKL_INT; const a:Ppointer; const
            lda:PMKL_INT; const x:Ppointer; const incx:PMKL_INT; const c:Ppointer; const ldc:PMKL_INT; const
            group_count:MKL_INT; const group_size:PMKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

procedure cblas_zdgmm_batch_strided(const layout:TCBLAS_LAYOUT; const side:TCBLAS_SIDE; const m:MKL_INT; const n:MKL_INT; const a:pointer; const
            lda:MKL_INT; const stridea:MKL_INT; const x:pointer; const incx:MKL_INT; const stridex:MKL_INT; const
            c:pointer; const ldc:MKL_INT; const stridec:MKL_INT; const batch_size:MKL_INT);winapi external {$ifdef libmkl} libmkl{$endif};

implementation

end.
