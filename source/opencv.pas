{M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M }
unit opencv;

interface
{$ifdef FPC}
{$if defined(MSWINDOWS)}
  {$Linklib opencv_core2413d.dll}
  {$LinkLib opencv_highgui2413d.dll}
  {$LinkLib opencv_imgproc2413d.dll}

{$elseif defined(DARWIN)}

{$else}
{$endif}

{$endif}

{* @brief This is the "metatype" used *only* as a function parameter.
It denotes that the function accepts arrays of multiple types, such as IplImage*, CvMat* or even
CvSeq* sometimes. The particular array type is determined at runtime by analyzing the first 4
bytes of the header. In C++ interface the role of CvArr is played by InputArray and OutputArray.
  }

const
  CV_MAX_DIM = 32;
  IPL_IMAGE_HEADER = 1;
  IPL_IMAGE_DATA = 2;
  IPL_IMAGE_ROI = 4;
{* extra border mode  }
  IPL_BORDER_REFLECT_101 = 4;
  IPL_BORDER_TRANSPARENT = 5;
  CV_TYPE_NAME_IMAGE = 'opencv-image';

{* for storing double-precision
   floating point data in IplImage's  }

   {***************************************************************************************\
   *                                  Image type (IplImage)                                 *
   \*************************************************************************************** }
   {
    * The following definitions (until #endif)
    * is an extract from IPL headers.
    * Copyright (c) 1995 Intel Corporation.
     }

  IPL_DEPTH_SIGN = $80000000;
  IPL_DEPTH_1U = 1;
  IPL_DEPTH_8U = 8;
  IPL_DEPTH_16U = 16;
  IPL_DEPTH_32F = 32;
  IPL_DEPTH_8S = IPL_DEPTH_SIGN or 8;
  IPL_DEPTH_16S = IPL_DEPTH_SIGN or 16;
  IPL_DEPTH_32S = IPL_DEPTH_SIGN or 32;
  IPL_DATA_ORDER_PIXEL = 0;
  IPL_DATA_ORDER_PLANE = 1;
  IPL_ORIGIN_TL = 0;
  IPL_ORIGIN_BL = 1;
  IPL_ALIGN_4BYTES = 4;
  IPL_ALIGN_8BYTES = 8;
  IPL_ALIGN_16BYTES = 16;
  IPL_ALIGN_32BYTES = 32;
  IPL_ALIGN_DWORD = IPL_ALIGN_4BYTES;
  IPL_ALIGN_QWORD = IPL_ALIGN_8BYTES;
  IPL_BORDER_CONSTANT = 0;
  IPL_BORDER_REPLICATE = 1;
  IPL_BORDER_REFLECT = 2;
  IPL_BORDER_WRAP = 3;
  IPL_DEPTH_64F = 64;
{***************************************************************************************\
*                                  Matrix type (CvMat)                                   *
\*************************************************************************************** }
  CV_AUTO_STEP = $7fffffff;



  CV_MAGIC_MASK = $FFFF0000;
  CV_MAT_MAGIC_VAL = $42420000;
  CV_TYPE_NAME_MAT = 'opencv-matrix';

{* Inline constructor. No data is allocated internally!!!
 * (Use together with cvCreateData, or use cvCreateMat instead to
 * get a matrix with allocated data):
  }
{***************************************************************************************\
*                       Multi-dimensional dense array (CvMatND)                          *
\*************************************************************************************** }

  CV_MATND_MAGIC_VAL = $42430000;
  CV_TYPE_NAME_MATND = 'opencv-nd-matrix';



{***************************************************************************************\
*                      Multi-dimensional sparse array (CvSparseMat)                      *
\*************************************************************************************** }

  CV_SPARSE_MAT_MAGIC_VAL = $42440000;
  CV_TYPE_NAME_SPARSE_MAT = 'opencv-sparse-matrix';


{*************** iteration through a sparse array **************** }



  CV_HIST_MAGIC_VAL = $42450000;
  CV_HIST_UNIFORM_FLAG = 1 shl 10;
{* indicates whether bin ranges are set already or not  }
  CV_HIST_RANGES_FLAG = 1 shl 11;
  CV_HIST_ARRAY = 0;
  CV_HIST_SPARSE = 1;
  CV_HIST_TREE = CV_HIST_SPARSE;
{* should be used as a parameter only,
   it turns to CV_HIST_UNIFORM_FLAG of hist->type  }
  CV_HIST_UNIFORM = 1;
{********************************** CvTermCriteria ************************************ }


  CV_TERMCRIT_ITER = 1;
  CV_TERMCRIT_NUMBER = CV_TERMCRIT_ITER;
  CV_TERMCRIT_EPS = 2;
{* @sa TermCriteria
  }
{*< may be combination of
                     CV_TERMCRIT_ITER
                     CV_TERMCRIT_EPS  }

{************************************ CvSlice ***************************************** }

  CV_WHOLE_SEQ_END_INDEX = $3fffffff;



  CV_STORAGE_MAGIC_VAL = $42890000;




  CV_TYPE_NAME_SEQ = 'opencv-sequence';
  CV_TYPE_NAME_SEQ_TREE = 'opencv-sequence-tree';
{************************************** Set ******************************************* }
{* @brief Set
  Order is not preserved. There can be gaps between sequence elements.
  After the element has been inserted it stays in the same place all the time.
  The MSB(most-significant or sign bit) of the first field (flags) is 0 iff the element exists.
 }


  CV_SET_ELEM_IDX_MASK = (1 shl 26)-1;



  CV_TYPE_NAME_GRAPH = 'opencv-graph';
{***************************************************************************************\
*                                    Sequence types                                      *
\*************************************************************************************** }


  CV_SEQ_MAGIC_VAL = $42990000;

  CV_SET_MAGIC_VAL = $42980000;

  CV_SEQ_ELTYPE_BITS = 12;
  CV_SEQ_ELTYPE_MASK = (1 shl CV_SEQ_ELTYPE_BITS)-1;
{*< (x,y)  }
  //CV_SEQ_ELTYPE_POINT = CV_32SC2;
{*< freeman code: 0..7  }
  //CV_SEQ_ELTYPE_CODE = CV_8UC1;
  CV_SEQ_ELTYPE_GENERIC = 0;
{sizeof(void*) }
{ was #define dname def_expr }

{*< &(x,y)  }
  //CV_SEQ_ELTYPE_PPOINT = CV_SEQ_ELTYPE_PTR;
{*< #(x,y)  }
  //CV_SEQ_ELTYPE_INDEX = CV_32SC1;
{*< &next_o, &next_d, &vtx_o, &vtx_d  }
  CV_SEQ_ELTYPE_GRAPH_EDGE = 0;
{*< first_edge, &(x,y)  }
  CV_SEQ_ELTYPE_GRAPH_VERTEX = 0;
{*< vertex of the binary tree    }
  CV_SEQ_ELTYPE_TRIAN_ATR = 0;
{*< connected component   }
  CV_SEQ_ELTYPE_CONNECTED_COMP = 0;
{*< (x,y,z)   }
  //CV_SEQ_ELTYPE_POINT3D = CV_32FC3;
  CV_SEQ_KIND_BITS = 2;
  CV_SEQ_KIND_MASK = ((1 shl CV_SEQ_KIND_BITS)-1) shl CV_SEQ_ELTYPE_BITS;
{* types of sequences  }
  CV_SEQ_KIND_GENERIC = 0 shl CV_SEQ_ELTYPE_BITS;
  CV_SEQ_KIND_CURVE = 1 shl CV_SEQ_ELTYPE_BITS;
  CV_SEQ_KIND_BIN_TREE = 2 shl CV_SEQ_ELTYPE_BITS;
{* types of sparse sequences (sets)  }
  CV_SEQ_KIND_GRAPH = 1 shl CV_SEQ_ELTYPE_BITS;
  CV_SEQ_KIND_SUBDIV2D = 2 shl CV_SEQ_ELTYPE_BITS;
  CV_SEQ_FLAG_SHIFT = CV_SEQ_KIND_BITS+CV_SEQ_ELTYPE_BITS;
{* flags for curves  }
  CV_SEQ_FLAG_CLOSED = 1 shl CV_SEQ_FLAG_SHIFT;
  CV_SEQ_FLAG_SIMPLE = 0 shl CV_SEQ_FLAG_SHIFT;
  CV_SEQ_FLAG_CONVEX = 0 shl CV_SEQ_FLAG_SHIFT;
  CV_SEQ_FLAG_HOLE = 2 shl CV_SEQ_FLAG_SHIFT;
{* flags for graphs  }
  CV_GRAPH_FLAG_ORIENTED = 1 shl CV_SEQ_FLAG_SHIFT;
  CV_GRAPH = CV_SEQ_KIND_GRAPH;
  CV_ORIENTED_GRAPH = CV_SEQ_KIND_GRAPH or CV_GRAPH_FLAG_ORIENTED;
{* point sets  }
  //CV_SEQ_POINT_SET = CV_SEQ_KIND_GENERIC or CV_SEQ_ELTYPE_POINT;
  //CV_SEQ_POINT3D_SET = CV_SEQ_KIND_GENERIC or CV_SEQ_ELTYPE_POINT3D;
  //CV_SEQ_POLYLINE = CV_SEQ_KIND_CURVE or CV_SEQ_ELTYPE_POINT;
  //CV_SEQ_POLYGON = CV_SEQ_FLAG_CLOSED or CV_SEQ_POLYLINE;
  //CV_SEQ_CONTOUR = CV_SEQ_POLYGON;
  //CV_SEQ_SIMPLE_POLYGON = CV_SEQ_FLAG_SIMPLE or CV_SEQ_POLYGON;
{* chain-coded curves  }
  //CV_SEQ_CHAIN = CV_SEQ_KIND_CURVE or CV_SEQ_ELTYPE_CODE;
  //CV_SEQ_CHAIN_CONTOUR = CV_SEQ_FLAG_CLOSED or CV_SEQ_CHAIN;
{* binary tree for the contour  }
  CV_SEQ_POLYGON_TREE = CV_SEQ_KIND_BIN_TREE or CV_SEQ_ELTYPE_TRIAN_ATR;
{* sequence of the connected components  }
  CV_SEQ_CONNECTED_COMP = CV_SEQ_KIND_GENERIC or CV_SEQ_ELTYPE_CONNECTED_COMP;
{* sequence of the integer numbers  }
  //CV_SEQ_INDEX = CV_SEQ_KIND_GENERIC or CV_SEQ_ELTYPE_INDEX;

  CV_AUTOSTEP = $7fffffff;

  CV_CMP_EQ = 0;
  CV_CMP_GT = 1;
  CV_CMP_GE = 2;
  CV_CMP_LT = 3;
  CV_CMP_LE = 4;
  CV_CMP_NE = 5;

  CV_CHECK_RANGE = 1;
  CV_CHECK_QUIET = 2;

  CV_RAND_UNI = 0;
  CV_RAND_NORMAL = 1;
  CV_SORT_EVERY_ROW = 0;
  CV_SORT_EVERY_COLUMN = 1;
  CV_SORT_ASCENDING = 0;
  CV_SORT_DESCENDING = 16;

  CV_GEMM_A_T = 1;
  CV_GEMM_B_T = 2;
  CV_GEMM_C_T = 4;

  CV_SVD_MODIFY_A = 1;
  CV_SVD_U_T = 2;
  CV_SVD_V_T = 4;

  CV_LU = 0;
  CV_SVD = 1;
  CV_SVD_SYM = 2;
  CV_CHOLESKY = 3;
  CV_QR = 4;
  CV_NORMAL = 16;

  CV_REDUCE_SUM = 0;
  CV_REDUCE_AVG = 1;
  CV_REDUCE_MAX = 2;
  CV_REDUCE_MIN = 3;

  CV_DXT_FORWARD = 0;
  CV_DXT_INVERSE = 1;
{*< divide result by size of array  }
  CV_DXT_SCALE = 2;
  CV_DXT_INV_SCALE = 3;
  CV_DXT_INVERSE_SCALE = CV_DXT_INV_SCALE;
{*< transform each row individually  }
  CV_DXT_ROWS = 4;
{*< conjugate the second argument of cvMulSpectrums  }
  CV_DXT_MUL_CONJ = 8;

  CV_FRONT = 1;
  CV_BACK = 0;

  CV_MAX_ARR = 10;
{* matrix iterator: used for n-ary operations on dense arrays  }
  CV_NO_DEPTH_CHECK = 1;
  CV_NO_CN_CHECK = 2;
  CV_NO_SIZE_CHECK = 4;
  CV_COVAR_SCRAMBLED = 0;
{* flag for cvCalcCovarMatrix, [v1-avg, v2-avg,...] * transpose([v1-avg,v2-avg,...])  }
  CV_COVAR_NORMAL = 1;
{* flag for cvCalcCovarMatrix, do not calc average (i.e. mean vector) - use the input vector instead
   (useful for calculating covariance matrix by parts)  }
  CV_COVAR_USE_AVG = 2;
{* flag for cvCalcCovarMatrix, scale the covariance matrix coefficients by number of the vectors  }
  CV_COVAR_SCALE = 4;
{* flag for cvCalcCovarMatrix, all the input vectors are stored in a single matrix, as its rows  }
  CV_COVAR_ROWS = 8;
{* flag for cvCalcCovarMatrix, all the input vectors are stored in a single matrix, as its columns  }
  CV_COVAR_COLS = 16;
  CV_PCA_DATA_AS_ROW = 0;
  CV_PCA_DATA_AS_COL = 1;
  CV_PCA_USE_AVG = 2;

  CV_C = 1;
  CV_L1 = 2;
  CV_L2 = 4;
  CV_NORM_MASK = 7;
  CV_RELATIVE = 8;
  CV_DIFF = 16;
  CV_MINMAX = 32;
  CV_DIFF_C = CV_DIFF or CV_C;
  CV_DIFF_L1 = CV_DIFF or CV_L1;
  CV_DIFF_L2 = CV_DIFF or CV_L2;
  CV_RELATIVE_C = CV_RELATIVE or CV_C;
  CV_RELATIVE_L1 = CV_RELATIVE or CV_L1;
  CV_RELATIVE_L2 = CV_RELATIVE or CV_L2;

  CV_GRAPH_VERTEX = 1;
  CV_GRAPH_TREE_EDGE = 2;
  CV_GRAPH_BACK_EDGE = 4;
  CV_GRAPH_FORWARD_EDGE = 8;
  CV_GRAPH_CROSS_EDGE = 16;
  CV_GRAPH_ANY_EDGE = 30;
  CV_GRAPH_NEW_TREE = 32;
  CV_GRAPH_BACKTRACKING = 64;
  CV_GRAPH_OVER = -(1);
  CV_GRAPH_ALL_ITEMS = -(1);
{* flags for graph vertices and edges  }
  CV_GRAPH_ITEM_VISITED_FLAG = 1 shl 30;

  CV_GRAPH_SEARCH_TREE_NODE_FLAG = 1 shl 29;
  CV_GRAPH_FORWARD_EDGE_FLAG = 1 shl 28;

  CV_KMEANS_USE_INITIAL_LABELS = 1;

  CV_ErrModeLeaf = 0;
{ Print error and continue  }
  CV_ErrModeParent = 1;
{ Don't print and continue  }
  CV_ErrModeSilent = 2;


  CV_RNG_COEFF = 4164903690;


type
  PPSingle    = ^PSingle;
  PPByte      = ^PByte;
  PIntPtr   = ^IntPtr;
  IntPtr    = NativeInt;
  PPIntPtr  = ^PIntPtr;
  PSize_t = ^size_t;
  size_t    = NativeUInt;
  PPCvArr = ^PCvArr;
  PCvArr  = pointer;

  PCvRNG = ^TCvRNG;
  TCvRNG = UInt64 ;


{* @see cv::Error::Code  }
{*< everything is ok                 }
{*< pseudo error for back trace      }
{*< unknown /unspecified error       }
{*< internal error (bad state)       }
{*< insufficient memory              }
{*< function arg/param is bad        }
{*< unsupported function             }
{*< iter. didn't converge            }
{*< tracing                          }
{*< image header is NULL             }
{*< image size is invalid            }
{*< offset is invalid                }
{ }
{*< image step is wrong, this may happen for a non-continuous matrix  }
{ }
{*< bad number of channels, for example, some functions accept only single channel matrices  }
{ }
{*< input image depth is not supported by the function  }
{ }
{*< number of dimensions is out of range  }
{*< incorrect input origin                }
{*< incorrect input align                 }
{ }
{ }
{*< input COI is not supported            }
{*< incorrect input roi                   }
{ }
{*< null pointer  }
{*< incorrect vector length  }
{*< incorrect filter structure content  }
{*< incorrect transform kernel content  }
{*< incorrect filter offset value  }
{*< the input/output structure size is incorrect   }
{*< division by zero  }
{*< in-place operation is not supported  }
{*< request can't be completed  }
{*< formats of input/output arrays differ  }
{*< flag is wrong or not supported  }
{*< bad CvPoint  }
{*< bad format of mask (neither 8uC1 nor 8sC1) }
{*< sizes of input/output structures do not match  }
{*< the data format/type is not supported by the function }
{*< some of parameters are out of range  }
{*< invalid syntax/structure of the parsed file  }
{*< the requested function/feature is not implemented  }
{*< an allocated block has been corrupted  }
{*< assertion failed    }
{*< no CUDA support     }
{*< GPU API call error  }
{*< no OpenGL support   }
{*< OpenGL API call error  }
{*< OpenCL API call error  }
{*< OpenCL initialization error  }
  TCvStatus = (CV_StsOk = 0,CV_StsBackTrace = -1,
    CV_StsError = -2,CV_StsInternal = -3,
    CV_StsNoMem = -4,CV_StsBadArg = -5,
    CV_StsBadFunc = -6,CV_StsNoConv = -7,
    CV_StsAutoTrace = -8,CV_HeaderIsNull = -9,
    CV_BadImageSize = -10,CV_BadOffset = -11,
    CV_BadDataPtr = -12,CV_BadStep = -13,
    CV_BadModelOrChSeq = -14,CV_BadNumChannels = -15,
    CV_BadNumChannel1U = -16,CV_BadDepth = -17,
    CV_BadAlphaChannel = -18,CV_BadOrder = -19,
    CV_BadOrigin = -20,CV_BadAlign = -21,
    CV_BadCallBack = -22,CV_BadTileSize = -23,
    CV_BadCOI = -24,CV_BadROISize = -25,CV_MaskIsTiled = -26,
    CV_StsNullPtr = -27,CV_StsVecLengthErr = -28,
    CV_StsFilterStructContentErr = -29,CV_StsKernelStructContentErr = -30,
    CV_StsFilterOffsetErr = -31,CV_StsBadSize = -201,
    CV_StsDivByZero = -202,CV_StsInplaceNotSupported = -203,
    CV_StsObjectNotFound = -204,CV_StsUnmatchedFormats = -205,
    CV_StsBadFlag = -206,CV_StsBadPoint = -207,
    CV_StsBadMask = -208,CV_StsUnmatchedSizes = -209,
    CV_StsUnsupportedFormat = -210,CV_StsOutOfRange = -211,
    CV_StsParseError = -212,CV_StsNotImplemented = -213,
    CV_StsBadMemBlock = -214,CV_StsAssert = -215,
    CV_GpuNotSupported = -216,CV_GpuApiCallError = -217,
    CV_OpenGlNotSupported = -218,CV_OpenGlApiCallError = -219,
    CV_OpenCLApiCallError = -220,CV_OpenCLDoubleNotSupported = -221,
    CV_OpenCLInitError = -222,CV_OpenCLNoAMDBlasFft = -223
    );


{* The IplImage is taken from the Intel Image Processing Library, in which the format is native. OpenCV
only supports a subset of possible IplImage formats, as outlined in the parameter list above.

In addition to the above restrictions, OpenCV handles ROIs differently. OpenCV functions require
that the image size or ROI size of all source and destination images match exactly. On the other
hand, the Intel Image Processing Library processes the area of intersection between the source and
destination images (or ROIs), allowing them to vary independently.
 }
{*< sizeof(IplImage)  }
{*< version (=0) }
{*< Most of OpenCV functions support 1,2,3 or 4 channels  }
{*< Ignored by OpenCV  }
{*< Pixel depth in bits: IPL_DEPTH_8U, IPL_DEPTH_8S, IPL_DEPTH_16S,
                               IPL_DEPTH_32S, IPL_DEPTH_32F and IPL_DEPTH_64F are supported.   }
{*< Ignored by OpenCV  }
{*< ditto  }
{*< 0 - interleaved color channels, 1 - separate color channels.
                               cvCreateImage can only create interleaved images  }
{*< 0 - top-left origin,
                               1 - bottom-left origin (Windows bitmaps style).   }
{*< Alignment of image rows (4 or 8).
                               OpenCV ignores it and uses widthStep instead.     }
{*< Image width in pixels.                            }
{*< Image height in pixels.                           }
{*< Image ROI. If NULL, the whole image is selected.  }
{*< Must be NULL.  }
{*< "           "  }
{*< "           "  }
{*< Image data size in bytes
                               (==image->height*image->widthStep
                               in case of interleaved data) }
{*< Pointer to aligned image data.          }
{*< Size of aligned image row in bytes.     }
{*< Ignored by OpenCV.                      }
{*< Ditto.                                  }
{*< Pointer to very origin of image data
                               (not necessarily aligned) -
                               needed for correct deallocation  }


  PIplTileInfo = Pointer;
  PCvSet = ^TCvSet;

  PIplROI = ^TIplROI;
  TIplROI = record
      coi : longint;
      xOffset : longint;
      yOffset : longint;
      width : longint;
      height : longint;
    end;
  PPIplImage = ^PIplImage;
  PIplImage  = ^TIplImage;
  TIplImage  = record
      nSize : longint;
      ID : longint;
      nChannels : longint;
      alphaChannel : longint;
      depth : longint;
      colorModel : array[0..3] of char;
      channelSeq : array[0..3] of char;
      dataOrder : longint;
      origin : longint;
      align : longint;
      width : longint;
      height : longint;
      roi : PIplROI;
      maskROI : PIplImage;
      imageId : pointer;
      tileInfo : PIplTileInfo;
      imageSize : longint;
      imageData : PChar;
      widthStep : longint;
      BorderMode : array[0..3] of longint;
      BorderConst : array[0..3] of longint;
      imageDataOrigin : PChar;
    end;
{*< 0 - no COI (all channels are selected), 1 - 0th channel is selected ... }
  PPIplConvKernel = ^PIplConvKernel;
  PIplConvKernel = ^TIplConvKernel;
  TIplConvKernel = record
      nCols : longint;
      nRows : longint;
      anchorX : longint;
      anchorY : longint;
      values : PLongint;
      nShiftR : longint;
    end;

  PIplConvKernelFP = ^TIplConvKernelFP;
  TIplConvKernelFP = record
      nCols : longint;
      nRows : longint;
      anchorX : longint;
      anchorY : longint;
      values : PSingle;
    end;

{* Matrix elements are stored row by row. Element (i, j) (i - 0-based row index, j - 0-based column
index) of a matrix can be retrieved or modified using CV_MAT_ELEM macro:

    uchar pixval = CV_MAT_ELEM(grayimg, uchar, i, j)
    CV_MAT_ELEM(cameraMatrix, float, 0, 2) = image.width*0.5f;

To access multiple-channel matrices, you can use
CV_MAT_ELEM(matrix, type, i, j\*nchannels + channel_idx).

@deprecated CvMat is now obsolete; consider using Mat instead.
  }
{ for internal use only  }
  PPPCvMat = ^PPCvMat;
  PPCvMat = ^PCvMat;
  PCvMat  = ^TCvMat;
  TCvMat  = record
      _type : longint;
      step : longint;
      refcount : PLongint;
      hdr_refcount : longint;
      data : record
        case longint of
            0 : ( ptr : PByte );
            1 : ( s : PSmallInt );
            2 : ( i : PLongint );
            3 : ( fl : PSingle );
            4 : ( db : PDouble );
      end;
      rows : longint;
      cols : longint;
  end;

  PPCvMatND = ^PCvMatND;
  PCvMatND = ^TCvMatND;
  TCvMatND = record
      _type : longint;
      dims : longint;
      refcount : PLongint;
      hdr_refcount : longint;
      data : record
          case longint of
            0 : ( ptr : PByte );
            1 : ( fl : PSingle );
            2 : ( db : PDouble );
            3 : ( i : PLongint );
            4 : ( s : PSmallInt );
          end;
      dim : array[0..(CV_MAX_DIM)-1] of record
          size : longint;
          step : longint;
        end;
    end;

  PPCvSparseMat = ^PCvSparseMat;
  PCvSparseMat = ^TCvSparseMat;
  TCvSparseMat = record
      _type : longint;
      dims : longint;
      refcount : PLongint;
      hdr_refcount : longint;
      heap : PCvSet;
      hashtable : PPointer;
      hashsize : longint;
      valoffset : longint;
      idxoffset : longint;
      size : array[0..(CV_MAX_DIM)-1] of longint;
    end;

  PCvSparseNode = ^TCvSparseNode;
  TCvSparseNode = record
      hashval : longword;
      next : PCvSparseNode;
    end;

  PCvSparseMatIterator = ^TCvSparseMatIterator;
  TCvSparseMatIterator = record
      mat : PCvSparseMat;
      node : PCvSparseNode;
      curidx : longint;
    end;


{***************************************************************************************\
*                                         Histogram                                      *
\*************************************************************************************** }


  PCvHistType = ^TCvHisttype;
  TCvHistType = longint;
{*< For uniform histograms.                       }
{*< For non-uniform histograms.                   }
{*< Embedded matrix header for array histograms.  }
  PPCvHistogram = ^PCvHistogram;
  PCvHistogram = ^TCvHistogram;
  TCvHistogram = record
      _type : longint;
      bins : PCvArr;
      thresh : array[0..(CV_MAX_DIM)-1] of array[0..1] of single;
      thresh2 : PPsingle;
      mat : TCvMatND;
    end;

{***************************************************************************************\
*                      Other supplementary data type definitions                         *
\*************************************************************************************** }
{************************************** CvRect **************************************** }
{* @sa Rect_  }


  PCvRect = ^TCvRect;
  TCvRect = record
    x : longint;
    y : longint;
    width : longint;
    height : longint;
  end;


  PCvTermCriteria = ^TCvTermCriteria;
  TCvTermCriteria = record
      _type : longint;
      max_iter : longint;
      epsilon : double;
    end;
{****************************** CvPoint and variants ********************************** }
  PPCvPoint =^PCvPoint;
  PCvPoint = ^TCvPoint;
  TCvPoint = record
      x : longint;
      y : longint;
    end;

  PCvPoint2D32f = ^TCvPoint2D32f;
  TCvPoint2D32f = record
      x : single;
      y : single;
    end;

  PCvPoint3D32f = ^TCvPoint3D32f;
  TCvPoint3D32f = record
      x : single;
      y : single;
      z : single;
    end;

  PCvPoint2D64f = ^TCvPoint2D64f;
  TCvPoint2D64f = record
      x : double;
      y : double;
    end;

  PCvPoint3D64f = ^TCvPoint3D64f;
  TCvPoint3D64f = record
      x : double;
      y : double;
      z : double;
    end;
{******************************* CvSize's & CvBox ************************************* }
  PCvSize = ^TCvSize;
  TCvSize = record
      width : longint;
      height : longint;
    end;

  PCvSize2D32f = ^TCvSize2D32f;
  TCvSize2D32f = record
      width : single;
      height : single;
    end;
{* @sa RotatedRect
  }
{*< Center of the box.                           }
{*< Box width and length.                        }
{*< Angle between the horizontal axis            }
{*< and the first side (i.e. length) in degrees  }
  PCvBox2D = ^TCvBox2D;
  TCvBox2D = record
      center : TCvPoint2D32f;
      size : TCvSize2D32f;
      angle : single;
    end;

  PCvNArrayIterator = ^TCvNArrayIterator;
  TCvNArrayIterator = record
    count : longint; (**< number of arrays *)
    dims : longint; (**< number of dimensions to iterate *)
    size : TCvSize; (**< maximal common linear size: { width = size, height = 1 } *)
    ptr : array[0..CV_MAX_ARR-1] of PByte; // /**< pointers to the array slices */
    stack : array[0..CV_MAX_DIM-1] of longint;// /**< for internal use */
    hdr : array[0..CV_MAX_ARR-1] of PCvMatND; // /**< pointers to the headers of the
                              //   matrices that are processed */
  end;

{* Line iterator state:  }
{* Pointer to the current point:  }
{ Bresenham algorithm state:  }
  PCvLineIterator = ^TCvLineIterator;
  TCvLineIterator = record
      ptr : PByte;
      err : longint;
      plus_delta : longint;
      minus_delta : longint;
      plus_step : longint;
      minus_step : longint;
    end;

  PCvSlice = ^TCvSlice;
  TCvSlice = record
      start_index : longint;
      end_index : longint;
    end;
{************************************ CvScalar **************************************** }
{* @sa Scalar_
  }
  PCvScalar = ^TCvScalar;
  TCvScalar = record
      val : array[0..3] of double;
    end;
{***************************************************************************************\
*                                   Dynamic Data structures                              *
\*************************************************************************************** }
{******************************* Memory storage *************************************** }
  PCvMemBlock = ^TCvMemBlock;
  TCvMemBlock = record
      prev : PCvMemBlock;
      next : PCvMemBlock;
    end;
{*< First allocated block.                    }
{*< Current memory block - top of the stack.  }
{*< We get new blocks from parent as needed.  }
{*< Block size.                               }
{*< Remaining free space in current block.    }
  PPCvMemStorage =^PCvMemStorage;
  PCvMemStorage = ^TCvMemStorage;
  TCvMemStorage = record
      signature : longint;
      bottom : PCvMemBlock;
      top : PCvMemBlock;
      parent : PCvMemStorage;
      block_size : longint;
      free_space : longint;
    end;

  PCvMemStoragePos = ^TCvMemStoragePos;
  TCvMemStoragePos = record
      top : PCvMemBlock;
      free_space : longint;
    end;
{********************************** Sequence ****************************************** }
{*< Previous sequence block.                    }
{*< Next sequence block.                        }
{*< Index of the first element in the block +   }
{*< sequence->first->start_index.               }
{*< Number of elements in the block.            }
{*< Pointer to the first element of the block.  }
  PPCvSeqBlock = ^PCvSeqBlock;
  PCvSeqBlock = ^TCvSeqBlock;
  TCvSeqBlock = record
      prev : PCvSeqBlock;
      next : PCvSeqBlock;
      start_index : longint;
      count : longint;
      data : PIntPtr;
    end;
{*
   Read/Write sequence.
   Elements can be dynamically inserted to or deleted from the sequence.
 }
{*< Miscellaneous flags.      }  {*< Size of sequence header.  }  {*< Previous sequence.        }  {*< Next sequence.            }  {*< 2nd previous sequence.    }  {*< 2nd next sequence.        }  {*< Total number of elements.             }  {*< Size of sequence element in bytes.    }  {*< Maximal bound of the last block.      }  {*< Current write pointer.                }  {*< Grow seq this many at a time.         }  {*< Where the seq is stored.              }  {*< Free blocks list.                     }  {*< Pointer to the first sequence block.  }
  PPCvSeq = ^PCvSeq;
  PCvSeq = ^TCvSeq;
  TCvSeq = record
      flags : longint;
      header_size : longint;
      h_prev : PCvSeq;
      h_next : PCvSeq;
      v_prev : PCvSeq;
      v_next : PCvSeq;
      total : longint;
      elem_size : longint;
      block_max : PIntPtr;
      ptr : PIntPtr;
      delta_elems : longint;
      storage : PCvMemStorage;
      free_blocks : PCvSeqBlock;
      first : PCvSeqBlock;
    end;

  PPCvSetElem = ^PCvSetElem;
  PCvSetElem = ^TCvSetElem;
  TCvSetElem = record
      flags : longint;
      next_free : PCvSetElem;
    end;
{*< Miscellaneous flags.      }  {*< Size of sequence header.  }  {*< Previous sequence.        }  {*< Next sequence.            }  {*< 2nd previous sequence.    }  {*< 2nd next sequence.        }  {*< Total number of elements.             }  {*< Size of sequence element in bytes.    }  {*< Maximal bound of the last block.      }  {*< Current write pointer.                }  {*< Grow seq this many at a time.         }  {*< Where the seq is stored.              }  {*< Free blocks list.                     }  {*< Pointer to the first sequence block.  }

  TCvSet = record
      flags : longint;
      header_size : longint;
      h_prev : PCvSeq;
      h_next : PCvSeq;
      v_prev : PCvSeq;
      v_next : PCvSeq;
      total : longint;
      elem_size : longint;
      block_max : PIntPtr;
      ptr : PIntPtr;
      delta_elems : longint;
      storage : PCvMemStorage;
      free_blocks : PCvSeqBlock;
      first : PCvSeqBlock;
      free_elems : PCvSetElem;
      active_count : longint;
    end;
{************************************ Graph ******************************************* }
{* @name Graph

We represent a graph as a set of vertices. Vertices contain their adjacency lists (more exactly,
pointers to first incoming or outcoming edge (or 0 if isolated vertex)). Edges are stored in
another set. There is a singly-linked list of incoming/outcoming edges for each vertex.

Each edge consists of:

- Two pointers to the starting and ending vertices (vtx[0] and vtx[1] respectively).

    A graph may be oriented or not. In the latter case, edges between vertex i to vertex j are not
distinguished during search operations.

- Two pointers to next edges for the starting and ending vertices, where next[0] points to the
next edge in the vtx[0] adjacency list and next[1] points to the next edge in the vtx[1]
adjacency list.

@see CvGraphEdge, CvGraphVtx, CvGraphVtx2D, CvGraph
@
 }
  PPCvGraphVtx = ^PCvGraphVtx;
  PCvGraphVtx = ^TCvGraphVtx;
  PPCvGraphEdge = ^PCvGraphEdge;
  PCvGraphEdge = ^TCvGraphEdge;
  TCvGraphVtx = record
      flags : longint;
      first : PCvGraphEdge;
    end;

  TCvGraphEdge = record
      flags : longint;
      weight : single;
      next : array[0..1] of PCvGraphEdge;
      vtx : array[0..1] of PCvGraphVtx;
    end;


  PCvGraphVtx2D = ^TCvGraphVtx2D;
  TCvGraphVtx2D = record
      flags : longint;
      first : PCvGraphEdge;
      ptr : PCvPoint2D32f;
    end;
{*
   Graph is "derived" from the set (this is set a of vertices)
   and includes another set (edges)
 }
{*< Miscellaneous flags.      }  {*< Size of sequence header.  }  {*< Previous sequence.        }  {*< Next sequence.            }  {*< 2nd previous sequence.    }  {*< 2nd next sequence.        }  {*< Total number of elements.             }  {*< Size of sequence element in bytes.    }  {*< Maximal bound of the last block.      }  {*< Current write pointer.                }  {*< Grow seq this many at a time.         }  {*< Where the seq is stored.              }  {*< Free blocks list.                     }  {*< Pointer to the first sequence block.  }
  PCvGraph = ^TCvGraph;
  TCvGraph = record
      flags : longint;
      header_size : longint;
      h_prev : PCvSeq;
      h_next : PCvSeq;
      v_prev : PCvSeq;
      v_next : PCvSeq;
      total : longint;
      elem_size : longint;
      block_max : PIntPtr;
      ptr : PIntPtr;
      delta_elems : longint;
      storage : PCvMemStorage;
      free_blocks : PCvSeqBlock;
      first : PCvSeqBlock;
      free_elems : PCvSetElem;
      active_count : longint;
      edges : PCvSet;
    end;
{* @  }


{ current graph vertex (or current edge origin)  }
{ current graph edge destination vertex  }
{ current edge  }
{ the graph  }
{ the graph vertex stack  }
{ the lower bound of certainly visited vertices  }
{ event mask  }
  PPCvGraphScanner = ^PCvGraphScanner;
  PCvGraphScanner = ^TCvGraphScanner;
  TCvGraphScanner = record
      vtx : PCvGraphVtx;
      dst : PCvGraphVtx;
      edge : PCvGraphEdge;
      graph : PCvGraph;
      stack : PCvSeq;
      index : longint;
      mask : longint;
    end;


  PCvTreeNodeIterator = ^TCvTreeNodeIterator;
  TCvTreeNodeIterator = record
      node : pointer;
      level : longint;
      max_level : longint;
    end;
{********************************** Chain/Contour ************************************ }
{*< Miscellaneous flags.      }  {*< Size of sequence header.  }  {*< Previous sequence.        }  {*< Next sequence.            }  {*< 2nd previous sequence.    }  {*< 2nd next sequence.        }  {*< Total number of elements.             }  {*< Size of sequence element in bytes.    }  {*< Maximal bound of the last block.      }  {*< Current write pointer.                }  {*< Grow seq this many at a time.         }  {*< Where the seq is stored.              }  {*< Free blocks list.                     }  {*< Pointer to the first sequence block.  }


  PCvChain = ^TCvChain;
  TCvChain = record
      flags : longint;
      header_size : longint;
      h_prev : PCvSeq;
      h_next : PCvSeq;
      v_prev : PCvSeq;
      v_next : PCvSeq;
      total : longint;
      elem_size : longint;
      block_max : PIntPtr;
      ptr : PIntPtr;
      delta_elems : longint;
      storage : PCvMemStorage;
      free_blocks : PCvSeqBlock;
      first : PCvSeqBlock;
      origin : TCvPoint;
    end;
{*< Miscellaneous flags.      }  {*< Size of sequence header.  }  {*< Previous sequence.        }  {*< Next sequence.            }  {*< 2nd previous sequence.    }  {*< 2nd next sequence.        }  {*< Total number of elements.             }  {*< Size of sequence element in bytes.    }  {*< Maximal bound of the last block.      }  {*< Current write pointer.                }  {*< Grow seq this many at a time.         }  {*< Where the seq is stored.              }  {*< Free blocks list.                     }  {*< Pointer to the first sequence block.  }

  PCvContour = ^TCvContour;
  TCvContour = record
      flags : longint;
      header_size : longint;
      h_prev : PCvSeq;
      h_next : PCvSeq;
      v_prev : PCvSeq;
      v_next : PCvSeq;
      total : longint;
      elem_size : longint;
      block_max : PIntPtr;
      ptr : PIntPtr;
      delta_elems : longint;
      storage : PCvMemStorage;
      free_blocks : PCvSeqBlock;
      first : PCvSeqBlock;
      rect : TCvRect;
      color : longint;
      reserved : array[0..2] of longint;
    end;

  PCvPoint2DSeq = ^TCvPoint2DSeq;
  TCvPoint2DSeq = TCvContour;

  PCvSeqWriter = ^TCvSeqWriter;
  TCvSeqWriter = record
      header_size : longint;
      seq : PCvSeq;
      block : PCvSeqBlock;
      ptr : PIntPtr;
      block_min : PIntPtr;
      block_max : PIntPtr;
    end;
{*< sequence, beign read  }  {*< current block  }  {*< pointer to element be read next  }  {*< pointer to the beginning of block  }
{*< pointer to the end of block  }  {*< = seq->first->start_index    }  {*< pointer to previous element  }
  PCvSeqReader = ^TCvSeqReader;
  TCvSeqReader = record
      header_size : longint;
      seq : PCvSeq;
      block : PCvSeqBlock;
      ptr : PIntPtr;
      block_min : PIntPtr;
      block_max : PIntPtr;
      delta_index : longint;
      prev_elem : PIntPtr;
    end;

  // a < b ? -1 : a > b ? 1 : 0
  TCvCmpFunc = function (const a, b, userdata: pointer):longint;//CV_CDCEL;

  TCv_iplCreateImageHeader = function (a, b, c:longint; d, e: PChar; f, g, h, i, j:longint;
                              ROI: PIplROI; iplim: PIplImage; k:pointer; info: PIplTileInfo):PIplImage;//CV_STDCALL;

  TCv_iplAllocateImageData = procedure(iplIm :PIplImage; a,b:longint); // CV_STDCALL;
  TCv_iplDeallocate = procedure(iplIm:PIplImage; a:longint); //CV_STDCALL;

  TCv_iplCreateROI = function (a,b,c,d,e: longint):PIplROI;//CV_STDCALL;
  TCv_iplCloneImage = function(iplIm: PIplImage) :PIplImage;// CV_STDCALL;

  TCvErrorCallback = function(startus :longint; const func_name, err_msg, file_name: PChar; line:longint; userdata:Pointer) : longint;// CL_DECL;

{.$ifdef CV_IMGPROC}

const
  CV_INPAINT_NS      =0;
  CV_INPAINT_TELEA   =1;


  CV_LOAD_IMAGE_UNCHANGED = -(1);
  CV_LOAD_IMAGE_GRAYSCALE = 0;
  CV_LOAD_IMAGE_COLOR = 1;
  CV_LOAD_IMAGE_ANYDEPTH = 2;
  CV_LOAD_IMAGE_ANYCOLOR = 4;
  CV_LOAD_IMAGE_IGNORE_ORIENTATION = 128;

  CV_IMWRITE_JPEG_QUALITY = 1;
  CV_IMWRITE_JPEG_PROGRESSIVE = 2;
  CV_IMWRITE_JPEG_OPTIMIZE = 3;
  CV_IMWRITE_JPEG_RST_INTERVAL = 4;
  CV_IMWRITE_JPEG_LUMA_QUALITY = 5;
  CV_IMWRITE_JPEG_CHROMA_QUALITY = 6;
  CV_IMWRITE_PNG_COMPRESSION = 16;
  CV_IMWRITE_PNG_STRATEGY = 17;
  CV_IMWRITE_PNG_BILEVEL = 18;
  CV_IMWRITE_PNG_STRATEGY_DEFAULT = 0;
  CV_IMWRITE_PNG_STRATEGY_FILTERED = 1;
  CV_IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY = 2;
  CV_IMWRITE_PNG_STRATEGY_RLE = 3;
  CV_IMWRITE_PNG_STRATEGY_FIXED = 4;
  CV_IMWRITE_PXM_BINARY = 32;
  CV_IMWRITE_EXR_TYPE = 48;
  CV_IMWRITE_WEBP_QUALITY = 64;
  CV_IMWRITE_PAM_TUPLETYPE = 128;
  CV_IMWRITE_PAM_FORMAT_NULL = 0;
  CV_IMWRITE_PAM_FORMAT_BLACKANDWHITE = 1;
  CV_IMWRITE_PAM_FORMAT_GRAYSCALE = 2;
  CV_IMWRITE_PAM_FORMAT_GRAYSCALE_ALPHA = 3;
  CV_IMWRITE_PAM_FORMAT_RGB = 4;
  CV_IMWRITE_PAM_FORMAT_RGB_ALPHA = 5;


  CV_GAUSSIAN_5x5 = 7;

  CV_SCHARR = -1;
  CV_MAX_SOBEL_KSIZE = 7;


  CV_BGR2BGRA = 0;
  CV_RGB2RGBA = CV_BGR2BGRA;
  CV_BGRA2BGR = 1;
  CV_RGBA2RGB = CV_BGRA2BGR;
  CV_BGR2RGBA = 2;
  CV_RGB2BGRA = CV_BGR2RGBA;
  CV_RGBA2BGR = 3;
  CV_BGRA2RGB = CV_RGBA2BGR;
  CV_BGR2RGB = 4;
  CV_RGB2BGR = CV_BGR2RGB;
  CV_BGRA2RGBA = 5;
  CV_RGBA2BGRA = CV_BGRA2RGBA;
  CV_BGR2GRAY = 6;
  CV_RGB2GRAY = 7;
  CV_GRAY2BGR = 8;
  CV_GRAY2RGB = CV_GRAY2BGR;
  CV_GRAY2BGRA = 9;
  CV_GRAY2RGBA = CV_GRAY2BGRA;
  CV_BGRA2GRAY = 10;
  CV_RGBA2GRAY = 11;
  CV_BGR2BGR565 = 12;
  CV_RGB2BGR565 = 13;
  CV_BGR5652BGR = 14;
  CV_BGR5652RGB = 15;
  CV_BGRA2BGR565 = 16;
  CV_RGBA2BGR565 = 17;
  CV_BGR5652BGRA = 18;
  CV_BGR5652RGBA = 19;
  CV_GRAY2BGR565 = 20;
  CV_BGR5652GRAY = 21;
  CV_BGR2BGR555 = 22;
  CV_RGB2BGR555 = 23;
  CV_BGR5552BGR = 24;
  CV_BGR5552RGB = 25;
  CV_BGRA2BGR555 = 26;
  CV_RGBA2BGR555 = 27;
  CV_BGR5552BGRA = 28;
  CV_BGR5552RGBA = 29;
  CV_GRAY2BGR555 = 30;
  CV_BGR5552GRAY = 31;
  CV_BGR2XYZ = 32;
  CV_RGB2XYZ = 33;
  CV_XYZ2BGR = 34;
  CV_XYZ2RGB = 35;
  CV_BGR2YCrCb = 36;
  CV_RGB2YCrCb = 37;
  CV_YCrCb2BGR = 38;
  CV_YCrCb2RGB = 39;
  CV_BGR2HSV = 40;
  CV_RGB2HSV = 41;
  CV_BGR2Lab = 44;
  CV_RGB2Lab = 45;
  CV_BayerBG2BGR = 46;
  CV_BayerGB2BGR = 47;
  CV_BayerRG2BGR = 48;
  CV_BayerGR2BGR = 49;
  CV_BayerBG2RGB = CV_BayerRG2BGR;
  CV_BayerGB2RGB = CV_BayerGR2BGR;
  CV_BayerRG2RGB = CV_BayerBG2BGR;
  CV_BayerGR2RGB = CV_BayerGB2BGR;
  CV_BGR2Luv = 50;
  CV_RGB2Luv = 51;
  CV_BGR2HLS = 52;
  CV_RGB2HLS = 53;
  CV_HSV2BGR = 54;
  CV_HSV2RGB = 55;
  CV_Lab2BGR = 56;
  CV_Lab2RGB = 57;
  CV_Luv2BGR = 58;
  CV_Luv2RGB = 59;
  CV_HLS2BGR = 60;
  CV_HLS2RGB = 61;
  CV_BayerBG2BGR_VNG = 62;
  CV_BayerGB2BGR_VNG = 63;
  CV_BayerRG2BGR_VNG = 64;
  CV_BayerGR2BGR_VNG = 65;
  CV_BayerBG2RGB_VNG = CV_BayerRG2BGR_VNG;
  CV_BayerGB2RGB_VNG = CV_BayerGR2BGR_VNG;
  CV_BayerRG2RGB_VNG = CV_BayerBG2BGR_VNG;
  CV_BayerGR2RGB_VNG = CV_BayerGB2BGR_VNG;
  CV_BGR2HSV_FULL = 66;
  CV_RGB2HSV_FULL = 67;
  CV_BGR2HLS_FULL = 68;
  CV_RGB2HLS_FULL = 69;
  CV_HSV2BGR_FULL = 70;
  CV_HSV2RGB_FULL = 71;
  CV_HLS2BGR_FULL = 72;
  CV_HLS2RGB_FULL = 73;
  CV_LBGR2Lab = 74;
  CV_LRGB2Lab = 75;
  CV_LBGR2Luv = 76;
  CV_LRGB2Luv = 77;
  CV_Lab2LBGR = 78;
  CV_Lab2LRGB = 79;
  CV_Luv2LBGR = 80;
  CV_Luv2LRGB = 81;
  CV_BGR2YUV = 82;
  CV_RGB2YUV = 83;
  CV_YUV2BGR = 84;
  CV_YUV2RGB = 85;
  CV_BayerBG2GRAY = 86;
  CV_BayerGB2GRAY = 87;
  CV_BayerRG2GRAY = 88;
  CV_BayerGR2GRAY = 89;
  CV_YUV2RGB_NV12 = 90;
  CV_YUV2BGR_NV12 = 91;
  CV_YUV2RGB_NV21 = 92;
  CV_YUV2BGR_NV21 = 93;
  CV_YUV420sp2RGB = CV_YUV2RGB_NV21;
  CV_YUV420sp2BGR = CV_YUV2BGR_NV21;
  CV_YUV2RGBA_NV12 = 94;
  CV_YUV2BGRA_NV12 = 95;
  CV_YUV2RGBA_NV21 = 96;
  CV_YUV2BGRA_NV21 = 97;
  CV_YUV420sp2RGBA = CV_YUV2RGBA_NV21;
  CV_YUV420sp2BGRA = CV_YUV2BGRA_NV21;
  CV_YUV2RGB_YV12 = 98;
  CV_YUV2BGR_YV12 = 99;
  CV_YUV2RGB_IYUV = 100;
  CV_YUV2BGR_IYUV = 101;
  CV_YUV2RGB_I420 = CV_YUV2RGB_IYUV;
  CV_YUV2BGR_I420 = CV_YUV2BGR_IYUV;
  CV_YUV420p2RGB = CV_YUV2RGB_YV12;
  CV_YUV420p2BGR = CV_YUV2BGR_YV12;
  CV_YUV2RGBA_YV12 = 102;
  CV_YUV2BGRA_YV12 = 103;
  CV_YUV2RGBA_IYUV = 104;
  CV_YUV2BGRA_IYUV = 105;
  CV_YUV2RGBA_I420 = CV_YUV2RGBA_IYUV;
  CV_YUV2BGRA_I420 = CV_YUV2BGRA_IYUV;
  CV_YUV420p2RGBA = CV_YUV2RGBA_YV12;
  CV_YUV420p2BGRA = CV_YUV2BGRA_YV12;
  CV_YUV2GRAY_420 = 106;
  CV_YUV2GRAY_NV21 = CV_YUV2GRAY_420;
  CV_YUV2GRAY_NV12 = CV_YUV2GRAY_420;
  CV_YUV2GRAY_YV12 = CV_YUV2GRAY_420;
  CV_YUV2GRAY_IYUV = CV_YUV2GRAY_420;
  CV_YUV2GRAY_I420 = CV_YUV2GRAY_420;
  CV_YUV420sp2GRAY = CV_YUV2GRAY_420;
  CV_YUV420p2GRAY = CV_YUV2GRAY_420;
  CV_YUV2RGB_UYVY = 107;
  CV_YUV2BGR_UYVY = 108;
  CV_YUV2RGB_Y422 = CV_YUV2RGB_UYVY;
  CV_YUV2BGR_Y422 = CV_YUV2BGR_UYVY;
  CV_YUV2RGB_UYNV = CV_YUV2RGB_UYVY;
  CV_YUV2BGR_UYNV = CV_YUV2BGR_UYVY;
  CV_YUV2RGBA_UYVY = 111;
  CV_YUV2BGRA_UYVY = 112;
  CV_YUV2RGBA_Y422 = CV_YUV2RGBA_UYVY;
  CV_YUV2BGRA_Y422 = CV_YUV2BGRA_UYVY;
  CV_YUV2RGBA_UYNV = CV_YUV2RGBA_UYVY;
  CV_YUV2BGRA_UYNV = CV_YUV2BGRA_UYVY;
  CV_YUV2RGB_YUY2 = 115;
  CV_YUV2BGR_YUY2 = 116;
  CV_YUV2RGB_YVYU = 117;
  CV_YUV2BGR_YVYU = 118;
  CV_YUV2RGB_YUYV = CV_YUV2RGB_YUY2;
  CV_YUV2BGR_YUYV = CV_YUV2BGR_YUY2;
  CV_YUV2RGB_YUNV = CV_YUV2RGB_YUY2;
  CV_YUV2BGR_YUNV = CV_YUV2BGR_YUY2;
  CV_YUV2RGBA_YUY2 = 119;
  CV_YUV2BGRA_YUY2 = 120;
  CV_YUV2RGBA_YVYU = 121;
  CV_YUV2BGRA_YVYU = 122;
  CV_YUV2RGBA_YUYV = CV_YUV2RGBA_YUY2;
  CV_YUV2BGRA_YUYV = CV_YUV2BGRA_YUY2;
  CV_YUV2RGBA_YUNV = CV_YUV2RGBA_YUY2;
  CV_YUV2BGRA_YUNV = CV_YUV2BGRA_YUY2;
  CV_YUV2GRAY_UYVY = 123;
  CV_YUV2GRAY_YUY2 = 124;
  CV_YUV2GRAY_Y422 = CV_YUV2GRAY_UYVY;
  CV_YUV2GRAY_UYNV = CV_YUV2GRAY_UYVY;
  CV_YUV2GRAY_YVYU = CV_YUV2GRAY_YUY2;
  CV_YUV2GRAY_YUYV = CV_YUV2GRAY_YUY2;
  CV_YUV2GRAY_YUNV = CV_YUV2GRAY_YUY2;
  CV_RGBA2mRGBA = 125;
  CV_mRGBA2RGBA = 126;
  CV_RGB2YUV_I420 = 127;
  CV_BGR2YUV_I420 = 128;
  CV_RGB2YUV_IYUV = CV_RGB2YUV_I420;
  CV_BGR2YUV_IYUV = CV_BGR2YUV_I420;
  CV_RGBA2YUV_I420 = 129;
  CV_BGRA2YUV_I420 = 130;
  CV_RGBA2YUV_IYUV = CV_RGBA2YUV_I420;
  CV_BGRA2YUV_IYUV = CV_BGRA2YUV_I420;
  CV_RGB2YUV_YV12 = 131;
  CV_BGR2YUV_YV12 = 132;
  CV_RGBA2YUV_YV12 = 133;
  CV_BGRA2YUV_YV12 = 134;
  CV_BayerBG2BGR_EA = 135;
  CV_BayerGB2BGR_EA = 136;
  CV_BayerRG2BGR_EA = 137;
  CV_BayerGR2BGR_EA = 138;
  CV_BayerBG2RGB_EA = CV_BayerRG2BGR_EA;
  CV_BayerGB2RGB_EA = CV_BayerGR2BGR_EA;
  CV_BayerRG2RGB_EA = CV_BayerBG2BGR_EA;
  CV_BayerGR2RGB_EA = CV_BayerGB2BGR_EA;
  CV_BayerBG2BGRA = 139;
  CV_BayerGB2BGRA = 140;
  CV_BayerRG2BGRA = 141;
  CV_BayerGR2BGRA = 142;
  CV_BayerBG2RGBA = CV_BayerRG2BGRA;
  CV_BayerGB2RGBA = CV_BayerGR2BGRA;
  CV_BayerRG2RGBA = CV_BayerBG2BGRA;
  CV_BayerGR2RGBA = CV_BayerGB2BGRA;
  CV_COLORCVT_MAX = 143;


  CV_INTER_NN = 0;
  CV_INTER_LINEAR = 1;
  CV_INTER_CUBIC = 2;
  CV_INTER_AREA = 3;
  CV_INTER_LANCZOS4 = 4;


  CV_WARP_FILL_OUTLIERS = 8;
  CV_WARP_INVERSE_MAP = 16;


  CV_SHAPE_RECT = 0;
  CV_SHAPE_CROSS = 1;
  CV_SHAPE_ELLIPSE = 2;
  CV_SHAPE_CUSTOM = 100;


  CV_MOP_ERODE = 0;
  CV_MOP_DILATE = 1;
  CV_MOP_OPEN = 2;
  CV_MOP_CLOSE = 3;
  CV_MOP_GRADIENT = 4;
  CV_MOP_TOPHAT = 5;
  CV_MOP_BLACKHAT = 6;

  CV_TM_SQDIFF = 0;
  CV_TM_SQDIFF_NORMED = 1;
  CV_TM_CCORR = 2;
  CV_TM_CCORR_NORMED = 3;
  CV_TM_CCOEFF = 4;
  CV_TM_CCOEFF_NORMED = 5;

  CV_RETR_EXTERNAL = 0;
  CV_RETR_LIST = 1;
  CV_RETR_CCOMP = 2;
  CV_RETR_TREE = 3;
  CV_RETR_FLOODFILL = 4;

  CV_CHAIN_CODE = 0;
  CV_CHAIN_APPROX_NONE = 1;
  CV_CHAIN_APPROX_SIMPLE = 2;
  CV_CHAIN_APPROX_TC89_L1 = 3;
  CV_CHAIN_APPROX_TC89_KCOS = 4;
  CV_LINK_RUNS = 5;

  CV_POLY_APPROX_DP = 0;


  CV_CONTOURS_MATCH_I1 = 1;
  CV_CONTOURS_MATCH_I2 = 2;
  CV_CONTOURS_MATCH_I3 = 3;


  CV_CLOCKWISE = 1;
  CV_COUNTER_CLOCKWISE = 2;

  CV_COMP_CORREL = 0;
  CV_COMP_CHISQR = 1;
  CV_COMP_INTERSECT = 2;
  CV_COMP_BHATTACHARYYA = 3;
  CV_COMP_HELLINGER = CV_COMP_BHATTACHARYYA;
  CV_COMP_CHISQR_ALT = 4;
  CV_COMP_KL_DIV = 5;


  CV_DIST_MASK_3 = 3;
  CV_DIST_MASK_5 = 5;
  CV_DIST_MASK_PRECISE = 0;


  CV_DIST_LABEL_CCOMP = 0;
  CV_DIST_LABEL_PIXEL = 1;


  CV_DIST_USER = -(1);
  CV_DIST_L1 = 1;
  CV_DIST_L2 = 2;
  CV_DIST_C = 3;
  CV_DIST_L12 = 4;
  CV_DIST_FAIR = 5;
  CV_DIST_WELSCH = 6;
  CV_DIST_HUBER = 7;

  CV_THRESH_BINARY = 0;
  CV_THRESH_BINARY_INV = 1;
  CV_THRESH_TRUNC = 2;
  CV_THRESH_TOZERO = 3;
  CV_THRESH_TOZERO_INV = 4;
  CV_THRESH_MASK = 7;
  CV_THRESH_OTSU = 8;
  CV_THRESH_TRIANGLE = 16;


  CV_ADAPTIVE_THRESH_MEAN_C = 0;
  CV_ADAPTIVE_THRESH_GAUSSIAN_C = 1;


  CV_FLOODFILL_FIXED_RANGE = 1 shl 16;
  CV_FLOODFILL_MASK_ONLY = 1 shl 17;


  CV_CANNY_L2_GRADIENT = 1 shl 31;


  CV_HOUGH_STANDARD = 0;
  CV_HOUGH_PROBABILISTIC = 1;
  CV_HOUGH_MULTI_SCALE = 2;
  CV_HOUGH_GRADIENT = 3;


type
  TSmoothMethod_c = (
    CV_BLUR_NO_SCALE = 0,
    CV_BLUR = 1,
    CV_GAUSSIAN = 2,
    CV_MEDIAN = 3,
    CV_BILATERAL = 4
  );

  PCvConnectedComp = ^TCvConnectedComp;
  TCvConnectedComp = record
      area : double;
      value : TCvScalar;
      rect : TCvRect;
      contour : PCvSeq;
    end;

  PCvMoments = ^TCvMoments;
  TCvMoments = record
      m00 : double;
      m10 : double;
      m01 : double;
      m20 : double;
      m11 : double;
      m02 : double;
      m30 : double;
      m21 : double;
      m12 : double;
      m03 : double;
      mu20 : double;
      mu11 : double;
      mu02 : double;
      mu30 : double;
      mu21 : double;
      mu12 : double;
      mu03 : double;
      inv_sqrt_m00 : double;
    end;

  PCvHuMoments = ^TCvHuMoments;
  TCvHuMoments = record
      hu1 : double;
      hu2 : double;
      hu3 : double;
      hu4 : double;
      hu5 : double;
      hu6 : double;
      hu7 : double;
    end;

  TCvDistanceFunction = function (a:PSingle; b:PSingle; user_param:pointer):single;//CV_CDECL:


  PCvContourScanner = ^TCvContourScanner;
  TCvContourScanner = Pointer;


  PCvChainPtReader = ^TCvChainPtReader;
  TCvChainPtReader = record
      header_size : longint;
      seq : PCvSeq;
      block : PCvSeqBlock;
      ptr : PIntPtr;
      block_min : PIntPtr;
      block_max : PIntPtr;
      delta_index : longint;
      prev_elem : PIntPtr;
      code : char;
      pt : TCvPoint;
      deltas : array[0..7] of array[0..1] of IntPtr;
    end;


  PCvConvexityDefect = ^TCvConvexityDefect;
  TCvConvexityDefect = record
      start : PCvPoint;
      _end : PCvPoint;
      depth_point : PCvPoint;
      depth : single;
    end;

  PCvFeatureTree = ^TCvFeatureTree;
  TCvFeatureTree = record
      {undefined structure}
    end;

  PCvLSH = ^TCvLSH;
  TCvLSH = record
      {undefined structure}
    end;

  PCvLSHOperations = ^TCvLSHOperations;
  TCvLSHOperations = record
      {undefined structure}
    end;
  TCvBoxPoint2d32f= array [0..3] of TCVPoint2d32f;

procedure cvAcc(image:PCvArr; sum:PCvArr; mask:PCvArr);winapi;external;

procedure cvSquareAcc(image:PCvArr; sqsum:PCvArr; mask:PCvArr);winapi;external;

procedure cvMultiplyAcc(image1:PCvArr; image2:PCvArr; acc:PCvArr; mask:PCvArr);winapi;external;

procedure cvRunningAvg(image:PCvArr; acc:PCvArr; alpha:double; mask:PCvArr);winapi;external;


procedure cvCopyMakeBorder(src:PCvArr; dst:PCvArr; offset:TCvPoint; bordertype:longint; value:TCvScalar);winapi;external;

procedure cvSmooth(src:PCvArr; dst:PCvArr; smoothtype:longint; size1:longint; size2:longint;
            sigma1:double; sigma2:double);winapi;external;

procedure cvFilter2D(src:PCvArr; dst:PCvArr; kernel:PCvMat; anchor:TCvPoint);winapi;external;

procedure cvIntegral(image:PCvArr; sum:PCvArr; sqsum:PCvArr; tilted_sum:PCvArr);winapi;external;

procedure cvPyrDown(src:PCvArr; dst:PCvArr; filter:longint);winapi;external;

procedure cvPyrUp(src:PCvArr; dst:PCvArr; filter:longint);winapi;external;

function cvCreatePyramid(img:PCvArr; extra_layers:longint; rate:double; layer_sizes:PCvSize; bufarr:PCvArr;
           calc:longint; filter:longint):PPCvMat;winapi;external;

procedure cvReleasePyramid(pyramid:PPPCvMat; extra_layers:longint);winapi;external;

procedure cvPyrMeanShiftFiltering(src:PCvArr; dst:PCvArr; sp:double; sr:double; max_level:longint;
            termcrit:TCvTermCriteria);winapi;external;

procedure cvWatershed(image:PCvArr; markers:PCvArr);winapi;external;

procedure cvSobel(src:PCvArr; dst:PCvArr; xorder:longint; yorder:longint; aperture_size:longint);winapi;external;

procedure cvLaplace(src:PCvArr; dst:PCvArr; aperture_size:longint);winapi;external;

procedure cvCvtColor(src:PCvArr; dst:PCvArr; code:longint);winapi;external;

procedure cvResize(src:PCvArr; dst:PCvArr; interpolation:longint);winapi;external;

procedure cvWarpAffine(src:PCvArr; dst:PCvArr; map_matrix:PCvMat; flags:longint; fillval:TCvScalar);winapi;external;

function cvGetAffineTransform(src:PCvPoint2D32f; dst:PCvPoint2D32f; map_matrix:PCvMat):PCvMat;winapi;external;

function cv2DRotationMatrix(center:TCvPoint2D32f; angle:double; scale:double; map_matrix:PCvMat):PCvMat;winapi;external;

procedure cvWarpPerspective(src:PCvArr; dst:PCvArr; map_matrix:PCvMat; flags:longint; fillval:TCvScalar);winapi;external;

function cvGetPerspectiveTransform(src:PCvPoint2D32f; dst:PCvPoint2D32f; map_matrix:PCvMat):PCvMat;winapi;external;

procedure cvRemap(src:PCvArr; dst:PCvArr; mapx:PCvArr; mapy:PCvArr; flags:longint;
            fillval:TCvScalar);winapi;external;

procedure cvConvertMaps(mapx:PCvArr; mapy:PCvArr; mapxy:PCvArr; mapalpha:PCvArr);winapi;external;

procedure cvLogPolar(src:PCvArr; dst:PCvArr; center:TCvPoint2D32f; M:double; flags:longint);winapi;external;

procedure cvLinearPolar(src:PCvArr; dst:PCvArr; center:TCvPoint2D32f; maxRadius:double; flags:longint);winapi;external;

function cvCreateStructuringElementEx(cols:longint; rows:longint; anchor_x:longint; anchor_y:longint; shape:longint;
           values:Plongint):PIplConvKernel;winapi;external;

procedure cvReleaseStructuringElement(element:PPIplConvKernel);winapi;external;

procedure cvErode(src:PCvArr; dst:PCvArr; element:PIplConvKernel; iterations:longint);winapi;external;

procedure cvDilate(src:PCvArr; dst:PCvArr; element:PIplConvKernel; iterations:longint);winapi;external;

procedure cvMorphologyEx(src:PCvArr; dst:PCvArr; temp:PCvArr; element:PIplConvKernel; operation:longint;
            iterations:longint);winapi;external;

procedure cvMoments(arr:PCvArr; moments:PCvMoments; binary:longint);winapi;external;

function cvGetSpatialMoment(moments:PCvMoments; x_order:longint; y_order:longint):double;winapi;external;

function cvGetCentralMoment(moments:PCvMoments; x_order:longint; y_order:longint):double;winapi;external;

function cvGetNormalizedCentralMoment(moments:PCvMoments; x_order:longint; y_order:longint):double;winapi;external;

procedure cvGetHuMoments(moments:PCvMoments; hu_moments:PCvHuMoments);winapi;external;


function cvSampleLine(image:PCvArr; pt1:TCvPoint; pt2:TCvPoint; buffer:pointer; connectivity:longint):longint;winapi;external;

procedure cvGetRectSubPix(src:PCvArr; dst:PCvArr; center:TCvPoint2D32f);winapi;external;

procedure cvGetQuadrangleSubPix(src:PCvArr; dst:PCvArr; map_matrix:PCvMat);winapi;external;

procedure cvMatchTemplate(image:PCvArr; templ:PCvArr; result:PCvArr; method:longint);winapi;external;

function cvCalcEMD2(signature1:PCvArr; signature2:PCvArr; distance_type:longint; distance_func:TCvDistanceFunction; cost_matrix:PCvArr;
           flow:PCvArr; lower_bound:Psingle; userdata:pointer):single;winapi;external;


function cvFindContours(image:PCvArr; storage:PCvMemStorage; first_contour:PPCvSeq; header_size:longint; mode:longint;
           method:longint; offset:TCvPoint):longint;winapi;external;

function cvStartFindContours(image:PCvArr; storage:PCvMemStorage; header_size:longint; mode:longint; method:longint;
           offset:TCvPoint):TCvContourScanner;winapi;external;

function cvFindNextContour(scanner:TCvContourScanner):PCvSeq;winapi;external;

procedure cvSubstituteContour(scanner:TCvContourScanner; new_contour:PCvSeq);winapi;external;

function cvEndFindContours(scanner:PCvContourScanner):PCvSeq;winapi;external;

function cvApproxChains(src_seq:PCvSeq; storage:PCvMemStorage; method:longint; parameter:double; minimal_perimeter:longint;
           recursive:longint):PCvSeq;winapi;external;

procedure cvStartReadChainPoints(chain:PCvChain; reader:PCvChainPtReader);winapi;external;

function cvReadChainPoint(reader:PCvChainPtReader):TCvPoint;winapi;external;


function cvApproxPoly(src_seq:pointer; header_size:longint; storage:PCvMemStorage; method:longint; eps:double;
           recursive:longint):PCvSeq;winapi;external;

function cvArcLength(curve:pointer; slice:TCvSlice; is_closed:longint):double;winapi;external;


function cvBoundingRect(points:PCvArr; update:longint):TCvRect;winapi;external;

function cvContourArea(contour:PCvArr; slice:TCvSlice; oriented:longint):double;winapi;external;

function cvMinAreaRect2(points:PCvArr; storage:PCvMemStorage):TCvBox2D;winapi;external;

function cvMinEnclosingCircle(points:PCvArr; center:PCvPoint2D32f; radius:Psingle):longint;winapi;external;

function cvMatchShapes(object1:pointer; object2:pointer; method:longint; parameter:double):double;winapi;external;

function cvConvexHull2(input:PCvArr; hull_storage:pointer; orientation:longint; return_points:longint):PCvSeq;winapi;external;

function cvCheckContourConvexity(contour:PCvArr):longint;winapi;external;

function cvConvexityDefects(contour:PCvArr; convexhull:PCvArr; storage:PCvMemStorage):PCvSeq;winapi;external;

function cvFitEllipse2(points:PCvArr):TCvBox2D;winapi;external;

function cvMaxRect(rect1:PCvRect; rect2:PCvRect):TCvRect;winapi;external;

procedure cvBoxPoints(box:TCvBox2D; pt:TCVBoxPoint2d32f);winapi;external;

function cvPointSeqFromMat(seq_kind:longint; mat:PCvArr; contour_header:PCvContour; block:PCvSeqBlock):PCvSeq;winapi;external;

function cvPointPolygonTest(contour:PCvArr; pt:TCvPoint2D32f; measure_dist:longint):double;winapi;external;


function cvCreateHist(dims:longint; sizes:Plongint; _type:longint; ranges:PPsingle; uniform:longint):PCvHistogram;winapi;external;

procedure cvSetHistBinRanges(hist:PCvHistogram; ranges:PPsingle; uniform:longint);winapi;external;

function cvMakeHistHeaderForArray(dims:longint; sizes:Plongint; hist:PCvHistogram; data:Psingle; ranges:PPsingle;
           uniform:longint):PCvHistogram;winapi;external;

procedure cvReleaseHist(hist:PPCvHistogram);winapi;external;

procedure cvClearHist(hist:PCvHistogram);winapi;external;

procedure cvGetMinMaxHistValue(hist:PCvHistogram; min_value:Psingle; max_value:Psingle; min_idx:Plongint; max_idx:Plongint);winapi;external;

procedure cvNormalizeHist(hist:PCvHistogram; factor:double);winapi;external;

procedure cvThreshHist(hist:PCvHistogram; threshold:double);winapi;external;

function cvCompareHist(hist1:PCvHistogram; hist2:PCvHistogram; method:longint):double;winapi;external;

procedure cvCopyHist(src:PCvHistogram; dst:PPCvHistogram);winapi;external;

procedure cvCalcBayesianProb(src:PPCvHistogram; number:longint; dst:PPCvHistogram);winapi;external;

procedure cvCalcArrHist(arr:PPCvArr; hist:PCvHistogram; accumulate:longint; mask:PCvArr);winapi;external;


procedure cvCalcArrBackProject(image:PPCvArr; dst:PCvArr; hist:PCvHistogram);winapi;external;

procedure cvCalcArrBackProjectPatch(image:PPCvArr; dst:PCvArr; range:TCvSize; hist:PCvHistogram; method:longint;
            factor:double);winapi;external;

procedure cvCalcProbDensity(hist1:PCvHistogram; hist2:PCvHistogram; dst_hist:PCvHistogram; scale:double);winapi;external;

procedure cvEqualizeHist(src:PCvArr; dst:PCvArr);winapi;external;

procedure cvDistTransform(src:PCvArr; dst:PCvArr; distance_type:longint; mask_size:longint; mask:Psingle;
            labels:PCvArr; labelType:longint);winapi;external;

function cvThreshold(src:PCvArr; dst:PCvArr; threshold:double; max_value:double; threshold_type:longint):double;winapi;external;

procedure cvAdaptiveThreshold(src:PCvArr; dst:PCvArr; max_value:double; adaptive_method:longint; threshold_type:longint;
            block_size:longint; param1:double);winapi;external;

procedure cvFloodFill(image:PCvArr; seed_point:TCvPoint; new_val:TCvScalar; lo_diff:TCvScalar; up_diff:TCvScalar;
            comp:PCvConnectedComp; flags:longint; mask:PCvArr);winapi;external;


procedure cvCanny(image:PCvArr; edges:PCvArr; threshold1:double; threshold2:double; aperture_size:longint);winapi;external;

procedure cvPreCornerDetect(image:PCvArr; corners:PCvArr; aperture_size:longint);winapi;external;

procedure cvCornerEigenValsAndVecs(image:PCvArr; eigenvv:PCvArr; block_size:longint; aperture_size:longint);winapi;external;

procedure cvCornerMinEigenVal(image:PCvArr; eigenval:PCvArr; block_size:longint; aperture_size:longint);winapi;external;

procedure cvCornerHarris(image:PCvArr; harris_response:PCvArr; block_size:longint; aperture_size:longint; k:double);winapi;external;

procedure cvFindCornerSubPix(image:PCvArr; corners:PCvPoint2D32f; count:longint; win:TCvSize; zero_zone:TCvSize;
            criteria:TCvTermCriteria);winapi;external;

procedure cvGoodFeaturesToTrack(image:PCvArr; eig_image:PCvArr; temp_image:PCvArr; corners:PCvPoint2D32f; corner_count:Plongint;
            quality_level:double; min_distance:double; mask:PCvArr; block_size:longint; use_harris:longint;
            k:double);winapi;external;

function cvHoughLines2(image:PCvArr; line_storage:pointer; method:longint; rho:double; theta:double;
           threshold:longint; param1:double; param2:double; min_theta:double; max_theta:double):PCvSeq;winapi;external;

function cvHoughCircles(image:PCvArr; circle_storage:pointer; method:longint; dp:double; min_dist:double;
           param1:double; param2:double; min_radius:longint; max_radius:longint):PCvSeq;winapi;external;

procedure cvFitLine(points:PCvArr; dist_type:longint; param:double; reps:double; aeps:double;
            line:Psingle);winapi;external;


const
  CV_FILLED = -(1);
  CV_AA = 16;


procedure cvLine(img:PCvArr; pt1:TCvPoint; pt2:TCvPoint; color:TCvScalar; thickness:longint;
            line_type:longint; shift:longint);winapi;external;

procedure cvRectangle(img:PCvArr; pt1:TCvPoint; pt2:TCvPoint; color:TCvScalar; thickness:longint;
            line_type:longint; shift:longint);winapi;external;

procedure cvRectangleR(img:PCvArr; r:TCvRect; color:TCvScalar; thickness:longint; line_type:longint;
            shift:longint);winapi;external;

procedure cvCircle(img:PCvArr; center:TCvPoint; radius:longint; color:TCvScalar; thickness:longint;
            line_type:longint; shift:longint);winapi;external;

procedure cvEllipse(img:PCvArr; center:TCvPoint; axes:TCvSize; angle:double; start_angle:double;
            end_angle:double; color:TCvScalar; thickness:longint; line_type:longint; shift:longint);winapi;external;

procedure cvFillConvexPoly(img:PCvArr; pts:PCvPoint; npts:longint; color:TCvScalar; line_type:longint;
            shift:longint);winapi;external;

procedure cvFillPoly(img:PCvArr; pts:PCvPoint; npts:Plongint; contours:longint; color:TCvScalar;
            line_type:longint; shift:longint);winapi;external;

procedure cvPolyLine(img:PCvArr; pts:PPCvPoint; npts:Plongint; contours:longint; is_closed:longint;
            color:TCvScalar; thickness:longint; line_type:longint; shift:longint);winapi;external;


function cvClipLine(img_size:TCvSize; pt1:PCvPoint; pt2:PCvPoint):longint;winapi;external;

function cvInitLineIterator(image:PCvArr; pt1:TCvPoint; pt2:TCvPoint; line_iterator:PCvLineIterator; connectivity:longint;
           left_to_right:longint):longint;winapi;external;
const
  CV_FONT_HERSHEY_SIMPLEX = 0;
  CV_FONT_HERSHEY_PLAIN = 1;
  CV_FONT_HERSHEY_DUPLEX = 2;
  CV_FONT_HERSHEY_COMPLEX = 3;
  CV_FONT_HERSHEY_TRIPLEX = 4;
  CV_FONT_HERSHEY_COMPLEX_SMALL = 5;
  CV_FONT_HERSHEY_SCRIPT_SIMPLEX = 6;
  CV_FONT_HERSHEY_SCRIPT_COMPLEX = 7;
  CV_FONT_ITALIC = 16;
  CV_FONT_VECTOR0 = CV_FONT_HERSHEY_SIMPLEX;

type
  PCvFont = ^TCvFont;
  TCvFont = record
      nameFont : Pchar;
      color : TCvScalar;
      font_face : longint;
      ascii : Plongint;
      greek : Plongint;
      cyrillic : Plongint;
      hscale : single;
      vscale : single;
      shear : single;
      thickness : longint;
      dx : single;
      line_type : longint;
    end;


procedure cvInitFont(font:PCvFont; font_face:longint; hscale:double; vscale:double; shear:double;
            thickness:longint; line_type:longint);winapi;external;

procedure cvPutText(img:PCvArr; text:Pchar; org:TCvPoint; font:PCvFont; color:TCvScalar);winapi;external;

procedure cvGetTextSize(text_string:Pchar; font:PCvFont; text_size:PCvSize; baseline:Plongint);winapi;external;

function cvColorToScalar(packed_color:double; arrtype:longint):TCvScalar;winapi;external;

function cvEllipse2Poly(center:TCvPoint; axes:TCvSize; angle:longint; arc_start:longint; arc_end:longint;
           pts:PCvPoint; delta:longint):longint;winapi;external;

procedure cvDrawContours(img:PCvArr; contour:PCvSeq; external_color:TCvScalar; hole_color:TCvScalar; max_level:longint;
            thickness:longint; line_type:longint; offset:TCvPoint);winapi;external;


{.$endif}

{.$ifdef CV_HIGHGUI}
Const
  CV_FONT_LIGHT = 25;
  CV_FONT_NORMAL = 50;
  CV_FONT_DEMIBOLD = 63;
  CV_FONT_BOLD = 75;
  CV_FONT_BLACK = 87;

  CV_STYLE_NORMAL = 0;
  CV_STYLE_ITALIC = 1;
  CV_STYLE_OBLIQUE = 2;

  CV_PUSH_BUTTON = 0;
  CV_CHECKBOX = 1;
  CV_RADIOBOX = 2;

  CV_WND_PROP_FULLSCREEN = 0;
  CV_WND_PROP_AUTOSIZE = 1;
  CV_WND_PROP_ASPECTRATIO = 2;
  CV_WND_PROP_OPENGL = 3;
  CV_WND_PROP_VISIBLE = 4;
  CV_WINDOW_NORMAL = $00000000;
  CV_WINDOW_AUTOSIZE = $00000001;
  CV_WINDOW_OPENGL = $00001000;
  CV_GUI_EXPANDED = $00000000;
  CV_GUI_NORMAL = $00000010;
  CV_WINDOW_FULLSCREEN = 1;
  CV_WINDOW_FREERATIO = $00000100;
  CV_WINDOW_KEEPRATIO = $00000000;

  CV_EVENT_MOUSEMOVE = 0;
  CV_EVENT_LBUTTONDOWN = 1;
  CV_EVENT_RBUTTONDOWN = 2;
  CV_EVENT_MBUTTONDOWN = 3;
  CV_EVENT_LBUTTONUP = 4;
  CV_EVENT_RBUTTONUP = 5;
  CV_EVENT_MBUTTONUP = 6;
  CV_EVENT_LBUTTONDBLCLK = 7;
  CV_EVENT_RBUTTONDBLCLK = 8;
  CV_EVENT_MBUTTONDBLCLK = 9;
  CV_EVENT_MOUSEWHEEL = 10;
  CV_EVENT_MOUSEHWHEEL = 11;

  CV_EVENT_FLAG_LBUTTON = 1;
  CV_EVENT_FLAG_RBUTTON = 2;
  CV_EVENT_FLAG_MBUTTON = 4;
  CV_EVENT_FLAG_CTRLKEY = 8;
  CV_EVENT_FLAG_SHIFTKEY = 16;
  CV_EVENT_FLAG_ALTKEY = 32;


type
  TCvTrackbarCallback = procedure (pos:longint);winapi;
  TCvTrackbarCallback2 = procedure (pos:longint; userdata:pointer);winapi;
  TCvMouseCallback = procedure (event:longint; x:longint; y:longint; flags:longint; param:pointer);winapi;
  TCvOpenGlDrawCallback = procedure (userdata:pointer);winapi;
  TCvGUILoopFunc = function (argc:longint; argv:PPchar):longint;
  TCvButtonCallback = procedure (state :longint; userdata:pointer);


  function cvFontQt(nameFont:Pchar; pointSize:longint; color:TCvScalar; weight:longint; style:longint;
             spacing:longint):TCvFont;winapi;external;
  procedure cvAddText(img:PCvArr; text:Pchar; org:TCvPoint; arg2:PCvFont);winapi;external;
  procedure cvDisplayOverlay(name:Pchar; text:Pchar; delayms:longint);winapi;external;
  procedure cvDisplayStatusBar(name:Pchar; text:Pchar; delayms:longint);winapi;external;
  procedure cvSaveWindowParameters(name:Pchar);winapi;external;
  procedure cvLoadWindowParameters(name:Pchar);winapi;external;
  function cvStartLoop(pt2Func:TCvGUILoopFunc ;argc:longint; argv:PPchar):longint;winapi;external;
  procedure cvStopLoop;winapi;external;

  function cvCreateButton(button_name:Pchar; on_change:TCvButtonCallback; userdata:pointer; button_type:longint; initial_button_state:longint):longint;winapi;external;

  function cvInitSystem(argc:longint; argv:PPchar):longint;winapi;external;
  function cvStartWindowThread:longint;winapi;external;


  function cvNamedWindow(name:Pchar; flags:longint):longint;winapi;external;

  procedure cvSetWindowProperty(name:Pchar; prop_id:longint; prop_value:double);winapi;external;
  function cvGetWindowProperty(name:Pchar; prop_id:longint):double;winapi;external;

  procedure cvShowImage(name:Pchar; image:PCvArr);winapi;external;

  procedure cvResizeWindow(name:Pchar; width:longint; height:longint);winapi;external;
  procedure cvMoveWindow(name:Pchar; x:longint; y:longint);winapi;external;

  procedure cvDestroyWindow(name:Pchar);winapi;external;
  procedure cvDestroyAllWindows;winapi;external;

  function cvGetWindowHandle(name:Pchar):pointer;winapi;external;

  function cvGetWindowName(window_handle:pointer):PChar;winapi;external;

  function cvCreateTrackbar(trackbar_name:Pchar; window_name:Pchar; value:Plongint; count:longint; on_change:TCvTrackbarCallback):longint;winapi;external;

  function cvCreateTrackbar2(trackbar_name:Pchar; window_name:Pchar; value:Plongint; count:longint; on_change:TCvTrackbarCallback2;
             userdata:pointer):longint;winapi;external;

  function cvGetTrackbarPos(trackbar_name:Pchar; window_name:Pchar):longint;winapi;external;
  procedure cvSetTrackbarPos(trackbar_name:Pchar; window_name:Pchar; pos:longint);winapi;external;
  procedure cvSetTrackbarMax(trackbar_name:Pchar; window_name:Pchar; maxval:longint);winapi;external;
  procedure cvSetTrackbarMin(trackbar_name:Pchar; window_name:Pchar; minval:longint);winapi;external;

  procedure cvSetMouseCallback(window_name:Pchar; on_mouse:TCvMouseCallback; param:pointer);winapi;external;

  function cvWaitKey(delay:longint):longint;winapi;external;

  procedure cvSetOpenGlDrawCallback(window_name:Pchar; callback:TCvOpenGlDrawCallback; userdata:pointer);winapi;external;
  procedure cvSetOpenGlContext(window_name:Pchar);winapi;external;
  procedure cvUpdateWindow(window_name:Pchar);winapi;external;

  {$ifdef MSWINDOWS}

  procedure cvSetPreprocessFuncWin32_(callback:pointer);winapi;external;
  procedure cvSetPostprocessFuncWin32_(callback:pointer);winapi;external;
  {$endif}


{.$endif}

{.$ifdef CV_VIDEOIO}

type
  PPCvCapture = ^PCvCapture;
  PCvCapture = ^TCvCapture;
  TCvCapture = record
  end;

  PPCvVideoWriter = ^PCvVideoWriter;
  PCvVideoWriter = ^TCvVideoWriter;
  TCvVideoWriter = record
  end;

const
  CV_LKFLOW_PYR_A_READY = 1       ;
  CV_LKFLOW_PYR_B_READY = 2       ;
  CV_LKFLOW_INITIAL_GUESSES = 4   ;
  CV_LKFLOW_GET_MIN_EIGENVALS = 8 ;


  CV_CAP_ANY = 0;
  CV_CAP_MIL = 100;
  CV_CAP_VFW = 200;
  CV_CAP_V4L = 200;
  CV_CAP_V4L2 = 200;
  CV_CAP_FIREWARE = 300;
  CV_CAP_FIREWIRE = 300;
  CV_CAP_IEEE1394 = 300;
  CV_CAP_DC1394 = 300;
  CV_CAP_CMU1394 = 300;
  CV_CAP_STEREO = 400;
  CV_CAP_TYZX = 400;
  CV_TYZX_LEFT = 400;
  CV_TYZX_RIGHT = 401;
  CV_TYZX_COLOR = 402;
  CV_TYZX_Z = 403;
  CV_CAP_QT = 500;
  CV_CAP_UNICAP = 600;
  CV_CAP_DSHOW = 700;
  CV_CAP_MSMF = 1400;
  CV_CAP_PVAPI = 800;
  CV_CAP_OPENNI = 900;
  CV_CAP_OPENNI_ASUS = 910;
  CV_CAP_ANDROID = 1000;
  CV_CAP_ANDROID_BACK = CV_CAP_ANDROID+99;
  CV_CAP_ANDROID_FRONT = CV_CAP_ANDROID+98;
  CV_CAP_XIAPI = 1100;
  CV_CAP_AVFOUNDATION = 1200;
  CV_CAP_GIGANETIX = 1300;
  CV_CAP_INTELPERC = 1500;
  CV_CAP_OPENNI2 = 1600;
  CV_CAP_GPHOTO2 = 1700;
  CV_CAP_GSTREAMER = 1800;
  CV_CAP_FFMPEG = 1900;
  CV_CAP_IMAGES = 2000;
  CV_CAP_ARAVIS = 2100;

  CV_CAP_PROP_DC1394_OFF = -(4);
  CV_CAP_PROP_DC1394_MODE_MANUAL = -(3);
  CV_CAP_PROP_DC1394_MODE_AUTO = -(2);
  CV_CAP_PROP_DC1394_MODE_ONE_PUSH_AUTO = -(1);
  CV_CAP_PROP_POS_MSEC = 0;
  CV_CAP_PROP_POS_FRAMES = 1;
  CV_CAP_PROP_POS_AVI_RATIO = 2;
  CV_CAP_PROP_FRAME_WIDTH = 3;
  CV_CAP_PROP_FRAME_HEIGHT = 4;
  CV_CAP_PROP_FPS = 5;
  CV_CAP_PROP_FOURCC = 6;
  CV_CAP_PROP_FRAME_COUNT = 7;
  CV_CAP_PROP_FORMAT = 8;
  CV_CAP_PROP_MODE = 9;
  CV_CAP_PROP_BRIGHTNESS = 10;
  CV_CAP_PROP_CONTRAST = 11;
  CV_CAP_PROP_SATURATION = 12;
  CV_CAP_PROP_HUE = 13;
  CV_CAP_PROP_GAIN = 14;
  CV_CAP_PROP_EXPOSURE = 15;
  CV_CAP_PROP_CONVERT_RGB = 16;
  CV_CAP_PROP_WHITE_BALANCE_BLUE_U = 17;
  CV_CAP_PROP_RECTIFICATION = 18;
  CV_CAP_PROP_MONOCHROME = 19;
  CV_CAP_PROP_SHARPNESS = 20;
  CV_CAP_PROP_AUTO_EXPOSURE = 21;
  CV_CAP_PROP_GAMMA = 22;
  CV_CAP_PROP_TEMPERATURE = 23;
  CV_CAP_PROP_TRIGGER = 24;
  CV_CAP_PROP_TRIGGER_DELAY = 25;
  CV_CAP_PROP_WHITE_BALANCE_RED_V = 26;
  CV_CAP_PROP_ZOOM = 27;
  CV_CAP_PROP_FOCUS = 28;
  CV_CAP_PROP_GUID = 29;
  CV_CAP_PROP_ISO_SPEED = 30;
  CV_CAP_PROP_MAX_DC1394 = 31;
  CV_CAP_PROP_BACKLIGHT = 32;
  CV_CAP_PROP_PAN = 33;
  CV_CAP_PROP_TILT = 34;
  CV_CAP_PROP_ROLL = 35;
  CV_CAP_PROP_IRIS = 36;
  CV_CAP_PROP_SETTINGS = 37;
  CV_CAP_PROP_BUFFERSIZE = 38;
  CV_CAP_PROP_AUTOFOCUS = 39;
  CV_CAP_PROP_SAR_NUM = 40;
  CV_CAP_PROP_SAR_DEN = 41;
  CV_CAP_PROP_AUTOGRAB = 1024;
  CV_CAP_PROP_SUPPORTED_PREVIEW_SIZES_STRING = 1025;
  CV_CAP_PROP_PREVIEW_FORMAT = 1026;
  CV_CAP_OPENNI_DEPTH_GENERATOR = 1 shl 31;
  CV_CAP_OPENNI_IMAGE_GENERATOR = 1 shl 30;
  CV_CAP_OPENNI_IR_GENERATOR = 1 shl 29;
  CV_CAP_OPENNI_GENERATORS_MASK = (CV_CAP_OPENNI_DEPTH_GENERATOR+CV_CAP_OPENNI_IMAGE_GENERATOR)+CV_CAP_OPENNI_IR_GENERATOR;
  CV_CAP_PROP_OPENNI_OUTPUT_MODE = 100;
  CV_CAP_PROP_OPENNI_FRAME_MAX_DEPTH = 101;
  CV_CAP_PROP_OPENNI_BASELINE = 102;
  CV_CAP_PROP_OPENNI_FOCAL_LENGTH = 103;
  CV_CAP_PROP_OPENNI_REGISTRATION = 104;
  CV_CAP_PROP_OPENNI_REGISTRATION_ON = CV_CAP_PROP_OPENNI_REGISTRATION;
  CV_CAP_PROP_OPENNI_APPROX_FRAME_SYNC = 105;
  CV_CAP_PROP_OPENNI_MAX_BUFFER_SIZE = 106;
  CV_CAP_PROP_OPENNI_CIRCLE_BUFFER = 107;
  CV_CAP_PROP_OPENNI_MAX_TIME_DURATION = 108;
  CV_CAP_PROP_OPENNI_GENERATOR_PRESENT = 109;
  CV_CAP_PROP_OPENNI2_SYNC = 110;
  CV_CAP_PROP_OPENNI2_MIRROR = 111;
  CV_CAP_OPENNI_IMAGE_GENERATOR_PRESENT = CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_OPENNI_GENERATOR_PRESENT;
  CV_CAP_OPENNI_IMAGE_GENERATOR_OUTPUT_MODE = CV_CAP_OPENNI_IMAGE_GENERATOR+CV_CAP_PROP_OPENNI_OUTPUT_MODE;
  CV_CAP_OPENNI_DEPTH_GENERATOR_PRESENT = CV_CAP_OPENNI_DEPTH_GENERATOR+CV_CAP_PROP_OPENNI_GENERATOR_PRESENT;
  CV_CAP_OPENNI_DEPTH_GENERATOR_BASELINE = CV_CAP_OPENNI_DEPTH_GENERATOR+CV_CAP_PROP_OPENNI_BASELINE;
  CV_CAP_OPENNI_DEPTH_GENERATOR_FOCAL_LENGTH = CV_CAP_OPENNI_DEPTH_GENERATOR+CV_CAP_PROP_OPENNI_FOCAL_LENGTH;
  CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION = CV_CAP_OPENNI_DEPTH_GENERATOR+CV_CAP_PROP_OPENNI_REGISTRATION;
  CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION_ON = CV_CAP_OPENNI_DEPTH_GENERATOR_REGISTRATION;
  CV_CAP_OPENNI_IR_GENERATOR_PRESENT = CV_CAP_OPENNI_IR_GENERATOR+CV_CAP_PROP_OPENNI_GENERATOR_PRESENT;
  CV_CAP_GSTREAMER_QUEUE_LENGTH = 200;
  CV_CAP_PROP_PVAPI_MULTICASTIP = 300;
  CV_CAP_PROP_PVAPI_FRAMESTARTTRIGGERMODE = 301;
  CV_CAP_PROP_PVAPI_DECIMATIONHORIZONTAL = 302;
  CV_CAP_PROP_PVAPI_DECIMATIONVERTICAL = 303;
  CV_CAP_PROP_PVAPI_BINNINGX = 304;
  CV_CAP_PROP_PVAPI_BINNINGY = 305;
  CV_CAP_PROP_PVAPI_PIXELFORMAT = 306;
  CV_CAP_PROP_XI_DOWNSAMPLING = 400;
  CV_CAP_PROP_XI_DATA_FORMAT = 401;
  CV_CAP_PROP_XI_OFFSET_X = 402;
  CV_CAP_PROP_XI_OFFSET_Y = 403;
  CV_CAP_PROP_XI_TRG_SOURCE = 404;
  CV_CAP_PROP_XI_TRG_SOFTWARE = 405;
  CV_CAP_PROP_XI_GPI_SELECTOR = 406;
  CV_CAP_PROP_XI_GPI_MODE = 407;
  CV_CAP_PROP_XI_GPI_LEVEL = 408;
  CV_CAP_PROP_XI_GPO_SELECTOR = 409;
  CV_CAP_PROP_XI_GPO_MODE = 410;
  CV_CAP_PROP_XI_LED_SELECTOR = 411;
  CV_CAP_PROP_XI_LED_MODE = 412;
  CV_CAP_PROP_XI_MANUAL_WB = 413;
  CV_CAP_PROP_XI_AUTO_WB = 414;
  CV_CAP_PROP_XI_AEAG = 415;
  CV_CAP_PROP_XI_EXP_PRIORITY = 416;
  CV_CAP_PROP_XI_AE_MAX_LIMIT = 417;
  CV_CAP_PROP_XI_AG_MAX_LIMIT = 418;
  CV_CAP_PROP_XI_AEAG_LEVEL = 419;
  CV_CAP_PROP_XI_TIMEOUT = 420;
  CV_CAP_PROP_XI_EXPOSURE = 421;
  CV_CAP_PROP_XI_EXPOSURE_BURST_COUNT = 422;
  CV_CAP_PROP_XI_GAIN_SELECTOR = 423;
  CV_CAP_PROP_XI_GAIN = 424;
  CV_CAP_PROP_XI_DOWNSAMPLING_TYPE = 426;
  CV_CAP_PROP_XI_BINNING_SELECTOR = 427;
  CV_CAP_PROP_XI_BINNING_VERTICAL = 428;
  CV_CAP_PROP_XI_BINNING_HORIZONTAL = 429;
  CV_CAP_PROP_XI_BINNING_PATTERN = 430;
  CV_CAP_PROP_XI_DECIMATION_SELECTOR = 431;
  CV_CAP_PROP_XI_DECIMATION_VERTICAL = 432;
  CV_CAP_PROP_XI_DECIMATION_HORIZONTAL = 433;
  CV_CAP_PROP_XI_DECIMATION_PATTERN = 434;
  CV_CAP_PROP_XI_TEST_PATTERN_GENERATOR_SELECTOR = 587;
  CV_CAP_PROP_XI_TEST_PATTERN = 588;
  CV_CAP_PROP_XI_IMAGE_DATA_FORMAT = 435;
  CV_CAP_PROP_XI_SHUTTER_TYPE = 436;
  CV_CAP_PROP_XI_SENSOR_TAPS = 437;
  CV_CAP_PROP_XI_AEAG_ROI_OFFSET_X = 439;
  CV_CAP_PROP_XI_AEAG_ROI_OFFSET_Y = 440;
  CV_CAP_PROP_XI_AEAG_ROI_WIDTH = 441;
  CV_CAP_PROP_XI_AEAG_ROI_HEIGHT = 442;
  CV_CAP_PROP_XI_BPC = 445;
  CV_CAP_PROP_XI_WB_KR = 448;
  CV_CAP_PROP_XI_WB_KG = 449;
  CV_CAP_PROP_XI_WB_KB = 450;
  CV_CAP_PROP_XI_WIDTH = 451;
  CV_CAP_PROP_XI_HEIGHT = 452;
  CV_CAP_PROP_XI_REGION_SELECTOR = 589;
  CV_CAP_PROP_XI_REGION_MODE = 595;
  CV_CAP_PROP_XI_LIMIT_BANDWIDTH = 459;
  CV_CAP_PROP_XI_SENSOR_DATA_BIT_DEPTH = 460;
  CV_CAP_PROP_XI_OUTPUT_DATA_BIT_DEPTH = 461;
  CV_CAP_PROP_XI_IMAGE_DATA_BIT_DEPTH = 462;
  CV_CAP_PROP_XI_OUTPUT_DATA_PACKING = 463;
  CV_CAP_PROP_XI_OUTPUT_DATA_PACKING_TYPE = 464;
  CV_CAP_PROP_XI_IS_COOLED = 465;
  CV_CAP_PROP_XI_COOLING = 466;
  CV_CAP_PROP_XI_TARGET_TEMP = 467;
  CV_CAP_PROP_XI_CHIP_TEMP = 468;
  CV_CAP_PROP_XI_HOUS_TEMP = 469;
  CV_CAP_PROP_XI_HOUS_BACK_SIDE_TEMP = 590;
  CV_CAP_PROP_XI_SENSOR_BOARD_TEMP = 596;
  CV_CAP_PROP_XI_CMS = 470;
  CV_CAP_PROP_XI_APPLY_CMS = 471;
  CV_CAP_PROP_XI_IMAGE_IS_COLOR = 474;
  CV_CAP_PROP_XI_COLOR_FILTER_ARRAY = 475;
  CV_CAP_PROP_XI_GAMMAY = 476;
  CV_CAP_PROP_XI_GAMMAC = 477;
  CV_CAP_PROP_XI_SHARPNESS = 478;
  CV_CAP_PROP_XI_CC_MATRIX_00 = 479;
  CV_CAP_PROP_XI_CC_MATRIX_01 = 480;
  CV_CAP_PROP_XI_CC_MATRIX_02 = 481;
  CV_CAP_PROP_XI_CC_MATRIX_03 = 482;
  CV_CAP_PROP_XI_CC_MATRIX_10 = 483;
  CV_CAP_PROP_XI_CC_MATRIX_11 = 484;
  CV_CAP_PROP_XI_CC_MATRIX_12 = 485;
  CV_CAP_PROP_XI_CC_MATRIX_13 = 486;
  CV_CAP_PROP_XI_CC_MATRIX_20 = 487;
  CV_CAP_PROP_XI_CC_MATRIX_21 = 488;
  CV_CAP_PROP_XI_CC_MATRIX_22 = 489;
  CV_CAP_PROP_XI_CC_MATRIX_23 = 490;
  CV_CAP_PROP_XI_CC_MATRIX_30 = 491;
  CV_CAP_PROP_XI_CC_MATRIX_31 = 492;
  CV_CAP_PROP_XI_CC_MATRIX_32 = 493;
  CV_CAP_PROP_XI_CC_MATRIX_33 = 494;
  CV_CAP_PROP_XI_DEFAULT_CC_MATRIX = 495;
  CV_CAP_PROP_XI_TRG_SELECTOR = 498;
  CV_CAP_PROP_XI_ACQ_FRAME_BURST_COUNT = 499;
  CV_CAP_PROP_XI_DEBOUNCE_EN = 507;
  CV_CAP_PROP_XI_DEBOUNCE_T0 = 508;
  CV_CAP_PROP_XI_DEBOUNCE_T1 = 509;
  CV_CAP_PROP_XI_DEBOUNCE_POL = 510;
  CV_CAP_PROP_XI_LENS_MODE = 511;
  CV_CAP_PROP_XI_LENS_APERTURE_VALUE = 512;
  CV_CAP_PROP_XI_LENS_FOCUS_MOVEMENT_VALUE = 513;
  CV_CAP_PROP_XI_LENS_FOCUS_MOVE = 514;
  CV_CAP_PROP_XI_LENS_FOCUS_DISTANCE = 515;
  CV_CAP_PROP_XI_LENS_FOCAL_LENGTH = 516;
  CV_CAP_PROP_XI_LENS_FEATURE_SELECTOR = 517;
  CV_CAP_PROP_XI_LENS_FEATURE = 518;
  CV_CAP_PROP_XI_DEVICE_MODEL_ID = 521;
  CV_CAP_PROP_XI_DEVICE_SN = 522;
  CV_CAP_PROP_XI_IMAGE_DATA_FORMAT_RGB32_ALPHA = 529;
  CV_CAP_PROP_XI_IMAGE_PAYLOAD_SIZE = 530;
  CV_CAP_PROP_XI_TRANSPORT_PIXEL_FORMAT = 531;
  CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_HZ = 532;
  CV_CAP_PROP_XI_SENSOR_CLOCK_FREQ_INDEX = 533;
  CV_CAP_PROP_XI_SENSOR_OUTPUT_CHANNEL_COUNT = 534;
  CV_CAP_PROP_XI_FRAMERATE = 535;
  CV_CAP_PROP_XI_COUNTER_SELECTOR = 536;
  CV_CAP_PROP_XI_COUNTER_VALUE = 537;
  CV_CAP_PROP_XI_ACQ_TIMING_MODE = 538;
  CV_CAP_PROP_XI_AVAILABLE_BANDWIDTH = 539;
  CV_CAP_PROP_XI_BUFFER_POLICY = 540;
  CV_CAP_PROP_XI_LUT_EN = 541;
  CV_CAP_PROP_XI_LUT_INDEX = 542;
  CV_CAP_PROP_XI_LUT_VALUE = 543;
  CV_CAP_PROP_XI_TRG_DELAY = 544;
  CV_CAP_PROP_XI_TS_RST_MODE = 545;
  CV_CAP_PROP_XI_TS_RST_SOURCE = 546;
  CV_CAP_PROP_XI_IS_DEVICE_EXIST = 547;
  CV_CAP_PROP_XI_ACQ_BUFFER_SIZE = 548;
  CV_CAP_PROP_XI_ACQ_BUFFER_SIZE_UNIT = 549;
  CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_SIZE = 550;
  CV_CAP_PROP_XI_BUFFERS_QUEUE_SIZE = 551;
  CV_CAP_PROP_XI_ACQ_TRANSPORT_BUFFER_COMMIT = 552;
  CV_CAP_PROP_XI_RECENT_FRAME = 553;
  CV_CAP_PROP_XI_DEVICE_RESET = 554;
  CV_CAP_PROP_XI_COLUMN_FPN_CORRECTION = 555;
  CV_CAP_PROP_XI_ROW_FPN_CORRECTION = 591;
  CV_CAP_PROP_XI_SENSOR_MODE = 558;
  CV_CAP_PROP_XI_HDR = 559;
  CV_CAP_PROP_XI_HDR_KNEEPOINT_COUNT = 560;
  CV_CAP_PROP_XI_HDR_T1 = 561;
  CV_CAP_PROP_XI_HDR_T2 = 562;
  CV_CAP_PROP_XI_KNEEPOINT1 = 563;
  CV_CAP_PROP_XI_KNEEPOINT2 = 564;
  CV_CAP_PROP_XI_IMAGE_BLACK_LEVEL = 565;
  CV_CAP_PROP_XI_HW_REVISION = 571;
  CV_CAP_PROP_XI_DEBUG_LEVEL = 572;
  CV_CAP_PROP_XI_AUTO_BANDWIDTH_CALCULATION = 573;
  CV_CAP_PROP_XI_FFS_FILE_ID = 594;
  CV_CAP_PROP_XI_FFS_FILE_SIZE = 580;
  CV_CAP_PROP_XI_FREE_FFS_SIZE = 581;
  CV_CAP_PROP_XI_USED_FFS_SIZE = 582;
  CV_CAP_PROP_XI_FFS_ACCESS_KEY = 583;
  CV_CAP_PROP_XI_SENSOR_FEATURE_SELECTOR = 585;
  CV_CAP_PROP_XI_SENSOR_FEATURE_VALUE = 586;
  CV_CAP_PROP_ANDROID_FLASH_MODE = 8001;
  CV_CAP_PROP_ANDROID_FOCUS_MODE = 8002;
  CV_CAP_PROP_ANDROID_WHITE_BALANCE = 8003;
  CV_CAP_PROP_ANDROID_ANTIBANDING = 8004;
  CV_CAP_PROP_ANDROID_FOCAL_LENGTH = 8005;
  CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_NEAR = 8006;
  CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_OPTIMAL = 8007;
  CV_CAP_PROP_ANDROID_FOCUS_DISTANCE_FAR = 8008;
  CV_CAP_PROP_ANDROID_EXPOSE_LOCK = 8009;
  CV_CAP_PROP_ANDROID_WHITEBALANCE_LOCK = 8010;
  CV_CAP_PROP_IOS_DEVICE_FOCUS = 9001;
  CV_CAP_PROP_IOS_DEVICE_EXPOSURE = 9002;
  CV_CAP_PROP_IOS_DEVICE_FLASH = 9003;
  CV_CAP_PROP_IOS_DEVICE_WHITEBALANCE = 9004;
  CV_CAP_PROP_IOS_DEVICE_TORCH = 9005;
  CV_CAP_PROP_GIGA_FRAME_OFFSET_X = 10001;
  CV_CAP_PROP_GIGA_FRAME_OFFSET_Y = 10002;
  CV_CAP_PROP_GIGA_FRAME_WIDTH_MAX = 10003;
  CV_CAP_PROP_GIGA_FRAME_HEIGH_MAX = 10004;
  CV_CAP_PROP_GIGA_FRAME_SENS_WIDTH = 10005;
  CV_CAP_PROP_GIGA_FRAME_SENS_HEIGH = 10006;
  CV_CAP_PROP_INTELPERC_PROFILE_COUNT = 11001;
  CV_CAP_PROP_INTELPERC_PROFILE_IDX = 11002;
  CV_CAP_PROP_INTELPERC_DEPTH_LOW_CONFIDENCE_VALUE = 11003;
  CV_CAP_PROP_INTELPERC_DEPTH_SATURATION_VALUE = 11004;
  CV_CAP_PROP_INTELPERC_DEPTH_CONFIDENCE_THRESHOLD = 11005;
  CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_HORZ = 11006;
  CV_CAP_PROP_INTELPERC_DEPTH_FOCAL_LENGTH_VERT = 11007;
  CV_CAP_INTELPERC_DEPTH_GENERATOR = 1 shl 29;
  CV_CAP_INTELPERC_IMAGE_GENERATOR = 1 shl 28;
  CV_CAP_INTELPERC_GENERATORS_MASK = CV_CAP_INTELPERC_DEPTH_GENERATOR+CV_CAP_INTELPERC_IMAGE_GENERATOR;

  
  CV_CAP_OPENNI_DEPTH_MAP = 0;
  CV_CAP_OPENNI_POINT_CLOUD_MAP = 1;
  CV_CAP_OPENNI_DISPARITY_MAP = 2;
  CV_CAP_OPENNI_DISPARITY_MAP_32F = 3;
  CV_CAP_OPENNI_VALID_DEPTH_MASK = 4;
  CV_CAP_OPENNI_BGR_IMAGE = 5;
  CV_CAP_OPENNI_GRAY_IMAGE = 6;
  CV_CAP_OPENNI_IR_IMAGE = 7;

  CV_CAP_OPENNI_VGA_30HZ = 0;
  CV_CAP_OPENNI_SXGA_15HZ = 1;
  CV_CAP_OPENNI_SXGA_30HZ = 2;
  CV_CAP_OPENNI_QVGA_30HZ = 3;
  CV_CAP_OPENNI_QVGA_60HZ = 4;

  CV_CAP_INTELPERC_DEPTH_MAP = 0;
  CV_CAP_INTELPERC_UVDEPTH_MAP = 1;
  CV_CAP_INTELPERC_IR_MAP = 2;
  CV_CAP_INTELPERC_IMAGE = 3;

  CV_CAP_PROP_GPHOTO2_PREVIEW = 17001;
  CV_CAP_PROP_GPHOTO2_WIDGET_ENUMERATE = 17002;
  CV_CAP_PROP_GPHOTO2_RELOAD_CONFIG = 17003;
  CV_CAP_PROP_GPHOTO2_RELOAD_ON_CHANGE = 17004;
  CV_CAP_PROP_GPHOTO2_COLLECT_MSGS = 17005;
  CV_CAP_PROP_GPHOTO2_FLUSH_MSGS = 17006;
  CV_CAP_PROP_SPEED = 17007;
  CV_CAP_PROP_APERTURE = 17008;
  CV_CAP_PROP_EXPOSUREPROGRAM = 17009;
  CV_CAP_PROP_VIEWFINDER = 17010;
  CV_FOURCC_PROMPT = -1;
  CV_FOURCC_DEFAULT = $56555949; // 'IYUV'
function CV_FOURCC(const c1, c2, c3, c4:char):longint;

function cvCreateFileCapture(filename:Pchar):PCvCapture;winapi;external;

function cvCreateFileCaptureWithPreference(filename:Pchar; apiPreference:longint):PCvCapture;winapi;external;

function cvCreateCameraCapture(index:longint):PCvCapture;winapi;external;

function cvGrabFrame(capture:PCvCapture):longint;winapi;external;

function cvRetrieveFrame(capture:PCvCapture; streamIdx:longint):PIplImage;winapi;external;

function cvQueryFrame(capture:PCvCapture):PIplImage;winapi;external;

procedure cvReleaseCapture(capture:PPCvCapture);winapi;external;

function cvGetCaptureProperty(capture:PCvCapture; property_id:longint):double;winapi;external;

function cvSetCaptureProperty(capture:PCvCapture; property_id:longint; value:double):longint;winapi;external;

function cvGetCaptureDomain(capture:PCvCapture):longint;winapi;external;

function cvCreateVideoWriter(filename:Pchar; fourcc:longint; fps:double; frame_size:TCvSize; is_color:longint):PCvVideoWriter;winapi;external;

function cvWriteFrame(writer:PCvVideoWriter; image:PIplImage):longint;winapi;external;

procedure cvReleaseVideoWriter(writer:PPCvVideoWriter);winapi;external;

{.$endif}

{* type checking macros  }
{************************************************************************************** }
{                            Sequence writer & reader                                   }
{************************************************************************************** }
{*< the sequence written  }  {*< current block  }  {*< pointer to free space  }  {*< pointer to the beginning of block }
{*< pointer to the end of block  }


  {***************************************************************************************\
*          Array allocation, deallocation, initialization and access to elements         *
\*************************************************************************************** }
{* `malloc` wrapper.
   If there is no enough memory, the function
   (as well as other OpenCV functions that call cvAlloc)
   raises an error.  }

function cvAlloc(size:size_t):Pointer;winapi;external;
{* `free` wrapper.
   Here and further all the memory releasing functions
   (that all call cvFree) take double pointer in order to
   to clear pointer to the data after releasing it.
   Passing pointer to NULL pointer is Ok: nothing happens in this case
 }
procedure cvFree_(ptr:pointer);winapi;external;
{* @brief Creates an image header but does not allocate the image data.

@param size Image width and height
@param depth Image depth (see cvCreateImage )
@param channels Number of channels (see cvCreateImage )
  }
function cvCreateImageHeader(size:TCvSize; depth:longint; channels:longint):PIplImage;winapi;external;
{* @brief Initializes an image header that was previously allocated.

The returned IplImage\* points to the initialized header.
@param image Image header to initialize
@param size Image width and height
@param depth Image depth (see cvCreateImage )
@param channels Number of channels (see cvCreateImage )
@param origin Top-left IPL_ORIGIN_TL or bottom-left IPL_ORIGIN_BL
@param align Alignment for image rows, typically 4 or 8 bytes
  }
function cvInitImageHeader(image:PIplImage; size:TCvSize; depth:longint; channels:longint; origin:longint; 
           align:longint):PIplImage;winapi;external;
{* @brief Creates an image header and allocates the image data.

This function call is equivalent to the following code:
@code
    header = cvCreateImageHeader(size, depth, channels);
    cvCreateData(header);
@endcode
@param size Image width and height
@param depth Bit depth of image elements. See IplImage for valid depths.
@param channels Number of channels per pixel. See IplImage for details. This function only creates
images with interleaved channels.
  }
function cvCreateImage(size:TCvSize; depth:longint; channels:longint):PIplImage;winapi;external;
{* @brief Deallocates an image header.

This call is an analogue of :
@code
    if(image )
    
        iplDeallocate(*image, IPL_IMAGE_HEADER | IPL_IMAGE_ROI);
        *image = 0;
    
@endcode
but it does not use IPL functions by default (see the CV_TURN_ON_IPL_COMPATIBILITY macro).
@param image Double pointer to the image header
  }
procedure cvReleaseImageHeader(image:PPIplImage);winapi;external;
{* @brief Deallocates the image header and the image data.

This call is a shortened form of :
@code
    if(*image )
    
        cvReleaseData(*image);
        cvReleaseImageHeader(image);
    
@endcode
@param image Double pointer to the image header
 }
procedure cvReleaseImage(image:PPIplImage);winapi;external;
{* Creates a copy of IPL image (widthStep may differ)  }
(* Const before type ignored *)
function cvCloneImage(image:PIplImage):PIplImage;winapi;external;
{* @brief Sets the channel of interest in an IplImage.

If the ROI is set to NULL and the coi is *not* 0, the ROI is allocated. Most OpenCV functions do
*not* support the COI setting, so to process an individual image/matrix channel one may copy (via
cvCopy or cvSplit) the channel to a separate image/matrix, process it and then copy the result
back (via cvCopy or cvMerge) if needed.
@param image A pointer to the image header
@param coi The channel of interest. 0 - all channels are selected, 1 - first channel is selected,
etc. Note that the channel indices become 1-based.
  }
procedure cvSetImageCOI(image:PIplImage; coi:longint);winapi;external;
{* @brief Returns the index of the channel of interest.

Returns the channel of interest of in an IplImage. Returned values correspond to the coi in
cvSetImageCOI.
@param image A pointer to the image header
  }
(* Const before type ignored *)
function cvGetImageCOI(image:PIplImage):longint;winapi;external;
{* @brief Sets an image Region Of Interest (ROI) for a given rectangle.

If the original image ROI was NULL and the rect is not the whole image, the ROI structure is
allocated.

Most OpenCV functions support the use of ROI and treat the image rectangle as a separate image. For
example, all of the pixel coordinates are counted from the top-left (or bottom-left) corner of the
ROI, not the original image.
@param image A pointer to the image header
@param rect The ROI rectangle
  }
procedure cvSetImageROI(image:PIplImage; rect:TCvRect);winapi;external;
{* @brief Resets the image ROI to include the entire image and releases the ROI structure.

This produces a similar result to the following, but in addition it releases the ROI structure. :
@code
    cvSetImageROI(image, cvRect(0, 0, image->width, image->height ));
    cvSetImageCOI(image, 0);
@endcode
@param image A pointer to the image header
  }
procedure cvResetImageROI(image:PIplImage);winapi;external;
{* @brief Returns the image ROI.

If there is no ROI set, cvRect(0,0,image-\>width,image-\>height) is returned.
@param image A pointer to the image header
  }
(* Const before type ignored *)
function cvGetImageROI(image:PIplImage):TCvRect;winapi;external;
{* @brief Creates a matrix header but does not allocate the matrix data.

The function allocates a new matrix header and returns a pointer to it. The matrix data can then be
allocated using cvCreateData or set explicitly to user-allocated data via cvSetData.
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type Type of the matrix elements, see cvCreateMat
  }
function cvCreateMatHeader(rows:longint; cols:longint; _type:longint):PCvMat;winapi;external;
{* @brief Initializes a pre-allocated matrix header.

This function is often used to process raw data with OpenCV matrix functions. For example, the
following code computes the matrix product of two matrices, stored as ordinary arrays:
@code
    double a[] =  1, 2, 3, 4,
                   5, 6, 7, 8,
                   9, 10, 11, 12 ;

    double b[] =  1, 5, 9,
                   2, 6, 10,
                   3, 7, 11,
                   4, 8, 12 ;

    double c[9];
    CvMat Ma, Mb, Mc ;

    cvInitMatHeader(&Ma, 3, 4, CV_64FC1, a);
    cvInitMatHeader(&Mb, 4, 3, CV_64FC1, b);
    cvInitMatHeader(&Mc, 3, 3, CV_64FC1, c);

    cvMatMulAdd(&Ma, &Mb, 0, &Mc);
    // the c array now contains the product of a (3x4) and b (4x3)
@endcode
@param mat A pointer to the matrix header to be initialized
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type Type of the matrix elements, see cvCreateMat .
@param data Optional: data pointer assigned to the matrix header
@param step Optional: full row width in bytes of the assigned data. By default, the minimal
possible step is used which assumes there are no gaps between subsequent rows of the matrix.
  }

function cvInitMatHeader(mat:PCvMat; rows:longint; cols:longint; _type:longint; data:pointer; 
           step:longint):PCvMat;winapi;external;
{* @brief Creates a matrix header and allocates the matrix data.

The function call is equivalent to the following code:
@code
    CvMat* mat = cvCreateMatHeader(rows, cols, type);
    cvCreateData(mat);
@endcode
@param rows Number of rows in the matrix
@param cols Number of columns in the matrix
@param type The type of the matrix elements in the form
CV_\<bit depth\>\<S|U|F\>C\<number of channels\> , where S=signed, U=unsigned, F=float. For
example, CV _ 8UC1 means the elements are 8-bit unsigned and the there is 1 channel, and CV _
32SC2 means the elements are 32-bit signed and there are 2 channels.
  }
function cvCreateMat(rows:longint; cols:longint; _type:longint):PCvMat;winapi;external;
{* @brief Deallocates a matrix.

The function decrements the matrix data reference counter and deallocates matrix header. If the data
reference counter is 0, it also deallocates the data. :
@code
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
@endcode
@param mat Double pointer to the matrix
  }
procedure cvReleaseMat(mat:PPCvMat);winapi;external;
{* @brief Decrements an array data reference counter.

The function decrements the data reference counter in a CvMat or CvMatND if the reference counter

pointer is not NULL. If the counter reaches zero, the data is deallocated. In the current
implementation the reference counter is not NULL only if the data was allocated using the
cvCreateData function. The counter will be NULL in other cases such as: external data was assigned
to the header using cvSetData, header is part of a larger matrix or image, or the header was
converted from an image or n-dimensional matrix header.
@param arr Pointer to an array header
  }
{* @brief Increments array data reference counter.

The function increments CvMat or CvMatND data reference counter and returns the new counter value if
the reference counter pointer is not NULL, otherwise it returns zero.
@param arr Array header
  }
{* Creates an exact copy of the input matrix (except, may be, step value)  }
(* Const before type ignored *)
function cvCloneMat(mat:PCvMat):PCvMat;winapi;external;
{* @brief Returns matrix header corresponding to the rectangular sub-array of input image or matrix.

The function returns header, corresponding to a specified rectangle of the input array. In other

words, it allows the user to treat a rectangular part of input array as a stand-alone array. ROI is
taken into account by the function so the sub-array of ROI is actually extracted.
@param arr Input array
@param submat Pointer to the resultant sub-array header
@param rect Zero-based coordinates of the rectangle of interest
  }
(* Const before type ignored *)
function cvGetSubRect(arr:PCvArr; submat:PCvMat; rect:TCvRect):PCvMat;winapi;external;

{* @brief Returns array row or row span.

The function returns the header, corresponding to a specified row/row span of the input array.
cvGetRow(arr, submat, row) is a shortcut for cvGetRows(arr, submat, row, row+1).
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param start_row Zero-based index of the starting row (inclusive) of the span
@param end_row Zero-based index of the ending row (exclusive) of the span
@param delta_row Index step in the row span. That is, the function extracts every delta_row -th
row from start_row and up to (but not including) end_row .
  }
(* Const before type ignored *)

function cvGetRows(arr:PCvArr; submat:PCvMat; start_row:longint; end_row:longint; delta_row:longint):PCvMat;winapi;external;
{* @overload
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param row Zero-based index of the selected row
 }
{* @brief Returns one of more array columns.

The function returns the header, corresponding to a specified column span of the input array. That

is, no data is copied. Therefore, any modifications of the submatrix will affect the original array.
If you need to copy the columns, use cvCloneMat. cvGetCol(arr, submat, col) is a shortcut for
cvGetCols(arr, submat, col, col+1).
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param start_col Zero-based index of the starting column (inclusive) of the span
@param end_col Zero-based index of the ending column (exclusive) of the span
  }
(* Const before type ignored *)
function cvGetCols(arr:PCvArr; submat:PCvMat; start_col:longint; end_col:longint):PCvMat;winapi;external;
{* @overload
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param col Zero-based index of the selected column
 }
{* @brief Returns one of array diagonals.

The function returns the header, corresponding to a specified diagonal of the input array.
@param arr Input array
@param submat Pointer to the resulting sub-array header
@param diag Index of the array diagonal. Zero value corresponds to the main diagonal, -1
corresponds to the diagonal above the main, 1 corresponds to the diagonal below the main, and so
forth.
  }
(* Const before type ignored *)
function cvGetDiag(arr:PCvArr; submat:PCvMat; diag:longint):PCvMat;winapi;external;
{* low-level scalar <-> raw data conversion functions  }
(* Const before type ignored *)
procedure cvScalarToRawData(scalar:PCvScalar; data:pointer; _type:longint; extend_to_12:longint);winapi;external;
(* Const before type ignored *)
procedure cvRawDataToScalar(data:pointer; _type:longint; scalar:PCvScalar);winapi;external;
{* @brief Creates a new matrix header but does not allocate the matrix data.

The function allocates a header for a multi-dimensional dense array. The array data can further be
allocated using cvCreateData or set explicitly to user-allocated data via cvSetData.
@param dims Number of array dimensions
@param sizes Array of dimension sizes
@param type Type of array elements, see cvCreateMat
  }
(* Const before type ignored *)
function cvCreateMatNDHeader(dims:longint; sizes:PLongint; _type:longint):PCvMatND;winapi;external;
{* @brief Creates the header and allocates the data for a multi-dimensional dense array.

This function call is equivalent to the following code:
@code
    CvMatND* mat = cvCreateMatNDHeader(dims, sizes, type);
    cvCreateData(mat);
@endcode
@param dims Number of array dimensions. This must not exceed CV_MAX_DIM (32 by default, but can be
changed at build time).
@param sizes Array of dimension sizes.
@param type Type of array elements, see cvCreateMat .
  }
(* Const before type ignored *)
function cvCreateMatND(dims:longint; sizes:PLongint; _type:longint):PCvMatND;winapi;external;
{* @brief Initializes a pre-allocated multi-dimensional array header.

@param mat A pointer to the array header to be initialized
@param dims The number of array dimensions
@param sizes An array of dimension sizes
@param type Type of array elements, see cvCreateMat
@param data Optional data pointer assigned to the matrix header
  }
(* Const before type ignored *)
function cvInitMatNDHeader(mat:PCvMatND; dims:longint; sizes:PLongint; _type:longint; data:pointer):PCvMatND;winapi;external;
{* @brief Deallocates a multi-dimensional array.

The function decrements the array data reference counter and releases the array header. If the
reference counter reaches 0, it also deallocates the data. :
@code
    if(*mat )
        cvDecRefData(*mat);
    cvFree((void**)mat);
@endcode
@param mat Double pointer to the array
  }
{* Creates a copy of CvMatND (except, may be, steps)  }
(* Const before type ignored *)
function cvCloneMatND(mat:PCvMatND):PCvMatND;winapi;external;
{* @brief Creates sparse array.

The function allocates a multi-dimensional sparse array. Initially the array contain no elements,
that is PtrND and other related functions will return 0 for every index.
@param dims Number of array dimensions. In contrast to the dense matrix, the number of dimensions is
practically unlimited (up to \f$2^16\f$ ).
@param sizes Array of dimension sizes
@param type Type of array elements. The same as for CvMat
  }
(* Const before type ignored *)
function cvCreateSparseMat(dims:longint; sizes:PLongint; _type:longint):PCvSparseMat;winapi;external;
{* @brief Deallocates sparse array.

The function releases the sparse array and clears the array pointer upon exit.
@param mat Double pointer to the array
  }
procedure cvReleaseSparseMat(mat:PPCvSparseMat);winapi;external;
{* Creates a copy of CvSparseMat (except, may be, zero items)  }
(* Const before type ignored *)
function cvCloneSparseMat(mat:PCvSparseMat):PCvSparseMat;winapi;external;
{* @brief Initializes sparse array elements iterator.

The function initializes iterator of sparse array elements and returns pointer to the first element,
or NULL if the array is empty.
@param mat Input array
@param mat_iterator Initialized iterator
  }
(* Const before type ignored *)
function cvInitSparseMatIterator(mat:PCvSparseMat; mat_iterator:PCvSparseMatIterator):PCvSparseNode;winapi;external;
{* @brief Returns the next sparse matrix element

The function moves iterator to the next sparse matrix element and returns pointer to it. In the
current version there is no any particular order of the elements, because they are stored in the
hash table. The sample below demonstrates how to iterate through the sparse matrix:
@code
    // print all the non-zero sparse matrix elements and compute their sum
    double sum = 0;
    int i, dims = cvGetDims(sparsemat);
    CvSparseMatIterator it;
    CvSparseNode* node = cvInitSparseMatIterator(sparsemat, &it);

    for(; node != 0; node = cvGetNextSparseNode(&it))
    
        int* idx = CV_NODE_IDX(array, node);
        float val = *(float*)CV_NODE_VAL(array, node);
        printf("M");
        for(i = 0; i < dims; i++ )
            printf("[%d]", idx[i]);
        printf("=%g\n", val);

        sum += val;
    

    printf("nTotal sum = %g\n", sum);
@endcode
@param mat_iterator Sparse array iterator
  }

{* initializes iterator that traverses through several arrays simultaneously
   (the function together with cvNextArraySlice is used for
    N-ari element-wise operations)  }
(* Const before type ignored *)

function cvInitNArrayIterator(count:longint; arrs:PPCvArr; mask:PCvArr; stubs:PCvMatND; array_iterator:PCvNArrayIterator; 
           flags:longint):longint;winapi;external;
{* returns zero value if iteration is finished, non-zero (slice length) otherwise  }
function cvNextNArraySlice(array_iterator:PCvNArrayIterator):longint;winapi;external;
{* @brief Returns type of array elements.

The function returns type of the array elements. In the case of IplImage the type is converted to
CvMat-like representation. For example, if the image has been created as:
@code
    IplImage* img = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);
@endcode
The code cvGetElemType(img) will return CV_8UC3.
@param arr Input array
  }
(* Const before type ignored *)
function cvGetElemType(arr:PCvArr):longint;winapi;external;
{* @brief Return number of array dimensions

The function returns the array dimensionality and the array of dimension sizes. In the case of
IplImage or CvMat it always returns 2 regardless of number of image/matrix rows. For example, the
following code calculates total number of array elements:
@code
    int sizes[CV_MAX_DIM];
    int i, total = 1;
    int dims = cvGetDims(arr, size);
    for(i = 0; i < dims; i++ )
        total *= sizes[i];
@endcode
@param arr Input array
@param sizes Optional output vector of the array dimension sizes. For 2d arrays the number of rows
(height) goes first, number of columns (width) next.
  }
(* Const before type ignored *)
function cvGetDims(arr:PCvArr; sizes:PLongint):longint;winapi;external;
{* @brief Returns array size along the specified dimension.

@param arr Input array
@param index Zero-based dimension index (for matrices 0 means number of rows, 1 means number of
columns; for images 0 means height, 1 means width)
  }
(* Const before type ignored *)
function cvGetDimSize(arr:PCvArr; index:longint):longint;winapi;external;
{* @brief Return pointer to a particular array element.

The functions return a pointer to a specific array element. Number of array dimension should match
to the number of indices passed to the function except for cvPtr1D function that can be used for
sequential access to 1D, 2D or nD dense arrays.

The functions can be used for sparse arrays as well - if the requested node does not exist they
create it and set it to zero.

All these as well as other functions accessing array elements ( cvGetND , cvGetRealND , cvSet
, cvSetND , cvSetRealND ) raise an error in case if the element index is out of range.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param type Optional output parameter: type of matrix elements
  }
(* Const before type ignored *)
function cvPtr1D(arr:PCvArr; idx0:longint; _type:PLongint):PByte;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvPtr2D(arr:PCvArr; idx0:longint; idx1:longint; _type:PLongint):PByte;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvPtr3D(arr:PCvArr; idx0:longint; idx1:longint; idx2:longint; _type:PLongint):PByte;winapi;external;
{* @overload
@param arr Input array
@param idx Array of the element indices
@param type Optional output parameter: type of matrix elements
@param create_node Optional input parameter for sparse matrices. Non-zero value of the parameter
means that the requested element is created if it does not exist already.
@param precalc_hashval Optional input parameter for sparse matrices. If the pointer is not NULL,
the function does not recalculate the node hash value, but takes it from the specified location.
It is useful for speeding up pair-wise operations (TODO: provide an example)
 }
(* Const before type ignored *)
(* Const before type ignored *)
function cvPtrND(arr:PCvArr; idx:PLongint; _type:PLongint; create_node:longint; precalc_hashval:PLongword):PByte;winapi;external;
{* @brief Return a specific array element.

The functions return a specific array element. In the case of a sparse array the functions return 0
if the requested node does not exist (no new node is created by the functions).
@param arr Input array
@param idx0 The first zero-based component of the element index
  }
(* Const before type ignored *)
function cvGet1D(arr:PCvArr; idx0:longint):TCvScalar;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvGet2D(arr:PCvArr; idx0:longint; idx1:longint):TCvScalar;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvGet3D(arr:PCvArr; idx0:longint; idx1:longint; idx2:longint):TCvScalar;winapi;external;
{* @overload
@param arr Input array
@param idx Array of the element indices
 }
(* Const before type ignored *)
(* Const before type ignored *)
function cvGetND(arr:PCvArr; idx:PLongint):TCvScalar;winapi;external;
{* @brief Return a specific element of single-channel 1D, 2D, 3D or nD array.

Returns a specific element of a single-channel array. If the array has multiple channels, a runtime
error is raised. Note that Get?D functions can be used safely for both single-channel and
multiple-channel arrays though they are a bit slower.

In the case of a sparse array the functions return 0 if the requested node does not exist (no new
node is created by the functions).
@param arr Input array. Must have a single channel.
@param idx0 The first zero-based component of the element index
  }
(* Const before type ignored *)
function cvGetReal1D(arr:PCvArr; idx0:longint):double;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvGetReal2D(arr:PCvArr; idx0:longint; idx1:longint):double;winapi;external;
{* @overload  }
(* Const before type ignored *)
function cvGetReal3D(arr:PCvArr; idx0:longint; idx1:longint; idx2:longint):double;winapi;external;
{* @overload
@param arr Input array. Must have a single channel.
@param idx Array of the element indices
 }
(* Const before type ignored *)
(* Const before type ignored *)
function cvGetRealND(arr:PCvArr; idx:PLongint):double;winapi;external;
{* @brief Change the particular array element.

The functions assign the new value to a particular array element. In the case of a sparse array the
functions create the node if it does not exist yet.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param value The assigned value
  }
procedure cvSet1D(arr:PCvArr; idx0:longint; value:TCvScalar);winapi;external;
{* @overload  }
procedure cvSet2D(arr:PCvArr; idx0:longint; idx1:longint; value:TCvScalar);winapi;external;
{* @overload  }
procedure cvSet3D(arr:PCvArr; idx0:longint; idx1:longint; idx2:longint; value:TCvScalar);winapi;external;
{* @overload
@param arr Input array
@param idx Array of the element indices
@param value The assigned value
 }
(* Const before type ignored *)
procedure cvSetND(arr:PCvArr; idx:PLongint; value:TCvScalar);winapi;external;
{* @brief Change a specific array element.

The functions assign a new value to a specific element of a single-channel array. If the array has
multiple channels, a runtime error is raised. Note that the Set\*D function can be used safely for
both single-channel and multiple-channel arrays, though they are a bit slower.

In the case of a sparse array the functions create the node if it does not yet exist.
@param arr Input array
@param idx0 The first zero-based component of the element index
@param value The assigned value
  }
procedure cvSetReal1D(arr:PCvArr; idx0:longint; value:double);winapi;external;
{* @overload  }
procedure cvSetReal2D(arr:PCvArr; idx0:longint; idx1:longint; value:double);winapi;external;
{* @overload  }
procedure cvSetReal3D(arr:PCvArr; idx0:longint; idx1:longint; idx2:longint; value:double);winapi;external;
{* @overload
@param arr Input array
@param idx Array of the element indices
@param value The assigned value
 }
(* Const before type ignored *)
procedure cvSetRealND(arr:PCvArr; idx:PLongint; value:double);winapi;external;
{* clears element of ND dense array,
   in case of sparse arrays it deletes the specified node  }
(* Const before type ignored *)
procedure cvClearND(arr:PCvArr; idx:PLongint);winapi;external;
{* @brief Returns matrix header for arbitrary array.

The function returns a matrix header for the input array that can be a matrix - CvMat, an image -
IplImage, or a multi-dimensional dense array - CvMatND (the third option is allowed only if
allowND != 0) . In the case of matrix the function simply returns the input pointer. In the case of
IplImage\* or CvMatND it initializes the header structure with parameters of the current image ROI
and returns &header. Because COI is not supported by CvMat, it is returned separately.

The function provides an easy way to handle both types of arrays - IplImage and CvMat using the same
code. Input array must have non-zero data pointer, otherwise the function will report an error.

@note If the input array is IplImage with planar data layout and COI set, the function returns the
pointer to the selected plane and COI == 0. This feature allows user to process IplImage structures
with planar data layout, even though OpenCV does not support such images.
@param arr Input array
@param header Pointer to CvMat structure used as a temporary buffer
@param coi Optional output parameter for storing COI
@param allowND If non-zero, the function accepts multi-dimensional dense arrays (CvMatND\*) and
returns 2D matrix (if CvMatND has two dimensions) or 1D matrix (when CvMatND has 1 dimension or
more than 2 dimensions). The CvMatND array must be continuous.
@sa cvGetImage, cvarrToMat.
  }
(* Const before type ignored *)
function cvGetMat(arr:PCvArr; header:PCvMat; coi:PLongint; allowND:longint):PCvMat;winapi;external;
{* @brief Returns image header for arbitrary array.

The function returns the image header for the input array that can be a matrix (CvMat) or image
(IplImage). In the case of an image the function simply returns the input pointer. In the case of
CvMat it initializes an image_header structure with the parameters of the input matrix. Note that
if we transform IplImage to CvMat using cvGetMat and then transform CvMat back to IplImage using
this function, we will get different headers if the ROI is set in the original image.
@param arr Input array
@param image_header Pointer to IplImage structure used as a temporary buffer
  }
(* Const before type ignored *)
function cvGetImage(arr:PCvArr; image_header:PIplImage):PIplImage;winapi;external;
{* @brief Changes the shape of a multi-dimensional array without copying the data.

The function is an advanced version of cvReshape that can work with multi-dimensional arrays as
well (though it can work with ordinary images and matrices) and change the number of dimensions.

Below are the two samples from the cvReshape description rewritten using cvReshapeMatND:
@code
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    IplImage gray_img_hdr, *gray_img;
    gray_img = (IplImage*)cvReshapeMatND(color_img, sizeof(gray_img_hdr), &gray_img_hdr, 1, 0, 0);
    ...
    int size[] =  2, 2, 2 ;
    CvMatND* mat = cvCreateMatND(3, size, CV_32F);
    CvMat row_header, *row;
    row = (CvMat*)cvReshapeMatND(mat, sizeof(row_header), &row_header, 0, 1, 0);
@endcode
In C, the header file for this function includes a convenient macro cvReshapeND that does away with
the sizeof_header parameter. So, the lines containing the call to cvReshapeMatND in the examples
may be replaced as follow:
@code
    gray_img = (IplImage*)cvReshapeND(color_img, &gray_img_hdr, 1, 0, 0);
    ...
    row = (CvMat*)cvReshapeND(mat, &row_header, 0, 1, 0);
@endcode
@param arr Input array
@param sizeof_header Size of output header to distinguish between IplImage, CvMat and CvMatND
output headers
@param header Output header to be filled
@param new_cn New number of channels. new_cn = 0 means that the number of channels remains
unchanged.
@param new_dims New number of dimensions. new_dims = 0 means that the number of dimensions
remains the same.
@param new_sizes Array of new dimension sizes. Only new_dims-1 values are used, because the
total number of elements must remain the same. Thus, if new_dims = 1, new_sizes array is not
used.
  }
(* Const before type ignored *)
function cvReshapeMatND(arr:PCvArr; sizeof_header:longint; header:PCvArr; new_cn:longint; new_dims:longint; 
           new_sizes:PLongint):PCvArr;winapi;external;
{* @brief Changes shape of matrix/image without copying data.

The function initializes the CvMat header so that it points to the same data as the original array
but has a different shape - different number of channels, different number of rows, or both.

The following example code creates one image buffer and two image headers, the first is for a
320x240x3 image and the second is for a 960x240x1 image:
@code
    IplImage* color_img = cvCreateImage(cvSize(320,240), IPL_DEPTH_8U, 3);
    CvMat gray_mat_hdr;
    IplImage gray_img_hdr, *gray_img;
    cvReshape(color_img, &gray_mat_hdr, 1);
    gray_img = cvGetImage(&gray_mat_hdr, &gray_img_hdr);
@endcode
And the next example converts a 3x3 matrix to a single 1x9 vector:
@code
    CvMat* mat = cvCreateMat(3, 3, CV_32F);
    CvMat row_header, *row;
    row = cvReshape(mat, &row_header, 0, 1);
@endcode
@param arr Input array
@param header Output header to be filled
@param new_cn New number of channels. 'new_cn = 0' means that the number of channels remains
unchanged.
@param new_rows New number of rows. 'new_rows = 0' means that the number of rows remains
unchanged unless it needs to be changed according to new_cn value.
 }
(* Const before type ignored *)
function cvReshape(arr:PCvArr; header:PCvMat; new_cn:longint; new_rows:longint):PCvMat;winapi;external;
{* Repeats source 2d array several times in both horizontal and
   vertical direction to fill destination array  }
(* Const before type ignored *)
procedure cvRepeat(src:PCvArr; dst:PCvArr);winapi;external;
{* @brief Allocates array data

The function allocates image, matrix or multi-dimensional dense array data. Note that in the case of
matrix types OpenCV allocation functions are used. In the case of IplImage they are used unless
CV_TURN_ON_IPL_COMPATIBILITY() has been called before. In the latter case IPL functions are used
to allocate the data.
@param arr Array header
  }
procedure cvCreateData(arr:PCvArr);winapi;external;
{* @brief Releases array data.

The function releases the array data. In the case of CvMat or CvMatND it simply calls
cvDecRefData(), that is the function can not deallocate external data. See also the note to
cvCreateData .
@param arr Array header
  }
procedure cvReleaseData(arr:PCvArr);winapi;external;
{* @brief Assigns user data to the array header.

The function assigns user data to the array header. Header should be initialized before using
cvCreateMatHeader, cvCreateImageHeader, cvCreateMatNDHeader, cvInitMatHeader,
cvInitImageHeader or cvInitMatNDHeader.
@param arr Array header
@param data User data
@param step Full row length in bytes
  }
procedure cvSetData(arr:PCvArr; data:pointer; step:longint);winapi;external;
{* @brief Retrieves low-level information about the array.

The function fills output variables with low-level information about the array data. All output

parameters are optional, so some of the pointers may be set to NULL. If the array is IplImage with
ROI set, the parameters of ROI are returned.

The following example shows how to get access to array elements. It computes absolute values of the
array elements :
@code
    float* data;
    int step;
    CvSize size;

    cvGetRawData(array, (uchar**)&data, &step, &size);
    step /= sizeof(data[0]);

    for(int y = 0; y < size.height; y++, data += step )
        for(int x = 0; x < size.width; x++ )
            data[x] = (float)fabs(data[x]);
@endcode
@param arr Array header
@param data Output pointer to the whole image origin or ROI origin if ROI is set
@param step Output full row length in bytes
@param roi_size Output ROI size
  }
(* Const before type ignored *)
procedure cvGetRawData(arr:PCvArr; data:PPByte; step:PLongint; roi_size:PCvSize);winapi;external;
{* @brief Returns size of matrix or image ROI.

The function returns number of rows (CvSize::height) and number of columns (CvSize::width) of the
input matrix or image. In the case of image the size of ROI is returned.
@param arr array header
  }
(* Const before type ignored *)
function cvGetSize(arr:PCvArr):TCvSize;winapi;external;
{* @brief Copies one array to another.

The function copies selected elements from an input array to an output array:

\f[\textttdst (I)= \textttsrc (I)  \quad \textif \quad \textttmask (I)  \ne 0.\f]

If any of the passed arrays is of IplImage type, then its ROI and COI fields are used. Both arrays
must have the same type, the same number of dimensions, and the same size. The function can also
copy sparse arrays (mask is not supported in this case).
@param src The source array
@param dst The destination array
@param mask Operation mask, 8-bit single channel array; specifies elements of the destination array
to be changed
  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvCopy(src:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* @brief Sets every element of an array to a given value.

The function copies the scalar value to every selected element of the destination array:
\f[\textttarr (I)= \textttvalue \quad \textif \quad \textttmask (I)  \ne 0\f]
If array arr is of IplImage type, then is ROI used, but COI must not be set.
@param arr The destination array
@param value Fill value
@param mask Operation mask, 8-bit single channel array; specifies elements of the destination
array to be changed
  }
(* Const before type ignored *)
procedure cvSet(arr:PCvArr; value:TCvScalar; mask:PCvArr);winapi;external;
{* @brief Clears the array.

The function clears the array. In the case of dense arrays (CvMat, CvMatND or IplImage),
cvZero(array) is equivalent to cvSet(array,cvScalarAll(0),0). In the case of sparse arrays all the
elements are removed.
@param arr Array to be cleared
  }
procedure cvSetZero(arr:PCvArr);winapi;external;
//const
//  cvZero = cvSetZero;  
{* Splits a multi-channel array into the set of single-channel arrays or
   extracts particular [color] plane  }
(* Const before type ignored *)

procedure cvSplit(src:PCvArr; dst0:PCvArr; dst1:PCvArr; dst2:PCvArr; dst3:PCvArr);winapi;external;
{* Merges a set of single-channel arrays into the single multi-channel array
   or inserts one particular [color] plane to the array  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMerge(src0:PCvArr; src1:PCvArr; src2:PCvArr; src3:PCvArr; dst:PCvArr);winapi;external;
{* Copies several channels from input arrays to
   certain channels of output arrays  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMixChannels(src:PPCvArr; src_count:longint; dst:PPCvArr; dst_count:longint; from_to:PLongint; 
            pair_count:longint);winapi;external;
{* @brief Converts one array to another with optional linear transformation.

The function has several different purposes, and thus has several different names. It copies one
array to another with optional scaling, which is performed first, and/or optional type conversion,
performed after:

\f[\textttdst (I) =  \textttscale \textttsrc (I) + ( \textttshift _0, \textttshift _1,...)\f]

All the channels of multi-channel arrays are processed independently.

The type of conversion is done with rounding and saturation, that is if the result of scaling +
conversion can not be represented exactly by a value of the destination array element type, it is
set to the nearest representable value on the real axis.
@param src Source array
@param dst Destination array
@param scale Scale factor
@param shift Value added to the scaled source array elements
  }
(* Const before type ignored *)
procedure cvConvertScale(src:PCvArr; dst:PCvArr; scale:double; shift:double);winapi;external;
//const
//  cvCvtScale = cvConvertScale;  
//  cvScale = cvConvertScale;  
{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   

//procedure cvConvert(src,dst : PCvArr) ;

{* Performs linear transformation on every source array element,
   stores absolute value of the result:
   dst(x,y,c) = abs(scale*src(x,y,c)+shift).
   destination array must have 8u type.
   In other cases one may use cvConvertScale + cvAbsDiffS  }
(* Const before type ignored *)
procedure cvConvertScaleAbs(src:PCvArr; dst:PCvArr; scale:double; shift:double);winapi;external;
//const
//  cvCvtScaleAbs = cvConvertScaleAbs;  
{* checks termination criteria validity and
   sets eps to default_eps (if it is not set),
   max_iter to default_max_iters (if it is not set)
 }

function cvCheckTermCriteria(criteria:TCvTermCriteria; default_eps:double; default_max_iters:longint):TCvTermCriteria;winapi;external;
{***************************************************************************************\
*                   Arithmetic, logic and comparison operations                          *
\*************************************************************************************** }
{* dst(mask) = src1(mask) + src2(mask)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAdd(src1:PCvArr; src2:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(mask) = src(mask) + value  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAddS(src:PCvArr; value:TCvScalar; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(mask) = src1(mask) - src2(mask)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvSub(src1:PCvArr; src2:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(mask) = value - src(mask)  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvSubRS(src:PCvArr; value:TCvScalar; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src1(idx) * src2(idx) * scale
   (scaled element-wise multiplication of 2 arrays)  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMul(src1:PCvArr; src2:PCvArr; dst:PCvArr; scale:double);winapi;external;
{* element-wise division/inversion with scaling:
    dst(idx) = src1(idx) * scale / src2(idx)
    or dst(idx) = scale / src2(idx) if src1 == 0  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvDiv(src1:PCvArr; src2:PCvArr; dst:PCvArr; scale:double);winapi;external;
{* dst = src1 * scale + src2  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvScaleAdd(src1:PCvArr; scale:TCvScalar; src2:PCvArr; dst:PCvArr);winapi;external;
{* dst = src1 * alpha + src2 * beta + gamma  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAddWeighted(src1:PCvArr; alpha:double; src2:PCvArr; beta:double; gamma:double; 
            dst:PCvArr);winapi;external;
{* @brief Calculates the dot product of two arrays in Euclidean metrics.

The function calculates and returns the Euclidean dot product of two arrays.

\f[src1  \bullet src2 =  \sum _I ( \textttsrc1 (I)  \textttsrc2 (I))\f]

In the case of multiple channel arrays, the results for all channels are accumulated. In particular,
cvDotProduct(a,a) where a is a complex vector, will return \f$||\texttta||^2\f$. The function can
process multi-dimensional arrays, row by row, layer by layer, and so on.
@param src1 The first source array
@param src2 The second source array
  }
(* Const before type ignored *)
(* Const before type ignored *)
function cvDotProduct(src1:PCvArr; src2:PCvArr):double;winapi;external;
{* dst(idx) = src1(idx) & src2(idx)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAnd(src1:PCvArr; src2:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src(idx) & value  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAndS(src:PCvArr; value:TCvScalar; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src1(idx) | src2(idx)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvOr(src1:PCvArr; src2:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src(idx) | value  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvOrS(src:PCvArr; value:TCvScalar; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src1(idx) ^ src2(idx)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvXor(src1:PCvArr; src2:PCvArr; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = src(idx) ^ value  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvXorS(src:PCvArr; value:TCvScalar; dst:PCvArr; mask:PCvArr);winapi;external;
{* dst(idx) = ~src(idx)  }
(* Const before type ignored *)
procedure cvNot(src:PCvArr; dst:PCvArr);winapi;external;
{* dst(idx) = lower(idx) <= src(idx) < upper(idx)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvInRange(src:PCvArr; lower:PCvArr; upper:PCvArr; dst:PCvArr);winapi;external;
{* dst(idx) = lower <= src(idx) < upper  }
(* Const before type ignored *)
procedure cvInRangeS(src:PCvArr; lower:TCvScalar; upper:TCvScalar; dst:PCvArr);winapi;external;

{* The comparison operation support single-channel arrays only.
   Destination image should be 8uC1 or 8sC1  }
{* dst(idx) = src1(idx) _cmp_op_ src2(idx)  }
(* Const before type ignored *)
(* Const before type ignored *)

procedure cvCmp(src1:PCvArr; src2:PCvArr; dst:PCvArr; cmp_op:longint);winapi;external;
{* dst(idx) = src1(idx) _cmp_op_ value  }
(* Const before type ignored *)
procedure cvCmpS(src:PCvArr; value:double; dst:PCvArr; cmp_op:longint);winapi;external;
{* dst(idx) = min(src1(idx),src2(idx))  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMin(src1:PCvArr; src2:PCvArr; dst:PCvArr);winapi;external;
{* dst(idx) = max(src1(idx),src2(idx))  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMax(src1:PCvArr; src2:PCvArr; dst:PCvArr);winapi;external;
{* dst(idx) = min(src(idx),value)  }
(* Const before type ignored *)
procedure cvMinS(src:PCvArr; value:double; dst:PCvArr);winapi;external;
{* dst(idx) = max(src(idx),value)  }
(* Const before type ignored *)
procedure cvMaxS(src:PCvArr; value:double; dst:PCvArr);winapi;external;
{* dst(x,y,c) = abs(src1(x,y,c) - src2(x,y,c))  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAbsDiff(src1:PCvArr; src2:PCvArr; dst:PCvArr);winapi;external;
{* dst(x,y,c) = abs(src(x,y,c) - value(c))  }
(* Const before type ignored *)
procedure cvAbsDiffS(src:PCvArr; dst:PCvArr; value:TCvScalar);winapi;external;
{***************************************************************************************\
*                                Math operations                                         *
\*************************************************************************************** }
{* Does cartesian->polar coordinates conversion.
   Either of output components (magnitude or angle) is optional  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvCartToPolar(x:PCvArr; y:PCvArr; magnitude:PCvArr; angle:PCvArr; angle_in_degrees:longint);winapi;external;
{* Does polar->cartesian coordinates conversion.
   Either of output components (magnitude or angle) is optional.
   If magnitude is missing it is assumed to be all 1's  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvPolarToCart(magnitude:PCvArr; angle:PCvArr; x:PCvArr; y:PCvArr; angle_in_degrees:longint);winapi;external;
{* Does powering: dst(idx) = src(idx)^power  }
(* Const before type ignored *)
procedure cvPow(src:PCvArr; dst:PCvArr; power:double);winapi;external;
{* Does exponention: dst(idx) = exp(src(idx)).
   Overflow is not handled yet. Underflow is handled.
   Maximal relative error is ~7e-6 for single-precision input  }
(* Const before type ignored *)
procedure cvExp(src:PCvArr; dst:PCvArr);winapi;external;
{* Calculates natural logarithms: dst(idx) = log(abs(src(idx))).
   Logarithm of 0 gives large negative number(~-700)
   Maximal relative error is ~3e-7 for single-precision output
 }
(* Const before type ignored *)
procedure cvLog(src:PCvArr; dst:PCvArr);winapi;external;
{* Fast arctangent calculation  }
function cvFastArctan(y:single; x:single):single;winapi;external;
{* Fast cubic root calculation  }
function cvCbrt(value:single):single;winapi;external;
{* Checks array values for NaNs, Infs or simply for too large numbers
   (if CV_CHECK_RANGE is set). If CV_CHECK_QUIET is set,
   no runtime errors is raised (function returns zero value in case of "bad" values).
   Otherwise cvError is called  }
(* Const before type ignored *)

function cvCheckArr(arr:PCvArr; flags:longint; min_val:double; max_val:double):longint;winapi;external;
{* @brief Fills an array with random numbers and updates the RNG state.

The function fills the destination array with uniformly or normally distributed random numbers.
@param rng CvRNG state initialized by cvRNG
@param arr The destination array
@param dist_type Distribution type
> -   **CV_RAND_UNI** uniform distribution
> -   **CV_RAND_NORMAL** normal or Gaussian distribution
@param param1 The first parameter of the distribution. In the case of a uniform distribution it is
the inclusive lower boundary of the random numbers range. In the case of a normal distribution it
is the mean value of the random numbers.
@param param2 The second parameter of the distribution. In the case of a uniform distribution it
is the exclusive upper boundary of the random numbers range. In the case of a normal distribution
it is the standard deviation of the random numbers.
@sa randu, randn, RNG::fill.
  }

procedure cvRandArr(rng:PCvRNG; arr:PCvArr; dist_type:longint; param1:TCvScalar; param2:TCvScalar);winapi;external;
procedure cvRandShuffle(mat:PCvArr; rng:PCvRNG; iter_factor:double);winapi;external;


procedure cvSort(src:PCvArr; dst:PCvArr; idxmat:PCvArr; flags:longint);winapi;external;
{* Finds real roots of a cubic equation  }
(* Const before type ignored *)
function cvSolveCubic(coeffs:PCvMat; roots:PCvMat):longint;winapi;external;
{* Finds all real and complex roots of a polynomial equation  }
(* Const before type ignored *)
procedure cvSolvePoly(coeffs:PCvMat; roots2:PCvMat; maxiter:longint; fig:longint);winapi;external;
{***************************************************************************************\
*                                Matrix operations                                       *
\*************************************************************************************** }
{* @brief Calculates the cross product of two 3D vectors.

The function calculates the cross product of two 3D vectors:
\f[\textttdst =  \textttsrc1 \times \textttsrc2\f]
or:
\f[\beginarrayl \textttdst _1 =  \textttsrc1 _2  \textttsrc2 _3 -  \textttsrc1 _3  \textttsrc2 _2 \\ \textttdst _2 =  \textttsrc1 _3  \textttsrc2 _1 -  \textttsrc1 _1  \textttsrc2 _3 \\ \textttdst _3 =  \textttsrc1 _1  \textttsrc2 _2 -  \textttsrc1 _2  \textttsrc2 _1 \endarray\f]
@param src1 The first source vector
@param src2 The second source vector
@param dst The destination vector
  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvCrossProduct(src1:PCvArr; src2:PCvArr; dst:PCvArr);winapi;external;
{* Matrix transform: dst = A*B + C, C is optional  }
{* Extended matrix transform:
   dst = alpha*op(A)*op(B) + beta*op(C), where op(X) is X or X^T  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)

procedure cvGEMM(src1:PCvArr; src2:PCvArr; alpha:double; src3:PCvArr; beta:double; 
            dst:PCvArr; tABC:longint);winapi;external;
//const
//  cvMatMulAddEx = cvGEMM;  
{* Transforms each element of source array and stores
   resultant vectors in destination array  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)

procedure cvTransform(src:PCvArr; dst:PCvArr; transmat:PCvMat; shiftvec:PCvMat);winapi;external;
//const
//  cvMatMulAddS = cvTransform;  
{* Does perspective transform on every element of input array  }
(* Const before type ignored *)
(* Const before type ignored *)

procedure cvPerspectiveTransform(src:PCvArr; dst:PCvArr; mat:PCvMat);winapi;external;
{* Calculates (A-delta)*(A-delta)^T (order=0) or (A-delta)^T*(A-delta) (order=1)  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMulTransposed(src:PCvArr; dst:PCvArr; order:longint; delta:PCvArr; scale:double);winapi;external;
{* Transposes matrix. Square matrices can be transposed in-place  }
(* Const before type ignored *)
procedure cvTranspose(src:PCvArr; dst:PCvArr);winapi;external;
//const
//  cvT = cvTranspose;  
{* Completes the symmetric matrix from the lower (LtoR=0) or from the upper (LtoR!=0) part  }

procedure cvCompleteSymm(matrix:PCvMat; LtoR:longint);winapi;external;
{* Mirror array data around horizontal (flip=0),
   vertical (flip=1) or both(flip=-1) axises:
   cvFlip(src) flips images vertically and sequences horizontally (inplace)  }
(* Const before type ignored *)
procedure cvFlip(src:PCvArr; dst:PCvArr; flip_mode:longint);winapi;external;
{* Performs Singular Value Decomposition of a matrix  }

procedure cvSVD(A:PCvArr; W:PCvArr; U:PCvArr; V:PCvArr; flags:longint);winapi;external;
{* Performs Singular Value Back Substitution (solves A*X = B):
   flags must be the same as in cvSVD  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvSVBkSb(W:PCvArr; U:PCvArr; V:PCvArr; B:PCvArr; X:PCvArr; 
            flags:longint);winapi;external;

{* Inverts matrix  }
(* Const before type ignored *)

function cvInvert(src:PCvArr; dst:PCvArr; method:longint):double;winapi;external;
//const
//  cvInv = cvInvert;  
{* Solves linear system (src1)*(dst) = (src2)
   (returns 0 if src1 is a singular and CV_LU method is used)  }
(* Const before type ignored *)
(* Const before type ignored *)

function cvSolve(src1:PCvArr; src2:PCvArr; dst:PCvArr; method:longint):longint;winapi;external;
{* Calculates determinant of input matrix  }
(* Const before type ignored *)
function cvDet(mat:PCvArr):double;winapi;external;
{* Calculates trace of the matrix (sum of elements on the main diagonal)  }
(* Const before type ignored *)
function cvTrace(mat:PCvArr):TCvScalar;winapi;external;
{* Finds eigen values and vectors of a symmetric matrix  }
procedure cvEigenVV(mat:PCvArr; evects:PCvArr; evals:PCvArr; eps:double; lowindex:longint; 
            highindex:longint);winapi;external;
{/* Finds selected eigen values and vectors of a symmetric matrix */ }
{void  cvSelectedEigenVV( CvArr* mat, CvArr* evects, CvArr* evals, }
{                                int lowindex, int highindex ); }
{* Makes an identity matrix (mat_ij = i == j)  }
procedure cvSetIdentity(mat:PCvArr; value:TCvScalar);winapi;external;
{* Fills matrix with given range of numbers  }
function cvRange(mat:PCvArr; start:double; &end:double):PCvArr;winapi;external;
{*   @anchor core_c_CovarFlags
@name Flags for cvCalcCovarMatrix
@see cvCalcCovarMatrix
  @
 }
{* flag for cvCalcCovarMatrix, transpose([v1-avg, v2-avg,...]) * [v1-avg,v2-avg,...]  }


{* @  }
{* Calculates covariation matrix for a set of vectors
@see @ref core_c_CovarFlags "flags"
 }
(* Const before type ignored *)

procedure cvCalcCovarMatrix(vects:PPCvArr; count:longint; cov_mat:PCvArr; avg:PCvArr; flags:longint);winapi;external;

(* Const before type ignored *)

procedure cvCalcPCA(data:PCvArr; mean:PCvArr; eigenvals:PCvArr; eigenvects:PCvArr; flags:longint);winapi;external;
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvProjectPCA(data:PCvArr; mean:PCvArr; eigenvects:PCvArr; result:PCvArr);winapi;external;
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvBackProjectPCA(proj:PCvArr; mean:PCvArr; eigenvects:PCvArr; result:PCvArr);winapi;external;
{* Calculates Mahalanobis(weighted) distance  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvMahalanobis(vec1:PCvArr; vec2:PCvArr; mat:PCvArr):double;winapi;external;
//const
//  cvMahalonobis = cvMahalanobis;  
{***************************************************************************************\
*                                    Array Statistics                                    *
\*************************************************************************************** }
{* Finds sum of array elements  }
(* Const before type ignored *)

function cvSum(arr:PCvArr):TCvScalar;winapi;external;
{* Calculates number of non-zero pixels  }
(* Const before type ignored *)
function cvCountNonZero(arr:PCvArr):longint;winapi;external;
{* Calculates mean value of array elements  }
(* Const before type ignored *)
(* Const before type ignored *)
function cvAvg(arr:PCvArr; mask:PCvArr):TCvScalar;winapi;external;
{* Calculates mean and standard deviation of pixel values  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvAvgSdv(arr:PCvArr; mean:PCvScalar; std_dev:PCvScalar; mask:PCvArr);winapi;external;
{* Finds global minimum, maximum and their positions  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvMinMaxLoc(arr:PCvArr; min_val:Pdouble; max_val:Pdouble; min_loc:PCvPoint; max_loc:PCvPoint; 
            mask:PCvArr);winapi;external;
{* @anchor core_c_NormFlags
  @name Flags for cvNorm and cvNormalize
  @
 }

{* @  }
{* Finds norm, difference norm or relative difference norm for an array (or two arrays)
@see ref core_c_NormFlags "flags"
 }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)

function cvNorm(arr1:PCvArr; arr2:PCvArr; norm_type:longint; mask:PCvArr):double;winapi;external;
{* @see ref core_c_NormFlags "flags"  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvNormalize(src:PCvArr; dst:PCvArr; a:double; b:double; norm_type:longint; 
            mask:PCvArr);winapi;external;
{* @anchor core_c_ReduceFlags
  @name Flags for cvReduce
  @
 }
{* @  }
{* @see @ref core_c_ReduceFlags "flags"  }
(* Const before type ignored *)

procedure cvReduce(src:PCvArr; dst:PCvArr; dim:longint; op:longint);winapi;external;
{***************************************************************************************\
*                      Discrete Linear Transforms and Related Functions                  *
\*************************************************************************************** }
{* @anchor core_c_DftFlags
  @name Flags for cvDFT, cvDCT and cvMulSpectrums
  @
   }
{* @  }
{* Discrete Fourier Transform:
    complex->complex,
    real->ccs (forward),
    ccs->real (inverse)
@see core_c_DftFlags "flags"
 }
(* Const before type ignored *)

procedure cvDFT(src:PCvArr; dst:PCvArr; flags:longint; nonzero_rows:longint);winapi;external;
{* Multiply results of DFTs: DFT(X)*DFT(Y) or DFT(X)*conj(DFT(Y))
@see core_c_DftFlags "flags"
 }
(* Const before type ignored *)
(* Const before type ignored *)

procedure cvMulSpectrums(src1:PCvArr; src2:PCvArr; dst:PCvArr; flags:longint);winapi;external;
{* Finds optimal DFT vector size >= size0  }
function cvGetOptimalDFTSize(size0:longint):longint;winapi;external;
{* Discrete Cosine Transform
@see core_c_DftFlags "flags"
 }
(* Const before type ignored *)
procedure cvDCT(src:PCvArr; dst:PCvArr; flags:longint);winapi;external;
{***************************************************************************************\
*                              Dynamic data structures                                   *
\*************************************************************************************** }
{* Calculates length of sequence slice (with support of negative indices).  }
(* Const before type ignored *)
function cvSliceLength(slice:TCvSlice; seq:PCvSeq):longint;winapi;external;
{* Creates new memory storage.
   block_size == 0 means that default,
   somewhat optimal size, is used (currently, it is 64K)  }
function cvCreateMemStorage(block_size:longint):PCvMemStorage;winapi;external;
{* Creates a memory storage that will borrow memory blocks from parent storage  }
function cvCreateChildMemStorage(parent:PCvMemStorage):PCvMemStorage;winapi;external;
{* Releases memory storage. All the children of a parent must be released before
   the parent. A child storage returns all the blocks to parent when it is released  }
procedure cvReleaseMemStorage(storage:PPCvMemStorage);winapi;external;
{* Clears memory storage. This is the only way(!!!) (besides cvRestoreMemStoragePos)
   to reuse memory allocated for the storage - cvClearSeq,cvClearSet ...
   do not free any memory.
   A child storage returns all the blocks to the parent when it is cleared  }
procedure cvClearMemStorage(storage:PCvMemStorage);winapi;external;
{* Remember a storage "free memory" position  }
(* Const before type ignored *)
procedure cvSaveMemStoragePos(storage:PCvMemStorage; pos:PCvMemStoragePos);winapi;external;
{* Restore a storage "free memory" position  }
procedure cvRestoreMemStoragePos(storage:PCvMemStorage; pos:PCvMemStoragePos);winapi;external;
{* Allocates continuous buffer of the specified size in the storage  }
function cvMemStorageAlloc(storage:PCvMemStorage; size:size_t):Pointer;winapi;external;
{* Allocates string in memory storage  }
{CvString cvMemStorageAllocString( CvMemStorage* storage, const char* ptr, }
{                                         int len CV_DEFAULT(-1) ); }
{* Creates new empty sequence that will reside in the specified storage  }
function cvCreateSeq(seq_flags:longint; header_size:size_t; elem_size:size_t; storage:PCvMemStorage):PCvSeq;winapi;external;
{* Changes default size (granularity) of sequence blocks.
   The default size is ~1Kbyte  }
procedure cvSetSeqBlockSize(seq:PCvSeq; delta_elems:longint);winapi;external;
{* Adds new element to the end of sequence. Returns pointer to the element  }
(* Const before type ignored *)
function cvSeqPush(seq:PCvSeq; element:pointer):PIntPtr;winapi;external;
{* Adds new element to the beginning of sequence. Returns pointer to it  }
(* Const before type ignored *)
function cvSeqPushFront(seq:PCvSeq; element:pointer):PIntPtr;winapi;external;
{* Removes the last element from sequence and optionally saves it  }
procedure cvSeqPop(seq:PCvSeq; element:pointer);winapi;external;
{* Removes the first element from sequence and optioanally saves it  }
procedure cvSeqPopFront(seq:PCvSeq; element:pointer);winapi;external;

{* Adds several new elements to the end of sequence  }
(* Const before type ignored *)

procedure cvSeqPushMulti(seq:PCvSeq; elements:pointer; count:longint; in_front:longint);winapi;external;
{* Removes several elements from the end of sequence and optionally saves them  }
procedure cvSeqPopMulti(seq:PCvSeq; elements:pointer; count:longint; in_front:longint);winapi;external;
{* Inserts a new element in the middle of sequence.
   cvSeqInsert(seq,0,elem) == cvSeqPushFront(seq,elem)  }
(* Const before type ignored *)
function cvSeqInsert(seq:PCvSeq; before_index:longint; element:pointer):PIntPtr;winapi;external;
{* Removes specified sequence element  }
procedure cvSeqRemove(seq:PCvSeq; index:longint);winapi;external;
{* Removes all the elements from the sequence. The freed memory
   can be reused later only by the same sequence unless cvClearMemStorage
   or cvRestoreMemStoragePos is called  }
procedure cvClearSeq(seq:PCvSeq);winapi;external;
{* Retrieves pointer to specified sequence element.
   Negative indices are supported and mean counting from the end
   (e.g -1 means the last sequence element)  }
(* Const before type ignored *)
function cvGetSeqElem(seq:PCvSeq; index:longint):PIntPtr;winapi;external;
{* Calculates index of the specified sequence element.
   Returns -1 if element does not belong to the sequence  }
(* Const before type ignored *)
(* Const before type ignored *)
function cvSeqElemIdx(seq:PCvSeq; element:pointer; block:PPCvSeqBlock):longint;winapi;external;
{* Initializes sequence writer. The new elements will be added to the end of sequence  }
procedure cvStartAppendToSeq(seq:PCvSeq; writer:PCvSeqWriter);winapi;external;
{* Combination of cvCreateSeq and cvStartAppendToSeq  }
procedure cvStartWriteSeq(seq_flags:longint; header_size:longint; elem_size:longint; storage:PCvMemStorage; writer:PCvSeqWriter);winapi;external;
{* Closes sequence writer, updates sequence header and returns pointer
   to the resultant sequence
   (which may be useful if the sequence was created using cvStartWriteSeq))
 }
function cvEndWriteSeq(writer:PCvSeqWriter):PCvSeq;winapi;external;
{* Updates sequence header. May be useful to get access to some of previously
   written elements via cvGetSeqElem or sequence reader  }
procedure cvFlushSeqWriter(writer:PCvSeqWriter);winapi;external;
{* Initializes sequence reader.
   The sequence can be read in forward or backward direction  }
(* Const before type ignored *)
procedure cvStartReadSeq(seq:PCvSeq; reader:PCvSeqReader; reverse:longint);winapi;external;
{* Returns current sequence reader position (currently observed sequence element)  }
function cvGetSeqReaderPos(reader:PCvSeqReader):longint;winapi;external;
{* Changes sequence reader position. It may seek to an absolute or
   to relative to the current position  }
procedure cvSetSeqReaderPos(reader:PCvSeqReader; index:longint; is_relative:longint);winapi;external;
{* Copies sequence content to a continuous piece of memory  }
(* Const before type ignored *)
function cvCvtSeqToArray(seq:PCvSeq; elements:pointer; slice:TCvSlice):Pointer;winapi;external;
{* Creates sequence header for array.
   After that all the operations on sequences that do not alter the content
   can be applied to the resultant sequence  }
function cvMakeSeqHeaderForArray(seq_type:longint; header_size:longint; elem_size:longint; elements:pointer; total:longint; 
           seq:PCvSeq; block:PCvSeqBlock):PCvSeq;winapi;external;
{* Extracts sequence slice (with or without copying sequence elements)  }
(* Const before type ignored *)
function cvSeqSlice(seq:PCvSeq; slice:TCvSlice; storage:PCvMemStorage; copy_data:longint):PCvSeq;winapi;external;
{* Removes sequence slice  }
procedure cvSeqRemoveSlice(seq:PCvSeq; slice:TCvSlice);winapi;external;
{* Inserts a sequence or array into another sequence  }
(* Const before type ignored *)
procedure cvSeqInsertSlice(seq:PCvSeq; before_index:longint; from_arr:PCvArr);winapi;external;
{* a < b ? -1 : a > b ? 1 : 0  }
{typedef int (CV_CDECL* CvCmpFunc)(const void* a, const void* b, void* userdata ); }
{* Sorts sequence in-place given element comparison function  }
procedure cvSeqSort(seq:PCvSeq; func:TCvCmpFunc; userdata:pointer);winapi;external;
{* Finds element in a [sorted] sequence  }
(* Const before type ignored *)
function cvSeqSearch(seq:PCvSeq; elem:pointer; func:TCvCmpFunc; is_sorted:longint; elem_idx:PLongint; 
           userdata:pointer):PIntPtr;winapi;external;
{* Reverses order of sequence elements in-place  }
procedure cvSeqInvert(seq:PCvSeq);winapi;external;
{* Splits sequence into one or more equivalence classes using the specified criteria  }
(* Const before type ignored *)
function cvSeqPartition(seq:PCvSeq; storage:PCvMemStorage; labels:PPCvSeq; is_equal:TCvCmpFunc; userdata:pointer):longint;winapi;external;
{*********** Internal sequence functions *********** }
procedure cvChangeSeqBlock(reader:pointer; direction:longint);winapi;external;
procedure cvCreateSeqBlock(writer:PCvSeqWriter);winapi;external;
{* Creates a new set  }
function cvCreateSet(set_flags:longint; header_size:longint; elem_size:longint; storage:PCvMemStorage):PCvSet;winapi;external;
{* Adds new element to the set and returns pointer to it  }
function cvSetAdd(set_header:PCvSet; elem:PCvSetElem; inserted_elem:PPCvSetElem):longint;winapi;external;
{* Removes element from the set by its index   }
procedure cvSetRemove(set_header:PCvSet; index:longint);winapi;external;
{* Returns a set element by index. If the element doesn't belong to the set,
   NULL is returned  }
{* Removes all the elements from the set  }
procedure cvClearSet(set_header:PCvSet);winapi;external;
{* Creates new graph  }
function cvCreateGraph(graph_flags:longint; header_size:longint; vtx_size:longint; edge_size:longint; storage:PCvMemStorage):PCvGraph;winapi;external;
{* Adds new vertex to the graph  }
(* Const before type ignored *)
function cvGraphAddVtx(graph:PCvGraph; vtx:PCvGraphVtx; inserted_vtx:PPCvGraphVtx):longint;winapi;external;
{* Removes vertex from the graph together with all incident edges  }
function cvGraphRemoveVtx(graph:PCvGraph; index:longint):longint;winapi;external;
function cvGraphRemoveVtxByPtr(graph:PCvGraph; vtx:PCvGraphVtx):longint;winapi;external;
{* Link two vertices specified by indices or pointers if they
   are not connected or return pointer to already existing edge
   connecting the vertices.
   Functions return 1 if a new edge was created, 0 otherwise  }
(* Const before type ignored *)
function cvGraphAddEdge(graph:PCvGraph; start_idx:longint; end_idx:longint; edge:PCvGraphEdge; inserted_edge:PPCvGraphEdge):longint;winapi;external;
(* Const before type ignored *)
function cvGraphAddEdgeByPtr(graph:PCvGraph; start_vtx:PCvGraphVtx; end_vtx:PCvGraphVtx; edge:PCvGraphEdge; inserted_edge:PPCvGraphEdge):longint;winapi;external;
{* Remove edge connecting two vertices  }
procedure cvGraphRemoveEdge(graph:PCvGraph; start_idx:longint; end_idx:longint);winapi;external;
procedure cvGraphRemoveEdgeByPtr(graph:PCvGraph; start_vtx:PCvGraphVtx; end_vtx:PCvGraphVtx);winapi;external;
{* Find edge connecting two vertices  }
(* Const before type ignored *)
function cvFindGraphEdge(graph:PCvGraph; start_idx:longint; end_idx:longint):PCvGraphEdge;winapi;external;
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvFindGraphEdgeByPtr(graph:PCvGraph; start_vtx:PCvGraphVtx; end_vtx:PCvGraphVtx):PCvGraphEdge;winapi;external;
//const
//  cvGraphFindEdge = cvFindGraphEdge;  
//  cvGraphFindEdgeByPtr = cvFindGraphEdgeByPtr;  
{* Remove all vertices and edges from the graph  }

procedure cvClearGraph(graph:PCvGraph);winapi;external;
{* Count number of edges incident to the vertex  }
(* Const before type ignored *)
function cvGraphVtxDegree(graph:PCvGraph; vtx_idx:longint):longint;winapi;external;
(* Const before type ignored *)
(* Const before type ignored *)
function cvGraphVtxDegreeByPtr(graph:PCvGraph; vtx:PCvGraphVtx):longint;winapi;external;
{* Retrieves graph vertex by given index  }
{ was #define dname(params) para_def_expr }
{ argument types are unknown }
//function cvGetGraphVtx(graph,idx : longint) : PCvGraphVtx;

{* Retrieves index of a graph vertex given its pointer  }
{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   
//function cvGraphVtxIdx(graph,vtx : longint) : longint;

{* Retrieves index of a graph edge given its pointer  }
{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   
//function cvGraphEdgeIdx(graph,edge : longint) : longint;

{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   
//function cvGraphGetVtxCount(graph : longint) : longint;

{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   
//function cvGraphGetEdgeCount(graph : longint) : longint;

{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   

//function CV_IS_GRAPH_VERTEX_VISITED(vtx : longint) : longint;

{ was #define dname(params) para_def_expr }
{ argument types are unknown }
{ return type might be wrong }   
//function CV_IS_GRAPH_EDGE_VISITED(edge : longint) : longint;


{* Creates new graph scanner.  }

function cvCreateGraphScanner(graph:PCvGraph; vtx:PCvGraphVtx; mask:longint):PCvGraphScanner;winapi;external;
{* Releases graph scanner.  }
procedure cvReleaseGraphScanner(scanner:PPCvGraphScanner);winapi;external;
{* Get next graph element  }
function cvNextGraphItem(scanner:PCvGraphScanner):longint;winapi;external;
{* Creates a copy of graph  }
(* Const before type ignored *)
function cvCloneGraph(graph:PCvGraph; storage:PCvMemStorage):PCvGraph;winapi;external;
{* Does look-up transformation. Elements of the source array
   (that should be 8uC1 or 8sC1) are used as indexes in lutarr 256-element table  }
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvLUT(src:PCvArr; dst:PCvArr; lut:PCvArr);winapi;external;
{****************** Iteration through the sequence tree **************** }
(* Const before type ignored *)

(* Const before type ignored *)

procedure cvInitTreeNodeIterator(tree_iterator:PCvTreeNodeIterator; first:pointer; max_level:longint);winapi;external;
function cvNextTreeNode(tree_iterator:PCvTreeNodeIterator):Pointer;winapi;external;
function cvPrevTreeNode(tree_iterator:PCvTreeNodeIterator):Pointer;winapi;external;
{* Inserts sequence into tree with specified "parent" sequence.
   If parent is equal to frame (e.g. the most external contour),
   then added contour will have null pointer to parent.  }
procedure cvInsertNodeIntoTree(node:pointer; parent:pointer; frame:pointer);winapi;external;
{* Removes contour from tree (together with the contour children).  }
procedure cvRemoveNodeFromTree(node:pointer; frame:pointer);winapi;external;
{* Gathers pointers to all the sequences,
   accessible from the `first`, to the single sequence  }
(* Const before type ignored *)
function cvTreeToNodeSeq(first:pointer; header_size:longint; storage:PCvMemStorage):PCvSeq;winapi;external;
{* The function implements the K-means algorithm for clustering an array of sample
   vectors in a specified number of classes  }


(* Const before type ignored *)

function cvKMeans2(samples:PCvArr; cluster_count:longint; labels:PCvArr; termcrit:TCvTermCriteria; attempts:longint; 
           rng:PCvRNG; flags:longint; _centers:PCvArr; compactness:Pdouble):longint;winapi;external;
{***************************************************************************************\
*                                    System functions                                    *
\*************************************************************************************** }
{* Loads optimized functions from IPP, MKL etc. or switches back to pure C code  }
function cvUseOptimized(on_off:longint):longint;winapi;external;
{* @brief Makes OpenCV use IPL functions for allocating IplImage and IplROI structures.

Normally, the function is not called directly. Instead, a simple macro
CV_TURN_ON_IPL_COMPATIBILITY() is used that calls cvSetIPLAllocators and passes there pointers
to IPL allocation functions. :
@code
    ...
    CV_TURN_ON_IPL_COMPATIBILITY()
    ...
@endcode
@param create_header pointer to a function, creating IPL image header.
@param allocate_data pointer to a function, allocating IPL image data.
@param deallocate pointer to a function, deallocating IPL image.
@param create_roi pointer to a function, creating IPL image ROI (i.e. Region of Interest).
@param clone_image pointer to a function, cloning an IPL image.
  }
procedure cvSetIPLAllocators(create_header:TCv_iplCreateImageHeader; allocate_data:TCv_iplAllocateImageData; deallocate:TCv_iplDeallocate; create_roi:TCv_iplCreateROI; clone_image:TCv_iplCloneImage);winapi;external;
{* @brief Releases an object.

 The function finds the type of a given object and calls release with the double pointer.
 @param struct_ptr Double pointer to the object
  }
procedure cvRelease(struct_ptr:Ppointer);winapi;external;
{* @brief Makes a clone of an object.

The function finds the type of a given object and calls clone with the passed object. Of course, if
you know the object type, for example, struct_ptr is CvMat\*, it is faster to call the specific
function, like cvCloneMat.
@param struct_ptr The object to clone
  }
(* Const before type ignored *)
function cvClone(struct_ptr:pointer):Pointer;winapi;external;
{********************************** Measuring Execution Time ************************** }
{* helper functions for RNG initialization and accurate time measurement:
   uses internal clock counter on x86  }
function cvGetTickCount:int64;winapi;external;
function cvGetTickFrequency:double;winapi;external;
{********************************** CPU capabilities ********************************** }
function cvCheckHardwareSupport(feature:longint):longint;winapi;external;
{********************************** Multi-Threading *********************************** }
{* retrieve/set the number of threads used in OpenMP implementations  }
function cvGetNumThreads:longint;winapi;external;
procedure cvSetNumThreads(threads:longint);winapi;external;
{* get index of the thread being executed  }
function cvGetThreadNum:longint;winapi;external;
{********************************* Error Handling ************************************* }
{* Get current OpenCV error status  }
function cvGetErrStatus:longint;winapi;external;
{* Sets error status silently  }
procedure cvSetErrStatus(status:longint);winapi;external;
{ Print error and exit program  }

{* Retrieves current error processing mode  }

function cvGetErrMode:longint;winapi;external;
{* Sets error processing mode, returns previously used mode  }
function cvSetErrMode(mode:longint):longint;winapi;external;
{* Sets error status and performs some additional actions (displaying message box,
 writing message to stderr, terminating application etc.)
 depending on the current error mode  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
procedure cvError(status:longint; func_name:PChar; err_msg:PChar; file_name:PChar; line:longint);winapi;external;
{* Retrieves textual description of the error given its code  }
function cvErrorStr(status:longint):PChar;winapi;external;
{* Retrieves detailed information about the last error occurred  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvGetErrInfo(errcode_desc:PPchar; description:PPchar; filename:PPchar; line:PLongint):longint;winapi;external;
{* Maps IPP error codes to the counterparts from OpenCV  }
function cvErrorFromIppStatus(ipp_status:longint):longint;winapi;external;
{* Assigns a new error-handling function  }
function cvRedirectError(error_handler:TCvErrorCallback; userdata:pointer; prev_userdata:Ppointer):TCvErrorCallback;winapi;external;
{* Output nothing  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvNulDevReport(status:longint; func_name:PChar; err_msg:PChar; file_name:PChar; line:longint; 
           userdata:pointer):longint;winapi;external;
{* Output to console(fprintf(stderr,...))  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvStdErrReport(status:longint; func_name:PChar; err_msg:PChar; file_name:PChar; line:longint; 
           userdata:pointer):longint;winapi;external;
{* Output to MessageBox(WIN32)  }
(* Const before type ignored *)
(* Const before type ignored *)
(* Const before type ignored *)
function cvGuiBoxReport(status:longint; func_name:PChar; err_msg:PChar; file_name:Pchar; line:longint; 
           userdata:pointer):longint;winapi;external;

implementation

function CV_FOURCC(const c1, c2, c3, c4: char): longint;
begin
  result:=((byte(c1) and 255) + ((byte(c2) and 255) shl 8) + ((byte(c3) and 255) shl 16) + ((byte(c4) and 255) shl 24))
end;

initialization


end.
