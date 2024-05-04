 {$Z4}
{$IFDEF FPC}
{$MACRO ON}
{$PACKRECORDS C}
{$ENDIF}

{$ifdef MSWINDOWS}
{$undef MKL_TBB}
{$undef MKL_OMP}
{$endif}            // Got errors while trying to staticly link libraries in WINDOWS only so
                                                                    // keeping it dynamic for now, however!, comment this if you know what you are doing

//{$define UPPERCASE_DECL}                                          // uncomment to use FORTRAN UpperCase blas functions

{$ifndef UPPERCASE_DECL}
  {$define LOWERCASE_DECL}
{$endif}

//{$define _CALLING:=cdecl;}

{$if defined(MKL_TBB) or defined(MKL_OMP) or defined(MKL_SEQUENTIAL)}
  {$ifdef MSWINDOWS}
      {$if defined(MKL_TBB)}
        {$linklib tbb12.lib}
        {$linklib mkl_tbb_thread.lib}
      {$elseif defined(MKL_OMP)}
        {$linklib mkl_intel_thread.lib}
        {$linklib libiomp5md.lib}
      {$else}
        {$linklib mkl_sequential.lib}
      {$endif}
      {$linklib mkl_intel_lp64.lib}
      {$linklib mkl_core.lib}
  {$endif}

  {$ifdef DARWIN}
    {$if defined(MKL_TBB)}                 // use (Thread Building Blocks) library for parallel comuting
                                           //https://en.wikipedia.org/wiki/Threading_Building_Blocks
      {$linklib libmkl_tbb_thread.a}
      {$linklib tbb}
      {$linklib c++}
    {$elseif defined(MKL_OMP)}            // use intel® version of (Open Multi Processing) library
                                          // for parallel computing  https://en.wikipedia.org/wiki/OpenMP
      {$linklib libmkl_intel_thread.a}
      {$linklib iomp5}
    {$else}
      {$linklib libmkl_sequential.a}      // use a single core computation
    {$endif}
      {.$linklib m}                        // Libraries were suggested by "./mkl_link_tool" helper
      {.$linklib dl}                       // Libraries were suggested by "./mkl_link_tool" helper
      {.$linklib pthread}                  // Libraries were suggested by "./mkl_link_tool" helper
      {$linklib libmkl_intel_lp64.a}      // Libraries were suggested by "./mkl_link_tool" helper
      {$linklib libmkl_core.a}            // Libraries were suggested by "./mkl_link_tool" helper
  {$endif}
  {$ifdef LINUX}
    {$if defined(MKL_TBB)}                //
      {$linklib libmkl_tbb_thread.a}
      {$linklib tbb}
      {$linklib stdc++}
    {$elseif defined(MKL_OMP)}
      {$linklib libmkl_intel_thread.a}
      {$linklib iomp5}
    {$else}
      {$linklib libmkl_sequential.a}
    {$endif}
      {$linklib m}
      {$linklib dl}
      {$linklib pthread}
      {$ifdef ilp64}
        {$linklib libmkl_intel_ilp64.a}    //use 8 bit addressing and sizes
      {$else}
        {$linklib libmkl_intel_lp64.a}     //use 4 bit addressing and sizes
      {$endif}
      {$linklib libmkl_core.a}
  {$endif}

{$else}
  {$define LIBMKL}
  {$ifdef MSWINDOWS}
    const libmkl='mkl_rt.2.dll';         // also comes along with other redistibutable dynamic link binaries in the same folder
  {$endif}

  {$ifdef darwin}
    const libmkl='libmkl_rt.dylib';
  {$endif}
  {$ifdef linux}
    const libmkl='libmkl_rt.so'    ;
  {$endif}
{$endif}

