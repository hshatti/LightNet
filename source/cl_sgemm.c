inline float mapX(const float x){
  return x*3-2.1F;
}
// Same purpose as mapX
// [0, 1] -> [-1.25, 1.25]
inline float mapY(const float y){
  return y*3 - 1.5F;
}

#define max_iteration  10000
#define _max           4.0f


__kernel void mandel(__global uchar *buf, const int w, const int h){

  const float lnxp1_max_iteration = log1p((float)max_iteration);

  int y = get_global_id(0);
  int x = get_global_id(1);
  float xx = mapX(x/(float)w);
  float yy = mapY(y/(float)h);

  y *= w * sizeof(uint);
  x *= sizeof(uint);

  float x0 = 0.0f; float y0 = 0.0f;
  int iteration = 0;
  float oldAbs = 0.0f;
  float coverageNum = max_iteration;
  buf += y;
  while (iteration < max_iteration) {
      float xtemp = x0 * x0 - y0 * y0;
      y0 = 2 * x0 * y0;
      x0 = xtemp;
      x0 = x0 + xx;
      y0 = y0 + yy;
      float currentAbs = x0*x0 + y0*y0;
      if (currentAbs>4.0f){
         float diffToLast  = currentAbs - oldAbs;
         float diffToMax   =       _max - oldAbs;
         coverageNum = iteration + diffToMax/diffToLast;
         break;
      }
      oldAbs = currentAbs;
      iteration++;
  }
  if (iteration == max_iteration)
//#if defined(__APPLE__)
  {
      buf[x] = 0xff;
      buf[x+1] = 0;
      buf[x+2] = 0;
      buf[x+3] = 0;
  } else
  {
      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
      buf[x+0] = 0xff;
      buf[x+1] = c;
      buf[x+2] = c;
      buf[x+3] = c;
   }
//#else
//  {
//      buf[x] = 0;
//      buf[x+1] = 0;
//      buf[x+2] = 0;
//      buf[x+3] = 0xff;
//  } else
//  {
//      uchar c = 0xff * log1p(coverageNum)/lnxp1_max_iteration;
//      buf[x+0] = c;
//      buf[x+1] = c;
//      buf[x+2] = c;
//      buf[x+3] = 0xff;
//  }
////#endif
}


//__kernel void sgemm(
//         long long M, long long N, long long K, float ALPHA,
//         __global float *A, long long lda,
//         __global float *B, long long ldb,
//         __global float *C, long long ldc) {
//
////
////    for (int i = 0; i<M ; i++)
////      for (int kk = 0 ; kk<K ; kk++) {
//          int i,kk;
//          i  = get_global_id(0);
//          kk = get_global_id(1);
//          float A_PART = ALPHA * A[i * lda + kk];
//          for (int j=0 ; j<N; j++)
//              C[i*ldc + j] += A_PART*B[kk*ldb + j];
//      //}
//
//}

