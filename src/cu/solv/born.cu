#include "ff/solv/born.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_born.h"
#include "seq/triangle.h"

namespace tinker
{
   #include "born_cu1.cc"

__global__
static void bornInit_cu1(int n, real pi43, const real* restrict rsolv, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real ri;
      ri = rsolv[i];
      real sum = 0.f;
      if (ri > 0.f) {
         sum = pi43 / REAL_POW(ri,3);
      }
      rborn[i] = sum; 
   }
}

__global__
static void bornFinal_cu1(int n, real pi43, const real* restrict rsolv, real* restrict rborn)
{
   real third = 0.333333333333333333;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real ri;
      ri = rsolv[i];
      if (ri > 0.f) {
         real rborni = rborn[i];
         rborni = REAL_POW((rborni/pi43),third);
         if (rborni < 0.) {
            rborni = 0.0001;
         }
         rborn[i] = 1. / rborni;
      }
   }
}

__global__
static void bornPrint_cu1(int n, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      # if __CUDA_ARCH__>=200
      printf("%d %f \n", i, rborn[i]);
      #endif  
   }
}

void born_cu(int vers)
{
   real pi43 = 4. * M_PI / 3;
   const auto& st = *mspatial_v2_unit;

   launch_k1s(g::s0, n, bornInit_cu1, n, pi43, rsolv, rborn);
   
   int ngrid = gpuGridSize(BLOCK_DIM);
   born_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
      rborn, rsolv, rdescr, shct, pi43);
   
   launch_k1s(g::s0, n, bornFinal_cu1, n, pi43, rsolv, rborn);

   // launch_k1s(g::s0, n, bornPrint_cu1, n, rborn);
}
}
