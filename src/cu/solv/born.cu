#include "ff/solv/born.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_born.h"
#include "seq/triangle.h"
#include <cmath>

namespace tinker
{

   #include "grycuk_cu1.cc"
   #include "hctobc_cu1.cc"

__global__
static void bornInit_cu1(int n, real doffset, const real* restrict rsolv, real* restrict roff, real* restrict drobc)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      roff[i] = rsolv[i] - doffset;
      drobc[i] = 1.f;
   }
}

__global__
static void grycukInit_cu1(int n, real pi43, const real* restrict rsolv, real* restrict rborn)
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
static void hctInit_cu1(int n, const real* restrict roff, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rborn[i] = 1.f / roff[i]; 
   }
}

__global__
static void obcInit_cu1(int n, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rborn[i] = 0.f; 
   }
}

__global__
static void grycukFinal_cu1(int n, real pi43, const real* restrict rsolv, real* restrict rborn)
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
         rborn[i] = 1.f / rborni;
      }
   }
}

__global__
static void hctFinal_cu1(int n, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rborn[i] = 1.f / rborn[i];
   }
}

__global__
static void obcFinal_cu1(int n, const real* restrict roff, const real* restrict rsolv, const real* restrict aobc, const real* restrict bobc, const real* restrict gobc, real* restrict rborn, real* restrict drobc)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real roi = roff[i];
      real rsi = rsolv[i];
      real alpha = aobc[i];
      real beta = bobc[i];
      real gamma = gobc[i];
      real sum = roi * (-rborn[i]);
      real sum2 = sum * sum;
      real sum3 = sum * sum2;
      real tsum = std::tanh(alpha*sum - beta*sum2 + gamma*sum3);
      real rborni = 1.f/roi - tsum/rsi;
      rborn[i] = 1.f / rborni;
      real tchain = roi * (alpha-2.f*beta*sum+3.f*gamma*sum2);
      drobc[i] = (1.f-tsum*tsum) * tchain / rsi;
   }
}

__global__
static void bornPrint_cu1(int n, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      # if __CUDA_ARCH__>=200
      printf("implicitsolvent %d %f \n", i, rborn[i]);
      #endif  
   }
}

void grycuk_cu(int vers)
{
   real pi43 = 4. * M_PI / 3.;
   const auto& st = *mspatial_v2_unit;

   launch_k1s(g::s0, n, grycukInit_cu1, n, pi43, rsolv, rborn);
   
   int ngrid = gpuGridSize(BLOCK_DIM);
   grycuk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
      rborn, rsolv, rdescr, shct, pi43);
   
   launch_k1s(g::s0, n, grycukFinal_cu1, n, pi43, rsolv, rborn);

   launch_k1s(g::s0, n, bornPrint_cu1, n, rborn);
}

void hct_cu(int vers)
{
   const auto& st = *mspatial_v2_unit;

   launch_k1s(g::s0, n, hctInit_cu1, n, roff, rborn);

   int ngrid = gpuGridSize(BLOCK_DIM);
   hctobc_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
      rborn, roff, shct);

   launch_k1s(g::s0, n, hctFinal_cu1, n, rborn);

   launch_k1s(g::s0, n, bornPrint_cu1, n, rborn);
}

void obc_cu(int vers)
{
   const auto& st = *mspatial_v2_unit;

   launch_k1s(g::s0, n, obcInit_cu1, n, rborn);

   int ngrid = gpuGridSize(BLOCK_DIM);
   hctobc_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
      rborn, roff, shct);

   launch_k1s(g::s0, n, obcFinal_cu1, n, roff, rsolv, aobc, bobc, gobc, rborn, drobc);

   launch_k1s(g::s0, n, bornPrint_cu1, n, rborn);
}

void born_cu(int vers)
{
   launch_k1s(g::s0, n, bornInit_cu1, n, doffset, rsolv, roff, drobc);

   if (borntyp == Born::GRYCUK) {
      grycuk_cu(vers);
   }
   else if (borntyp == Born::HCT) {
      hct_cu(vers);
   }
   else if (borntyp == Born::OBC) {
      obc_cu(vers);
   }
}
}
