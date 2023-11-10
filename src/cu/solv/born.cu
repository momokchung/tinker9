#include "ff/solv/solute.h"
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
static void grycukFinal_cu1(int n, real pi43, bool usetanh, const real* restrict rsolv, real* restrict rborn, real* restrict bornint)
{
   real third = 0.333333333333333333;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real ri = rsolv[i];
      if (ri > 0.f) {
         real rborni = rborn[i];
         if (usetanh) {
            bornint[i] = rborni;
            tanhrsc(rborni,ri,pi43);
         }
         rborni = pi43 / REAL_POW(ri,3) + rborni;
         rborni = REAL_POW((rborni/pi43),third);
         if (rborni < 0.) rborni = 0.0001;
         rborn[i] = 1.f / rborni;
      }
   }
}

__global__
static void hctFinal_cu1(int n, const real* restrict roff, real* restrict rborn)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real rborni = rborn[i] + 1.f/roff[i];
      rborn[i] = 1.f / rborni;
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
      real rborni = rborn[i];
      real sum = roi * (-rborni);
      real sum2 = sum * sum;
      real sum3 = sum * sum2;
      real tsum = std::tanh(alpha*sum - beta*sum2 + gamma*sum3);
      rborni = 1.f/roi - tsum/rsi;
      rborn[i] = 1.f / rborni;
      real tchain = roi * (alpha-2.f*beta*sum+3.f*gamma*sum2);
      drobc[i] = (1.f-tsum*tsum) * tchain / rsi;
   }
}

// __global__
// static void bornPrint_cu1(int n, const real* restrict rborn)
// {
//    for (int i = ITHREAD; i < n; i += STRIDE) {
//       real rborni = rborn[i];
//       # if __CUDA_ARCH__>=200
//       printf("implicitsolvent %d %10.6e \n", i, rborni);
//       #endif  
//    }
// }

void bornInit_cu(int vers)
{
   if (borntyp == Born::HCT or borntyp == Born::OBC) {
      launch_k1s(g::s0, n, bornInit_cu1, n, doffset, rsolv, roff, drobc);
   }
}

void born_cu(int vers)
{
   const auto& st = *mspatial_v2_unit;

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (borntyp == Born::GRYCUK) {
      real pi43 = 4./3. * pi;
      grycuk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
                  descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);
   }
   else if (borntyp == Born::HCT) {
      hctobc_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
                  rborn, roff, shct);
   }
   else if (borntyp == Born::OBC) {
      hctobc_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
                  rborn, roff, shct);
   }
}

void bornFinal_cu(int vers)
{
   if (borntyp == Born::GRYCUK) {
      real pi43 = 4./3. * pi;
      launch_k1s(g::s0, n, grycukFinal_cu1, n, pi43, usetanh, rsolv, rborn, bornint);
   }
   else if (borntyp == Born::HCT) {
      launch_k1s(g::s0, n, hctFinal_cu1, n, roff, rborn);
   }
   else if (borntyp == Born::OBC) {
      launch_k1s(g::s0, n, obcFinal_cu1, n, roff, rsolv, aobc, bobc, gobc, rborn, drobc);
   }
}
}

namespace tinker
{
#include "grycuk1_cu1.cc"

template <class Ver>
static void born1_cu2(bool use_gk)
{
   const auto& st = *mspatial_v2_unit;

   int ngrid = gpuGridSize(BLOCK_DIM);

   real third = 0.333333333333333333;
   real pi43 = 4. * third * pi;
   real factor = -REAL_POW(pi,third) * REAL_POW(6.,2.*third) / 9.;

   grycuk1_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, vir_es, desx, desy, desz, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, descoff, pi43, factor, useneck, usetanh, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint, use_gk);
}

void born1_cu(int vers)
{
   bool use_gk = false;
   if (solvtyp == Solv::GK) use_gk = true;

   if (vers == calc::v1)
      born1_cu2<calc::V1>(use_gk);
   else if (vers == calc::v4)
      born1_cu2<calc::V4>(use_gk);
   else if (vers == calc::v5)
      born1_cu2<calc::V5>(use_gk);
   else if (vers == calc::v6)
      born1_cu2<calc::V6>(use_gk);
}
}