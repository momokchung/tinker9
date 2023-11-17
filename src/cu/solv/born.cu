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

template <class Ver>
static void grycuk_cu2()
{
   const auto& st = *mspatial_v2_unit;

   int ngrid = gpuGridSize(BLOCK_DIM);

   real pi43 = 4./3. * pi;
   grycuk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
                descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);

   launch_k1s(g::s0, n, grycukFinal_cu1, n, pi43, usetanh, rsolv, rborn, bornint);
}

// __global__
// static void bornPrint_cu1(int n, const real* restrict rborn)
// {
//    for (int i = ITHREAD; i < n; i += STRIDE) {
//       real rborni = rborn[i];
//       # if __CUDA_ARCH__>=200
//       printf("implicitsolvent %5d %10.6e \n", i, rborni);
//       #endif  
//    }
// }

void born_cu(int vers)
{
   if (borntyp == Born::GRYCUK) {
      if (vers == calc::v0)
         grycuk_cu2<calc::V0>();
      else if (vers == calc::v3)
         grycuk_cu2<calc::V3>();
      else if (vers == calc::v4)
         grycuk_cu2<calc::V4>();
      else if (vers == calc::v5)
         grycuk_cu2<calc::V5>();
   } else {
      throwExceptionMissingFunction("born_cu", __FILE__, __LINE__);
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

   if (borntyp == Born::GRYCUK) {
      if (vers == calc::v4)
         born1_cu2<calc::V4>(use_gk);
      else if (vers == calc::v5)
         born1_cu2<calc::V5>(use_gk);
   } else {
      throwExceptionMissingFunction("born_cu", __FILE__, __LINE__);
   }
}
}
