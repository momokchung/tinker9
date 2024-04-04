#include "ff/solv/solute.h"
#include "ff/dloop.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_born.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>
#include <cmath>

namespace tinker
{
#include "grycuk_cu1.cc"
#include "grycukN2_cu1.cc"

template <class Ver>
static void grycuk_cu2()
{
   int ngrid = gpuGridSize(BLOCK_DIM);

   real pi43 = (real)4/3 * pi;

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      grycuk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);
   } else {
      const auto& st = *mn2_unit;
      grycukN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, x, y, z, st.nakp, st.iakp,
         descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);
   }

   real maxbrad = 30;
   launch_k1s(g::s0, n, grycukFinal_cu1, n, pi43, maxbrad, usetanh, rsolv, rborn, bornint);
}

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
#include "grycuk1N2_cu1.cc"

template <class Ver>
static void born1_cu2()
{
   int ngrid = gpuGridSize(BLOCK_DIM);

   bool use_gk = false;
   if (solvtyp == Solv::GK) use_gk = true;

   real third = (real)0.333333333333333333;
   real pi43 = 4 * third * pi;
   real factor = -REAL_POW(pi,third) * REAL_POW((real)6.,2*third) / 9;

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      grycuk1_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, desx, desy, desz, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         descoff, pi43, factor, useneck, usetanh, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint, use_gk);
   } else {
      const auto& st = *mn2_unit;
      grycuk1N2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, desx, desy, desz, x, y, z, st.nakp, st.iakp,
         descoff, pi43, factor, useneck, usetanh, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint, use_gk);
   }
}

void born1_cu(int vers)
{
   if (borntyp == Born::GRYCUK) {
      if (vers == calc::v4)
         born1_cu2<calc::V4>();
      else if (vers == calc::v5)
         born1_cu2<calc::V5>();
   } else {
      throwExceptionMissingFunction("born1_cu", __FILE__, __LINE__);
   }
}
}
