#include "ff/solv/nblistgk.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_born.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>
#include <cmath>

namespace tinker {
__global__
void grycukFinal_cu1(int n, real pi43, real maxbrad, bool usetanh, const real* restrict rsolv, real* restrict rborn, real* restrict bornint)
{
   real third = (real)0.333333333333333333;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real ri = rsolv[i];
      if (ri > 0) {
         real rborni = rborn[i];
         if (usetanh) {
            bornint[i] = rborni;
            tanhrsc(rborni,ri,pi43);
         }
         rborni = pi43 / REAL_POW(ri,3) + rborni;
         if (rborni < 0) {
            rborn[i] = maxbrad;
         } else {
            rborni = REAL_POW((rborni/pi43),third);
            rborni = 1 / rborni;
            rborn[i] = rborni;
            if (rborni < ri) {
               rborn[i] = ri;
            } else if (rborni > maxbrad) {
               rborn[i] = maxbrad;
            } else if (isinf(rborni) or isnan(rborni)) {
               rborn[i] = ri;
            }
         }
      }
   }
}

#include "grycuk_cu1.cc"
#include "grycukN2_cu1.cc"

static void grycuk_cu2()
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   real third = (real)0.333333333333333333;
   real pi43 = 4 * third * pi;

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      grycuk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);
   } else {
      const auto& st = *mdloop_unit;
      grycukN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, off, x, y, z, st.nakp, st.iakp,
         descoff, pi43, useneck, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck);
   }

   real maxbrad = 30;
   launch_k1s(g::s0, n, grycukFinal_cu1, n, pi43, maxbrad, usetanh, rsolv, rborn, bornint);
}

void born_cu()
{
   if (borntyp == Born::GRYCUK) {
      grycuk_cu2();
   } else {
      throwExceptionMissingFunction("born_cu", __FILE__, __LINE__);
   }
}
}

namespace tinker {
#include "grycuk1_cu1.cc"
#include "grycuk1N2_cu1.cc"

static void born1_cu2()
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   bool use_gk = false;
   if (solvtyp == Solv::GK) use_gk = true;

   real third = (real)0.333333333333333333;
   real pi43 = 4 * third * pi;
   real factor = -REAL_POW(pi,third) * REAL_POW((real)6.,2*third) / 9;

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      grycuk1_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         descoff, pi43, factor, useneck, usetanh, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint, use_gk);
   } else {
      const auto& st = *mdloop_unit;
      grycuk1N2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, desx, desy, desz, off, x, y, z, st.nakp, st.iakp,
         descoff, pi43, factor, useneck, usetanh, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint, use_gk);
   }
}

void born1_cu(int vers)
{
   auto do_g = vers & calc::grad;
   if (borntyp == Born::GRYCUK) {
      if (do_g)
         born1_cu2();
   } else {
      throwExceptionMissingFunction("born1_cu", __FILE__, __LINE__);
   }
}
}
