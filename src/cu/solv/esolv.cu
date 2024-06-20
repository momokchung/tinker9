#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/cumodamoeba.h"
#include "ff/modamoeba.h"
#include "ff/solv/nblistgk.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_solv.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>
#include <cmath>

namespace tinker {
#include "ewca_cu1.cc"
#include "ewcaN2_cu1.cc"

template <class Ver>
static void ewca_cu2()
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      ewca_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         epsdsp, raddsp, epso, epsh, rmino, rminh, shctd, dspoff, slevy, awater);
   } else {
      const auto& st = *mdloop_unit;
      ewcaN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, es, desx, desy, desz, off, x, y, z, st.nakp, st.iakp,
         epsdsp, raddsp, epso, epsh, rmino, rminh, shctd, dspoff, slevy, awater);
   }

   launch_k1s(g::s0, n, ewcaFinal_cu1<Ver>, n, cdsp, nes, es);
}

void ewca_cu(int vers)
{
   if (vers == calc::v0)
      ewca_cu2<calc::V0>();
   else if (vers == calc::v3)
      ewca_cu2<calc::V3>();
   else if (vers == calc::v4)
      ewca_cu2<calc::V4>();
   else if (vers == calc::v5)
      ewca_cu2<calc::V5>();
}
}

namespace tinker {
#include "egka_cu1.cc"
#include "egkaN2_cu1.cc"

template <class Ver>
static void egka_cu2()
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   real dwater = (real)78.3;
   real fc = electric * 1 * (1-dwater)/(0+1*dwater);
   real fd = electric * 2 * (1-dwater)/(1+2*dwater);
   real fq = electric * 3 * (1-dwater)/(2+3*dwater);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      egka_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
   } else {
      const auto& st = *mdloop_unit;
      egkaN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, nes, es, desx, desy, desz, off, x, y, z, st.nakp, st.iakp,
         trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
   }

   launch_k1s(g::s0, n, egkaFinal_cu1<Ver>, n, nes, es, drb, drbp, trqx, trqy, trqz, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
}

void egka_cu(int vers)
{
   if (vers == calc::v0)
      egka_cu2<calc::V0>();
   else if (vers == calc::v3)
      egka_cu2<calc::V3>();
   else if (vers == calc::v4)
      egka_cu2<calc::V4>();
   else if (vers == calc::v5)
      egka_cu2<calc::V5>();
}
}

namespace tinker {
#include "ediff_cu1.cc"
#include "ediffN2_cu1.cc"

template <class Ver>
static void ediff_cu2()
{
   const real off = switchOff(Switch::MPOLE);
   const real f = electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      ediff_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, desx, desy, desz,
         off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
         st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
   } else {
      const auto& st = *mdloop_unit;
      ediffN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nes, es, desx, desy, desz,
         off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, x, y, z, st.nakpl, st.iakpl,
         st.nakpa, st.iakpa, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
   }
}

void ediff_cu(int vers)
{
   if (vers == calc::v0)
      ediff_cu2<calc::V0>();
   else if (vers == calc::v3)
      ediff_cu2<calc::V3>();
   else if (vers == calc::v4)
      ediff_cu2<calc::V4>();
   else if (vers == calc::v5)
      ediff_cu2<calc::V5>();
}
}

namespace tinker {
void addToEnrgy_cu()
{
   addToEnrgy_cu1<<<1, 1, 0, g::s0>>>(es, cave);
}

void addToGrad_cu()
{
   launch_k1s(g::s0, n, addToGrad_cu1, n, desx, desy, desz, decvx, decvy, decvz);
}
}
