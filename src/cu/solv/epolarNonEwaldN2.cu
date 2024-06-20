#include "ff/modamoeba.h"
#include "ff/cumodamoeba.h"
#include "ff/solv/nblistgk.h"
#include "ff/switch.h"
#include "seq/epolartorque.h"
#include "seq/launch.h"
#include "seq/pair_polar.h"
#include "seq/triangle.h"

namespace tinker {
#include "epolarNonEwaldN2_cu1.cc"

template <class Ver>
static void epolarNonEwaldN2_cu(const real (*uind)[3], const real (*uinp)[3])
{
   constexpr bool do_g = Ver::g;

   const auto& st = *mdloop_unit;
   real off = switchOff(Switch::MPOLE);

   const real f = 0.5f * electric / dielec;

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, ufld, dufld);
   int ngrid = gpuGridSize(BLOCK_DIM);
   epolarNonEwaldN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nep, ep, vir_ep, depx, depy, depz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, x, y, z, st.nakpl, st.iakpl,
      st.nakpa, st.iakpa, ufld, dufld, rpole, uind, uinp, f);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorque_cu, //
         trqx, trqy, trqz, n, rpole, ufld, dufld);
   }
}

void epolarNonEwaldN2_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   if (vers == calc::v0) {
      epolarNonEwaldN2_cu<calc::V0>(uind, uinp);
   } else if (vers == calc::v3) {
      epolarNonEwaldN2_cu<calc::V3>(uind, uinp);
   } else if (vers == calc::v4) {
      epolarNonEwaldN2_cu<calc::V4>(uind, uinp);
   } else if (vers == calc::v5) {
      epolarNonEwaldN2_cu<calc::V5>(uind, uinp);
   }
}
}
