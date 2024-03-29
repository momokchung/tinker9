#include "ff/modamoeba.h"
#include "ff/cumodamoeba.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/epolartorque.h"
#include "seq/launch.h"
#include "seq/pair_polar.h"
#include "seq/triangle.h"

namespace tinker {
#include "epolarN2_cu1.cc"

template <class Ver>
static void epolarN2_cu(const real (*uind)[3], const real (*uinp)[3])
{
   constexpr bool do_g = Ver::g;

   const auto& st = *mspatial_v2_unit;
   real off = switchOff(Switch::MPOLE);

   const real f = 0.5f * electric / dielec;

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, ufld, dufld);
   int ngrid = gpuGridSize(BLOCK_DIM);
   epolarN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nep, ep, vir_ep, depx, depy, depz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niakp, st.iakp, ufld, dufld, rpole, uind, uinp, f);

   // torque
   if CONSTEXPR (do_g) {
      launch_k1s(g::s0, n, epolarTorque_cu, //
         trqx, trqy, trqz, n, rpole, ufld, dufld);
   }
}

void epolarNonEwaldN2_cu(int vers, const real (*uind)[3], const real (*uinp)[3])
{
   if (vers == calc::v0) {
      epolarN2_cu<calc::V0>(uind, uinp);
   } else if (vers == calc::v1) {
      epolarN2_cu<calc::V1>(uind, uinp);
   } else if (vers == calc::v3) {
      epolarN2_cu<calc::V3>(uind, uinp);
   } else if (vers == calc::v4) {
      epolarN2_cu<calc::V4>(uind, uinp);
   } else if (vers == calc::v5) {
      epolarN2_cu<calc::V5>(uind, uinp);
   } else if (vers == calc::v6) {
      epolarN2_cu<calc::V6>(uind, uinp);
   }
}
}
