#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "seq/pair_mpole.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>

namespace tinker {
#include "empoleNonEwaldgk_cu1.cc"
#include "empoleNonEwaldgkN2_cu1.cc"

template <class Ver>
static void empoleNonEwaldgk_cu2()
{
   const auto& st = *mspatial_v2_unit;
   real off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      empoleNonEwaldgk_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nem, em, vir_em,
         demx, demy, demz, off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale,
         st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, f);
   } else {
      empoleNonEwaldgkN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nem, em, vir_em,
         demx, demy, demz, off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale,
         st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niakp, st.iakp, trqx, trqy, trqz, rpole, f);
   }
}

void empoleNonEwaldgk_cu(int vers)
{
   if (vers == calc::v0) {
      empoleNonEwaldgk_cu2<calc::V0>();
   } else if (vers == calc::v1) {
      empoleNonEwaldgk_cu2<calc::V1>();
   } else if (vers == calc::v3) {
      empoleNonEwaldgk_cu2<calc::V3>();
   } else if (vers == calc::v4) {
      empoleNonEwaldgk_cu2<calc::V4>();
   } else if (vers == calc::v5) {
      empoleNonEwaldgk_cu2<calc::V5>();
   } else if (vers == calc::v6) {
      empoleNonEwaldgk_cu2<calc::V6>();
   }
}
}
