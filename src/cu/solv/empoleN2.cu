#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "seq/pair_mpole.h"
#include "seq/triangle.h"
#include <iostream>

namespace tinker {
#include "empoleN2_cu1.cc"

template <class Ver>
static void empoleN2_cu2()
{
   const auto& st = *mspatial_v2_unit;
   real off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;
   int ngrid = gpuGridSize(BLOCK_DIM);
   empoleN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nem, em, vir_em,
      demx, demy, demz, off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale,
      st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niakp, st.iakp, trqx, trqy, trqz, rpole, f);
}

void empoleN2_cu(int vers)
{
   if (vers == calc::v0) {
      empoleN2_cu2<calc::V0>();
   } else if (vers == calc::v1) {
      empoleN2_cu2<calc::V1>();
   } else if (vers == calc::v3) {
      empoleN2_cu2<calc::V3>();
   } else if (vers == calc::v4) {
      empoleN2_cu2<calc::V4>();
   } else if (vers == calc::v5) {
      empoleN2_cu2<calc::V5>();
   } else if (vers == calc::v6) {
      empoleN2_cu2<calc::V6>();
   }
}
}
