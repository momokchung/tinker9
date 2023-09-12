#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "seq/launch.h"
#include "seq/pair_mpole.h"
#include "seq/triangle.h"
#include <iostream>

namespace tinker {
#include "empole_cu1.cc"

template <class Ver>
static void empole_cu()
{
   const auto& st = *mspatial_v2_unit;

   const real f = electric / dielec;
   int ngrid = gpuGridSize(BLOCK_DIM);
   empole_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nem, em, vir_em,
      demx, demy, demz, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale,
      st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.ndopair, st.dopair, trqx, trqy, trqz, rpole, f);
}

void empoleN2_cu(int vers)
{
   if (vers == calc::v0) {
      empole_cu<calc::V0>();
   } else if (vers == calc::v1) {
      empole_cu<calc::V1>();
   } else if (vers == calc::v3) {
      empole_cu<calc::V3>();
   } else if (vers == calc::v4) {
      empole_cu<calc::V4>();
   } else if (vers == calc::v5) {
      empole_cu<calc::V5>();
   } else if (vers == calc::v6) {
      empole_cu<calc::V6>();
   }
}
}
