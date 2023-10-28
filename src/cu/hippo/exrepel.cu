#include "ff/hippo/erepel.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/xrepel.h"
#include "seq/triangle.h"

namespace tinker {
__global__
static void solvcoeff_cu1(int n, const real (*restrict xrepole)[MPL_TOTAL], const real* restrict crpxr, real (*restrict cpxr)[4])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      scoeffAtomI(i, xrepole, crpxr, cpxr);
   }
}

__global__
static void rotcoeff_cu1(int n, const LocalFrame* restrict zaxis,
   const real* restrict x, const real* restrict y, const real* restrict z,
   const real (*restrict cpxr)[4], real (*restrict rcpxr)[4])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rotcoeffAtomI(i, zaxis, x, y, z, cpxr, rcpxr);
   }
}

#include "exrepel_cu1.cc"
template <class Ver>
static void exrepel_cu2()
{
   const auto& st = *mspatial_v2_unit;
   real cut = switchCut(Switch::REPULS);
   real off = switchOff(Switch::REPULS);

   int ngrid = gpuGridSize(BLOCK_DIM);
   exrepel_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nrep, er, vir_er, derx, dery, derz, cut,
      off, st.si2.bit0, nrepexclude, repexclude, repexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, trqx, trqy, trqz, zpxr, dmppxr, rcpxr, mut, vlam, vcouple);
}

void solvcoeff_cu()
{
   launch_k1s(g::s0, n, solvcoeff_cu1, n, xrepole, crpxr, cpxr);
}

void rotcoeff_cu()
{
   launch_k1s(g::s0, n, rotcoeff_cu1, n, zaxis, x, y, z, cpxr, rcpxr);
}

void exrepel_cu(int vers)
{
   if (vers == calc::v0)
      exrepel_cu2<calc::V0>();
   else if (vers == calc::v1)
      exrepel_cu2<calc::V1>();
   else if (vers == calc::v3)
      exrepel_cu2<calc::V3>();
   else if (vers == calc::v4)
      exrepel_cu2<calc::V4>();
   else if (vers == calc::v5)
      exrepel_cu2<calc::V5>();
   else if (vers == calc::v6)
      exrepel_cu2<calc::V6>();
}
}
