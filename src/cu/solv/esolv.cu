#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_born.h"
#include "seq/triangle.h"
#include <cmath>

namespace tinker
{
template <class Ver>
static void esolv_cu2()
{
   const auto& st = *mspatial_v2_unit;

   // int ngrid = gpuGridSize(BLOCK_DIM);
   // erepel_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nrep, er, vir_er, derx, dery, derz, cut,
   //    off, st.si2.bit0, nrepexclude, repexclude, repexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
   //    st.niak, st.iak, st.lst, trqx, trqy, trqz, rrepole, sizpr, elepr, dmppr, mut, vlam, vcouple);
}

void esolv_cu(int vers)
{
   // if (solvtyp == Solv::GK or solvtyp == Solv::PB) {
   //    enp_cu(vers);
   // }
   // if (vers == calc::v0)
   //    esolv_cu2<calc::V0>();
   // else if (vers == calc::v3)
   //    esolv_cu2<calc::V3>();
   // else if (vers == calc::v4)
   //    esolv_cu2<calc::V4>();
   // else if (vers == calc::v5)
   //    esolv_cu2<calc::V5>();
}
}
