#include "ff/amoeba/empole.h"
#include "ff/amoeba/epolar.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/cumodamoeba.h"
#include "ff/elec.h"
#include "ff/modamoeba.h"
#include "ff/solv/inducegk.h"
#include "ff/solv/nblistgk.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/damp.h"
#include "seq/emselfamoeba.h"
#include "seq/launch.h"
#include "seq/pair_emplar.h"
#include "seq/triangle.h"

namespace tinker {
#include "emplargkN2_cu1.cc"

template <class Ver>
static void emplargkN2_cu(const real (*uind)[3], const real (*uinp)[3])
{
   const auto& st = *mdloop_unit;
   real off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);

   auto kera = emplargkN2_cu1a<Ver>;
   kera<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, x, y, z, em, vir_em, demx, demy, demz, off,
      trqx, trqy, trqz, rpole, uind, uinp, f, st.nakpa, st.iakpa);
   auto kerb = emplargkN2_cu1b<Ver>;
   kerb<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, x, y, z, em, vir_em, demx, demy, demz, off,
      trqx, trqy, trqz, rpole, uind, uinp, f, st.nakpl, st.iakpl);
   auto kerc = emplargkN2_cu1c<Ver>;
   kerc<<<ngrid, BLOCK_DIM, 0, g::s0>>>(nmdpuexclude, mdpuexclude, mdpuexclude_scale, x, y, z,
      em, vir_em, demx, demy, demz, off, trqx, trqy, trqz, rpole, uind, uinp, f);
}

template <class Ver>
static void emplarNonEwaldgk_cu(int vers)
{
   // induce
   inducegk(uind, uinp, uinds, uinps, vers);
   // empole and epolar
   emplargkN2_cu<Ver>(uinds, uinps);
   if CONSTEXPR (Ver::e)
      epolar0DotProd(uinds, udirp);
}

void emplargkN2_cu(int vers)
{
   if (vers == calc::v0)
      emplarNonEwaldgk_cu<calc::V0>(vers);
   else if (vers == calc::v1)
      emplarNonEwaldgk_cu<calc::V1>(vers);
   // else if (vers == calc::v3)
   //    emplarNonEwaldgk_cu<calc::V3>(vers);
   else if (vers == calc::v4)
      emplarNonEwaldgk_cu<calc::V4>(vers);
   else if (vers == calc::v5)
      emplarNonEwaldgk_cu<calc::V5>(vers);
   else if (vers == calc::v6)
      emplarNonEwaldgk_cu<calc::V6>(vers);
}
}
