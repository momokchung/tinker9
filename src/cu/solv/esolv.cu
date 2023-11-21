#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/cumodamoeba.h"
#include "ff/modamoeba.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_solv.h"
#include "seq/triangle.h"
#include <cmath>

namespace tinker
{
#include "ewca_cu1.cc"

template <class Ver>
static void ewca_cu2()
{   
   const auto& st = *mspatial_v2_unit;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ewca_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, es, desx, desy, desz, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niakp, st.iakp, epsdsp, raddsp, epso, epsh, rmino, rminh, shctd, dspoff, slevy, awater);
   
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

template <class Ver>
static void egka_cu2(real fc, real fd, real fq)
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);
   egka_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nes, es, vir_es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niakp, st.iakp, trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);

   launch_k1s(g::s0, n, egkaFinal_cu1<Ver>, n, nes, es, drb, drbp, trqx, trqy, trqz, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
}

void egka_cu(int vers)
{
   real dwater = 78.3;
   real fc = electric * 1. * (1.-dwater)/(0.+1.*dwater);
   real fd = electric * 2. * (1.-dwater)/(1.+2.*dwater);
   real fq = electric * 3. * (1.-dwater)/(2.+3.*dwater);

   if (vers == calc::v0)
      egka_cu2<calc::V0>(fc, fd, fq);
   else if (vers == calc::v3)
      egka_cu2<calc::V3>(fc, fd, fq);
   else if (vers == calc::v4)
      egka_cu2<calc::V4>(fc, fd, fq);
   else if (vers == calc::v5)
      egka_cu2<calc::V5>(fc, fd, fq);
}
}

namespace tinker {
#include "ediff_cu1.cc"

template <class Ver>
static void ediff_cu2()
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   const real f = electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ediff_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nes, es, desx, desy, desz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niakp, st.iakp, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
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
