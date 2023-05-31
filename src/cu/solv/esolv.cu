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
   ewca_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, epsvdw, radvdw, epso, epsh, rmino, rminh, shctd, dispoff, slevy, awater);
}

__global__
static void ewcaFinal_cu1(int n, const real* restrict cdisp, EnergyBuffer restrict es)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real cdispi = cdisp[i];
      using ebuf_prec = EnergyBufferTraits::type;
      ebuf_prec estl;
      estl = floatTo<ebuf_prec>(cdispi);
      atomic_add(estl, es, i);
}
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

   launch_k1s(g::s0, n, ewcaFinal_cu1, n, cdisp, es);
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
   egka_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, rborn, rpole, uinds, gkc, fc, fd, fq);
}

__global__
static void egkaFinal_cu1(int n, EnergyBuffer restrict es, const real* restrict rborn,
   const real (*restrict rpole)[10], const real (*restrict uinds)[3], real gkc, real fc, real fd, real fq)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real ci = rpole[i][MPL_PME_0];
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real qixx = rpole[i][MPL_PME_XX];
      real qixy = rpole[i][MPL_PME_XY];
      real qixz = rpole[i][MPL_PME_XZ];
      real qiyy = rpole[i][MPL_PME_YY];
      real qiyz = rpole[i][MPL_PME_YZ];
      real qizz = rpole[i][MPL_PME_ZZ];
      real uidx = uinds[i][0];
      real uidy = uinds[i][1];
      real uidz = uinds[i][2];
      real rbi = rborn[i];

      real rb2 = rbi * rbi;
      real expc = 1.0 / gkc;
      real gf2 = 1.0 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real gf5 = gf3 * gf2;

      real expc1 = 1.0 - expc;
      real a00 = fc * gf;
      real a01 = -fc * expc1 * gf3;
      real a10 = -fd * gf3;
      real a20 = 3.0 * fq * gf5;

      real gc1 = a00;
      real gux2 = a10;
      real guy3 = a10;
      real guz3 = a10;
      real gc5 = a01;
      real gc8 = a01;
      real gc10 = a01;
      real gqxx5 = 2.0 * a20;
      real gqyy8 = 2.0 * a20;
      real gqzz10 = 2.0 * a20;
      real gqxy6 = a20;
      real gqxz7 = a20;
      real gqyz9 = a20;

      real esym = ci * ci * gc1 - dix * dix * gux2 - diy * diy * guy3 - diz * diz * guz3;
      real ewi = ci * (qixx * gc5 + qiyy * gc8 + qizz * gc10)
         + qixx * qixx * gqxx5 + qiyy * qiyy * gqyy8 + qizz * qizz * gqzz10
         + 4.0 * (qixy * qixy * gqxy6 + qixz * qixz * gqxz7 + qiyz * qiyz * gqyz9);
      real e = esym + ewi;

      real ei = -dix * uidx * gux2 - diy * uidy * guy3 - diz * uidz * guz3;

      e += ei;
      e *= 0.5;

      using ebuf_prec = EnergyBufferTraits::type;
      ebuf_prec estl;
      estl = floatTo<ebuf_prec>(e);
      atomic_add(estl, es, i);
}
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

   launch_k1s(g::s0, n, egkaFinal_cu1, n, es, rborn, rpole, uinds, gkc, fc, fd, fq);
}
}

namespace tinker {
#include "ediff_cu1.cc"

template <class Ver>
static void ediff_cu2()
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   const real f = 0.5 * electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ediff_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, rpole, uind, uinds, f);
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
