#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/cumodamoeba.h"
#include "ff/modamoeba.h"
#include "ff/solv/nblistgk.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_solv.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>
#include <cmath>

namespace tinker {
template <class Ver>
__global__
static void ewcaFinal_cu1(int n, const real* restrict cdsp, CountBuffer restrict nes, EnergyBuffer restrict es)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if CONSTEXPR (do_e) {
         real cdspi = cdsp[i];
         using ebuf_prec = EnergyBufferTraits::type;
         ebuf_prec estl;
         estl = floatTo<ebuf_prec>(cdspi);
         atomic_add(estl, es, i);
      }
      if CONSTEXPR (do_a) atomic_add(2, nes, i);
   }
}

#include "ewca_cu1.cc"
#include "ewcaN2_cu1.cc"

template <class Ver>
static void ewca_cu2()
{
   const real off = switchOff(Switch::MPOLE);
   real epsosqrt = REAL_SQRT(epso);
   real epshsqrt = REAL_SQRT(epsh);
   real rmino2 = rmino*rmino;
   real rmino3 = rmino2*rmino;
   real rminh2 = rminh*rminh;
   real rminh3 = rminh2*rminh;
   real slwater = slevy * awater;

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      ewca_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         epsdsp, raddsp, epso, epsosqrt, epsh, epshsqrt, rmino2, rmino3, rminh2, rminh3, shctd, dspoff, slwater);
   } else {
      const auto& st = *mdloop_unit;
      ewcaN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, es, desx, desy, desz, off, x, y, z, st.nakp, st.iakp,
         epsdsp, raddsp, epso, epsosqrt, epsh, epshsqrt, rmino2, rmino3, rminh2, rminh3, shctd, dspoff, slwater);
   }

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
template <class Ver>
__global__
static void egkaFinal_cu1(int n, CountBuffer restrict nes, EnergyBuffer restrict es, real* restrict drb, real* restrict drbp, real* restrict trqx, real* restrict trqy, real* restrict trqz,
   const real* restrict rborn, const real (*restrict rpole)[10], const real (*restrict uinds)[3], const real (*restrict uinps)[3], real gkc, real fc, real fd, real fq)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

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
      real uipx = uinps[i][0];
      real uipy = uinps[i][1];
      real uipz = uinps[i][2];
      real rbi = rborn[i];

      real e,txi,tyi,tzi,drbi,dpbi;

      self_egka<Ver>(ci, dix, diy, diz, qixx, qixy, qixz,
         qiyy, qiyz, qizz, uidx, uidy, uidz,
         uipx, uipy, uipz, rbi, gkc, fc, fd, fq,
         e, txi, tyi, tzi, drbi, dpbi);

      if CONSTEXPR (do_e) {
         using ebuf_prec = EnergyBufferTraits::type;
         ebuf_prec estl;
         estl = floatTo<ebuf_prec>(e);
         atomic_add(estl, es, i);
      }

      if CONSTEXPR (do_a) atomic_add(1, nes, i);

      if CONSTEXPR (do_g) {
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(drbi, drb, i);
         atomic_add(dpbi, drbp, i);
      }
   }
}

#include "egka_cu1.cc"
#include "egkaN2_cu1.cc"

template <class Ver>
static void egka_cu2()
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   real dwater = (real)78.3;
   real fc = electric * 1 * (1-dwater)/(0+1*dwater);
   real fd = electric * 2 * (1-dwater)/(1+2*dwater);
   real fq = electric * 3 * (1-dwater)/(2+3*dwater);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      egka_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niak, st.iak, st.lst,
         trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
   } else {
      const auto& st = *mdloop_unit;
      egkaN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, nes, es, desx, desy, desz, off, x, y, z, st.nakp, st.iakp,
         trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
   }

   launch_k1s(g::s0, n, egkaFinal_cu1<Ver>, n, nes, es, drb, drbp, trqx, trqy, trqz, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);
}

void egka_cu(int vers)
{
   if (vers == calc::v0)
      egka_cu2<calc::V0>();
   else if (vers == calc::v3)
      egka_cu2<calc::V3>();
   else if (vers == calc::v4)
      egka_cu2<calc::V4>();
   else if (vers == calc::v5)
      egka_cu2<calc::V5>();
}
}

namespace tinker {
#include "ediff_cu1.cc"
#include "ediffN2_cu1.cc"

template <class Ver>
static void ediff_cu2()
{
   const real off = switchOff(Switch::MPOLE);
   const real f = electric / dielec;

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      ediff_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, desx, desy, desz,
         off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
         st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
   } else {
      const auto& st = *mdloop_unit;
      ediffN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nes, es, desx, desy, desz,
         off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, x, y, z, st.nakpl, st.iakpl,
         st.nakpa, st.iakpa, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
   }
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

namespace tinker {
__global__
static void addToEnrgy_cu1(EnergyBuffer restrict es, const real cave)
{
   atomic_add(cave, es, ITHREAD);
}

__global__
static void addToGrad_cu1(int n, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, const real* restrict gxi, const real* restrict gyi, const real* restrict gzi)
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      atomic_add(gxi[i], gx, i);
      atomic_add(gyi[i], gy, i);
      atomic_add(gzi[i], gz, i);
   }
}

void addToEnrgy_cu()
{
   addToEnrgy_cu1<<<1, 1, 0, g::s0>>>(es, cave);
}

void addToGrad_cu()
{
   launch_k1s(g::s0, n, addToGrad_cu1, n, desx, desy, desz, decvx, decvy, decvz);
}
}
