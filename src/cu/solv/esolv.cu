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

template <class Ver>
__global__
static void ewcaFinal_cu1(int n, const real* restrict cdisp, CountBuffer restrict nes, EnergyBuffer restrict es)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if CONSTEXPR (do_e) {
         real cdispi = cdisp[i];
         using ebuf_prec = EnergyBufferTraits::type;
         ebuf_prec estl;
         estl = floatTo<ebuf_prec>(cdispi);
         atomic_add(estl, es, i);
      }
      if CONSTEXPR (do_a) atomic_add(2, nes, i);
}
}

#include "ewca_cu1.cc"

template <class Ver>
static void ewca_cu2()
{   
   const auto& st = *mspatial_v2_unit;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ewca_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, es, desx, desy, desz, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, epsvdw, radvdw, epso, epsh, rmino, rminh, shctd, dispoff, slevy, awater);
   
   launch_k1s(g::s0, n, ewcaFinal_cu1<Ver>, n, cdisp, nes, es);
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
      real guz4 = a10;
      real gc5 = a01;
      real gc8 = a01;
      real gc10 = a01;
      real gqxx5 = 2.0 * a20;
      real gqyy8 = 2.0 * a20;
      real gqzz10 = 2.0 * a20;
      real gqxy6 = a20;
      real gqxz7 = a20;
      real gqyz9 = a20;

      real esym = ci * ci * gc1 - dix * dix * gux2 - diy * diy * guy3 - diz * diz * guz4;
      real ewi = ci * (qixx * gc5 + qiyy * gc8 + qizz * gc10)
         + qixx * qixx * gqxx5 + qiyy * qiyy * gqyy8 + qizz * qizz * gqzz10
         + 4.0 * (qixy * qixy * gqxy6 + qixz * qixz * gqxz7 + qiyz * qiyz * gqyz9);
      real e = esym + ewi;

      real ei = -dix * uidx * gux2 - diy * uidy * guy3 - diz * uidz * guz4;

      e += ei;
      e *= 0.5;

      if CONSTEXPR (do_e) {
         using ebuf_prec = EnergyBufferTraits::type;
         ebuf_prec estl;
         estl = floatTo<ebuf_prec>(e);
         atomic_add(estl, es, i);
      }

      if CONSTEXPR (do_a) atomic_add(1, nes, i);

      if CONSTEXPR (do_g) {
         real uipx = uinps[i][0];
         real uipy = uinps[i][1];
         real uipz = uinps[i][2];
         real uix = uidx + uipx;
         real uiy = uidy + uipy;
         real uiz = uidz + uipz;

         real gf7 = gf5 * gf2;
         real dgfdr =  0.5;
         real a11 = 3.0 * fd * expc1 * gf5;
         real b00 = -fc * dgfdr * gf3;
         real b10 = 3.0 * dgfdr * gf5;
         real b20 = -15.0 * fq *dgfdr * gf7;
         real b01 = b10 - expc*b10;
         b01 = fc * b01;
         b10 = fd * b10;

         real gc21 = b00;
         real gc25 = b01;
         real gc28 = b01;
         real gc30 = b01;
         real gux11 = 3.0*a11;
         real guy17 = 3.0*a11 ;
         real guz20 = 3.0*a11;
         real gux22 = b10;
         real guy23 = b10;
         real guz24 = b10;
         real gqxx25 = 2.0*b20;
         real gqxy26 = b20;
         real gqxz27 = b20;
         real gqyy28 = 2.0*b20;
         real gqyz29 = b20;
         real gqzz30 = 2.0*b20;

         real desymdr = ci*ci*gc21 - (dix*dix*gux22 + diy*diy*guy23 + diz*diz*guz24);
         real dewidr = ci*(qixx*gc25 + qiyy*gc28 + qizz*gc30)
                     + qixx*qixx*gqxx25 + qiyy*qiyy*gqyy28 + qizz*qizz*gqzz30
                     + 4.0*(qixy*qixy*gqxy26 + qixz*qixz*gqxz27 + qiyz*qiyz*gqyz29);
         real dsumdr = desymdr + dewidr;
         real drbi = rbi*dsumdr;

         real dsymdr = -dix*uix*gux22 - diy*uiy*guy23 - diz*uiz*guz24;
         real dpbi = rbi*dsymdr;

         real duvdr = uidx*uipx*gux22 + uidy*uipy*guy23 + uidz*uipz*guz24;
         dpbi -= rbi*duvdr;

         real fid[3];
         fid[0] = 0.5 * (uix * gux2);
         fid[1] = 0.5 * (uiy * guy3);
         fid[2] = 0.5 * (uiz * guz4);
         real txi = diy * fid[2] - diz * fid[1];
         real tyi = diz * fid[0] - dix * fid[2];
         real tzi = dix * fid[1] - diy * fid[0];

         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(drbi, drb, i);
         atomic_add(dpbi, drbp, i);
      }
}
}

#include "egka_cu1.cc"

template <class Ver>
static void egka_cu2(real fc, real fd, real fq)
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);
   egka_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, vir_es, desx, desy, desz, off, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, trqx, trqy, trqz, drb, drbp, rborn, rpole, uinds, uinps, gkc, fc, fd, fq);

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
   else if (vers == calc::v1)
      egka_cu2<calc::V1>(fc, fd, fq);
   else if (vers == calc::v3)
      egka_cu2<calc::V3>(fc, fd, fq);
   else if (vers == calc::v4)
      egka_cu2<calc::V4>(fc, fd, fq);
   else if (vers == calc::v5)
      egka_cu2<calc::V5>(fc, fd, fq);
   else if (vers == calc::v6)
      egka_cu2<calc::V6>(fc, fd, fq);
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
   ediff_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, nes, es, desx, desy, desz,
      off, st.si1.bit0, nmdpuexclude, mdpuexclude, mdpuexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl,
      st.niak, st.iak, st.lst, trqx, trqy, trqz, rpole, uind, uinds, uinp, uinps, f);
}

// __global__
// static void arrayPrint_cu1(int n, const real (*restrict rpole)[10])
// {
//    for (int i = ITHREAD; i < n; i += STRIDE) {
//       # if __CUDA_ARCH__>=200
//       printf("pole %d %10.6e %10.6e %10.6e\n", i, rpole[i][MPL_PME_0], rpole[i][MPL_PME_X], rpole[i][MPL_PME_XX]);
//       #endif  
//    }
// }

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
