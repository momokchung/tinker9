#include "ff/modamoeba.h"
#include "ff/atom.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/nblist.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/pair_solv.h"
#include "tool/gpucard.h"

namespace tinker {
#define EWCA_DPTRS                               \
   x, y, z, desx, desy, desz, es, epsdsp, raddsp
template <class Ver>
static void ewca_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   auto bufsize = bufferSize();

   real epsosqrt = REAL_SQRT(epso);
   real epshsqrt = REAL_SQRT(epsh);
   real rmino2 = rmino*rmino;
   real rmino3 = rmino2*rmino;
   real rminh2 = rminh*rminh;
   real rminh3 = rminh2*rminh;
   real slwater = slevy * awater;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(EWCA_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real epsi = epsdsp[i];
      real rmin = raddsp[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real epsk = epsdsp[k];
            real rmkn = raddsp[k];
            real r = REAL_SQRT(r2);
            real r3 = r2 * r;

            real epsisqrt = REAL_SQRT(epsi);
            real term1 = epsosqrt + epsisqrt;
            real term12 = term1 * term1;
            real rmin2 = rmin * rmin;
            real rmin3 = rmin2 * rmin;
            real emixo = 4 * epso * epsi / term12;
            real rmixo = 2 * (rmino3 + rmin3) / (rmino2 + rmin2);
            real rmixo7 = REAL_POW(rmixo, 7);
            real aoi = emixo * rmixo7;
            real term2 = epshsqrt + epsisqrt;
            real term22 = term2 * term2;
            real emixh = 4 * epsh * epsi / term22;
            real rmixh = 2 * (rminh3 + rmin3) / (rminh2 + rmin2);
            real rmixh7 = REAL_POW(rmixh, 7);
            real ahi = emixh * rmixh7;
            real rio = rmixo / 2 + dspoff;
            real rih = rmixh / 2 + dspoff;
            real si = rmin * shctd;
            real si2 = si * si;

            real epsksqrt = REAL_SQRT(epsk);
            real term3 = epsosqrt + epsksqrt;
            real term32 = term3 * term3;
            real emkxo = 4 * epso * epsk / term32;
            real rmkn2 = rmkn * rmkn;
            real rmkn3 = rmkn2 * rmkn;
            real rmkxo = 2 * (rmino3 + rmkn3) / (rmino2 + rmkn2);
            real rmkxo7 = REAL_POW(rmkxo, 7);
            real aok = emkxo * rmkxo7;
            real term4 = epshsqrt + epsksqrt;
            real term42 = term4 * term4;
            real emkxh = 4 * epsh * epsk / term42;
            real rmkxh = 2 * (rminh3 + rmkn3) / (rminh2 + rmkn2);
            real rmkxh7 = REAL_POW(rmkxh, 7);
            real ahk = emkxh * rmkxh7;
            real rko = rmkxo / 2 + dspoff;
            real rkh = rmkxh / 2 + dspoff;
            real sk = rmkn * shctd;
            real sk2 = sk * sk;

            real sum1,sum2;
            real e,de,de1,de2;
            real de11,de12,de21,de22;

            pair_ewca<Ver>(r, r2, r3, rio, rmixo, rmixo7, sk, sk2, aoi, emixo, sum1, de11, true);
            pair_ewca<Ver>(r, r2, r3, rih, rmixh, rmixh7, sk, sk2, ahi, emixh, sum2, de12, false);
            e = sum1 + sum2;

            pair_ewca<Ver>(r, r2, r3, rko, rmkxo, rmkxo7, si, si2, aok, emkxo, sum1, de21, true);
            pair_ewca<Ver>(r, r2, r3, rkh, rmkxh, rmkxh7, si, si2, ahk, emkxh, sum2, de22, false);
            e += sum1 + sum2;

            e *= -slwater;

            if CONSTEXPR (do_e)
               atomic_add(e, es, offset);
            if CONSTEXPR (do_g) {
               de1 = de11 + de12;
               de2 = de21 + de22;
               de = de1 + de2;
               de *= slwater / r;
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;
               gxi += dedx;
               gyi += dedy;
               gzi += dedz;
               atomic_add(-dedx, desx, k);
               atomic_add(-dedy, desy, k);
               atomic_add(-dedz, desz, k);
            }
         }
      } // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, desx, i);
         atomic_add(gyi, desy, i);
         atomic_add(gzi, desz, i);
      }
   } // end for (int i)

   #pragma acc parallel loop independent async\
               deviceptr(cdsp, es, nes)
   for (int i = 0; i < n; ++i) {
      int offset = i & (bufsize - 1);
      if CONSTEXPR (do_e)
         atomic_add(cdsp[i], es, offset);
      if CONSTEXPR (do_a)
         atomic_add(2, nes, offset);
   }
}

void ewca_acc(int vers)
{
   if (vers == calc::v0)
      ewca_acc1<calc::V0>();
   else if (vers == calc::v3)
      ewca_acc1<calc::V3>();
   else if (vers == calc::v4)
      ewca_acc1<calc::V4>();
   else if (vers == calc::v5)
      ewca_acc1<calc::V5>();
}
}

namespace tinker {
#define EGKA_DPTRS1                                                                            \
   x, y, z, desx, desy, desz, drb, drbp, rborn, rpole, uinds, uinps, nes, es, trqx, trqy, trqz
#define EGKA_DPTRS2                                                 \
   drb, drbp, rborn, rpole, uinds, uinps, nes, es, trqx, trqy, trqz
template <class Ver>
static void egka_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   auto bufsize = bufferSize();
   PairSolvGrad pgrad;

   real dwater = (real)78.3;
   real fc = electric * 1 * (1-dwater)/(0+1*dwater);
   real fd = electric * 2 * (1-dwater)/(1+2*dwater);
   real fq = electric * 3 * (1-dwater)/(2+3*dwater);

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(EGKA_DPTRS1,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
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
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;
      MAYBE_UNUSED real drbi = 0, dpbi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi,drbi,dpbi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real xr2 = xr * xr;
         real yr2 = yr * yr;
         real zr2 = zr * zr;

         real r2 = xr2 + yr2 + zr2;
         if (r2 <= off2) {
            real ck = rpole[k][MPL_PME_0];
            real dkx = rpole[k][MPL_PME_X];
            real dky = rpole[k][MPL_PME_Y];
            real dkz = rpole[k][MPL_PME_Z];
            real qkxx = rpole[k][MPL_PME_XX];
            real qkxy = rpole[k][MPL_PME_XY];
            real qkxz = rpole[k][MPL_PME_XZ];
            real qkyy = rpole[k][MPL_PME_YY];
            real qkyz = rpole[k][MPL_PME_YZ];
            real qkzz = rpole[k][MPL_PME_ZZ];
            real ukdx = uinds[k][0];
            real ukdy = uinds[k][1];
            real ukdz = uinds[k][2];
            real ukpx = uinps[k][0];
            real ukpy = uinps[k][1];
            real ukpz = uinps[k][2];
            real rbk = rborn[k];

            real e,tdrbi,tdpbi,tdrbk,tdpbk;
            zero(pgrad);
            pair_egka<Ver>(r2, xr, yr, zr, xr2, yr2, zr2, ci, dix, diy, diz, qixx,
               qixy, qixz, qiyy, qiyz, qizz, uidx, uidy, uidz,
               uipx, uipy, uipz, rbi, ck, dkx, dky,
               dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy,
               ukdz, ukpx, ukpy, ukpz, rbk, gkc, fc, fd, fq, e,
               pgrad, tdrbi, tdpbi, tdrbk, tdpbk);

            if CONSTEXPR (do_e) {
               atomic_add(e, es, offset);
               if CONSTEXPR (do_a)
                  atomic_add(1, nes, offset);
            }

            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, desx, k);
               atomic_add(-pgrad.frcy, desy, k);
               atomic_add(-pgrad.frcz, desz, k);

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               atomic_add(pgrad.ttqk[0], trqx, k);
               atomic_add(pgrad.ttqk[1], trqy, k);
               atomic_add(pgrad.ttqk[2], trqz, k);

               drbi += tdrbi;
               dpbi += tdpbi;
               atomic_add(tdrbk, drb, k);
               atomic_add(tdpbk, drbp, k);
            }
         }
      } // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, desx, i);
         atomic_add(gyi, desy, i);
         atomic_add(gzi, desz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(drbi, drb, i);
         atomic_add(dpbi, drbp, i);
      }
   } // end for (int i)

   #pragma acc parallel loop independent async\
               deviceptr(EGKA_DPTRS2)
   for (int i = 0; i < n; ++i) {
      int offset = i & (bufsize - 1);
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
      if CONSTEXPR (do_e)
         atomic_add(e, es, offset);
      if CONSTEXPR (do_a)
         atomic_add(1, nes, offset);
      if CONSTEXPR (do_g) {
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(drbi, drb, i);
         atomic_add(dpbi, drbp, i);
      }
   }
}

void egka_acc(int vers)
{
   if (vers == calc::v0)
      egka_acc1<calc::V0>();
   else if (vers == calc::v3)
      egka_acc1<calc::V3>();
   else if (vers == calc::v4)
      egka_acc1<calc::V4>();
   else if (vers == calc::v5)
      egka_acc1<calc::V5>();
}
}

namespace tinker {
#define EDIFF_DPTRS                                                                                             \
   x, y, z, desx, desy, desz, pdamp, jpolar, thlval, rpole, uind, uinds, uinp, uinps, nes, es, trqx, trqy, trqz
template <class Ver>
static void ediff_acc1()
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;

   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   auto bufsize = bufferSize();
   PairSolvGrad pgrad;

   const real f = electric / dielec;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(EDIFF_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
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
      real uidx = uind[i][0];
      real uidy = uind[i][1];
      real uidz = uind[i][2];
      real uidsx = uinds[i][0];
      real uidsy = uinds[i][1];
      real uidsz = uinds[i][2];
      real uipx = uinp[i][0];
      real uipy = uinp[i][1];
      real uipz = uinp[i][2];
      real uipsx = uinps[i][0];
      real uipsy = uinps[i][1];
      real uipsz = uinps[i][2];
      real pdi = pdamp[i];
      int jpi = jpolar[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;
      MAYBE_UNUSED real txi = 0, tyi = 0, tzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent private(pgrad)\
                  reduction(+:gxi,gyi,gzi,txi,tyi,tzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int offset = (kk + i * n) & (bufsize - 1);
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;
         real xr2 = xr * xr;
         real yr2 = yr * yr;
         real zr2 = zr * zr;

         real r2 = xr2 + yr2 + zr2;
         if (r2 <= off2) {
            real ck = rpole[k][MPL_PME_0];
            real dkx = rpole[k][MPL_PME_X];
            real dky = rpole[k][MPL_PME_Y];
            real dkz = rpole[k][MPL_PME_Z];
            real qkxx = rpole[k][MPL_PME_XX];
            real qkxy = rpole[k][MPL_PME_XY];
            real qkxz = rpole[k][MPL_PME_XZ];
            real qkyy = rpole[k][MPL_PME_YY];
            real qkyz = rpole[k][MPL_PME_YZ];
            real qkzz = rpole[k][MPL_PME_ZZ];
            real ukdx = uind[k][0];
            real ukdy = uind[k][1];
            real ukdz = uind[k][2];
            real ukdsx = uinds[k][0];
            real ukdsy = uinds[k][1];
            real ukdsz = uinds[k][2];
            real ukpx = uinp[k][0];
            real ukpy = uinp[k][1];
            real ukpz = uinp[k][2];
            real ukpsx = uinps[k][0];
            real ukpsy = uinps[k][1];
            real ukpsz = uinps[k][2];
            real pdk = pdamp[k];
            int jpk = jpolar[k];
            real pga = thlval[njpolar * jpi + jpk];

            real e;
            zero(pgrad);
            pair_ediff<Ver>(r2, xr, yr, zr, 1, 1, 1,
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, uidx, uidy, uidz,
               uidsx, uidsy, uidsz, uipx, uipy, uipz, uipsx, uipsy, uipsz, pdi, pga,
               ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz,
               ukdsx, ukdsy, ukdsz, ukpx, ukpy, ukpz, ukpsx, ukpsy, ukpsz, pdk, pga,
               f, e, pgrad);

            if CONSTEXPR (do_e) {
               atomic_add(e, es, offset);
               if CONSTEXPR (do_a)
                  atomic_add(1, nes, offset);
            }

            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               atomic_add(-pgrad.frcx, desx, k);
               atomic_add(-pgrad.frcy, desy, k);
               atomic_add(-pgrad.frcz, desz, k);

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               atomic_add(pgrad.ttqk[0], trqx, k);
               atomic_add(pgrad.ttqk[1], trqy, k);
               atomic_add(pgrad.ttqk[2], trqz, k);
            }
         }
      } // end for (int kk)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, desx, i);
         atomic_add(gyi, desy, i);
         atomic_add(gzi, desz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
      }
   } // end for (int i)

   #pragma acc parallel async\
               deviceptr(EDIFF_DPTRS,mdpuexclude,mdpuexclude_scale)
   #pragma acc loop independent private(pgrad)
   for (int ii = 0; ii < nmdpuexclude; ++ii) {
      int offset = ii & (bufsize - 1);

      int i = mdpuexclude[ii][0];
      int k = mdpuexclude[ii][1];
      real scaleb = mdpuexclude_scale[ii][1] - 1;
      real scalec = mdpuexclude_scale[ii][2] - 1;
      real scaled = mdpuexclude_scale[ii][3] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
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
      real uidx = uind[i][0];
      real uidy = uind[i][1];
      real uidz = uind[i][2];
      real uidsx = uinds[i][0];
      real uidsy = uinds[i][1];
      real uidsz = uinds[i][2];
      real uipx = uinp[i][0];
      real uipy = uinp[i][1];
      real uipz = uinp[i][2];
      real uipsx = uinps[i][0];
      real uipsy = uinps[i][1];
      real uipsz = uinps[i][2];
      real pdi = pdamp[i];
      int jpi = jpolar[i];

      real xr = x[k] - xi;
      real yr = y[k] - yi;
      real zr = z[k] - zi;

      real r2 = xr * xr + yr * yr + zr * zr;
      if (r2 <= off2) {
         real ck = rpole[k][MPL_PME_0];
         real dkx = rpole[k][MPL_PME_X];
         real dky = rpole[k][MPL_PME_Y];
         real dkz = rpole[k][MPL_PME_Z];
         real qkxx = rpole[k][MPL_PME_XX];
         real qkxy = rpole[k][MPL_PME_XY];
         real qkxz = rpole[k][MPL_PME_XZ];
         real qkyy = rpole[k][MPL_PME_YY];
         real qkyz = rpole[k][MPL_PME_YZ];
         real qkzz = rpole[k][MPL_PME_ZZ];
         real ukdx = uind[k][0];
         real ukdy = uind[k][1];
         real ukdz = uind[k][2];
         real ukdsx = uinds[k][0];
         real ukdsy = uinds[k][1];
         real ukdsz = uinds[k][2];
         real ukpx = uinp[k][0];
         real ukpy = uinp[k][1];
         real ukpz = uinp[k][2];
         real ukpsx = uinps[k][0];
         real ukpsy = uinps[k][1];
         real ukpsz = uinps[k][2];
         real pdk = pdamp[k];
         int jpk = jpolar[k];
         real pga = thlval[njpolar * jpi + jpk];

         real e;
         zero(pgrad);
         pair_ediff<Ver>(r2, xr, yr, zr, scaleb, scalec, scaled,
            ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, uidx, uidy, uidz,
            uidsx, uidsy, uidsz, uipx, uipy, uipz, uipsx, uipsy, uipsz, pdi, pga,
            ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx, ukdy, ukdz,
            ukdsx, ukdsy, ukdsz, ukpx, ukpy, ukpz, ukpsx, ukpsy, ukpsz, pdk, pga,
            f, e, pgrad);

         if CONSTEXPR (do_e)
            atomic_add(e, es, offset);

         if CONSTEXPR (do_g) {
            atomic_add(pgrad.frcx, desx, i);
            atomic_add(pgrad.frcy, desy, i);
            atomic_add(pgrad.frcz, desz, i);
            atomic_add(-pgrad.frcx, desx, k);
            atomic_add(-pgrad.frcy, desy, k);
            atomic_add(-pgrad.frcz, desz, k);

            atomic_add(pgrad.ttqi[0], trqx, i);
            atomic_add(pgrad.ttqi[1], trqy, i);
            atomic_add(pgrad.ttqi[2], trqz, i);
            atomic_add(pgrad.ttqk[0], trqx, k);
            atomic_add(pgrad.ttqk[1], trqy, k);
            atomic_add(pgrad.ttqk[2], trqz, k);
         }
      }
   }
}

void ediff_acc(int vers)
{
   if (vers == calc::v0)
      ediff_acc1<calc::V0>();
   else if (vers == calc::v3)
      ediff_acc1<calc::V3>();
   else if (vers == calc::v4)
      ediff_acc1<calc::V4>();
   else if (vers == calc::v5)
      ediff_acc1<calc::V5>();
}
}

namespace tinker {
void addToEnrgy_acc()
{
   auto bufsize = bufferSize();
   int offset = 0 & (bufsize - 1);
   {
      #pragma acc parallel deviceptr(es)
      atomic_add(cave, es, offset);
   }
}

void addToGrad_acc()
{
   #pragma acc parallel loop independent async\
               deviceptr(desx, desy, desz, decvx, decvy, decvz)
   for (int i = 0; i < n; ++i) {
      atomic_add(decvx[i], desx, i);
      atomic_add(decvy[i], desy, i);
      atomic_add(decvz[i], desz, i);
   }
}
}
