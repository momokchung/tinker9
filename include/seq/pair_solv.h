#pragma once
#include "ff/solv/solute.h"
#include "seq/damp.h"
#include "seq/seq.h"
#include <algorithm>
#include <cmath>

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_ROUTINE
inline void pair_ewca(real r, real r2, real r3, real rio, real rmixo, real rmixo7, real sk, real sk2, real aoi, real emixo, real& sum, real& de, bool ifo)
{
   constexpr bool do_g = Ver::g;

   sum = 0;
   de = 0;
   real scale;
   real lik,lik2,lik3,lik4,lik5,lik6;
   real uik,uik2,uik3,uik4,uik5,uik6;
   real lik10,lik11,lik12,lik13;
   real uik10,uik11,uik12,uik13;
   real dl,du,term,term1,term2,rmax;
   real iwca,idisp,irepl;
   if (ifo) scale = 1;
   else scale = 2;
   if (rio < r+sk) {
      rmax = REAL_MAX(rio,r-sk);
      lik = rmax;
      if (lik < rmixo) {
         lik2 = lik * lik;
         lik3 = lik2 * lik;
         lik4 = lik3 * lik;
         uik = REAL_MIN(r+sk,rmixo);
         uik2 = uik * uik;
         uik3 = uik2 * uik;
         uik4 = uik3 * uik;
         term = 4 * pi / (48*r) * (3*(lik4-uik4) - 8*r*(lik3-uik3) + 6*(r2-sk2)*(lik2-uik2));
         iwca = -emixo * term;
         sum += iwca;
         if CONSTEXPR (do_g) {
            if (rio > r-sk) {
               dl = -lik2 + 2*r2 + 2*sk2;
               dl = dl * lik2;
            } else {
               dl = -lik3 + 4*lik2*r - 6*lik*r2 + 2*lik*sk2 + 4*r3 - 4*r*sk2;
               dl = dl * lik;
            }
            if (r+sk > rmixo) {
               du = -uik2 + 2*r2 + 2*sk2;
               du = -du * uik2;
            } else {
               du = -uik3 + 4*uik2*r - 6*uik*r2 + 2*uik*sk2 + 4*r3 - 4*r*sk2;
               du = -du * uik;
            }
            de += -emixo*pi*(dl+du)/(4*r2);
         }
      }
      uik = r + sk;
      if (uik > rmixo) {
         uik2 = uik * uik;
         uik3 = uik2 * uik;
         uik4 = uik3 * uik;
         uik5 = uik4 * uik;
         uik10 = uik5 * uik5;
         uik11 = uik10 * uik;
         uik12 = uik11 * uik;
         lik = REAL_MAX(rmax,rmixo);
         lik2 = lik * lik;
         lik3 = lik2 * lik;
         lik4 = lik3 * lik;
         lik5 = lik4 * lik;
         lik10 = lik5 * lik5;
         lik11 = lik10 * lik;
         lik12 = lik11 * lik;
         term1 = 4 * pi / (120*r*lik5*uik5) * (15*uik*lik*r*(uik4-lik4) - 10*uik2*lik2*(uik3-lik3) + 6*(sk2-r2)*(uik5-lik5));
         idisp = -2 * aoi * term1;
         term2 = 4 * pi / (2640*r*lik12*uik12) * (120*uik*lik*r*(uik11-lik11) - 66*uik2*lik2*(uik10-lik10) + 55*(sk2-r2)*(uik12-lik12));
         irepl = aoi * rmixo7 * term2;
         sum += irepl + idisp;
         if CONSTEXPR (do_g) {
            uik6 = uik5 * uik;
            uik13 = uik12 * uik;
            lik6 = lik5 * lik;
            lik13 = lik12 * lik;
            if ((rio > r-sk) or (rmax < rmixo)) {
               dl = -5*lik2 + 3*r2 + 3*sk2;
               dl = -dl / lik5;
            } else {
               dl = 5*lik3 - 33*lik*r2 - 3*lik*sk2 + 15*(lik2*r+r3-r*sk2);
               dl = dl / lik6;
            }
            du = 5*uik3 - 33*uik*r2 - 3*uik*sk2 + 15*(uik2*r+r3-r*sk2);
            du = -du / uik6;
            de = de -2*aoi*pi*(dl + du)/(15*r2);
            if ((rio>r-sk) or (rmax<rmixo)) {
               dl = -6*lik2 + 5*r2 + 5*sk2;
               dl = -dl / lik12;
            } else {
               dl = 6*lik3 - 125*lik*r2 - 5*lik*sk2 + 60*(lik2*r+r3-r*sk2);
               dl = dl / lik13;
            }
            du = 6*uik3 - 125*uik*r2 -5*uik*sk2 + 60*(uik2*r+r3-r*sk2);
            du = -du / uik13;
            de += aoi*rmixo7*pi*(dl + du)/(60*r2);
         }
      }
   }
   sum *= scale;
   de *= scale;
}

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
}

namespace tinker {
struct PairSolvGrad
{
   real frcx, frcy, frcz;
   real ttqi[3];
   real ttqk[3];
};

SEQ_ROUTINE
inline void zero(PairSolvGrad& pgrad)
{
   pgrad.frcx = 0;
   pgrad.frcy = 0;
   pgrad.frcz = 0;
   pgrad.ttqi[0] = 0;
   pgrad.ttqi[1] = 0;
   pgrad.ttqi[2] = 0;
   pgrad.ttqk[0] = 0;
   pgrad.ttqk[1] = 0;
   pgrad.ttqk[2] = 0;
}

#pragma acc routine seq
template <class Ver>
SEQ_ROUTINE
inline void pair_egka(real r2, real xr, real yr, real zr, real xr2, real yr2, real zr2,
                     real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
                     real qiyy, real qiyz, real qizz, real uidx, real uidy, real uidz, 
                     real uipx, real uipy, real uipz, real rbi,
                     real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
                     real qkyy, real qkyz, real qkzz, real ukdx, real ukdy, real ukdz, 
                     real ukpx, real ukpy, real ukpz, real rbk,
                     real gkc, real fc, real fd, real fq,
                     real& e, PairSolvGrad& pgrad, real& dedx, real& dedy, real& dedz, real& drbi, real& dpbi, real& drbk, real& dpbk)
{
   constexpr bool do_g = Ver::g;

   real uix = uidx + uipx;
   real uiy = uidy + uipy;
   real uiz = uidz + uipz;
   real ukx = ukdx + ukpx;
   real uky = ukdy + ukpy;
   real ukz = ukdz + ukpz;

   real rb2 = rbi * rbk;
   real expterm = REAL_EXP(-r2 / (gkc * rb2));
   real expc = expterm / gkc;
   real dexpc = -2/(gkc * rb2);
   real gf2 = 1/(r2 + rb2*expterm);
   real gf = REAL_SQRT(gf2);
   real gf3 = gf2 * gf;
   real gf5 = gf3 * gf2;
   real gf7 = gf5 * gf2;
   real gf9 = gf7 * gf2;

   real expcr,dexpcr,dgfdr,gf11;
   if CONSTEXPR (do_g) {
      expcr = r2*expterm / (gkc*gkc*rb2*rb2);
      dexpcr = 2 / (gkc*rb2*rb2);
      dgfdr = 0.5f * expterm * (1+r2/(rb2*gkc));
      gf11 = gf9 * gf2;
   }

   real a[6][4];
   real gc[30];
   real gux[30];
   real guy[30];
   real guz[30];
   real gqxx[30];
   real gqxy[30];
   real gqxz[30];
   real gqyy[30];
   real gqyz[30];
   real gqzz[30];

   a[0][0] = gf;
   a[1][0] = -gf3;
   a[2][0] = 3 * gf5;
   a[3][0] = -15 * gf7;
   a[4][0] = 105 * gf9;

   real b[5][3];
   if CONSTEXPR (do_g) {
      a[5][0] = -945 * gf11;

      b[0][0] = dgfdr * a[1][0];
      b[1][0] = dgfdr * a[2][0];
      b[2][0] = dgfdr * a[3][0];
      b[3][0] = dgfdr * a[4][0];
      b[4][0] = dgfdr * a[5][0];
   }

   real expc1 = 1 - expc;
   a[0][1] = expc1 * a[1][0];
   a[1][1] = expc1 * a[2][0];
   a[2][1] = expc1 * a[3][0];
   a[3][1] = expc1 * a[4][0];

   if CONSTEXPR (do_g) {
      a[4][1] = expc1 * a[5][0];

      b[0][1] = b[1][0] - expcr * a[1][0] - expc * b[1][0];
      b[1][1] = b[2][0] - expcr * a[2][0] - expc * b[2][0];
      b[2][1] = b[3][0] - expcr * a[3][0] - expc * b[3][0];
      b[3][1] = b[4][0] - expcr * a[4][0] - expc * b[4][0];
   }

   real expcdexpc = -expc * dexpc;
   a[0][2] = expc1*a[1][1] + expcdexpc*a[1][0];
   a[1][2] = expc1*a[2][1] + expcdexpc*a[2][0];
   a[2][2] = expc1*a[3][1] + expcdexpc*a[3][0];

   if CONSTEXPR (do_g) {
      a[3][2] = expc1 * a[4][1] + expcdexpc * a[4][0];
      b[0][2] = b[1][1] - (expcr * (a[1][1] + dexpc * a[1][0]) + expc * (b[1][1] + dexpcr * a[1][0] + dexpc * b[1][0]));
      b[1][2] = b[2][1] - (expcr * (a[2][1] + dexpc * a[2][0]) + expc * (b[2][1] + dexpcr * a[2][0] + dexpc * b[2][0]));
      b[2][2] = b[3][1] - (expcr * (a[3][1] + dexpc * a[3][0]) + expc * (b[3][1] + dexpcr * a[3][0] + dexpc * b[3][0]));
      
      expcdexpc = 2 * expcdexpc;
      a[0][3] = expc1 * a[1][2] + expcdexpc * a[1][1];
      a[1][3] = expc1 * a[2][2] + expcdexpc * a[2][1];
      a[2][3] = expc1 * a[3][2] + expcdexpc * a[3][1];

      expcdexpc = -expc * dexpc * dexpc;
      a[0][3] = a[0][3] + expcdexpc * a[1][0];
      a[1][3] = a[1][3] + expcdexpc * a[2][0];
      a[2][3] = a[2][3] + expcdexpc * a[3][0];
   }

   a[0][0] = fc * a[0][0];
   a[0][1] = fc * a[0][1];
   a[0][2] = fc * a[0][2];
   a[1][0] = fd * a[1][0];
   a[1][1] = fd * a[1][1];
   a[1][2] = fd * a[1][2];
   a[2][0] = fq * a[2][0];
   a[2][1] = fq * a[2][1];
   a[2][2] = fq * a[2][2];

   if CONSTEXPR (do_g) {
      a[0][3] = fc * a[0][3];
      a[1][3] = fd * a[1][3];
      a[2][3] = fq * a[2][3];

      b[0][0] = fc * b[0][0];
      b[0][1] = fc * b[0][1];
      b[0][2] = fc * b[0][2];
      b[1][0] = fd * b[1][0];
      b[1][1] = fd * b[1][1];
      b[1][2] = fd * b[1][2];
      b[2][0] = fq * b[2][0];
      b[2][1] = fq * b[2][1];
      b[2][2] = fq * b[2][2];
   }

   gc[0] = a[0][0];
   gux[0] = xr * a[1][0];
   guy[0] = yr * a[1][0];
   guz[0] = zr * a[1][0];
   gqxx[0] = xr2 * a[2][0];
   gqyy[0] = yr2 * a[2][0];
   gqzz[0] = zr2 * a[2][0];
   gqxy[0] = xr * yr * a[2][0];
   gqxz[0] = xr * zr * a[2][0];
   gqyz[0] = yr * zr * a[2][0];

   if CONSTEXPR (do_g) {
      gc[20] = b[0][0];
      gux[20] = xr * b[1][0];
      guy[20] = yr * b[1][0];
      guz[20] = zr * b[1][0];
      gqxx[20] = xr2 * b[2][0];
      gqyy[20] = yr2 * b[2][0];
      gqzz[20] = zr2 * b[2][0];
      gqxy[20] = xr * yr * b[2][0];
      gqxz[20] = xr * zr * b[2][0];
      gqyz[20] = yr * zr * b[2][0];
   }

   gc[1] = xr * a[0][1];
   gc[2] = yr * a[0][1];
   gc[3] = zr * a[0][1];
   gux[1] = a[1][0] + xr2 * a[1][1];
   gux[2] = xr * yr * a[1][1];
   gux[3] = xr * zr * a[1][1];
   guy[1] = gux[2];
   guy[2] = a[1][0] + yr2 * a[1][1];
   guy[3] = yr * zr * a[1][1];
   guz[1] = gux[3];
   guz[2] = guy[3];
   guz[3] = a[1][0] + zr2 * a[1][1];
   gqxx[1] = xr * (2 * a[2][0] + xr2 * a[2][1]);
   gqxx[2] = yr * xr2 * a[2][1];
   gqxx[3] = zr * xr2 * a[2][1];
   gqyy[1] = xr * yr2 * a[2][1];
   gqyy[2] = yr * (2 * a[2][0] + yr2 * a[2][1]);
   gqyy[3] = zr * yr2 * a[2][1];
   gqzz[1] = xr * zr2 * a[2][1];
   gqzz[2] = yr * zr2 * a[2][1];
   gqzz[3] = zr * (2 * a[2][0] + zr2 * a[2][1]);
   gqxy[1] = yr * (a[2][0] + xr2 * a[2][1]);
   gqxy[2] = xr * (a[2][0] + yr2 * a[2][1]);
   gqxy[3] = zr * xr * yr * a[2][1];
   gqxz[1] = zr * (a[2][0] + xr2 * a[2][1]);
   gqxz[2] = gqxy[3];
   gqxz[3] = xr * (a[2][0] + zr2 * a[2][1]);
   gqyz[1] = gqxy[3];
   gqyz[2] = zr * (a[2][0] + yr2 * a[2][1]);
   gqyz[3] = yr * (a[2][0] + zr2 * a[2][1]);

   if CONSTEXPR (do_g) {
      gc[21] = xr * b[0][1];
      gc[22] = yr * b[0][1];
      gc[23] = zr * b[0][1];
      gux[21] = b[1][0] + xr2 * b[1][1];
      gux[22] = xr * yr * b[1][1];
      gux[23] = xr * zr * b[1][1];
      guy[21] = gux[22];
      guy[22] = b[1][0] + yr2 * b[1][1];
      guy[23] = yr * zr * b[1][1];
      guz[21] = gux[23];
      guz[22] = guy[23];
      guz[23] = b[1][0] + zr2 * b[1][1];
      gqxx[21] = xr * (2 * b[2][0] + xr2 * b[2][1]);
      gqxx[22] = yr * xr2 * b[2][1];
      gqxx[23] = zr * xr2 * b[2][1];
      gqyy[21] = xr * yr2 * b[2][1];
      gqyy[22] = yr * (2 * b[2][0] + yr2 * b[2][1]);
      gqyy[23] = zr * yr2 * b[2][1];
      gqzz[21] = xr * zr2 * b[2][1];
      gqzz[22] = yr * zr2 * b[2][1];
      gqzz[23] = zr * (2 * b[2][0] + zr2 * b[2][1]);
      gqxy[21] = yr * (b[2][0] + xr2 * b[2][1]);
      gqxy[22] = xr * (b[2][0] + yr2 * b[2][1]);
      gqxy[23] = zr * xr * yr * b[2][1];
      gqxz[21] = zr * (b[2][0] + xr2 * b[2][1]);
      gqxz[22] = gqxy[23];
      gqxz[23] = xr * (b[2][0] + zr2 * b[2][1]);
      gqyz[21] = gqxy[23];
      gqyz[22] = zr * (b[2][0] + yr2 * b[2][1]);
      gqyz[23] = yr * (b[2][0] + zr2 * b[2][1]);
   }

   gc[4] = a[0][1] + xr2 * a[0][2];
   gc[5] = xr * yr * a[0][2];
   gc[6] = xr * zr * a[0][2];
   gc[7] = a[0][1] + yr2 * a[0][2];
   gc[8] = yr * zr * a[0][2];
   gc[9] = a[0][1] + zr2 * a[0][2];
   gux[4] = xr * (a[1][1] + 2 * a[1][1] + xr2 * a[1][2]);
   gux[5] = yr * (a[1][1] + xr2 * a[1][2]);
   gux[6] = zr * (a[1][1] + xr2 * a[1][2]);
   gux[7] = xr * (a[1][1] + yr2 * a[1][2]);
   gux[8] = zr * xr * yr * a[1][2];
   gux[9] = xr * (a[1][1] + zr2 * a[1][2]);
   guy[4] = yr * (a[1][1] + xr2 * a[1][2]);
   guy[5] = xr * (a[1][1] + yr2 * a[1][2]);
   guy[6] = gux[8];
   guy[7] = yr * (a[1][1] + 2 * a[1][1] + yr2 * a[1][2]);
   guy[8] = zr * (a[1][1] + yr2 * a[1][2]);
   guy[9] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[4] = zr * (a[1][1] + xr2 * a[1][2]);
   guz[5] = gux[8];
   guz[6] = xr * (a[1][1] + zr2 * a[1][2]);
   guz[7] = zr * (a[1][1] + yr2 * a[1][2]);
   guz[8] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[9] = zr * (a[1][1] + 2 * a[1][1] + zr2 * a[1][2]);

   gqxx[4] = 2 * a[2][0] + xr2 * (5 * a[2][1] + xr2 * a[2][2]);
   gqxx[5] = yr * xr * (2 * a[2][1] + xr2 * a[2][2]);
   gqxx[6] = zr * xr * (2 * a[2][1] + xr2 * a[2][2]);
   gqxx[7] = xr2 * (a[2][1] + yr2 * a[2][2]);
   gqxx[8] = zr * yr * xr2 * a[2][2];
   gqxx[9] = xr2 * (a[2][1] + zr2 * a[2][2]);
   gqyy[4] = yr2 * (a[2][1] + xr2 * a[2][2]);
   gqyy[5] = xr * yr * (2 * a[2][1] + yr2 * a[2][2]);
   gqyy[6] = xr * zr * yr2 * a[2][2];
   gqyy[7] = 2 * a[2][0] + yr2 * (5 * a[2][1] + yr2 * a[2][2]);
   gqyy[8] = yr * zr * (2 * a[2][1] + yr2 * a[2][2]);
   gqyy[9] = yr2 * (a[2][1] + zr2 * a[2][2]);
   gqzz[4] = zr2 * (a[2][1] + xr2 * a[2][2]);
   gqzz[5] = xr * yr * zr2 * a[2][2];
   gqzz[6] = xr * zr * (2 * a[2][1] + zr2 * a[2][2]);
   gqzz[7] = zr2 * (a[2][1] + yr2 * a[2][2]);
   gqzz[8] = yr * zr * (2 * a[2][1] + zr2 * a[2][2]);
   gqzz[9] = 2 * a[2][0] + zr2 * (5 * a[2][1] + zr2 * a[2][2]);
   
   gqxy[4] = xr * yr * (3 * a[2][1] + xr2 * a[2][2]);
   gqxy[5] = a[2][0] + (xr2 + yr2) * a[2][1] + xr2 * yr2 * a[2][2];
   gqxy[6] = zr * yr * (a[2][1] + xr2 * a[2][2]);
   gqxy[7] = xr * yr * (3 * a[2][1] + yr2 * a[2][2]);
   gqxy[8] = zr * xr * (a[2][1] + yr2 * a[2][2]);
   gqxy[9] = xr * yr * (a[2][1] + zr2 * a[2][2]);

   gqxz[4] = xr * zr * (3 * a[2][1] + xr2 * a[2][2]);
   gqxz[5] = yr * zr * (a[2][1] + xr2 * a[2][2]);
   gqxz[6] = a[2][0] + (xr2 + zr2) * a[2][1] + xr2 * zr2 * a[2][2];
   gqxz[7] = xr * zr * (a[2][1] + yr2 * a[2][2]);
   gqxz[8] = xr * yr * (a[2][1] + zr2 * a[2][2]);
   gqxz[9] = xr * zr * (3 * a[2][1] + zr2 * a[2][2]);

   gqyz[4] = zr * yr * (a[2][1] + xr2 * a[2][2]);
   gqyz[5] = xr * zr * (a[2][1] + yr2 * a[2][2]);
   gqyz[6] = xr * yr * (a[2][1] + zr2 * a[2][2]);
   gqyz[7] = yr * zr * (3 * a[2][1] + yr2 * a[2][2]);
   gqyz[8] = a[2][0] + (yr2 + zr2) * a[2][1] + yr2 * zr2 * a[2][2];
   gqyz[9] = yr * zr * (3 * a[2][1] + zr2 * a[2][2]);

   if CONSTEXPR (do_g) {
      gc[24] = b[0][1] + xr2 * b[0][2];
      gc[25] = xr * yr * b[0][2];
      gc[26] = xr * zr * b[0][2];
      gc[27] = b[0][1] + yr2 * b[0][2];
      gc[28] = yr * zr * b[0][2];
      gc[29] = b[0][1] + zr2 * b[0][2];
      gux[24] = xr * (3 * b[1][1] + xr2 * b[1][2]);
      gux[25] = yr * (b[1][1] + xr2 * b[1][2]);
      gux[26] = zr * (b[1][1] + xr2 * b[1][2]);
      gux[27] = xr * (b[1][1] + yr2 * b[1][2]);
      gux[28] = zr * xr * yr * b[1][2];
      gux[29] = xr * (b[1][1] + zr2 * b[1][2]);
      guy[24] = yr * (b[1][1] + xr2 * b[1][2]);
      guy[25] = xr * (b[1][1] + yr2 * b[1][2]);
      guy[26] = gux[28];
      guy[27] = yr * (3 * b[1][1] + yr2 * b[1][2]);
      guy[28] = zr * (b[1][1] + yr2 * b[1][2]);
      guy[29] = yr * (b[1][1] + zr2 * b[1][2]);
      guz[24] = zr * (b[1][1] + xr2 * b[1][2]);
      guz[25] = gux[28];
      guz[26] = xr * (b[1][1] + zr2 * b[1][2]);
      guz[27] = zr * (b[1][1] + yr2 * b[1][2]);
      guz[28] = yr * (b[1][1] + zr2 * b[1][2]);
      guz[29] = zr * (3 * b[1][1] + zr2 * b[1][2]);
      gqxx[24] = 2 * b[2][0] + xr2 * (5 * b[2][1] + xr2 * b[2][2]);
      gqxx[25] = yr * xr * (2 * b[2][1] + xr2 * b[2][2]);
      gqxx[26] = zr * xr * (2 * b[2][1] + xr2 * b[2][2]);
      gqxx[27] = xr2 * (b[2][1] + yr2 * b[2][2]);
      gqxx[28] = zr * yr * xr2 * b[2][2];
      gqxx[29] = xr2 * (b[2][1] + zr2 * b[2][2]);
      gqyy[24] = yr2 * (b[2][1] + xr2 * b[2][2]);
      gqyy[25] = xr * yr * (2 * b[2][1] + yr2 * b[2][2]);
      gqyy[26] = xr * zr * yr2 * b[2][2];
      gqyy[27] = 2 * b[2][0] + yr2 * (5 * b[2][1] + yr2 * b[2][2]);
      gqyy[28] = yr * zr * (2 * b[2][1] + yr2 * b[2][2]);
      gqyy[29] = yr2 * (b[2][1] + zr2 * b[2][2]);
      gqzz[24] = zr2 * (b[2][1] + xr2 * b[2][2]);
      gqzz[25] = xr * yr * zr2 * b[2][2];
      gqzz[26] = xr * zr * (2 * b[2][1] + zr2 * b[2][2]);
      gqzz[27] = zr2 * (b[2][1] + yr2 * b[2][2]);
      gqzz[28] = yr * zr * (2 * b[2][1] + zr2 * b[2][2]);
      gqzz[29] = 2 * b[2][0] + zr2 * (5 * b[2][1] + zr2 * b[2][2]);
      gqxy[24] = xr * yr * (3 * b[2][1] + xr2 * b[2][2]);
      gqxy[25] = b[2][0] + (xr2 + yr2) * b[2][1] + xr2 * yr2 * b[2][2];
      gqxy[26] = zr * yr * (b[2][1] + xr2 * b[2][2]);
      gqxy[27] = xr * yr * (3 * b[2][1] + yr2 * b[2][2]);
      gqxy[28] = zr * xr * (b[2][1] + yr2 * b[2][2]);
      gqxy[29] = xr * yr * (b[2][1] + zr2 * b[2][2]);
      gqxz[24] = xr * zr * (3 * b[2][1] + xr2 * b[2][2]);
      gqxz[25] = yr * zr * (b[2][1] + xr2 * b[2][2]);
      gqxz[26] = b[2][0] + (xr2 + zr2) * b[2][1] + xr2 * zr2 * b[2][2];
      gqxz[27] = xr * zr * (b[2][1] + yr2 * b[2][2]);
      gqxz[28] = xr * yr * (b[2][1] + zr2 * b[2][2]);
      gqxz[29] = xr * zr * (3 * b[2][1] + zr2 * b[2][2]);
      gqyz[24] = zr * yr * (b[2][1] + xr2 * b[2][2]);
      gqyz[25] = xr * zr * (b[2][1] + yr2 * b[2][2]);
      gqyz[26] = xr * yr * (b[2][1] + zr2 * b[2][2]);
      gqyz[27] = yr * zr * (3 * b[2][1] + yr2 * b[2][2]);
      gqyz[28] = b[2][0] + (yr2 + zr2) * b[2][1] + yr2 * zr2 * b[2][2];
      gqyz[29] = yr * zr * (3 * b[2][1] + zr2 * b[2][2]);

      gc[10] = xr * (3 * a[0][2] + xr2 * a[0][3]);
      gc[11] = yr * (a[0][2] + xr2 * a[0][3]);
      gc[12] = zr * (a[0][2] + xr2 * a[0][3]);
      gc[13] = xr * (a[0][2] + yr2 * a[0][3]);
      gc[14] = xr * yr * zr * a[0][3];
      gc[15] = xr * (a[0][2] + zr2 * a[0][3]);
      gc[16] = yr * (3 * a[0][2] + yr2 * a[0][3]);
      gc[17] = zr * (a[0][2] + yr2 * a[0][3]);
      gc[18] = yr * (a[0][2] + zr2 * a[0][3]);
      gc[19] = zr * (3 * a[0][2] + zr2 * a[0][3]);
      gux[10] = 3 * a[1][1] + xr2 * (6 * a[1][2] + xr2 * a[1][3]);
      gux[11] = xr * yr * (3 * a[1][2] + xr2 * a[1][3]);
      gux[12] = xr * zr * (3 * a[1][2] + xr2 * a[1][3]);
      gux[13] = a[1][1] + (xr2 + yr2) * a[1][2] + xr2 * yr2 * a[1][3];
      gux[14] = yr * zr * (a[1][2] + xr2 * a[1][3]);
      gux[15] = a[1][1] + (xr2 + zr2) * a[1][2] + xr2 * zr2 * a[1][3];
      gux[16] = xr * yr * (3 * a[1][2] + yr2 * a[1][3]);
      gux[17] = xr * zr * (a[1][2] + yr2 * a[1][3]);
      gux[18] = xr * yr * (a[1][2] + zr2 * a[1][3]);
      gux[19] = xr * zr * (3 * a[1][2] + zr2 * a[1][3]);
      guy[10] = gux[11];
      guy[11] = gux[13];
      guy[12] = gux[14];
      guy[13] = gux[16];
      guy[14] = gux[17];
      guy[15] = gux[18];
      guy[16] = 3 * a[1][1] + yr2 * (6 * a[1][2] + yr2 * a[1][3]);
      guy[17] = yr * zr * (3 * a[1][2] + yr2 * a[1][3]);
      guy[18] = a[1][1] + (yr2 + zr2) * a[1][2] + yr2 * zr2 * a[1][3];
      guy[19] = yr * zr * (3 * a[1][2] + zr2 * a[1][3]);
      guz[10] = gux[12];
      guz[11] = gux[14];
      guz[12] = gux[15];
      guz[13] = gux[17];
      guz[14] = gux[18];
      guz[15] = gux[19];
      guz[16] = guy[17];
      guz[17] = guy[18];
      guz[18] = guy[19];
      guz[19] = 3 * a[1][1] + zr2 * (6 * a[1][2] + zr2 * a[1][3]);
      gqxx[10] = xr * (12 * a[2][1] + xr2 * (9 * a[2][2] + xr2 * a[2][3]));
      gqxx[11] = yr * (2 * a[2][1] + xr2 * (5 * a[2][2] + xr2 * a[2][3]));
      gqxx[12] = zr * (2 * a[2][1] + xr2 * (5 * a[2][2] + xr2 * a[2][3]));
      gqxx[13] = xr * (2 * a[2][1] + yr2 * 2 * a[2][2] + xr2 * (a[2][2] + yr2 * a[2][3]));
      gqxx[14] = xr * yr * zr * (2 * a[2][2] + xr2 * a[2][3]);
      gqxx[15] = xr * (2 * a[2][1] + zr2 * 2 * a[2][2] + xr2 * (a[2][2] + zr2 * a[2][3]));
      gqxx[16] = yr * xr2 * (3 * a[2][2] + yr2 * a[2][3]);
      gqxx[17] = zr * xr2 * (a[2][2] + yr2 * a[2][3]);
      gqxx[18] = yr * xr2 * (a[2][2] + zr2 * a[2][3]);
      gqxx[19] = zr * xr2 * (3 * a[2][2] + zr2 * a[2][3]);
      gqxy[10] = yr * (3 * a[2][1] + xr2 * (6 * a[2][2] + xr2 * a[2][3]));
      gqxy[11] = xr * (3 * (a[2][1] + yr2 * a[2][2]) + xr2 * (a[2][2] + yr2 * a[2][3]));
      gqxy[12] = xr * yr * zr * (3 * a[2][2] + xr2 * a[2][3]);
      gqxy[13] = yr * (3 * (a[2][1] + xr2 * a[2][2]) + yr2 * (a[2][2] + xr2 * a[2][3]));
      gqxy[14] = zr * (a[2][1] + (yr2 + xr2) * a[2][2] + yr2 * xr2 * a[2][3]);
      gqxy[15] = yr * (a[2][1] + (xr2 + zr2) * a[2][2] + xr2 * zr2 * a[2][3]);
      gqxy[16] = xr * (3 * (a[2][1] + yr2 * a[2][2]) + yr2 * (3 * a[2][2] + yr2 * a[2][3]));
      gqxy[17] = xr * yr * zr * (3 * a[2][2] + yr2 * a[2][3]);
      gqxy[18] = xr * (a[2][1] + (yr2 + zr2) * a[2][2] + yr2 * zr2 * a[2][3]);
      gqxy[19] = xr * yr * zr * (3 * a[2][2] + zr2 * a[2][3]);
      gqxz[10] = zr * (3 * a[2][1] + xr2 * (6 * a[2][2] + xr2 * a[2][3]));
      gqxz[11] = xr * yr * zr * (3 * a[2][2] + xr2 * a[2][3]);
      gqxz[12] = xr * (3 * (a[2][1] + zr2 * a[2][2]) + xr2 * (a[2][2] + zr2 * a[2][3]));
      gqxz[13] = zr * (a[2][1] + (xr2 + yr2) * a[2][2] + xr2 * yr2 * a[2][3]);
      gqxz[14] = yr * (a[2][1] + (xr2 + zr2) * a[2][2] + zr2 * xr2 * a[2][3]);
      gqxz[15] = zr * (3 * (a[2][1] + xr2 * a[2][2]) + zr2 * (a[2][2] + xr2 * a[2][3]));
      gqxz[16] = xr * yr * zr * (3 * a[2][2] + yr2 * a[2][3]);
      gqxz[17] = xr * (a[2][1] + (zr2 + yr2) * a[2][2] + zr2 * yr2 * a[2][3]);
      gqxz[18] = xr * yr * zr * (3 * a[2][2] + zr2 * a[2][3]);
      gqxz[19] = xr * (3 * a[2][1] + zr2 * (6 * a[2][2] + zr2 * a[2][3]));
      gqyy[10] = xr * yr2 * (3 * a[2][2] + xr2 * a[2][3]);
      gqyy[11] = yr * (2 * a[2][1] + xr2 * 2 * a[2][2] + yr2 * (a[2][2] + xr2 * a[2][3]));
      gqyy[12] = zr * yr2 * (a[2][2] + xr2 * a[2][3]);
      gqyy[13] = xr * (2 * a[2][1] + yr2 * (5 * a[2][2] + yr2 * a[2][3]));
      gqyy[14] = xr * yr * zr * (2 * a[2][2] + yr2 * a[2][3]);
      gqyy[15] = xr * yr2 * (a[2][2] + zr2 * a[2][3]);
      gqyy[16] = yr * (12 * a[2][1] + yr2 * (9 * a[2][2] + yr2 * a[2][3]));
      gqyy[17] = zr * (2 * a[2][1] + yr2 * (5 * a[2][2] + yr2 * a[2][3]));
      gqyy[18] = yr * (2 * a[2][1] + zr2 * 2 * a[2][2] + yr2 * (a[2][2] + zr2 * a[2][3]));
      gqyy[19] = zr * yr2 * (3 * a[2][2] + zr2 * a[2][3]);
      gqyz[10] = xr * yr * zr * (3 * a[2][2] + xr2 * a[2][3]);
      gqyz[11] = zr * (a[2][1] + (xr2 + yr2) * a[2][2] + xr2 * yr2 * a[2][3]);
      gqyz[12] = yr * (a[2][1] + (xr2 + zr2) * a[2][2] + xr2 * zr2 * a[2][3]);
      gqyz[13] = xr * yr * zr * (3 * a[2][2] + yr2 * a[2][3]);
      gqyz[14] = xr * (a[2][1] + (yr2 + zr2) * a[2][2] + yr2 * zr2 * a[2][3]);
      gqyz[15] = xr * yr * zr * (3 * a[2][2] + zr2 * a[2][3]);
      gqyz[16] = zr * (3 * a[2][1] + yr2 * (6 * a[2][2] + yr2 * a[2][3]));
      gqyz[17] = yr * (3 * (a[2][1] + zr2 * a[2][2]) + yr2 * (a[2][2] + zr2 * a[2][3]));
      gqyz[18] = zr * (3 * (a[2][1] + yr2 * a[2][2]) + zr2 * (a[2][2] + yr2 * a[2][3]));
      gqyz[19] = yr * (3 * a[2][1] + zr2 * (6 * a[2][2] + zr2 * a[2][3]));
      gqzz[10] = xr * zr2 * (3 * a[2][2] + xr2 * a[2][3]);
      gqzz[11] = yr * (zr2 * a[2][2] + xr2 * (zr2 * a[2][3]));
      gqzz[12] = zr * (2 * a[2][1] + xr2 * 2 * a[2][2] + zr2 * (a[2][2] + xr2 * a[2][3]));
      gqzz[13] = xr * zr2 * (a[2][2] + yr2 * a[2][3]);
      gqzz[14] = xr * yr * zr * (2 * a[2][2] + zr2 * a[2][3]);
      gqzz[15] = xr * (2 * a[2][1] + zr2 * (5 * a[2][2] + zr2 * a[2][3]));
      gqzz[16] = yr * zr2 * (3 * a[2][2] + yr2 * a[2][3]);
      gqzz[17] = zr * (2 * a[2][1] + yr2 * 2 * a[2][2] + zr2 * (a[2][2] + yr2 * a[2][3]));
      gqzz[18] = yr * (2 * a[2][1] + zr2 * (5 * a[2][2] + zr2 * a[2][3]));
      gqzz[19] = zr * (12 * a[2][1] + zr2 * (9 * a[2][2] + zr2 * a[2][3]));
   }

   real esym = ci * ck * gc[0]
         - dix * (dkx * gux[1] + dky * guy[1] + dkz * guz[1])
         - diy * (dkx * gux[2] + dky * guy[2] + dkz * guz[2])
         - diz * (dkx * gux[3] + dky * guy[3] + dkz * guz[3]);

   real ewi = ci * (dkx * gc[1] + dky * gc[2] + dkz * gc[3])
      - ck * (dix * gux[0] + diy * guy[0] + diz * guz[0])
      + ci * (qkxx * gc[4] + qkyy * gc[7] + qkzz * gc[9]
              + 2 * (qkxy * gc[5] + qkxz * gc[6] + qkyz * gc[8]))
      + ck * (qixx * gqxx[0] + qiyy * gqyy[0] + qizz * gqzz[0]
              + 2 * (qixy * gqxy[0] + qixz * gqxz[0] + qiyz * gqyz[0]))
      - dix * (qkxx * gux[4] + qkyy * gux[7] + qkzz * gux[9]
               + 2 * (qkxy * gux[5] + qkxz * gux[6] + qkyz * gux[8]))
      - diy * (qkxx * guy[4] + qkyy * guy[7] + qkzz * guy[9]
               + 2 * (qkxy * guy[5] + qkxz * guy[6] + qkyz * guy[8]))
      - diz * (qkxx * guz[4] + qkyy * guz[7] + qkzz * guz[9]
               + 2 * (qkxy * guz[5] + qkxz * guz[6] + qkyz * guz[8]))
      + dkx * (qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
               + 2 * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]))
      + dky * (qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
               + 2 * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]))
      + dkz * (qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
               + 2 * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]))
      + qixx * (qkxx * gqxx[4] + qkyy * gqxx[7] + qkzz * gqxx[9]
                + 2 * (qkxy * gqxx[5] + qkxz * gqxx[6] + qkyz * gqxx[8]))
      + qiyy * (qkxx * gqyy[4] + qkyy * gqyy[7] + qkzz * gqyy[9]
                + 2 * (qkxy * gqyy[5] + qkxz * gqyy[6] + qkyz * gqyy[8]))
      + qizz * (qkxx * gqzz[4] + qkyy * gqzz[7] + qkzz * gqzz[9]
                + 2 * (qkxy * gqzz[5] + qkxz * gqzz[6] + qkyz * gqzz[8]))
      + 2 * (qixy * (qkxx * gqxy[4] + qkyy * gqxy[7] + qkzz * gqxy[9]
                       + 2 * (qkxy * gqxy[5] + qkxz * gqxy[6] + qkyz * gqxy[8]))
               + qixz * (qkxx * gqxz[4] + qkyy * gqxz[7] + qkzz * gqxz[9]
                         + 2 * (qkxy * gqxz[5] + qkxz * gqxz[6] + qkyz * gqxz[8]))
               + qiyz * (qkxx * gqyz[4] + qkyy * gqyz[7] + qkzz * gqyz[9]
                         + 2 * (qkxy * gqyz[5] + qkxz * gqyz[6] + qkyz * gqyz[8])));

   real ewk = ci * (dkx * gux[0] + dky * guy[0] + dkz * guz[0])
      - ck * (dix * gc[1] + diy * gc[2] + diz * gc[3])
      + ci * (qkxx * gqxx[0] + qkyy * gqyy[0] + qkzz * gqzz[0]
              + 2 * (qkxy * gqxy[0] + qkxz * gqxz[0] + qkyz * gqyz[0]))
      + ck * (qixx * gc[4] + qiyy * gc[7] + qizz * gc[9]
              + 2 * (qixy * gc[5] + qixz * gc[6] + qiyz * gc[8]))
      - dix * (qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
               + 2 * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]))
      - diy * (qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
               + 2 * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]))
      - diz * (qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
               + 2 * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]))
      + dkx * (qixx * gux[4] + qiyy * gux[7] + qizz * gux[9]
               + 2 * (qixy * gux[5] + qixz * gux[6] + qiyz * gux[8]))
      + dky * (qixx * guy[4] + qiyy * guy[7] + qizz * guy[9]
               + 2 * (qixy * guy[5] + qixz * guy[6] + qiyz * guy[8]))
      + dkz * (qixx * guz[4] + qiyy * guz[7] + qizz * guz[9]
               + 2 * (qixy * guz[5] + qixz * guz[6] + qiyz * guz[8]))
      + qixx * (qkxx * gqxx[4] + qkyy * gqyy[4] + qkzz * gqzz[4]
                + 2 * (qkxy * gqxy[4] + qkxz * gqxz[4] + qkyz * gqyz[4]))
      + qiyy * (qkxx * gqxx[7] + qkyy * gqyy[7] + qkzz * gqzz[7]
                + 2 * (qkxy * gqxy[7] + qkxz * gqxz[7] + qkyz * gqyz[7]))
      + qizz * (qkxx * gqxx[9] + qkyy * gqyy[9] + qkzz * gqzz[9]
                + 2 * (qkxy * gqxy[9] + qkxz * gqxz[9] + qkyz * gqyz[9]))
      + 2 * (qixy * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                       + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
               + qixz * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                         + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
               + qiyz * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                         + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8])));

   real esymi = -dix * (ukdx * gux[1] + ukdy * guy[1] + ukdz * guz[1])
        - diy * (ukdx * gux[2] + ukdy * guy[2] + ukdz * guz[2])
        - diz * (ukdx * gux[3] + ukdy * guy[3] + ukdz * guz[3])
        - dkx * (uidx * gux[1] + uidy * guy[1] + uidz * guz[1])
        - dky * (uidx * gux[2] + uidy * guy[2] + uidz * guz[2])
        - dkz * (uidx * gux[3] + uidy * guy[3] + uidz * guz[3]);

   real ewii = ci * (ukdx * gc[1] + ukdy * gc[2] + ukdz * gc[3])
       - ck * (uidx * gux[0] + uidy * guy[0] + uidz * guz[0])
       - uidx * (qkxx * gux[4] + qkyy * gux[7] + qkzz * gux[9]
                 + 2 * (qkxy * gux[5] + qkxz * gux[6] + qkyz * gux[8]))
       - uidy * (qkxx * guy[4] + qkyy * guy[7] + qkzz * guy[9]
                 + 2 * (qkxy * guy[5] + qkxz * guy[6] + qkyz * guy[8]))
       - uidz * (qkxx * guz[4] + qkyy * guz[7] + qkzz * guz[9]
                 + 2 * (qkxy * guz[5] + qkxz * guz[6] + qkyz * guz[8]))
       + ukdx * (qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
                 + 2 * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]))
       + ukdy * (qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
                 + 2 * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]))
       + ukdz * (qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
                 + 2 * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]));

   real ewki = ci * (ukdx * gux[0] + ukdy * guy[0] + ukdz * guz[0])
       - ck * (uidx * gc[1] + uidy * gc[2] + uidz * gc[3])
       - uidx * (qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
                 + 2 * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]))
       - uidy * (qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
                 + 2 * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]))
       - uidz * (qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
                 + 2 * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]))
       + ukdx * (qixx * gux[4] + qiyy * gux[7] + qizz * gux[9]
                 + 2 * (qixy * gux[5] + qixz * gux[6] + qiyz * gux[8]))
       + ukdy * (qixx * guy[4] + qiyy * guy[7] + qizz * guy[9]
                 + 2 * (qixy * guy[5] + qixz * guy[6] + qiyz * guy[8]))
       + ukdz * (qixx * guz[4] + qiyy * guz[7] + qizz * guz[9]
                 + 2 * (qixy * guz[5] + qixz * guz[6] + qiyz * guz[8]));

   e = esym + 0.5f*(ewi+ewk);
   e += 0.5f * (esymi + 0.5f*(ewii+ewki));

   if CONSTEXPR (do_g) {
      real desymdx = ci * ck * gc[1]
            - (dix * (dkx * gux[4] + dky * guy[4] + dkz * guz[4])
            +  diy * (dkx * gux[5] + dky * guy[5] + dkz * guz[5])
            +  diz * (dkx * gux[6] + dky * guy[6] + dkz * guz[6]));

      real dewidx = ci * (dkx * gc[4] + dky * gc[5] + dkz * gc[6])
                  - ck * (dix * gux[1] + diy * guy[1] + diz * guz[1])
                  + ci * (qkxx * gc[10] + qkyy * gc[13] + qkzz * gc[15]
                 + 2 * (qkxy * gc[11] + qkxz * gc[12] + qkyz * gc[14]))
                  + ck * (qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
                 + 2 * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]))
                 - dix * (qkxx * gux[10] + qkyy * gux[13] + qkzz * gux[15]
                 + 2 * (qkxy * gux[11] + qkxz * gux[12] + qkyz * gux[14]))
                 - diy * (qkxx * guy[10] + qkyy * guy[13] + qkzz * guy[15]
                 + 2 * (qkxy * guy[11] + qkxz * guy[12] + qkyz * guy[14]))
                 - diz * (qkxx * guz[10] + qkyy * guz[13] + qkzz * guz[15]
                 + 2 * (qkxy * guz[11] + qkxz * guz[12] + qkyz * guz[14]))
                 + dkx * (qixx * gqxx[4] + qiyy * gqyy[4] + qizz * gqzz[4]
                 + 2 * (qixy * gqxy[4] + qixz * gqxz[4] + qiyz * gqyz[4]))
                 + dky * (qixx * gqxx[5] + qiyy * gqyy[5] + qizz * gqzz[5]
                 + 2 * (qixy * gqxy[5] + qixz * gqxz[5] + qiyz * gqyz[5]))
                 + dkz * (qixx * gqxx[6] + qiyy * gqyy[6] + qizz * gqzz[6]
                 + 2 * (qixy * gqxy[6] + qixz * gqxz[6] + qiyz * gqyz[6]))
                + qixx * (qkxx * gqxx[10] + qkyy * gqxx[13] + qkzz * gqxx[15]
                 + 2 * (qkxy * gqxx[11] + qkxz * gqxx[12] + qkyz * gqxx[14]))
                + qiyy * (qkxx * gqyy[10] + qkyy * gqyy[13] + qkzz * gqyy[15]
                 + 2 * (qkxy * gqyy[11] + qkxz * gqyy[12] + qkyz * gqyy[14]))
                + qizz * (qkxx * gqzz[10] + qkyy * gqzz[13] + qkzz * gqzz[15]
                 + 2 * (qkxy * gqzz[11] + qkxz * gqzz[12] + qkyz * gqzz[14]))
         + 2 * (qixy * (qkxx * gqxy[10] + qkyy * gqxy[13] + qkzz * gqxy[15]
                 + 2 * (qkxy * gqxy[11] + qkxz * gqxy[12] + qkyz * gqxy[14]))
                + qixz * (qkxx * gqxz[10] + qkyy * gqxz[13] + qkzz * gqxz[15]
                 + 2 * (qkxy * gqxz[11] + qkxz * gqxz[12] + qkyz * gqxz[14]))
                + qiyz * (qkxx * gqyz[10] + qkyy * gqyz[13] + qkzz * gqyz[15]
                 + 2 * (qkxy * gqyz[11] + qkxz * gqyz[12] + qkyz * gqyz[14])));

      real dewkdx = ci * (dkx * gux[1] + dky * guy[1] + dkz * guz[1])
                  - ck * (dix * gc[4] + diy * gc[5] + diz * gc[6])
                  + ci * (qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
                 + 2 * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]))
                  + ck * (qixx * gc[10] + qiyy * gc[13] + qizz * gc[15]
                 + 2 * (qixy * gc[11] + qixz * gc[12] + qiyz * gc[14]))
                 - dix * (qkxx * gqxx[4] + qkyy * gqyy[4] + qkzz * gqzz[4]
                 + 2 * (qkxy * gqxy[4] + qkxz * gqxz[4] + qkyz * gqyz[4]))
                 - diy * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                 + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
                 - diz * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                 + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
                 + dkx * (qixx * gux[10] + qiyy * gux[13] + qizz * gux[15]
                 + 2 * (qixy * gux[11] + qixz * gux[12] + qiyz * gux[14]))
                 + dky * (qixx * guy[10] + qiyy * guy[13] + qizz * guy[15]
                 + 2 * (qixy * guy[11] + qixz * guy[12] + qiyz * guy[14]))
                 + dkz * (qixx * guz[10] + qiyy * guz[13] + qizz * guz[15]
                 + 2 * (qixy * guz[11] + qixz * guz[12] + qiyz * guz[14]))
                + qixx * (qkxx * gqxx[10] + qkyy * gqyy[10] + qkzz * gqzz[10]
                 + 2 * (qkxy * gqxy[10] + qkxz * gqxz[10] + qkyz * gqyz[10]))
                + qiyy * (qkxx * gqxx[13] + qkyy * gqyy[13] + qkzz * gqzz[13]
                 + 2 * (qkxy * gqxy[13] + qkxz * gqxz[13] + qkyz * gqyz[13]))
                + qizz * (qkxx * gqxx[15] + qkyy * gqyy[15] + qkzz * gqzz[15]
                 + 2 * (qkxy * gqxy[15] + qkxz * gqxz[15] + qkyz * gqyz[15]))
         + 2 * (qixy * (qkxx * gqxx[11] + qkyy * gqyy[11] + qkzz * gqzz[11]
                 + 2 * (qkxy * gqxy[11] + qkxz * gqxz[11] + qkyz * gqyz[11]))
                + qixz * (qkxx * gqxx[12] + qkyy * gqyy[12] + qkzz * gqzz[12]
                 + 2 * (qkxy * gqxy[12] + qkxz * gqxz[12] + qkyz * gqyz[12]))
                + qiyz * (qkxx * gqxx[14] + qkyy * gqyy[14] + qkzz * gqzz[14]
                 + 2 * (qkxy * gqxy[14] + qkxz * gqxz[14] + qkyz * gqyz[14])));

      dedx = desymdx + 0.5f*(dewidx+dewkdx);

      real desymdy = ci * ck * gc[2]
               -(dix * (dkx * gux[5] + dky * guy[5] + dkz * guz[5])
               + diy * (dkx * gux[7] + dky * guy[7] + dkz * guz[7])
               + diz * (dkx * gux[8] + dky * guy[8] + dkz * guz[8]));

      real dewidy = ci * (dkx * gc[5] + dky * gc[7] + dkz * gc[8])
                  - ck * (dix * gux[2] + diy * guy[2] + diz * guz[2])
                  + ci * (qkxx * gc[11] + qkyy * gc[16] + qkzz * gc[18]
                 + 2 * (qkxy * gc[13] + qkxz * gc[14] + qkyz * gc[17]))
                  + ck * (qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
                 + 2 * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]))
                 - dix * (qkxx * gux[11] + qkyy * gux[16] + qkzz * gux[18]
                 + 2 * (qkxy * gux[13] + qkxz * gux[14] + qkyz * gux[17]))
                 - diy * (qkxx * guy[11] + qkyy * guy[16] + qkzz * guy[18]
                 + 2 * (qkxy * guy[13] + qkxz * guy[14] + qkyz * guy[17]))
                 - diz * (qkxx * guz[11] + qkyy * guz[16] + qkzz * guz[18]
                 + 2 * (qkxy * guz[13] + qkxz * guz[14] + qkyz * guz[17]))
                 + dkx * (qixx * gqxx[5] + qiyy * gqyy[5] + qizz * gqzz[5]
                 + 2 * (qixy * gqxy[5] + qixz * gqxz[5] + qiyz * gqyz[5]))
                 + dky * (qixx * gqxx[7] + qiyy * gqyy[7] + qizz * gqzz[7]
                 + 2 * (qixy * gqxy[7] + qixz * gqxz[7] + qiyz * gqyz[7]))
                 + dkz * (qixx * gqxx[8] + qiyy * gqyy[8] + qizz * gqzz[8]
                 + 2 * (qixy * gqxy[8] + qixz * gqxz[8] + qiyz * gqyz[8]))
                + qixx * (qkxx * gqxx[11] + qkyy * gqxx[16] + qkzz * gqxx[18]
                 + 2 * (qkxy * gqxx[13] + qkxz * gqxx[14] + qkyz * gqxx[17]))
                + qiyy * (qkxx * gqyy[11] + qkyy * gqyy[16] + qkzz * gqyy[18]
                 + 2 * (qkxy * gqyy[13] + qkxz * gqyy[14] + qkyz * gqyy[17]))
                + qizz * (qkxx * gqzz[11] + qkyy * gqzz[16] + qkzz * gqzz[18]
                 + 2 * (qkxy * gqzz[13] + qkxz * gqzz[14] + qkyz * gqzz[17]))
         + 2 * (qixy * (qkxx * gqxy[11] + qkyy * gqxy[16] + qkzz * gqxy[18]
                 + 2 * (qkxy * gqxy[13] + qkxz * gqxy[14] + qkyz * gqxy[17]))
                + qixz * (qkxx * gqxz[11] + qkyy * gqxz[16] + qkzz * gqxz[18]
                 + 2 * (qkxy * gqxz[13] + qkxz * gqxz[14] + qkyz * gqxz[17]))
                + qiyz * (qkxx * gqyz[11] + qkyy * gqyz[16] + qkzz * gqyz[18]
                 + 2 * (qkxy * gqyz[13] + qkxz * gqyz[14] + qkyz * gqyz[17])));

      real dewkdy = ci * (dkx * gux[2] + dky * guy[2] + dkz * guz[2])
                  - ck * (dix * gc[5] + diy * gc[7] + diz * gc[8])
                  + ci * (qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
                 + 2 * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]))
                  + ck * (qixx * gc[11] + qiyy * gc[16] + qizz * gc[18]
                 + 2 * (qixy * gc[13] + qixz * gc[14] + qiyz * gc[17]))
                 - dix * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                 + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
                 - diy * (qkxx * gqxx[7] + qkyy * gqyy[7] + qkzz * gqzz[7]
                 + 2 * (qkxy * gqxy[7] + qkxz * gqxz[7] + qkyz * gqyz[7]))
                 - diz * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                 + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8]))
                 + dkx * (qixx * gux[11] + qiyy * gux[16] + qizz * gux[18]
                 + 2 * (qixy * gux[13] + qixz * gux[14] + qiyz * gux[17]))
                 + dky * (qixx * guy[11] + qiyy * guy[16] + qizz * guy[18]
                 + 2 * (qixy * guy[13] + qixz * guy[14] + qiyz * guy[17]))
                 + dkz * (qixx * guz[11] + qiyy * guz[16] + qizz * guz[18]
                 + 2 * (qixy * guz[13] + qixz * guz[14] + qiyz * guz[17]))
                + qixx * (qkxx * gqxx[11] + qkyy * gqyy[11] + qkzz * gqzz[11]
                 + 2 * (qkxy * gqxy[11] + qkxz * gqxz[11] + qkyz * gqyz[11]))
                + qiyy * (qkxx * gqxx[16] + qkyy * gqyy[16] + qkzz * gqzz[16]
                 + 2 * (qkxy * gqxy[16] + qkxz * gqxz[16] + qkyz * gqyz[16]))
                + qizz * (qkxx * gqxx[18] + qkyy * gqyy[18] + qkzz * gqzz[18]
                 + 2 * (qkxy * gqxy[18] + qkxz * gqxz[18] + qkyz * gqyz[18]))
         + 2 * (qixy * (qkxx * gqxx[13] + qkyy * gqyy[13] + qkzz * gqzz[13]
                 + 2 * (qkxy * gqxy[13] + qkxz * gqxz[13] + qkyz * gqyz[13]))
                + qixz * (qkxx * gqxx[14] + qkyy * gqyy[14] + qkzz * gqzz[14]
                 + 2 * (qkxy * gqxy[14] + qkxz * gqxz[14] + qkyz * gqyz[14]))
                + qiyz * (qkxx * gqxx[17] + qkyy * gqyy[17] + qkzz * gqzz[17]
                 + 2 * (qkxy * gqxy[17] + qkxz * gqxz[17] + qkyz * gqyz[17])));

      dedy = desymdy + 0.5f*(dewidy+dewkdy);

      real desymdz = ci * ck * gc[3]
                  -(dix * (dkx * gux[6] + dky * guy[6] + dkz * guz[6])
                  + diy * (dkx * gux[8] + dky * guy[8] + dkz * guz[8])
                  + diz * (dkx * gux[9] + dky * guy[9] + dkz * guz[9]));

      real dewidz = ci * (dkx * gc[6] + dky * gc[8] + dkz * gc[9])
                  - ck * (dix * gux[3] + diy * guy[3] + diz * guz[3])
                  + ci * (qkxx * gc[12] + qkyy * gc[17] + qkzz * gc[19]
                 + 2 * (qkxy * gc[14] + qkxz * gc[15] + qkyz * gc[18]))
                  + ck * (qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
                 + 2 * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]))
                 - dix * (qkxx * gux[12] + qkyy * gux[17] + qkzz * gux[19]
                 + 2 * (qkxy * gux[14] + qkxz * gux[15] + qkyz * gux[18]))
                 - diy * (qkxx * guy[12] + qkyy * guy[17] + qkzz * guy[19]
                 + 2 * (qkxy * guy[14] + qkxz * guy[15] + qkyz * guy[18]))
                 - diz * (qkxx * guz[12] + qkyy * guz[17] + qkzz * guz[19]
                 + 2 * (qkxy * guz[14] + qkxz * guz[15] + qkyz * guz[18]))
                 + dkx * (qixx * gqxx[6] + qiyy * gqyy[6] + qizz * gqzz[6]
                 + 2 * (qixy * gqxy[6] + qixz * gqxz[6] + qiyz * gqyz[6]))
                 + dky * (qixx * gqxx[8] + qiyy * gqyy[8] + qizz * gqzz[8]
                 + 2 * (qixy * gqxy[8] + qixz * gqxz[8] + qiyz * gqyz[8]))
                 + dkz * (qixx * gqxx[9] + qiyy * gqyy[9] + qizz * gqzz[9]
                 + 2 * (qixy * gqxy[9] + qixz * gqxz[9] + qiyz * gqyz[9]))
                + qixx * (qkxx * gqxx[12] + qkyy * gqxx[17] + qkzz * gqxx[19]
                 + 2 * (qkxy * gqxx[14] + qkxz * gqxx[15] + qkyz * gqxx[18]))
                + qiyy * (qkxx * gqyy[12] + qkyy * gqyy[17] + qkzz * gqyy[19]
                 + 2 * (qkxy * gqyy[14] + qkxz * gqyy[15] + qkyz * gqyy[18]))
                + qizz * (qkxx * gqzz[12] + qkyy * gqzz[17] + qkzz * gqzz[19]
                 + 2 * (qkxy * gqzz[14] + qkxz * gqzz[15] + qkyz * gqzz[18]))
         + 2 * (qixy * (qkxx * gqxy[12] + qkyy * gqxy[17] + qkzz * gqxy[19]
                 + 2 * (qkxy * gqxy[14] + qkxz * gqxy[15] + qkyz * gqxy[18]))
                + qixz * (qkxx * gqxz[12] + qkyy * gqxz[17] + qkzz * gqxz[19]
                 + 2 * (qkxy * gqxz[14] + qkxz * gqxz[15] + qkyz * gqxz[18]))
                + qiyz * (qkxx * gqyz[12] + qkyy * gqyz[17] + qkzz * gqyz[19]
                 + 2 * (qkxy * gqyz[14] + qkxz * gqyz[15] + qkyz * gqyz[18])));

      real dewkdz = ci * (dkx * gux[3] + dky * guy[3] + dkz * guz[3])
                  - ck * (dix * gc[6] + diy * gc[8] + diz * gc[9])
                  + ci * (qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
                 + 2 * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]))
                  + ck * (qixx * gc[12] + qiyy * gc[17] + qizz * gc[19]
                 + 2 * (qixy * gc[14] + qixz * gc[15] + qiyz * gc[18]))
                 - dix * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                 + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
                 - diy * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                 + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8]))
                 - diz * (qkxx * gqxx[9] + qkyy * gqyy[9] + qkzz * gqzz[9]
                 + 2 * (qkxy * gqxy[9] + qkxz * gqxz[9] + qkyz * gqyz[9]))
                 + dkx * (qixx * gux[12] + qiyy * gux[17] + qizz * gux[19]
                 + 2 * (qixy * gux[14] + qixz * gux[15] + qiyz * gux[18]))
                 + dky * (qixx * guy[12] + qiyy * guy[17] + qizz * guy[19]
                 + 2 * (qixy * guy[14] + qixz * guy[15] + qiyz * guy[18]))
                 + dkz * (qixx * guz[12] + qiyy * guz[17] + qizz * guz[19]
                 + 2 * (qixy * guz[14] + qixz * guz[15] + qiyz * guz[18]))
                + qixx * (qkxx * gqxx[12] + qkyy * gqyy[12] + qkzz * gqzz[12]
                 + 2 * (qkxy * gqxy[12] + qkxz * gqxz[12] + qkyz * gqyz[12]))
                + qiyy * (qkxx * gqxx[17] + qkyy * gqyy[17] + qkzz * gqzz[17]
                 + 2 * (qkxy * gqxy[17] + qkxz * gqxz[17] + qkyz * gqyz[17]))
                + qizz * (qkxx * gqxx[19] + qkyy * gqyy[19] + qkzz * gqzz[19]
                 + 2 * (qkxy * gqxy[19] + qkxz * gqxz[19] + qkyz * gqyz[19]))
         + 2 * (qixy * (qkxx * gqxx[14] + qkyy * gqyy[14] + qkzz * gqzz[14]
                 + 2 * (qkxy * gqxy[14] + qkxz * gqxz[14] + qkyz * gqyz[14]))
                + qixz * (qkxx * gqxx[15] + qkyy * gqyy[15] + qkzz * gqzz[15]
                 + 2 * (qkxy * gqxy[15] + qkxz * gqxz[15] + qkyz * gqyz[15]))
                + qiyz * (qkxx * gqxx[18] + qkyy * gqyy[18] + qkzz * gqzz[18]
                 + 2 * (qkxy * gqxy[18] + qkxz * gqxz[18] + qkyz * gqyz[18])));
                 
      dedz = desymdz + 0.5f*(dewidz+dewkdz);

      real desymdr = ci * ck * gc[20]
                   -(dix * (dkx * gux[21] + dky * guy[21] + dkz * guz[21])
                   + diy * (dkx * gux[22] + dky * guy[22] + dkz * guz[22])
                   + diz * (dkx * gux[23] + dky * guy[23] + dkz * guz[23]));

      real dewidr = ci * (dkx * gc[21] + dky * gc[22] + dkz * gc[23])
                  - ck * (dix * gux[20] + diy * guy[20] + diz * guz[20])
                  + ci * (qkxx * gc[24] + qkyy * gc[27] + qkzz * gc[29]
                 + 2 * (qkxy * gc[25] + qkxz * gc[26] + qkyz * gc[28]))
                  + ck * (qixx * gqxx[20] + qiyy * gqyy[20] + qizz * gqzz[20]
                 + 2 * (qixy * gqxy[20] + qixz * gqxz[20] + qiyz * gqyz[20]))
                 - dix * (qkxx * gux[24] + qkyy * gux[27] + qkzz * gux[29]
                 + 2 * (qkxy * gux[25] + qkxz * gux[26] + qkyz * gux[28]))
                 - diy * (qkxx * guy[24] + qkyy * guy[27] + qkzz * guy[29]
                 + 2 * (qkxy * guy[25] + qkxz * guy[26] + qkyz * guy[28]))
                 - diz * (qkxx * guz[24] + qkyy * guz[27] + qkzz * guz[29]
                 + 2 * (qkxy * guz[25] + qkxz * guz[26] + qkyz * guz[28]))
                 + dkx * (qixx * gqxx[21] + qiyy * gqyy[21] + qizz * gqzz[21]
                 + 2 * (qixy * gqxy[21] + qixz * gqxz[21] + qiyz * gqyz[21]))
                 + dky * (qixx * gqxx[22] + qiyy * gqyy[22] + qizz * gqzz[22]
                 + 2 * (qixy * gqxy[22] + qixz * gqxz[22] + qiyz * gqyz[22]))
                 + dkz * (qixx * gqxx[23] + qiyy * gqyy[23] + qizz * gqzz[23]
                 + 2 * (qixy * gqxy[23] + qixz * gqxz[23] + qiyz * gqyz[23]))
                + qixx * (qkxx * gqxx[24] + qkyy * gqxx[27] + qkzz * gqxx[29]
                 + 2 * (qkxy * gqxx[25] + qkxz * gqxx[26] + qkyz * gqxx[28]))
                + qiyy * (qkxx * gqyy[24] + qkyy * gqyy[27] + qkzz * gqyy[29]
                 + 2 * (qkxy * gqyy[25] + qkxz * gqyy[26] + qkyz * gqyy[28]))
                + qizz * (qkxx * gqzz[24] + qkyy * gqzz[27] + qkzz * gqzz[29]
                 + 2 * (qkxy * gqzz[25] + qkxz * gqzz[26] + qkyz * gqzz[28]))
         + 2 * (qixy * (qkxx * gqxy[24] + qkyy * gqxy[27] + qkzz * gqxy[29]
                 + 2 * (qkxy * gqxy[25] + qkxz * gqxy[26] + qkyz * gqxy[28]))
                + qixz * (qkxx * gqxz[24] + qkyy * gqxz[27] + qkzz * gqxz[29]
                 + 2 * (qkxy * gqxz[25] + qkxz * gqxz[26] + qkyz * gqxz[28]))
                + qiyz * (qkxx * gqyz[24] + qkyy * gqyz[27] + qkzz * gqyz[29]
                 + 2 * (qkxy * gqyz[25] + qkxz * gqyz[26] + qkyz * gqyz[28])));

      real dewkdr = ci * (dkx * gux[20] + dky * guy[20] + dkz * guz[20])
                  - ck * (dix * gc[21] + diy * gc[22] + diz * gc[23])
                  + ci * (qkxx * gqxx[20] + qkyy * gqyy[20] + qkzz * gqzz[20]
                 + 2 * (qkxy * gqxy[20] + qkxz * gqxz[20] + qkyz * gqyz[20]))
                  + ck * (qixx * gc[24] + qiyy * gc[27] + qizz * gc[29]
                 + 2 * (qixy * gc[25] + qixz * gc[26] + qiyz * gc[28]))
                 - dix * (qkxx * gqxx[21] + qkyy * gqyy[21] + qkzz * gqzz[21]
                 + 2 * (qkxy * gqxy[21] + qkxz * gqxz[21] + qkyz * gqyz[21]))
                 - diy * (qkxx * gqxx[22] + qkyy * gqyy[22] + qkzz * gqzz[22]
                 + 2 * (qkxy * gqxy[22] + qkxz * gqxz[22] + qkyz * gqyz[22]))
                 - diz * (qkxx * gqxx[23] + qkyy * gqyy[23] + qkzz * gqzz[23]
                 + 2 * (qkxy * gqxy[23] + qkxz * gqxz[23] + qkyz * gqyz[23]))
                 + dkx * (qixx * gux[24] + qiyy * gux[27] + qizz * gux[29]
                 + 2 * (qixy * gux[25] + qixz * gux[26] + qiyz * gux[28]))
                 + dky * (qixx * guy[24] + qiyy * guy[27] + qizz * guy[29]
                 + 2 * (qixy * guy[25] + qixz * guy[26] + qiyz * guy[28]))
                 + dkz * (qixx * guz[24] + qiyy * guz[27] + qizz * guz[29]
                 + 2 * (qixy * guz[25] + qixz * guz[26] + qiyz * guz[28]))
                + qixx * (qkxx * gqxx[24] + qkyy * gqyy[24] + qkzz * gqzz[24]
                 + 2 * (qkxy * gqxy[24] + qkxz * gqxz[24] + qkyz * gqyz[24]))
                + qiyy * (qkxx * gqxx[27] + qkyy * gqyy[27] + qkzz * gqzz[27]
                 + 2 * (qkxy * gqxy[27] + qkxz * gqxz[27] + qkyz * gqyz[27]))
                + qizz * (qkxx * gqxx[29] + qkyy * gqyy[29] + qkzz * gqzz[29]
                 + 2 * (qkxy * gqxy[29] + qkxz * gqxz[29] + qkyz * gqyz[29]))
         + 2 * (qixy * (qkxx * gqxx[25] + qkyy * gqyy[25] + qkzz * gqzz[25]
                 + 2 * (qkxy * gqxy[25] + qkxz * gqxz[25] + qkyz * gqyz[25]))
                + qixz * (qkxx * gqxx[26] + qkyy * gqyy[26] + qkzz * gqzz[26]
                 + 2 * (qkxy * gqxy[26] + qkxz * gqxz[26] + qkyz * gqyz[26]))
                + qiyz * (qkxx * gqxx[28] + qkyy * gqyy[28] + qkzz * gqzz[28]
                 + 2 * (qkxy * gqxy[28] + qkxz * gqxz[28] + qkyz * gqyz[28])));
   
      real dsumdr = desymdr + 0.5f*(dewidr+dewkdr);
      drbi = rbk*dsumdr;
      drbk = rbi*dsumdr;

      real fid[3],fkd[3];
      real fidg[3][3],fkdg[3][3];

      fid[0] = dkx * gux[1] + dky * gux[2] + dkz * gux[3]
          + 0.5f * (ck * gux[0] + qkxx * gux[4] + qkyy * gux[7] + qkzz * gux[9]
          + 2 * (qkxy * gux[5] + qkxz * gux[6] + qkyz * gux[8])
          + ck * gc[1] + qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
          + 2 * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]));
      
      fid[1] = dkx * guy[1] + dky * guy[2] + dkz * guy[3]
          + 0.5f * (ck * guy[0] + qkxx * guy[4] + qkyy * guy[7] + qkzz * guy[9]
          + 2 * (qkxy * guy[5] + qkxz * guy[6] + qkyz * guy[8])
          + ck * gc[2] + qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
          + 2 * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]));
      
      fid[2] = dkx * guz[1] + dky * guz[2] + dkz * guz[3]
          + 0.5f * (ck * guz[0] + qkxx * guz[4] + qkyy * guz[7] + qkzz * guz[9]
          + 2 * (qkxy * guz[5] + qkxz * guz[6] + qkyz * guz[8])
          + ck * gc[3] + qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
          + 2 * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]));

      fkd[0] = dix * gux[1] + diy * gux[2] + diz * gux[3]
          - 0.5f * (ci * gux[0] + qixx * gux[4] + qiyy * gux[7] + qizz * gux[9]
          + 2 * (qixy * gux[5] + qixz * gux[6] + qiyz * gux[8])
          + ci * gc[1] + qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
          + 2 * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]));
      
      fkd[1] = dix * guy[1] + diy * guy[2] + diz * guy[3]
          - 0.5f * (ci * guy[0] + qixx * guy[4] + qiyy * guy[7] + qizz * guy[9]
          + 2 * (qixy * guy[5] + qixz * guy[6] + qiyz * guy[8])
          + ci * gc[2] + qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
          + 2 * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]));
      
      fkd[2] = dix * guz[1] + diy * guz[2] + diz * guz[3]
          - 0.5f * (ci * guz[0] + qixx * guz[4] + qiyy * guz[7] + qizz * guz[9]
          + 2 * (qixy * guz[5] + qixz * guz[6] + qiyz * guz[8])
          + ci * gc[3] + qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
          + 2 * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]));

      pgrad.ttqi[0] = diy*fid[2] - diz*fid[1];
      pgrad.ttqi[1] = diz*fid[0] - dix*fid[2];
      pgrad.ttqi[2] = dix*fid[1] - diy*fid[0];
      pgrad.ttqk[0] = dky*fkd[2] - dkz*fkd[1];
      pgrad.ttqk[1] = dkz*fkd[0] - dkx*fkd[2];
      pgrad.ttqk[2] = dkx*fkd[1] - dky*fkd[0];

      fidg[0][0] = -0.5f * (ck * gqxx[0] + dkx * gqxx[1] + dky * gqxx[2] + dkz * gqxx[3]
          + qkxx * gqxx[4] + qkyy * gqxx[7] + qkzz * gqxx[9]
          + 2 * (qkxy * gqxx[5] + qkxz * gqxx[6] + qkyz * gqxx[8])
          + ck * gc[4] + dkx * gux[4] + dky * guy[4] + dkz * guz[4]
          + qkxx * gqxx[4] + qkyy * gqyy[4] + qkzz * gqzz[4]
          + 2 * (qkxy * gqxy[4] + qkxz * gqxz[4] + qkyz * gqyz[4]));
      
      fidg[0][1] = -0.5f * (ck * gqxy[0] + dkx * gqxy[1] + dky * gqxy[2] + dkz * gqxy[3]
          + qkxx * gqxy[4] + qkyy * gqxy[7] + qkzz * gqxy[9]
          + 2 * (qkxy * gqxy[5] + qkxz * gqxy[6] + qkyz * gqxy[8])
          + ck * gc[5] + dkx * gux[5] + dky * guy[5] + dkz * guz[5]
          + qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
          + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]));
      
      fidg[0][2] = -0.5f * (ck * gqxz[0] + dkx * gqxz[1] + dky * gqxz[2] + dkz * gqxz[3]
          + qkxx * gqxz[4] + qkyy * gqxz[7] + qkzz * gqxz[9]
          + 2 * (qkxy * gqxz[5] + qkxz * gqxz[6] + qkyz * gqxz[8])
          + ck * gc[6] + dkx * gux[6] + dky * guy[6] + dkz * guz[6]
          + qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
          + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]));

      fidg[1][1] = -0.5f * (ck * gqyy[0] + dkx * gqyy[1] + dky * gqyy[2] + dkz * gqyy[3]
          + qkxx * gqyy[4] + qkyy * gqyy[7] + qkzz * gqyy[9]
          + 2 * (qkxy * gqyy[5] + qkxz * gqyy[6] + qkyz * gqyy[8])
          + ck * gc[7] + dkx * gux[7] + dky * guy[7] + dkz * guz[7]
          + qkxx * gqxx[7] + qkyy * gqyy[7] + qkzz * gqzz[7]
          + 2 * (qkxy * gqxy[7] + qkxz * gqxz[7] + qkyz * gqyz[7]));
      
      fidg[1][2] = -0.5f * (ck * gqyz[0] + dkx * gqyz[1] + dky * gqyz[2] + dkz * gqyz[3]
          + qkxx * gqyz[4] + qkyy * gqyz[7] + qkzz * gqyz[9]
          + 2 * (qkxy * gqyz[5] + qkxz * gqyz[6] + qkyz * gqyz[8])
          + ck * gc[8] + dkx * gux[8] + dky * guy[8] + dkz * guz[8]
          + qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
          + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8]));
      
      fidg[2][2] = -0.5f * (ck * gqzz[0] + dkx * gqzz[1] + dky * gqzz[2] + dkz * gqzz[3]
          + qkxx * gqzz[4] + qkyy * gqzz[7] + qkzz * gqzz[9]
          + 2 * (qkxy * gqzz[5] + qkxz * gqzz[6] + qkyz * gqzz[8])
          + ck * gc[9] + dkx * gux[9] + dky * guy[9] + dkz * guz[9]
          + qkxx * gqxx[9] + qkyy * gqyy[9] + qkzz * gqzz[9]
          + 2 * (qkxy * gqxy[9] + qkxz * gqxz[9] + qkyz * gqyz[9]));

      fidg[1][0] = fidg[0][1];
      fidg[2][0] = fidg[0][2];
      fidg[2][1] = fidg[1][2];

      fkdg[0][0] = -0.5f * (ci * gqxx[0] - dix * gqxx[1] - diy * gqxx[2] - diz * gqxx[3]
          + qixx * gqxx[4] + qiyy * gqxx[7] + qizz * gqxx[9]
          + 2 * (qixy * gqxx[5] + qixz * gqxx[6] + qiyz * gqxx[8])
          + ci * gc[4] - dix * gux[4] - diy * guy[4] - diz * guz[4]
          + qixx * gqxx[4] + qiyy * gqyy[4] + qizz * gqzz[4]
          + 2 * (qixy * gqxy[4] + qixz * gqxz[4] + qiyz * gqyz[4]));
      
      fkdg[0][1] = -0.5f * (ci * gqxy[0] - dix * gqxy[1] - diy * gqxy[2] - diz * gqxy[3]
          + qixx * gqxy[4] + qiyy * gqxy[7] + qizz * gqxy[9]
          + 2 * (qixy * gqxy[5] + qixz * gqxy[6] + qiyz * gqxy[8])
          + ci * gc[5] - dix * gux[5] - diy * guy[5] - diz * guz[5]
          + qixx * gqxx[5] + qiyy * gqyy[5] + qizz * gqzz[5]
          + 2 * (qixy * gqxy[5] + qixz * gqxz[5] + qiyz * gqyz[5]));
      
      fkdg[0][2] = -0.5f * (ci * gqxz[0] - dix * gqxz[1] - diy * gqxz[2] - diz * gqxz[3]
          + qixx * gqxz[4] + qiyy * gqxz[7] + qizz * gqxz[9]
          + 2 * (qixy * gqxz[5] + qixz * gqxz[6] + qiyz * gqxz[8])
          + ci * gc[6] - dix * gux[6] - diy * guy[6] - diz * guz[6]
          + qixx * gqxx[6] + qiyy * gqyy[6] + qizz * gqzz[6]
          + 2 * (qixy * gqxy[6] + qixz * gqxz[6] + qiyz * gqyz[6]));

      fkdg[1][1] = -0.5f * (ci * gqyy[0] - dix * gqyy[1] - diy * gqyy[2] - diz * gqyy[3]
          + qixx * gqyy[4] + qiyy * gqyy[7] + qizz * gqyy[9]
          + 2 * (qixy * gqyy[5] + qixz * gqyy[6] + qiyz * gqyy[8])
          + ci * gc[7] - dix * gux[7] - diy * guy[7] - diz * guz[7]
          + qixx * gqxx[7] + qiyy * gqyy[7] + qizz * gqzz[7]
          + 2 * (qixy * gqxy[7] + qixz * gqxz[7] + qiyz * gqyz[7]));
      
      fkdg[1][2] = -0.5f * (ci * gqyz[0] - dix * gqyz[1] - diy * gqyz[2] - diz * gqyz[3]
          + qixx * gqyz[4] + qiyy * gqyz[7] + qizz * gqyz[9]
          + 2 * (qixy * gqyz[5] + qixz * gqyz[6] + qiyz * gqyz[8])
          + ci * gc[8] - dix * gux[8] - diy * guy[8] - diz * guz[8]
          + qixx * gqxx[8] + qiyy * gqyy[8] + qizz * gqzz[8]
          + 2 * (qixy * gqxy[8] + qixz * gqxz[8] + qiyz * gqyz[8]));
      
      fkdg[2][2] = -0.5f * (ci * gqzz[0] - dix * gqzz[1] - diy * gqzz[2] - diz * gqzz[3]
          + qixx * gqzz[4] + qiyy * gqzz[7] + qizz * gqzz[9]
          + 2 * (qixy * gqzz[5] + qixz * gqzz[6] + qiyz * gqzz[8])
          + ci * gc[9] - dix * gux[9] - diy * guy[9] - diz * guz[9]
          + qixx * gqxx[9] + qiyy * gqyy[9] + qizz * gqzz[9]
          + 2 * (qixy * gqxy[9] + qixz * gqxz[9] + qiyz * gqyz[9]));

      fkdg[1][0] = fkdg[0][1];
      fkdg[2][0] = fkdg[0][2];
      fkdg[2][1] = fkdg[1][2];

      pgrad.ttqi[0] += 2 * (qixy * fidg[0][2] + qiyy * fidg[1][2] + qiyz * fidg[2][2]
                            - qixz * fidg[0][1] - qiyz * fidg[1][1] - qizz * fidg[2][1]);
      pgrad.ttqi[1] += 2 * (qixz * fidg[0][0] + qiyz * fidg[1][0] + qizz * fidg[2][0]
                            - qixx * fidg[0][2] - qixy * fidg[1][2] - qixz * fidg[2][2]);
      pgrad.ttqi[2] += 2 * (qixx * fidg[0][1] + qixy * fidg[1][1] + qixz * fidg[2][1]
                            - qixy * fidg[0][0] - qiyy * fidg[1][0] - qiyz * fidg[2][0]);
      pgrad.ttqk[0] += 2 * (qkxy * fkdg[0][2] + qkyy * fkdg[1][2] + qkyz * fkdg[2][2]
                            - qkxz * fkdg[0][1] - qkyz * fkdg[1][1] - qkzz * fkdg[2][1]);
      pgrad.ttqk[1] += 2 * (qkxz * fkdg[0][0] + qkyz * fkdg[1][0] + qkzz * fkdg[2][0]
                            - qkxx * fkdg[0][2] - qkxy * fkdg[1][2] - qkxz * fkdg[2][2]);
      pgrad.ttqk[2] += 2 * (qkxx * fkdg[0][1] + qkxy * fkdg[1][1] + qkxz * fkdg[2][1]
                            - qkxy * fkdg[0][0] - qkyy * fkdg[1][0] - qkyz * fkdg[2][0]);

      real dpsymdx = -dix * (ukx * gux[4] + uky * guy[4] + ukz * guz[4])
                    - diy * (ukx * gux[5] + uky * guy[5] + ukz * guz[5])
                    - diz * (ukx * gux[6] + uky * guy[6] + ukz * guz[6])
                    - dkx * (uix * gux[4] + uiy * guy[4] + uiz * guz[4])
                    - dky * (uix * gux[5] + uiy * guy[5] + uiz * guz[5])
                    - dkz * (uix * gux[6] + uiy * guy[6] + uiz * guz[6]);

      real dpwidx = ci * (ukx * gc[4] + uky * gc[5] + ukz * gc[6])
                  - ck * (uix * gux[1] + uiy * guy[1] + uiz * guz[1])
                  - uix * (qkxx * gux[10] + qkyy * gux[13] + qkzz * gux[15]
                  + 2 * (qkxy * gux[11] + qkxz * gux[12] + qkyz * gux[14]))
                  - uiy * (qkxx * guy[10] + qkyy * guy[13] + qkzz * guy[15]
                  + 2 * (qkxy * guy[11] + qkxz * guy[12] + qkyz * guy[14]))
                  - uiz * (qkxx * guz[10] + qkyy * guz[13] + qkzz * guz[15]
                  + 2 * (qkxy * guz[11] + qkxz * guz[12] + qkyz * guz[14]))
                  + ukx * (qixx * gqxx[4] + qiyy * gqyy[4] + qizz * gqzz[4]
                  + 2 * (qixy * gqxy[4] + qixz * gqxz[4] + qiyz * gqyz[4]))
                  + uky * (qixx * gqxx[5] + qiyy * gqyy[5] + qizz * gqzz[5]
                  + 2 * (qixy * gqxy[5] + qixz * gqxz[5] + qiyz * gqyz[5]))
                  + ukz * (qixx * gqxx[6] + qiyy * gqyy[6] + qizz * gqzz[6]
                  + 2 * (qixy * gqxy[6] + qixz * gqxz[6] + qiyz * gqyz[6]));

      real dpwkdx = ci * (ukx * gux[1] + uky * guy[1] + ukz * guz[1])
                  - ck * (uix * gc[4] + uiy * gc[5] + uiz * gc[6])
                  - uix * (qkxx * gqxx[4] + qkyy * gqyy[4] + qkzz * gqzz[4]
                  + 2 * (qkxy * gqxy[4] + qkxz * gqxz[4] + qkyz * gqyz[4]))
                  - uiy * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                  + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
                  - uiz * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                  + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
                  + ukx * (qixx * gux[10] + qiyy * gux[13] + qizz * gux[15]
                  + 2 * (qixy * gux[11] + qixz * gux[12] + qiyz * gux[14]))
                  + uky * (qixx * guy[10] + qiyy * guy[13] + qizz * guy[15]
                  + 2 * (qixy * guy[11] + qixz * guy[12] + qiyz * guy[14]))
                  + ukz * (qixx * guz[10] + qiyy * guz[13] + qizz * guz[15]
                  + 2 * (qixy * guz[11] + qixz * guz[12] + qiyz * guz[14]));

      real dpdx = 0.5f * (dpsymdx + 0.5f*(dpwidx + dpwkdx));

      real dpsymdy = -dix * (ukx * gux[5] + uky * guy[5] + ukz * guz[5])
                    - diy * (ukx * gux[7] + uky * guy[7] + ukz * guz[7])
                    - diz * (ukx * gux[8] + uky * guy[8] + ukz * guz[8])
                    - dkx * (uix * gux[5] + uiy * guy[5] + uiz * guz[5])
                    - dky * (uix * gux[7] + uiy * guy[7] + uiz * guz[7])
                    - dkz * (uix * gux[8] + uiy * guy[8] + uiz * guz[8]);

      real dpwidy = ci * (ukx * gc[5] + uky * gc[7] + ukz * gc[8])
                  - ck * (uix * gux[2] + uiy * guy[2] + uiz * guz[2])
                  - uix * (qkxx * gux[11] + qkyy * gux[16] + qkzz * gux[18]
                  + 2 * (qkxy * gux[13] + qkxz * gux[14] + qkyz * gux[17]))
                  - uiy * (qkxx * guy[11] + qkyy * guy[16] + qkzz * guy[18]
                  + 2 * (qkxy * guy[13] + qkxz * guy[14] + qkyz * guy[17]))
                  - uiz * (qkxx * guz[11] + qkyy * guz[16] + qkzz * guz[18]
                  + 2 * (qkxy * guz[13] + qkxz * guz[14] + qkyz * guz[17]))
                  + ukx * (qixx * gqxx[5] + qiyy * gqyy[5] + qizz * gqzz[5]
                  + 2 * (qixy * gqxy[5] + qixz * gqxz[5] + qiyz * gqyz[5]))
                  + uky * (qixx * gqxx[7] + qiyy * gqyy[7] + qizz * gqzz[7]
                  + 2 * (qixy * gqxy[7] + qixz * gqxz[7] + qiyz * gqyz[7]))
                  + ukz * (qixx * gqxx[8] + qiyy * gqyy[8] + qizz * gqzz[8]
                  + 2 * (qixy * gqxy[8] + qixz * gqxz[8] + qiyz * gqyz[8]));

      real dpwkdy = ci * (ukx * gux[2] + uky * guy[2] + ukz * guz[2])
                  - ck * (uix * gc[5] + uiy * gc[7] + uiz * gc[8])
                  - uix * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                  + 2 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
                  - uiy * (qkxx * gqxx[7] + qkyy * gqyy[7] + qkzz * gqzz[7]
                  + 2 * (qkxy * gqxy[7] + qkxz * gqxz[7] + qkyz * gqyz[7]))
                  - uiz * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                  + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8]))
                  + ukx * (qixx * gux[11] + qiyy * gux[16] + qizz * gux[18]
                  + 2 * (qixy * gux[13] + qixz * gux[14] + qiyz * gux[17]))
                  + uky * (qixx * guy[11] + qiyy * guy[16] + qizz * guy[18]
                  + 2 * (qixy * guy[13] + qixz * guy[14] + qiyz * guy[17]))
                  + ukz * (qixx * guz[11] + qiyy * guz[16] + qizz * guz[18]
                  + 2 * (qixy * guz[13] + qixz * guz[14] + qiyz * guz[17]));

      real dpdy = 0.5f * (dpsymdy + 0.5f*(dpwidy + dpwkdy));

      real dpsymdz = -dix * (ukx * gux[6] + uky * guy[6] + ukz * guz[6])
                    - diy * (ukx * gux[8] + uky * guy[8] + ukz * guz[8])
                    - diz * (ukx * gux[9] + uky * guy[9] + ukz * guz[9])
                    - dkx * (uix * gux[6] + uiy * guy[6] + uiz * guz[6])
                    - dky * (uix * gux[8] + uiy * guy[8] + uiz * guz[8])
                    - dkz * (uix * gux[9] + uiy * guy[9] + uiz * guz[9]);

      real dpwidz = ci * (ukx * gc[6] + uky * gc[8] + ukz * gc[9])
                  - ck * (uix * gux[3] + uiy * guy[3] + uiz * guz[3])
                  - uix * (qkxx * gux[12] + qkyy * gux[17] + qkzz * gux[19]
                  + 2 * (qkxy * gux[14] + qkxz * gux[15] + qkyz * gux[18]))
                  - uiy * (qkxx * guy[12] + qkyy * guy[17] + qkzz * guy[19]
                  + 2 * (qkxy * guy[14] + qkxz * guy[15] + qkyz * guy[18]))
                  - uiz * (qkxx * guz[12] + qkyy * guz[17] + qkzz * guz[19]
                  + 2 * (qkxy * guz[14] + qkxz * guz[15] + qkyz * guz[18]))
                  + ukx * (qixx * gqxx[6] + qiyy * gqyy[6] + qizz * gqzz[6]
                  + 2 * (qixy * gqxy[6] + qixz * gqxz[6] + qiyz * gqyz[6]))
                  + uky * (qixx * gqxx[8] + qiyy * gqyy[8] + qizz * gqzz[8]
                  + 2 * (qixy * gqxy[8] + qixz * gqxz[8] + qiyz * gqyz[8]))
                  + ukz * (qixx * gqxx[9] + qiyy * gqyy[9] + qizz * gqzz[9]
                  + 2 * (qixy * gqxy[9] + qixz * gqxz[9] + qiyz * gqyz[9]));

      real dpwkdz = ci * (ukx * gux[3] + uky * guy[3] + ukz * guz[3])
                  - ck * (uix * gc[6] + uiy * gc[8] + uiz * gc[9])
                  - uix * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                  + 2 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
                  - uiy * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                  + 2 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8]))
                  - uiz * (qkxx * gqxx[9] + qkyy * gqyy[9] + qkzz * gqzz[9]
                  + 2 * (qkxy * gqxy[9] + qkxz * gqxz[9] + qkyz * gqyz[9]))
                  + ukx * (qixx * gux[12] + qiyy * gux[17] + qizz * gux[19]
                  + 2 * (qixy * gux[14] + qixz * gux[15] + qiyz * gux[18]))
                  + uky * (qixx * guy[12] + qiyy * guy[17] + qizz * guy[19]
                  + 2 * (qixy * guy[14] + qixz * guy[15] + qiyz * guy[18]))
                  + ukz * (qixx * guz[12] + qiyy * guz[17] + qizz * guz[19]
                  + 2 * (qixy * guz[14] + qixz * guz[15] + qiyz * guz[18]));

      real dpdz = 0.5f * (dpsymdz + 0.5f*(dpwidz+dpwkdz));

      real dsymdr = -dix * (ukx * gux[21] + uky * guy[21] + ukz * guz[21])
                   - diy * (ukx * gux[22] + uky * guy[22] + ukz * guz[22])
                   - diz * (ukx * gux[23] + uky * guy[23] + ukz * guz[23])
                   - dkx * (uix * gux[21] + uiy * guy[21] + uiz * guz[21])
                   - dky * (uix * gux[22] + uiy * guy[22] + uiz * guz[22])
                   - dkz * (uix * gux[23] + uiy * guy[23] + uiz * guz[23]);

      real dwipdr = ci * (ukx * gc[21] + uky * gc[22] + ukz * gc[23])
                  - ck * (uix * gux[20] + uiy * guy[20] + uiz * guz[20])
                 - uix * (qkxx * gux[24] + qkyy * gux[27] + qkzz * gux[29]
                 + 2 * (qkxy * gux[25] + qkxz * gux[26] + qkyz * gux[28]))
                 - uiy * (qkxx * guy[24] + qkyy * guy[27] + qkzz * guy[29]
                 + 2 * (qkxy * guy[25] + qkxz * guy[26] + qkyz * guy[28]))
                 - uiz * (qkxx * guz[24] + qkyy * guz[27] + qkzz * guz[29]
                 + 2 * (qkxy * guz[25] + qkxz * guz[26] + qkyz * guz[28]))
                 + ukx * (qixx * gqxx[21] + qiyy * gqyy[21] + qizz * gqzz[21]
                 + 2 * (qixy * gqxy[21] + qixz * gqxz[21] + qiyz * gqyz[21]))
                 + uky * (qixx * gqxx[22] + qiyy * gqyy[22] + qizz * gqzz[22]
                 + 2 * (qixy * gqxy[22] + qixz * gqxz[22] + qiyz * gqyz[22]))
                 + ukz * (qixx * gqxx[23] + qiyy * gqyy[23] + qizz * gqzz[23]
                 + 2 * (qixy * gqxy[23] + qixz * gqxz[23] + qiyz * gqyz[23]));

      real dwkpdr = ci * (ukx * gux[20] + uky * guy[20] + ukz * guz[20])
                  - ck * (uix * gc[21] + uiy * gc[22] + uiz * gc[23])
                 - uix * (qkxx * gqxx[21] + qkyy * gqyy[21] + qkzz * gqzz[21]
                 + 2 * (qkxy * gqxy[21] + qkxz * gqxz[21] + qkyz * gqyz[21]))
                 - uiy * (qkxx * gqxx[22] + qkyy * gqyy[22] + qkzz * gqzz[22]
                 + 2 * (qkxy * gqxy[22] + qkxz * gqxz[22] + qkyz * gqyz[22]))
                 - uiz * (qkxx * gqxx[23] + qkyy * gqyy[23] + qkzz * gqzz[23]
                 + 2 * (qkxy * gqxy[23] + qkxz * gqxz[23] + qkyz * gqyz[23]))
                 + ukx * (qixx * gux[24] + qiyy * gux[27] + qizz * gux[29]
                 + 2 * (qixy * gux[25] + qixz * gux[26] + qiyz * gux[28]))
                 + uky * (qixx * guy[24] + qiyy * guy[27] + qizz * guy[29]
                 + 2 * (qixy * guy[25] + qixz * guy[26] + qiyz * guy[28]))
                 + ukz * (qixx * guz[24] + qiyy * guz[27] + qizz * guz[29]
                 + 2 * (qixy * guz[25] + qixz * guz[26] + qiyz * guz[28]));

      dsumdr = dsymdr + 0.5f*(dwipdr+dwkpdr);
      dpbi = 0.5f*rbk*dsumdr;
      dpbk = 0.5f*rbi*dsumdr;

      dpdx = dpdx - 0.5f * (
             uidx * (ukpx * gux[4] + ukpy * gux[5] + ukpz * gux[6])
           + uidy * (ukpx * guy[4] + ukpy * guy[5] + ukpz * guy[6])
           + uidz * (ukpx * guz[4] + ukpy * guz[5] + ukpz * guz[6])
           + ukdx * (uipx * gux[4] + uipy * gux[5] + uipz * gux[6])
           + ukdy * (uipx * guy[4] + uipy * guy[5] + uipz * guy[6])
           + ukdz * (uipx * guz[4] + uipy * guz[5] + uipz * guz[6]));

      dpdy = dpdy - 0.5f * (
             uidx * (ukpx * gux[5] + ukpy * gux[7] + ukpz * gux[8])
           + uidy * (ukpx * guy[5] + ukpy * guy[7] + ukpz * guy[8])
           + uidz * (ukpx * guz[5] + ukpy * guz[7] + ukpz * guz[8])
           + ukdx * (uipx * gux[5] + uipy * gux[7] + uipz * gux[8])
           + ukdy * (uipx * guy[5] + uipy * guy[7] + uipz * guy[8])
           + ukdz * (uipx * guz[5] + uipy * guz[7] + uipz * guz[8]));

      dpdz = dpdz - 0.5f * (
             uidx * (ukpx * gux[6] + ukpy * gux[8] + ukpz * gux[9])
           + uidy * (ukpx * guy[6] + ukpy * guy[8] + ukpz * guy[9])
           + uidz * (ukpx * guz[6] + ukpy * guz[8] + ukpz * guz[9])
           + ukdx * (uipx * gux[6] + uipy * gux[8] + uipz * gux[9])
           + ukdy * (uipx * guy[6] + uipy * guy[8] + uipz * guy[9])
           + ukdz * (uipx * guz[6] + uipy * guz[8] + uipz * guz[9]));

      real duvdr = uidx * (ukpx * gux[21] + ukpy * gux[22] + ukpz * gux[23])
                 + uidy * (ukpx * guy[21] + ukpy * guy[22] + ukpz * guy[23])
                 + uidz * (ukpx * guz[21] + ukpy * guz[22] + ukpz * guz[23])
                 + ukdx * (uipx * gux[21] + uipy * gux[22] + uipz * gux[23])
                 + ukdy * (uipx * guy[21] + uipy * guy[22] + uipz * guy[23])
                 + ukdz * (uipx * guz[21] + uipy * guz[22] + uipz * guz[23]);

      dpbi = dpbi - 0.5f*rbk*duvdr;
      dpbk = dpbk - 0.5f*rbi*duvdr;

      fid[0] = 0.5f * (ukx * gux[1] + uky * guy[1] + ukz * guz[1]);
      fid[1] = 0.5f * (ukx * gux[2] + uky * guy[2] + ukz * guz[2]);
      fid[2] = 0.5f * (ukx * gux[3] + uky * guy[3] + ukz * guz[3]);
      fkd[0] = 0.5f * (uix * gux[1] + uiy * guy[1] + uiz * guz[1]);
      fkd[1] = 0.5f * (uix * gux[2] + uiy * guy[2] + uiz * guz[2]);
      fkd[2] = 0.5f * (uix * gux[3] + uiy * guy[3] + uiz * guz[3]);

      pgrad.ttqi[0] += diy * fid[2] - diz * fid[1];
      pgrad.ttqi[1] += diz * fid[0] - dix * fid[2];
      pgrad.ttqi[2] += dix * fid[1] - diy * fid[0];
      pgrad.ttqk[0] += dky * fkd[2] - dkz * fkd[1];
      pgrad.ttqk[1] += dkz * fkd[0] - dkx * fkd[2];
      pgrad.ttqk[2] += dkx * fkd[1] - dky * fkd[0];

      fidg[0][0] = -0.25f * ((ukx * gqxx[1] + uky * gqxx[2] + ukz * gqxx[3])
                          + (ukx * gux[4] + uky * guy[4] + ukz * guz[4]));
      fidg[0][1] = -0.25f * ((ukx * gqxy[1] + uky * gqxy[2] + ukz * gqxy[3])
                          + (ukx * gux[5] + uky * guy[5] + ukz * guz[5]));
      fidg[0][2] = -0.25f * ((ukx * gqxz[1] + uky * gqxz[2] + ukz * gqxz[3])
                          + (ukx * gux[6] + uky * guy[6] + ukz * guz[6]));
      fidg[1][1] = -0.25f * ((ukx * gqyy[1] + uky * gqyy[2] + ukz * gqyy[3])
                          + (ukx * gux[7] + uky * guy[7] + ukz * guz[7]));
      fidg[1][2] = -0.25f * ((ukx * gqyz[1] + uky * gqyz[2] + ukz * gqyz[3])
                          + (ukx * gux[8] + uky * guy[8] + ukz * guz[8]));
      fidg[2][2] = -0.25f * ((ukx * gqzz[1] + uky * gqzz[2] + ukz * gqzz[3])
                          + (ukx * gux[9] + uky * guy[9] + ukz * guz[9]));
      fidg[1][0] = fidg[0][1];
      fidg[2][0] = fidg[0][2];
      fidg[2][1] = fidg[1][2];

      fkdg[0][0] = 0.25f * ((uix * gqxx[1] + uiy * gqxx[2] + uiz * gqxx[3])
                         + (uix * gux[4] + uiy * guy[4] + uiz * guz[4]));
      fkdg[0][1] = 0.25f * ((uix * gqxy[1] + uiy * gqxy[2] + uiz * gqxy[3])
                         + (uix * gux[5] + uiy * guy[5] + uiz * guz[5]));
      fkdg[0][2] = 0.25f * ((uix * gqxz[1] + uiy * gqxz[2] + uiz * gqxz[3])
                         + (uix * gux[6] + uiy * guy[6] + uiz * guz[6]));
      fkdg[1][1] = 0.25f * ((uix * gqyy[1] + uiy * gqyy[2] + uiz * gqyy[3])
                         + (uix * gux[7] + uiy * guy[7] + uiz * guz[7]));
      fkdg[1][2] = 0.25f * ((uix * gqyz[1] + uiy * gqyz[2] + uiz * gqyz[3])
                         + (uix * gux[8] + uiy * guy[8] + uiz * guz[8]));
      fkdg[2][2] = 0.25f * ((uix * gqzz[1] + uiy * gqzz[2] + uiz * gqzz[3])
                         + (uix * gux[9] + uiy * guy[9] + uiz * guz[9]));
      fkdg[1][0] = fkdg[0][1];
      fkdg[2][0] = fkdg[0][2];
      fkdg[2][1] = fkdg[1][2];

      pgrad.ttqi[0] += 2 * (qixy * fidg[0][2] + qiyy * fidg[1][2] + qiyz * fidg[2][2]
                            - qixz * fidg[0][1] - qiyz * fidg[1][1] - qizz * fidg[2][1]);
      pgrad.ttqi[1] += 2 * (qixz * fidg[0][0] + qiyz * fidg[1][0] + qizz * fidg[2][0]
                            - qixx * fidg[0][2] - qixy * fidg[1][2] - qixz * fidg[2][2]);
      pgrad.ttqi[2] += 2 * (qixx * fidg[0][1] + qixy * fidg[1][1] + qixz * fidg[2][1]
                            - qixy * fidg[0][0] - qiyy * fidg[1][0] - qiyz * fidg[2][0]);
      pgrad.ttqk[0] += 2 * (qkxy * fkdg[0][2] + qkyy * fkdg[1][2] + qkyz * fkdg[2][2]
                            - qkxz * fkdg[0][1] - qkyz * fkdg[1][1] - qkzz * fkdg[2][1]);
      pgrad.ttqk[1] += 2 * (qkxz * fkdg[0][0] + qkyz * fkdg[1][0] + qkzz * fkdg[2][0]
                            - qkxx * fkdg[0][2] - qkxy * fkdg[1][2] - qkxz * fkdg[2][2]);
      pgrad.ttqk[2] += 2 * (qkxx * fkdg[0][1] + qkxy * fkdg[1][1] + qkxz * fkdg[2][1]
                            - qkxy * fkdg[0][0] - qkyy * fkdg[1][0] - qkyz * fkdg[2][0]);

      pgrad.frcx = -(dedx + dpdx);
      pgrad.frcy = -(dedy + dpdy);
      pgrad.frcz = -(dedz + dpdz);
   }
}

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
      real expc = 1 / gkc;
      real gf2 = 1 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real gf5 = gf3 * gf2;

      real expc1 = 1 - expc;
      real a00 = fc * gf;
      real a01 = -fc * expc1 * gf3;
      real a10 = -fd * gf3;
      real a20 = 3 * fq * gf5;

      real gc1 = a00;
      real gux2 = a10;
      real guy3 = a10;
      real guz4 = a10;
      real gc5 = a01;
      real gc8 = a01;
      real gc10 = a01;
      real gqxx5 = 2 * a20;
      real gqyy8 = 2 * a20;
      real gqzz10 = 2 * a20;
      real gqxy6 = a20;
      real gqxz7 = a20;
      real gqyz9 = a20;

      real esym = ci * ci * gc1 - dix * dix * gux2 - diy * diy * guy3 - diz * diz * guz4;
      real ewi = ci * (qixx * gc5 + qiyy * gc8 + qizz * gc10)
         + qixx * qixx * gqxx5 + qiyy * qiyy * gqyy8 + qizz * qizz * gqzz10
         + 4 * (qixy * qixy * gqxy6 + qixz * qixz * gqxz7 + qiyz * qiyz * gqyz9);
      real e = esym + ewi;

      real ei = -dix * uidx * gux2 - diy * uidy * guy3 - diz * uidz * guz4;

      e += ei;
      e *= 0.5f;

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
         real dgfdr =  0.5f;
         real b00 = -fc * dgfdr * gf3;
         real b10 = 3 * dgfdr * gf5;
         real b20 = -15 * fq *dgfdr * gf7;
         real b01 = b10 - expc*b10;
         b01 = fc * b01;
         b10 = fd * b10;

         real gc21 = b00;
         real gc25 = b01;
         real gc28 = b01;
         real gc30 = b01;
         real gux22 = b10;
         real guy23 = b10;
         real guz24 = b10;
         real gqxx25 = 2*b20;
         real gqxy26 = b20;
         real gqxz27 = b20;
         real gqyy28 = 2*b20;
         real gqyz29 = b20;
         real gqzz30 = 2*b20;

         real desymdr = ci*ci*gc21 - (dix*dix*gux22 + diy*diy*guy23 + diz*diz*guz24);
         real dewidr = ci*(qixx*gc25 + qiyy*gc28 + qizz*gc30)
                     + qixx*qixx*gqxx25 + qiyy*qiyy*gqyy28 + qizz*qizz*gqzz30
                     + 4*(qixy*qixy*gqxy26 + qixz*qixz*gqxz27 + qiyz*qiyz*gqyz29);
         real dsumdr = desymdr + dewidr;
         real drbi = rbi*dsumdr;

         real dsymdr = -dix*uix*gux22 - diy*uiy*guy23 - diz*uiz*guz24;
         real dpbi = rbi*dsymdr;

         real duvdr = uidx*uipx*gux22 + uidy*uipy*guy23 + uidz*uipz*guz24;
         dpbi -= rbi*duvdr;

         real fid[3];
         fid[0] = 0.5f * (uix * gux2);
         fid[1] = 0.5f * (uiy * guy3);
         fid[2] = 0.5f * (uiz * guz4);
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
}

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_ROUTINE
inline void pair_ediff(
   real r2, real xr, real yr, real zr, real dscale, real pscale, real uscale,
   real ci, real dix, real diy, real diz,
   real qixx, real qixy, real qixz, real qiyy, real qiyz, real qizz,
   real uidx, real uidy, real uidz, real uidsx, real uidsy, real uidsz,
   real uipx, real uipy, real uipz, real uipsx, real uipsy, real uipsz, 
   real pdi, real pti, //
   real ck, real dkx, real dky, real dkz,
   real qkxx, real qkxy, real qkxz, real qkyy, real qkyz, real qkzz,
   real ukdx, real ukdy, real ukdz, real ukdsx, real ukdsy, real ukdsz,
   real ukpx, real ukpy, real ukpz, real ukpsx, real ukpsy, real ukpsz,
   real pdk, real ptk, //
   real f, real& restrict e, PairSolvGrad& pgrad)
{
   constexpr bool do_g = Ver::g;

   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   MAYBE_UNUSED real rr9;
   if CONSTEXPR (do_g) rr9 = 7 * rr7 * rr2;
   real scale3 = 1;
   real scale5 = 1;
   real scale7 = 1;
   real ddsc3[3],ddsc5[3],ddsc7[3];
   for (int i = 0; i < 3; i++) {
      ddsc3[i] = 0;
      ddsc5[i] = 0;
      ddsc7[i] = 0;
   }

   real pgamma = REAL_MIN(pti, ptk);
   real damp = pdi * pdk;
   real ratio = r * REAL_RECIP(damp);
   damp = (damp == 0 ? ((real)1e16) : pgamma * ratio * ratio * ratio);
   real expdamp = REAL_EXP(-damp);
   scale3 = 1 - expdamp;
   scale5 = 1 - expdamp * (1 + damp);
   scale7 = 1 - expdamp * (1 + damp + 0.6f * damp * damp);
   ddsc3[0] = 3 * damp * expdamp * xr / r2;
   ddsc3[1] = 3 * damp * expdamp * yr / r2;
   ddsc3[2] = 3 * damp * expdamp * zr / r2;
   ddsc5[0] = damp * ddsc3[0];
   ddsc5[1] = damp * ddsc3[1];
   ddsc5[2] = damp * ddsc3[2];
   ddsc7[0] = (-0.2f + 0.6f * damp) * ddsc5[0];
   ddsc7[1] = (-0.2f + 0.6f * damp) * ddsc5[1];
   ddsc7[2] = (-0.2f + 0.6f * damp) * ddsc5[2];

   real scale3i = scale3 * uscale;
   real scale5i = scale5 * uscale;
   real dsc3 = scale3 * dscale;
   real dsc5 = scale5 * dscale;
   real dsc7 = scale7 * dscale;
   real psc3 = scale3 * pscale;
   real psc5 = scale5 * pscale;
   real psc7 = scale7 * pscale;
   real qir[3];
   qir[0] = qixx * xr + qixy * yr + qixz * zr;
   qir[1] = qixy * xr + qiyy * yr + qiyz * zr;
   qir[2] = qixz * xr + qiyz * yr + qizz * zr;
   real qkr[3];
   qkr[0] = qkxx * xr + qkxy * yr + qkxz * zr;
   qkr[1] = qkxy * xr + qkyy * yr + qkyz * zr;
   qkr[2] = qkxz * xr + qkyz * yr + qkzz * zr;
   real sc[10];
   sc[2] = dix * xr + diy * yr + diz * zr;
   sc[3] = dkx * xr + dky * yr + dkz * zr;
   sc[4] = qir[0] * xr + qir[1] * yr + qir[2] * zr;
   sc[5] = qkr[0] * xr + qkr[1] * yr + qkr[2] * zr;
   real sci[8];
   sci[0] = uidsx * dkx + uidsy * dky + uidsz * dkz + dix * ukdsx + diy * ukdsy + diz * ukdsz;
   sci[1] = uidsx * ukdsx + uidsy * ukdsy + uidsz * ukdsz;
   sci[2] = uidsx * xr + uidsy * yr + uidsz * zr;
   sci[3] = ukdsx * xr + ukdsy * yr + ukdsz * zr;
   sci[6] = qir[0] * ukdsx + qir[1] * ukdsy + qir[2] * ukdsz;
   sci[7] = qkr[0] * uidsx + qkr[1] * uidsy + qkr[2] * uidsz;
   real gli[7];
   gli[0] = ck * sci[2] - ci * sci[3];
   gli[1] = -sc[2] * sci[3] - sci[2] * sc[3];
   gli[2] = sci[2] * sc[5] - sci[3] * sc[4];
   gli[5] = sci[0];
   gli[6] = 2 * (sci[6] - sci[7]);
   real ei = 0.5f * (rr3 * (gli[0] + gli[5]) * psc3
               + rr5 * (gli[1] + gli[6]) * psc5
               + rr7 * gli[2] * psc7);
   e = f * ei;

   MAYBE_UNUSED real dixr[3], dkxr[3];
   MAYBE_UNUSED real ttm2i[3], ttm3i[3];
   MAYBE_UNUSED real rxqir[3], rxqkr[3];
   MAYBE_UNUSED real dixuk[3], dkxui[3], dixukp[3], dkxuip[3];
   MAYBE_UNUSED real qiuk[3], qkui[3], qiukp[3], qkuip[3];
   MAYBE_UNUSED real uixqkr[3], ukxqir[3], uixqkrp[3], ukxqirp[3];
   MAYBE_UNUSED real rxqiuk[3], rxqkui[3], rxqiukp[3], rxqkuip[3];
   MAYBE_UNUSED real scip[8], glip[7], gfi[6], gti[6];
   MAYBE_UNUSED real ftm2i[3], fridmp[3], findmp[3];
   if CONSTEXPR (do_g) {
      dixr[0] = diy * zr - diz * yr;
      dixr[1] = diz * xr - dix * zr;
      dixr[2] = dix * yr - diy * xr;
      dkxr[0] = dky * zr - dkz * yr;
      dkxr[1] = dkz * xr - dkx * zr;
      dkxr[2] = dkx * yr - dky * xr;
      rxqir[0] = yr * qir[2] - zr * qir[1];
      rxqir[1] = zr * qir[0] - xr * qir[2];
      rxqir[2] = xr * qir[1] - yr * qir[0];
      rxqkr[0] = yr * qkr[2] - zr * qkr[1];
      rxqkr[1] = zr * qkr[0] - xr * qkr[2];
      rxqkr[2] = xr * qkr[1] - yr * qkr[0];
      dixuk[0] = diy * ukdsz - diz * ukdsy;
      dixuk[1] = diz * ukdsx - dix * ukdsz;
      dixuk[2] = dix * ukdsy - diy * ukdsx;
      dkxui[0] = dky * uidsz - dkz * uidsy;
      dkxui[1] = dkz * uidsx - dkx * uidsz;
      dkxui[2] = dkx * uidsy - dky * uidsx;
      dixukp[0] = diy * ukpsz - diz * ukpsy;
      dixukp[1] = diz * ukpsx - dix * ukpsz;
      dixukp[2] = dix * ukpsy - diy * ukpsx;
      dkxuip[0] = dky * uipsz - dkz * uipsy;
      dkxuip[1] = dkz * uipsx - dkx * uipsz;
      dkxuip[2] = dkx * uipsy - dky * uipsx;
      
      qiuk[0] = qixx * ukdsx + qixy * ukdsy + qixz * ukdsz;
      qiuk[1] = qixy * ukdsx + qiyy * ukdsy + qiyz * ukdsz;
      qiuk[2] = qixz * ukdsx + qiyz * ukdsy + qizz * ukdsz;
      
      qkui[0] = qkxx * uidsx + qkxy * uidsy + qkxz * uidsz;
      qkui[1] = qkxy * uidsx + qkyy * uidsy + qkyz * uidsz;
      qkui[2] = qkxz * uidsx + qkyz * uidsy + qkzz * uidsz;
      
      qiukp[0] = qixx * ukpsx + qixy * ukpsy + qixz * ukpsz;
      qiukp[1] = qixy * ukpsx + qiyy * ukpsy + qiyz * ukpsz;
      qiukp[2] = qixz * ukpsx + qiyz * ukpsy + qizz * ukpsz;
      
      qkuip[0] = qkxx * uipsx + qkxy * uipsy + qkxz * uipsz;
      qkuip[1] = qkxy * uipsx + qkyy * uipsy + qkyz * uipsz;
      qkuip[2] = qkxz * uipsx + qkyz * uipsy + qkzz * uipsz;
      
      uixqkr[0] = uidsy * qkr[2] - uidsz * qkr[1];
      uixqkr[1] = uidsz * qkr[0] - uidsx * qkr[2];
      uixqkr[2] = uidsx * qkr[1] - uidsy * qkr[0];
      
      ukxqir[0] = ukdsy * qir[2] - ukdsz * qir[1];
      ukxqir[1] = ukdsz * qir[0] - ukdsx * qir[2];
      ukxqir[2] = ukdsx * qir[1] - ukdsy * qir[0];
      
      uixqkrp[0] = uipsy * qkr[2] - uipsz * qkr[1];
      uixqkrp[1] = uipsz * qkr[0] - uipsx * qkr[2];
      uixqkrp[2] = uipsx * qkr[1] - uipsy * qkr[0];
      
      ukxqirp[0] = ukpsy * qir[2] - ukpsz * qir[1];
      ukxqirp[1] = ukpsz * qir[0] - ukpsx * qir[2];
      ukxqirp[2] = ukpsx * qir[1] - ukpsy * qir[0];
      
      rxqiuk[0] = yr * qiuk[2] - zr * qiuk[1];
      rxqiuk[1] = zr * qiuk[0] - xr * qiuk[2];
      rxqiuk[2] = xr * qiuk[1] - yr * qiuk[0];
      
      rxqkui[0] = yr * qkui[2] - zr * qkui[1];
      rxqkui[1] = zr * qkui[0] - xr * qkui[2];
      rxqkui[2] = xr * qkui[1] - yr * qkui[0];
      
      rxqiukp[0] = yr * qiukp[2] - zr * qiukp[1];
      rxqiukp[1] = zr * qiukp[0] - xr * qiukp[2];
      rxqiukp[2] = xr * qiukp[1] - yr * qiukp[0];
      
      rxqkuip[0] = yr * qkuip[2] - zr * qkuip[1];
      rxqkuip[1] = zr * qkuip[0] - xr * qkuip[2];
      rxqkuip[2] = xr * qkuip[1] - yr * qkuip[0];
      
      scip[0] = uipsx * dkx + uipsy * dky + uipsz * dkz + dix * ukpsx + diy * ukpsy + diz * ukpsz;
      scip[1] = uidsx * ukpsx + uidsy * ukpsy + uidsz * ukpsz + uipsx * ukdsx + uipsy * ukdsy + uipsz * ukdsz;
      scip[2] = uipsx * xr + uipsy * yr + uipsz * zr;
      scip[3] = ukpsx * xr + ukpsy * yr + ukpsz * zr;
      scip[6] = qir[0] * ukpsx + qir[1] * ukpsy + qir[2] * ukpsz;
      scip[7] = qkr[0] * uipsx + qkr[1] * uipsy + qkr[2] * uipsz;
      
      glip[0] = ck * scip[2] - ci * scip[3];
      glip[1] = -sc[2] * scip[3] - scip[2] * sc[3];
      glip[2] = scip[2] * sc[5] - scip[3] * sc[4];
      glip[5] = scip[0];
      glip[6] = 2 * (scip[6] - scip[7]);
      
      gfi[0] = 0.5f * rr5 * ((gli[0] + gli[5]) * psc3
         + (glip[0] + glip[5]) * dsc3 + scip[1] * scale3i)
         + 0.5f * rr7 * ((gli[6] + gli[1]) * psc5
         + (glip[6] + glip[1]) * dsc5
         - (sci[2] * scip[3] + scip[2] * sci[3]) * scale5i)
         + 0.5f * rr9 * (gli[2] * psc7 + glip[2] * dsc7);
      gfi[1] = -rr3 * ck + rr5 * sc[3] - rr7 * sc[5];
      gfi[2] = rr3 * ci + rr5 * sc[2] + rr7 * sc[4];
      gfi[3] = 2 * rr5;
      gfi[4] = rr7 * (sci[3] * psc7 + scip[3] * dsc7);
      gfi[5] = -rr7 * (sci[2] * psc7 + scip[2] * dsc7);
      
      ftm2i[0] = gfi[0] * xr + 0.5f * (-rr3 * ck * (uidsx * psc3 + uipsx * dsc3)
         + rr5 * sc[3] * (uidsx * psc5 + uipsx * dsc5)
         - rr7 * sc[5] * (uidsx * psc7 + uipsx * dsc7))
         + (rr3 * ci * (ukdsx * psc3 + ukpsx * dsc3)
         + rr5 * sc[2] * (ukdsx * psc5 + ukpsx * dsc5)
         + rr7 * sc[4] * (ukdsx * psc7 + ukpsx * dsc7)) * 0.5f
         + rr5 * scale5i * (sci[3] * uipsx + scip[3] * uidsx
         + sci[2] * ukpsx + scip[2] * ukdsx) * 0.5f
         + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * dix
         + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dkx
         + 0.5f * gfi[3] * ((qkui[0] - qiuk[0]) * psc5
         + (qkuip[0] - qiukp[0]) * dsc5)
         + gfi[4] * qir[0] + gfi[5] * qkr[0];
      ftm2i[1] = gfi[0] * yr + 0.5f * (-rr3 * ck * (uidsy * psc3 + uipsy * dsc3)
         + rr5 * sc[3] * (uidsy * psc5 + uipsy * dsc5)
         - rr7 * sc[5] * (uidsy * psc7 + uipsy * dsc7))
         + (rr3 * ci * (ukdsy * psc3 + ukpsy * dsc3)
         + rr5 * sc[2] * (ukdsy * psc5 + ukpsy * dsc5)
         + rr7 * sc[4] * (ukdsy * psc7 + ukpsy * dsc7)) * 0.5f
         + rr5 * scale5i * (sci[3] * uipsy + scip[3] * uidsy
         + sci[2] * ukpsy + scip[2] * ukdsy) * 0.5f
         + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * diy
         + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dky
         + 0.5f * gfi[3] * ((qkui[1] - qiuk[1]) * psc5
         + (qkuip[1] - qiukp[1]) * dsc5)
         + gfi[4] * qir[1] + gfi[5] * qkr[1];
      ftm2i[2] = gfi[0] * zr + 0.5f * (-rr3 * ck * (uidsz * psc3 + uipsz * dsc3)
         + rr5 * sc[3] * (uidsz * psc5 + uipsz * dsc5)
         - rr7 * sc[5] * (uidsz * psc7 + uipsz * dsc7))
         + (rr3 * ci * (ukdsz * psc3 + ukpsz * dsc3)
         + rr5 * sc[2] * (ukdsz * psc5 + ukpsz * dsc5)
         + rr7 * sc[4] * (ukdsz * psc7 + ukpsz * dsc7)) * 0.5f
         + rr5 * scale5i * (sci[3] * uipsz + scip[3] * uidsz
         + sci[2] * ukpsz + scip[2] * ukdsz) * 0.5f
         + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * diz
         + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dkz
         + 0.5f * gfi[3] * ((qkui[2] - qiuk[2]) * psc5
         + (qkuip[2] - qiukp[2]) * dsc5)
         + gfi[4] * qir[2] + gfi[5] * qkr[2];
      
      fridmp[0] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
         + (glip[0] + glip[5]) * dscale) * ddsc3[0]
         + rr5 * ((gli[1] + gli[6]) * pscale
         + (glip[1] + glip[6]) * dscale) * ddsc5[0]
         + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[0]);
      fridmp[1] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
         + (glip[0] + glip[5]) * dscale) * ddsc3[1]
         + rr5 * ((gli[1] + gli[6]) * pscale
         + (glip[1] + glip[6]) * dscale) * ddsc5[1]
         + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[1]);
      fridmp[2] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
         + (glip[0] + glip[5]) * dscale) * ddsc3[2]
         + rr5 * ((gli[1] + gli[6]) * pscale
         + (glip[1] + glip[6]) * dscale) * ddsc5[2]
         + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[2]);
      
      findmp[0] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[0]
         - rr5 * ddsc5[0] * (sci[2] * scip[3] + scip[2] * sci[3]));
      findmp[1] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[1]
         - rr5 * ddsc5[1] * (sci[2] * scip[3] + scip[2] * sci[3]));
      findmp[2] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[2]
         - rr5 * ddsc5[2] * (sci[2] * scip[3] + scip[2] * sci[3]));
      ftm2i[0] -= fridmp[0] + findmp[0];
      ftm2i[1] -= fridmp[1] + findmp[1];
      ftm2i[2] -= fridmp[2] + findmp[2];
      
      gti[1] = 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5;
      gti[2] = 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5;
      gti[3] = gfi[3];
      gti[4] = gfi[4];
      gti[5] = gfi[5];
      ttm2i[0] = -rr3 * (dixuk[0] * psc3 + dixukp[0] * dsc3) * 0.5f
         + gti[1] * dixr[0] + gti[3] * ((ukxqir[0] + rxqiuk[0]) * psc5
         + (ukxqirp[0] + rxqiukp[0]) * dsc5) * 0.5f - gti[4] * rxqir[0];
      ttm2i[1] = -rr3 * (dixuk[1] * psc3 + dixukp[1] * dsc3) * 0.5f
         + gti[1] * dixr[1] + gti[3] * ((ukxqir[1] + rxqiuk[1]) * psc5
         + (ukxqirp[1] + rxqiukp[1]) * dsc5) * 0.5f - gti[4] * rxqir[1];
      ttm2i[2] = -rr3 * (dixuk[2] * psc3 + dixukp[2] * dsc3) * 0.5f
         + gti[1] * dixr[2] + gti[3] * ((ukxqir[2] + rxqiuk[2]) * psc5
         + (ukxqirp[2] + rxqiukp[2]) * dsc5) * 0.5f - gti[4] * rxqir[2];
      ttm3i[0] = -rr3 * (dkxui[0] * psc3 + dkxuip[0] * dsc3) * 0.5f
         + gti[2] * dkxr[0] - gti[3] * ((uixqkr[0] + rxqkui[0]) * psc5
         + (uixqkrp[0] + rxqkuip[0]) * dsc5) * 0.5f - gti[5] * rxqkr[0];
      ttm3i[1] = -rr3 * (dkxui[1] * psc3 + dkxuip[1] * dsc3) * 0.5f
         + gti[2] * dkxr[1] - gti[3] * ((uixqkr[1] + rxqkui[1]) * psc5
         + (uixqkrp[1] + rxqkuip[1]) * dsc5) * 0.5f - gti[5] * rxqkr[1];
      ttm3i[2] = -rr3 * (dkxui[2] * psc3 + dkxuip[2] * dsc3) * 0.5f
         + gti[2] * dkxr[2] - gti[3] * ((uixqkr[2] + rxqkui[2]) * psc5
         + (uixqkrp[2] + rxqkuip[2]) * dsc5) * 0.5f - gti[5] * rxqkr[2];
      pgrad.frcx = f * ftm2i[0];
      pgrad.frcy = f * ftm2i[1];
      pgrad.frcz = f * ftm2i[2];
      pgrad.ttqi[0] = f*ttm2i[0];
      pgrad.ttqi[1] = f*ttm2i[1];
      pgrad.ttqi[2] = f*ttm2i[2];
      pgrad.ttqk[0] = f*ttm3i[0];
      pgrad.ttqk[1] = f*ttm3i[1];
      pgrad.ttqk[2] = f*ttm3i[2];
   }

   sci[0] = uidx * dkx + uidy * dky + uidz * dkz + dix * ukdx + diy * ukdy + diz * ukdz;
   sci[1] = uidx * ukdx + uidy * ukdy + uidz * ukdz;
   sci[2] = uidx * xr + uidy * yr + uidz * zr;
   sci[3] = ukdx * xr + ukdy * yr + ukdz * zr;
   sci[6] = qir[0] * ukdx + qir[1] * ukdy + qir[2] * ukdz;
   sci[7] = qkr[0] * uidx + qkr[1] * uidy + qkr[2] * uidz;
   gli[0] = ck * sci[2] - ci * sci[3];
   gli[1] = -sc[2] * sci[3] - sci[2] * sc[3];
   gli[2] = sci[2] * sc[5] - sci[3] * sc[4];
   gli[5] = sci[0];
   gli[6] = 2 * (sci[6] - sci[7]);
   ei = -0.5f * (rr3 * (gli[0] + gli[5]) * psc3
               + rr5 * (gli[1] + gli[6]) * psc5
               + rr7 * gli[2] * psc7);
   e += f * ei;

   if CONSTEXPR (do_g) {
      dixuk[0] = diy * ukdz - diz * ukdy;
      dixuk[1] = diz * ukdx - dix * ukdz;
      dixuk[2] = dix * ukdy - diy * ukdx;
      dkxui[0] = dky * uidz - dkz * uidy;
      dkxui[1] = dkz * uidx - dkx * uidz;
      dkxui[2] = dkx * uidy - dky * uidx;
      dixukp[0] = diy * ukpz - diz * ukpy;
      dixukp[1] = diz * ukpx - dix * ukpz;
      dixukp[2] = dix * ukpy - diy * ukpx;
      dkxuip[0] = dky * uipz - dkz * uipy;
      dkxuip[1] = dkz * uipx - dkx * uipz;
      dkxuip[2] = dkx * uipy - dky * uipx;
      qiuk[0] = qixx * ukdx + qixy * ukdy + qixz * ukdz;
      qiuk[1] = qixy * ukdx + qiyy * ukdy + qiyz * ukdz;
      qiuk[2] = qixz * ukdx + qiyz * ukdy + qizz * ukdz;
      qkui[0] = qkxx * uidx + qkxy * uidy + qkxz * uidz;
      qkui[1] = qkxy * uidx + qkyy * uidy + qkyz * uidz;
      qkui[2] = qkxz * uidx + qkyz * uidy + qkzz * uidz;
      qiukp[0] = qixx * ukpx + qixy * ukpy + qixz * ukpz;
      qiukp[1] = qixy * ukpx + qiyy * ukpy + qiyz * ukpz;
      qiukp[2] = qixz * ukpx + qiyz * ukpy + qizz * ukpz;
      qkuip[0] = qkxx * uipx + qkxy * uipy + qkxz * uipz;
      qkuip[1] = qkxy * uipx + qkyy * uipy + qkyz * uipz;
      qkuip[2] = qkxz * uipx + qkyz * uipy + qkzz * uipz;
      uixqkr[0] = uidy * qkr[2] - uidz * qkr[1];
      uixqkr[1] = uidz * qkr[0] - uidx * qkr[2];
      uixqkr[2] = uidx * qkr[1] - uidy * qkr[0];
      ukxqir[0] = ukdy * qir[2] - ukdz * qir[1];
      ukxqir[1] = ukdz * qir[0] - ukdx * qir[2];
      ukxqir[2] = ukdx * qir[1] - ukdy * qir[0];
      uixqkrp[0] = uipy * qkr[2] - uipz * qkr[1];
      uixqkrp[1] = uipz * qkr[0] - uipx * qkr[2];
      uixqkrp[2] = uipx * qkr[1] - uipy * qkr[0];
      ukxqirp[0] = ukpy * qir[2] - ukpz * qir[1];
      ukxqirp[1] = ukpz * qir[0] - ukpx * qir[2];
      ukxqirp[2] = ukpx * qir[1] - ukpy * qir[0];
      rxqiuk[0] = yr * qiuk[2] - zr * qiuk[1];
      rxqiuk[1] = zr * qiuk[0] - xr * qiuk[2];
      rxqiuk[2] = xr * qiuk[1] - yr * qiuk[0];
      rxqkui[0] = yr * qkui[2] - zr * qkui[1];
      rxqkui[1] = zr * qkui[0] - xr * qkui[2];
      rxqkui[2] = xr * qkui[1] - yr * qkui[0];
      rxqiukp[0] = yr * qiukp[2] - zr * qiukp[1];
      rxqiukp[1] = zr * qiukp[0] - xr * qiukp[2];
      rxqiukp[2] = xr * qiukp[1] - yr * qiukp[0];
      rxqkuip[0] = yr * qkuip[2] - zr * qkuip[1];
      rxqkuip[1] = zr * qkuip[0] - xr * qkuip[2];
      rxqkuip[2] = xr * qkuip[1] - yr * qkuip[0];
      scip[0] = uipx * dkx + uipy * dky + uipz * dkz + dix * ukpx + diy * ukpy + diz * ukpz;
      scip[1] = uidx * ukpx + uidy * ukpy + uidz * ukpz + uipx * ukdx + uipy * ukdy + uipz * ukdz;
      scip[2] = uipx * xr + uipy * yr + uipz * zr;
      scip[3] = ukpx * xr + ukpy * yr + ukpz * zr;
      scip[6] = qir[0] * ukpx + qir[1] * ukpy + qir[2] * ukpz;
      scip[7] = qkr[0] * uipx + qkr[1] * uipy + qkr[2] * uipz;
      glip[0] = ck * scip[2] - ci * scip[3];
      glip[1] = -sc[2] * scip[3] - scip[2] * sc[3];
      glip[2] = scip[2] * sc[5] - scip[3] * sc[4];
      glip[5] = scip[0];
      glip[6] = 2 * (scip[6] - scip[7]);
      gfi[0] = 0.5f * rr5 * ((gli[0] + gli[5]) * psc3
                           + (glip[0] + glip[5]) * dsc3 + scip[1] * scale3i)
               + 0.5f * rr7 * ((gli[6] + gli[1]) * psc5
                              + (glip[6] + glip[1]) * dsc5
                              - (sci[2] * scip[3] + scip[2] * sci[3]) * scale5i)
               + 0.5f * rr9 * (gli[2] * psc7 + glip[2] * dsc7);
      gfi[1] = -rr3 * ck + rr5 * sc[3] - rr7 * sc[5];
      gfi[2] = rr3 * ci + rr5 * sc[2] + rr7 * sc[4];
      gfi[3] = 2 * rr5;
      gfi[4] = rr7 * (sci[3] * psc7 + scip[3] * dsc7);
      gfi[5] = -rr7 * (sci[2] * psc7 + scip[2] * dsc7);
      ftm2i[0] = gfi[0] * xr + 0.5f * (-rr3 * ck * (uidx * psc3 + uipx * dsc3)
                                    + rr5 * sc[3] * (uidx * psc5 + uipx * dsc5)
                                    - rr7 * sc[5] * (uidx * psc7 + uipx * dsc7))
               + (rr3 * ci * (ukdx * psc3 + ukpx * dsc3)
                  + rr5 * sc[2] * (ukdx * psc5 + ukpx * dsc5)
                  + rr7 * sc[4] * (ukdx * psc7 + ukpx * dsc7)) * 0.5f
               + rr5 * scale5i * (sci[3] * uipx + scip[3] * uidx
                                    + sci[2] * ukpx + scip[2] * ukdx) * 0.5f
               + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * dix
               + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dkx
               + 0.5f * gfi[3] * ((qkui[0] - qiuk[0]) * psc5
                                 + (qkuip[0] - qiukp[0]) * dsc5)
               + gfi[4] * qir[0] + gfi[5] * qkr[0];
      ftm2i[1] = gfi[0] * yr + 0.5f * (-rr3 * ck * (uidy * psc3 + uipy * dsc3)
                                    + rr5 * sc[3] * (uidy * psc5 + uipy * dsc5)
                                    - rr7 * sc[5] * (uidy * psc7 + uipy * dsc7))
               + (rr3 * ci * (ukdy * psc3 + ukpy * dsc3)
                  + rr5 * sc[2] * (ukdy * psc5 + ukpy * dsc5)
                  + rr7 * sc[4] * (ukdy * psc7 + ukpy * dsc7)) * 0.5f
               + rr5 * scale5i * (sci[3] * uipy + scip[3] * uidy
                                    + sci[2] * ukpy + scip[2] * ukdy) * 0.5f
               + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * diy
               + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dky
               + 0.5f * gfi[3] * ((qkui[1] - qiuk[1]) * psc5
                                 + (qkuip[1] - qiukp[1]) * dsc5)
               + gfi[4] * qir[1] + gfi[5] * qkr[1];
      ftm2i[2] = gfi[0] * zr + 0.5f * (-rr3 * ck * (uidz * psc3 + uipz * dsc3)
                                    + rr5 * sc[3] * (uidz * psc5 + uipz * dsc5)
                                    - rr7 * sc[5] * (uidz * psc7 + uipz * dsc7))
               + (rr3 * ci * (ukdz * psc3 + ukpz * dsc3)
                  + rr5 * sc[2] * (ukdz * psc5 + ukpz * dsc5)
                  + rr7 * sc[4] * (ukdz * psc7 + ukpz * dsc7)) * 0.5f
               + rr5 * scale5i * (sci[3] * uipz + scip[3] * uidz
                                    + sci[2] * ukpz + scip[2] * ukdz) * 0.5f
               + 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5 * diz
               + 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5 * dkz
               + 0.5f * gfi[3] * ((qkui[2] - qiuk[2]) * psc5
                                 + (qkuip[2] - qiukp[2]) * dsc5)
               + gfi[4] * qir[2] + gfi[5] * qkr[2];
      fridmp[0] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
                              + (glip[0] + glip[5]) * dscale) * ddsc3[0]
                        + rr5 * ((gli[1] + gli[6]) * pscale
                                 + (glip[1] + glip[6]) * dscale) * ddsc5[0]
                        + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[0]);

      fridmp[1] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
                              + (glip[0] + glip[5]) * dscale) * ddsc3[1]
                        + rr5 * ((gli[1] + gli[6]) * pscale
                                 + (glip[1] + glip[6]) * dscale) * ddsc5[1]
                        + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[1]);

      fridmp[2] = 0.5f * (rr3 * ((gli[0] + gli[5]) * pscale
                              + (glip[0] + glip[5]) * dscale) * ddsc3[2]
                        + rr5 * ((gli[1] + gli[6]) * pscale
                                 + (glip[1] + glip[6]) * dscale) * ddsc5[2]
                        + rr7 * (gli[2] * pscale + glip[2] * dscale) * ddsc7[2]);
      findmp[0] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[0]
                                    - rr5 * ddsc5[0] * (sci[2] * scip[3] + scip[2] * sci[3]));
      findmp[1] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[1]
                                    - rr5 * ddsc5[1] * (sci[2] * scip[3] + scip[2] * sci[3]));
      findmp[2] = 0.5f * uscale * (scip[1] * rr3 * ddsc3[2]
                                    - rr5 * ddsc5[2] * (sci[2] * scip[3] + scip[2] * sci[3]));
      ftm2i[0] = ftm2i[0] - fridmp[0] - findmp[0];
      ftm2i[1] = ftm2i[1] - fridmp[1] - findmp[1];
      ftm2i[2] = ftm2i[2] - fridmp[2] - findmp[2];
      gti[1] = 0.5f * (sci[3] * psc5 + scip[3] * dsc5) * rr5;
      gti[2] = 0.5f * (sci[2] * psc5 + scip[2] * dsc5) * rr5;
      gti[3] = gfi[3];
      gti[4] = gfi[4];
      gti[5] = gfi[5];
      ttm2i[0] = -rr3 * (dixuk[0] * psc3 + dixukp[0] * dsc3) * 0.5f
         + gti[1] * dixr[0] + gti[3] * ((ukxqir[0] + rxqiuk[0]) * psc5
         + (ukxqirp[0] + rxqiukp[0]) * dsc5) * 0.5f - gti[4] * rxqir[0];
      ttm2i[1] = -rr3 * (dixuk[1] * psc3 + dixukp[1] * dsc3) * 0.5f
         + gti[1] * dixr[1] + gti[3] * ((ukxqir[1] + rxqiuk[1]) * psc5
         + (ukxqirp[1] + rxqiukp[1]) * dsc5) * 0.5f - gti[4] * rxqir[1];
      ttm2i[2] = -rr3 * (dixuk[2] * psc3 + dixukp[2] * dsc3) * 0.5f
         + gti[1] * dixr[2] + gti[3] * ((ukxqir[2] + rxqiuk[2]) * psc5
         + (ukxqirp[2] + rxqiukp[2]) * dsc5) * 0.5f - gti[4] * rxqir[2];
      ttm3i[0] = -rr3 * (dkxui[0] * psc3 + dkxuip[0] * dsc3) * 0.5f
         + gti[2] * dkxr[0] - gti[3] * ((uixqkr[0] + rxqkui[0]) * psc5
         + (uixqkrp[0] + rxqkuip[0]) * dsc5) * 0.5f - gti[5] * rxqkr[0];
      ttm3i[1] = -rr3 * (dkxui[1] * psc3 + dkxuip[1] * dsc3) * 0.5f
         + gti[2] * dkxr[1] - gti[3] * ((uixqkr[1] + rxqkui[1]) * psc5
         + (uixqkrp[1] + rxqkuip[1]) * dsc5) * 0.5f - gti[5] * rxqkr[1];
      ttm3i[2] = -rr3 * (dkxui[2] * psc3 + dkxuip[2] * dsc3) * 0.5f
         + gti[2] * dkxr[2] - gti[3] * ((uixqkr[2] + rxqkui[2]) * psc5
         + (uixqkrp[2] + rxqkuip[2]) * dsc5) * 0.5f - gti[5] * rxqkr[2];
      pgrad.frcx -= f * ftm2i[0];
      pgrad.frcy -= f * ftm2i[1];
      pgrad.frcz -= f * ftm2i[2];
      pgrad.ttqi[0] -= f * ttm2i[0];
      pgrad.ttqi[1] -= f * ttm2i[1];
      pgrad.ttqi[2] -= f * ttm2i[2];
      pgrad.ttqk[0] -= f * ttm3i[0];
      pgrad.ttqk[1] -= f * ttm3i[1];
      pgrad.ttqk[2] -= f * ttm3i[2];
   }
}

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
}
