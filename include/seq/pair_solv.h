#pragma once
#include "ff/solv/solute.h"
#include "seq/damp.h"
#include "seq/seq.h"
#include <algorithm>
#include <cmath>

namespace tinker {
#pragma acc routine seq
template <bool do_g>
SEQ_ROUTINE
inline void pair_ewca(real r, real r2, real rio, real rmixo, real rmixo7, real sk, real sk2, real aoi, real emixo, real& sum, bool ifo)
{
   sum = 0.;
   real scale;
   real lik,lik2,lik3,lik4,lik5;
   real uik,uik2,uik3,uik4,uik5;
   real lik10,lik11,lik12;
   real uik10,uik11,uik12;
   real term,rmax;
   real iwca,idisp,irepl;
   if (ifo) scale = 1.;
   else scale = 2.;
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
         term = 4. * pi / (48.*r) * (3.*(lik4-uik4) - 8.*r*(lik3-uik3) + 6.*(r2-sk2)*(lik2-uik2));
         iwca = -emixo * term;
         sum += iwca;
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
         term = 4. * pi / (120.*r*lik5*uik5) * (15.*uik*lik*r*(uik4-lik4) - 10.*uik2*lik2*(uik3-lik3) + 6.*(sk2-r2)*(uik5-lik5));
         idisp = -2. * aoi * term;
         term = 4. * pi / (2640.*r*lik12*uik12) * (120.*uik*lik*r*(uik11-lik11) - 66.*uik2*lik2*(uik10-lik10) + 55.*(sk2-r2)*(uik12-lik12));
         irepl = aoi * rmixo7 * term;
         sum += irepl + idisp;
      }
   }
   sum = sum * scale;
}
}

namespace tinker {
#pragma acc routine seq
template <bool do_g>
SEQ_ROUTINE
inline void pair_egka(real r2, real xr, real yr, real zr, real xr2, real yr2, real zr2,
                     real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
                     real qiyy, real qiyz, real qizz, real uidx, real uidy, real uidz, real rbi,
                     real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
                     real qkyy, real qkyz, real qkzz, real ukdx, real ukdy, real ukdz, real rbk,
                     real gkc, real fc, real fd, real fq, real& e)
{
   real rb2 = rbi * rbk;
   real expterm = REAL_EXP(-r2 / (gkc * rb2));
   real expc = expterm / gkc;
   real dexpc = -2./(gkc*rbi*rbk);
   real gf2 = 1. / (r2 + rb2*expterm);
   real gf = REAL_SQRT(gf2);
   real gf3 = gf2 * gf;
   real gf5 = gf3 * gf2;
   real gf7 = gf5 * gf2;
   real gf9 = gf7 * gf2;

   real a[5][3];
   real gc[10];
   real gux[10];
   real guy[10];
   real guz[10];
   real gqxx[10];
   real gqxy[10];
   real gqxz[10];
   real gqyy[10];
   real gqyz[10];
   real gqzz[10];

   a[0][0] = gf;
   a[1][0] = -gf3;
   a[2][0] = 3. * gf5;
   a[3][0] = -15. * gf7;
   a[4][0] = 105. * gf9;

   real expc1 = 1. - expc;
   a[0][1] = expc1 * a[1][0];
   a[1][1] = expc1 * a[2][0];
   a[2][1] = expc1 * a[3][0];
   a[3][1] = expc1 * a[4][0];

   real expcdexpc = -expc * dexpc;
   a[0][2] = expc1*a[1][1] + expcdexpc*a[1][0];
   a[1][2] = expc1*a[2][1] + expcdexpc*a[2][0];
   a[2][2] = expc1*a[3][1] + expcdexpc*a[3][0];

   a[0][0] = fc * a[0][0];
   a[0][1] = fc * a[0][1];
   a[0][2] = fc * a[0][2];
   a[1][0] = fd * a[1][0];
   a[1][1] = fd * a[1][1];
   a[1][2] = fd * a[1][2];
   a[2][0] = fq * a[2][0];
   a[2][1] = fq * a[2][1];
   a[2][2] = fq * a[2][2];

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
   gqxx[1] = xr * (2.0 * a[2][0] + xr2 * a[2][1]);
   gqxx[2] = yr * xr2 * a[2][1];
   gqxx[3] = zr * xr2 * a[2][1];
   gqyy[1] = xr * yr2 * a[2][1];
   gqyy[2] = yr * (2.0 * a[2][0] + yr2 * a[2][1]);
   gqyy[3] = zr * yr2 * a[2][1];
   gqzz[1] = xr * zr2 * a[2][1];
   gqzz[2] = yr * zr2 * a[2][1];
   gqzz[3] = zr * (2.0 * a[2][0] + zr2 * a[2][1]);
   gqxy[1] = yr * (a[2][0] + xr2 * a[2][1]);
   gqxy[2] = xr * (a[2][0] + yr2 * a[2][1]);
   gqxy[3] = zr * xr * yr * a[2][1];
   gqxz[1] = zr * (a[2][0] + xr2 * a[2][1]);
   gqxz[2] = gqxy[3];
   gqxz[3] = xr * (a[2][0] + zr2 * a[2][1]);
   gqyz[1] = gqxy[3];
   gqyz[2] = zr * (a[2][0] + yr2 * a[2][1]);
   gqyz[3] = yr * (a[2][0] + zr2 * a[2][1]);

   gc[4] = a[0][1] + xr2 * a[0][2];
   gc[5] = xr * yr * a[0][2];
   gc[6] = xr * zr * a[0][2];
   gc[7] = a[0][1] + yr2 * a[0][2];
   gc[8] = yr * zr * a[0][2];
   gc[9] = a[0][1] + zr2 * a[0][2];
   gux[4] = xr * (a[1][1] + 2.0 * a[1][1] + xr2 * a[1][2]);
   gux[5] = yr * (a[1][1] + xr2 * a[1][2]);
   gux[6] = zr * (a[1][1] + xr2 * a[1][2]);
   gux[7] = xr * (a[1][1] + yr2 * a[1][2]);
   gux[8] = zr * xr * yr * a[1][2];
   gux[9] = xr * (a[1][1] + zr2 * a[1][2]);
   guy[4] = yr * (a[1][1] + xr2 * a[1][2]);
   guy[5] = xr * (a[1][1] + yr2 * a[1][2]);
   guy[6] = gux[8];
   guy[7] = yr * (a[1][1] + 2.0 * a[1][1] + yr2 * a[1][2]);
   guy[8] = zr * (a[1][1] + yr2 * a[1][2]);
   guy[9] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[4] = zr * (a[1][1] + xr2 * a[1][2]);
   guz[5] = gux[8];
   guz[6] = xr * (a[1][1] + zr2 * a[1][2]);
   guz[7] = zr * (a[1][1] + yr2 * a[1][2]);
   guz[8] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[9] = zr * (a[1][1] + 2.0 * a[1][1] + zr2 * a[1][2]);

   gqxx[4] = 2.0 * a[2][0] + xr2 * (5.0 * a[2][1] + xr2 * a[2][2]);
   gqxx[5] = yr * xr * (2.0 * a[2][1] + xr2 * a[2][2]);
   gqxx[6] = zr * xr * (2.0 * a[2][1] + xr2 * a[2][2]);
   gqxx[7] = xr2 * (a[2][1] + yr2 * a[2][2]);
   gqxx[8] = zr * yr * xr2 * a[2][2];
   gqxx[9] = xr2 * (a[2][1] + zr2 * a[2][2]);
   gqyy[4] = yr2 * (a[2][1] + xr2 * a[2][2]);
   gqyy[5] = xr * yr * (2.0 * a[2][1] + yr2 * a[2][2]);
   gqyy[6] = xr * zr * yr2 * a[2][2];
   gqyy[7] = 2.0 * a[2][0] + yr2 * (5.0 * a[2][1] + yr2 * a[2][2]);
   gqyy[8] = yr * zr * (2.0 * a[2][1] + yr2 * a[2][2]);
   gqyy[9] = yr2 * (a[2][1] + zr2 * a[2][2]);
   gqzz[4] = zr2 * (a[2][1] + xr2 * a[2][2]);
   gqzz[5] = xr * yr * zr2 * a[2][2];
   gqzz[6] = xr * zr * (2.0 * a[2][1] + zr2 * a[2][2]);
   gqzz[7] = zr2 * (a[2][1] + yr2 * a[2][2]);
   gqzz[8] = yr * zr * (2.0 * a[2][1] + zr2 * a[2][2]);
   gqzz[9] = 2.0 * a[2][0] + zr2 * (5.0 * a[2][1] + zr2 * a[2][2]);
   
   gqxy[4] = xr * yr * (3.0 * a[2][1] + xr2 * a[2][2]);
   gqxy[5] = a[2][0] + (xr2 + yr2) * a[2][1] + xr2 * yr2 * a[2][2];
   gqxy[6] = zr * yr * (a[2][1] + xr2 * a[2][2]);
   gqxy[7] = xr * yr * (3.0 * a[2][1] + yr2 * a[2][2]);
   gqxy[8] = zr * xr * (a[2][1] + yr2 * a[2][2]);
   gqxy[9] = xr * yr * (a[2][1] + zr2 * a[2][2]);

   gqxz[4] = xr * zr * (3.0 * a[2][1] + xr2 * a[2][2]);
   gqxz[5] = yr * zr * (a[2][1] + xr2 * a[2][2]);
   gqxz[6] = a[2][0] + (xr2 + zr2) * a[2][1] + xr2 * zr2 * a[2][2];
   gqxz[7] = xr * zr * (a[2][1] + yr2 * a[2][2]);
   gqxz[8] = xr * yr * (a[2][1] + zr2 * a[2][2]);
   gqxz[9] = xr * zr * (3.0 * a[2][1] + zr2 * a[2][2]);

   gqyz[4] = zr * yr * (a[2][1] + xr2 * a[2][2]);
   gqyz[5] = xr * zr * (a[2][1] + yr2 * a[2][2]);
   gqyz[6] = xr * yr * (a[2][1] + zr2 * a[2][2]);
   gqyz[7] = yr * zr * (3.0 * a[2][1] + yr2 * a[2][2]);
   gqyz[8] = a[2][0] + (yr2 + zr2) * a[2][1] + yr2 * zr2 * a[2][2];
   gqyz[9] = yr * zr * (3.0 * a[2][1] + zr2 * a[2][2]);


   real esym = ci * ck * gc[0]
         - dix * (dkx * gux[1] + dky * guy[1] + dkz * guz[1])
         - diy * (dkx * gux[2] + dky * guy[2] + dkz * guz[2])
         - diz * (dkx * gux[3] + dky * guy[3] + dkz * guz[3]);

   real ewi = ci * (dkx * gc[1] + dky * gc[2] + dkz * gc[3])
      - ck * (dix * gux[0] + diy * guy[0] + diz * guz[0])
      + ci * (qkxx * gc[4] + qkyy * gc[7] + qkzz * gc[9]
              + 2.0 * (qkxy * gc[5] + qkxz * gc[6] + qkyz * gc[8]))
      + ck * (qixx * gqxx[0] + qiyy * gqyy[0] + qizz * gqzz[0]
              + 2.0 * (qixy * gqxy[0] + qixz * gqxz[0] + qiyz * gqyz[0]))
      - dix * (qkxx * gux[4] + qkyy * gux[7] + qkzz * gux[9]
               + 2.0 * (qkxy * gux[5] + qkxz * gux[6] + qkyz * gux[8]))
      - diy * (qkxx * guy[4] + qkyy * guy[7] + qkzz * guy[9]
               + 2.0 * (qkxy * guy[5] + qkxz * guy[6] + qkyz * guy[8]))
      - diz * (qkxx * guz[4] + qkyy * guz[7] + qkzz * guz[9]
               + 2.0 * (qkxy * guz[5] + qkxz * guz[6] + qkyz * guz[8]))
      + dkx * (qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
               + 2.0 * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]))
      + dky * (qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
               + 2.0 * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]))
      + dkz * (qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
               + 2.0 * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]))
      + qixx * (qkxx * gqxx[4] + qkyy * gqxx[7] + qkzz * gqxx[9]
                + 2.0 * (qkxy * gqxx[5] + qkxz * gqxx[6] + qkyz * gqxx[8]))
      + qiyy * (qkxx * gqyy[4] + qkyy * gqyy[7] + qkzz * gqyy[9]
                + 2.0 * (qkxy * gqyy[5] + qkxz * gqyy[6] + qkyz * gqyy[8]))
      + qizz * (qkxx * gqzz[4] + qkyy * gqzz[7] + qkzz * gqzz[9]
                + 2.0 * (qkxy * gqzz[5] + qkxz * gqzz[6] + qkyz * gqzz[8]))
      + 2.0 * (qixy * (qkxx * gqxy[4] + qkyy * gqxy[7] + qkzz * gqxy[9]
                       + 2.0 * (qkxy * gqxy[5] + qkxz * gqxy[6] + qkyz * gqxy[8]))
               + qixz * (qkxx * gqxz[4] + qkyy * gqxz[7] + qkzz * gqxz[9]
                         + 2.0 * (qkxy * gqxz[5] + qkxz * gqxz[6] + qkyz * gqxz[8]))
               + qiyz * (qkxx * gqyz[4] + qkyy * gqyz[7] + qkzz * gqyz[9]
                         + 2.0 * (qkxy * gqyz[5] + qkxz * gqyz[6] + qkyz * gqyz[8])));

   real ewk = ci * (dkx * gux[0] + dky * guy[0] + dkz * guz[0])
      - ck * (dix * gc[1] + diy * gc[2] + diz * gc[3])
      + ci * (qkxx * gqxx[0] + qkyy * gqyy[0] + qkzz * gqzz[0]
              + 2.0 * (qkxy * gqxy[0] + qkxz * gqxz[0] + qkyz * gqyz[0]))
      + ck * (qixx * gc[4] + qiyy * gc[7] + qizz * gc[9]
              + 2.0 * (qixy * gc[5] + qixz * gc[6] + qiyz * gc[8]))
      - dix * (qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
               + 2.0 * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]))
      - diy * (qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
               + 2.0 * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]))
      - diz * (qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
               + 2.0 * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]))
      + dkx * (qixx * gux[4] + qiyy * gux[7] + qizz * gux[9]
               + 2.0 * (qixy * gux[5] + qixz * gux[6] + qiyz * gux[8]))
      + dky * (qixx * guy[4] + qiyy * guy[7] + qizz * guy[9]
               + 2.0 * (qixy * guy[5] + qixz * guy[6] + qiyz * guy[8]))
      + dkz * (qixx * guz[4] + qiyy * guz[7] + qizz * guz[9]
               + 2.0 * (qixy * guz[5] + qixz * guz[6] + qiyz * guz[8]))
      + qixx * (qkxx * gqxx[4] + qkyy * gqyy[4] + qkzz * gqzz[4]
                + 2.0 * (qkxy * gqxy[4] + qkxz * gqxz[4] + qkyz * gqyz[4]))
      + qiyy * (qkxx * gqxx[7] + qkyy * gqyy[7] + qkzz * gqzz[7]
                + 2.0 * (qkxy * gqxy[7] + qkxz * gqxz[7] + qkyz * gqyz[7]))
      + qizz * (qkxx * gqxx[9] + qkyy * gqyy[9] + qkzz * gqzz[9]
                + 2.0 * (qkxy * gqxy[9] + qkxz * gqxz[9] + qkyz * gqyz[9]))
      + 2.0 * (qixy * (qkxx * gqxx[5] + qkyy * gqyy[5] + qkzz * gqzz[5]
                       + 2.0 * (qkxy * gqxy[5] + qkxz * gqxz[5] + qkyz * gqyz[5]))
               + qixz * (qkxx * gqxx[6] + qkyy * gqyy[6] + qkzz * gqzz[6]
                         + 2.0 * (qkxy * gqxy[6] + qkxz * gqxz[6] + qkyz * gqyz[6]))
               + qiyz * (qkxx * gqxx[8] + qkyy * gqyy[8] + qkzz * gqzz[8]
                         + 2.0 * (qkxy * gqxy[8] + qkxz * gqxz[8] + qkyz * gqyz[8])));

   real esymi = -dix * (ukdx * gux[1] + ukdy * guy[1] + ukdz * guz[1])
        - diy * (ukdx * gux[2] + ukdy * guy[2] + ukdz * guz[2])
        - diz * (ukdx * gux[3] + ukdy * guy[3] + ukdz * guz[3])
        - dkx * (uidx * gux[1] + uidy * guy[1] + uidz * guz[1])
        - dky * (uidx * gux[2] + uidy * guy[2] + uidz * guz[2])
        - dkz * (uidx * gux[3] + uidy * guy[3] + uidz * guz[3]);

   real ewii = ci * (ukdx * gc[1] + ukdy * gc[2] + ukdz * gc[3])
       - ck * (uidx * gux[0] + uidy * guy[0] + uidz * guz[0])
       - uidx * (qkxx * gux[4] + qkyy * gux[7] + qkzz * gux[9]
                 + 2. * (qkxy * gux[5] + qkxz * gux[6] + qkyz * gux[8]))
       - uidy * (qkxx * guy[4] + qkyy * guy[7] + qkzz * guy[9]
                 + 2. * (qkxy * guy[5] + qkxz * guy[6] + qkyz * guy[8]))
       - uidz * (qkxx * guz[4] + qkyy * guz[7] + qkzz * guz[9]
                 + 2. * (qkxy * guz[5] + qkxz * guz[6] + qkyz * guz[8]))
       + ukdx * (qixx * gqxx[1] + qiyy * gqyy[1] + qizz * gqzz[1]
                 + 2. * (qixy * gqxy[1] + qixz * gqxz[1] + qiyz * gqyz[1]))
       + ukdy * (qixx * gqxx[2] + qiyy * gqyy[2] + qizz * gqzz[2]
                 + 2. * (qixy * gqxy[2] + qixz * gqxz[2] + qiyz * gqyz[2]))
       + ukdz * (qixx * gqxx[3] + qiyy * gqyy[3] + qizz * gqzz[3]
                 + 2. * (qixy * gqxy[3] + qixz * gqxz[3] + qiyz * gqyz[3]));

   real ewki = ci * (ukdx * gux[0] + ukdy * guy[0] + ukdz * guz[0])
       - ck * (uidx * gc[1] + uidy * gc[2] + uidz * gc[3])
       - uidx * (qkxx * gqxx[1] + qkyy * gqyy[1] + qkzz * gqzz[1]
                 + 2. * (qkxy * gqxy[1] + qkxz * gqxz[1] + qkyz * gqyz[1]))
       - uidy * (qkxx * gqxx[2] + qkyy * gqyy[2] + qkzz * gqzz[2]
                 + 2. * (qkxy * gqxy[2] + qkxz * gqxz[2] + qkyz * gqyz[2]))
       - uidz * (qkxx * gqxx[3] + qkyy * gqyy[3] + qkzz * gqzz[3]
                 + 2. * (qkxy * gqxy[3] + qkxz * gqxz[3] + qkyz * gqyz[3]))
       + ukdx * (qixx * gux[4] + qiyy * gux[7] + qizz * gux[9]
                 + 2. * (qixy * gux[5] + qixz * gux[6] + qiyz * gux[8]))
       + ukdy * (qixx * guy[4] + qiyy * guy[7] + qizz * guy[9]
                 + 2. * (qixy * guy[5] + qixz * guy[6] + qiyz * guy[8]))
       + ukdz * (qixx * guz[4] + qiyy * guz[7] + qizz * guz[9]
                 + 2. * (qixy * guz[5] + qixz * guz[6] + qiyz * guz[8]));

   e = esym + 0.5*(ewi+ewk);
   e += 0.5 * (esymi + 0.5*(ewii+ewki));
}
}

namespace tinker {
#pragma acc routine seq
template <class Ver>
SEQ_ROUTINE
inline void pair_ediff(real r2, real xr, real yr, real zr, real dscale, real pscale, real uscale,
                       real ci, real dix, real diy, real diz,
                       real qixx, real qixy, real qixz, real qiyy, real qiyz, real qizz,
                       real uidx, real uidy, real uidz, real uidsx, real uidsy, real uidsz, 
                       real pdi, real pti, //
                       real ck, real dkx, real dky, real dkz,
                       real qkxx, real qkxy, real qkxz, real qkyy, real qkyz, real qkzz,
                       real ukdx, real ukdy, real ukdz, real ukdsx, real ukdsy, real ukdsz,
                       real pdk, real ptk, //
                       real f, real& e)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;

   real uix = uidsx - uidx;
   real uiy = uidsy - uidy;
   real uiz = uidsz - uidz;
   real ukx = ukdsx - ukdx;
   real uky = ukdsy - ukdy;
   real ukz = ukdsz - ukdz;

   real dir = dix * xr + diy * yr + diz * zr;
   real qix = qixx * xr + qixy * yr + qixz * zr;
   real qiy = qixy * xr + qiyy * yr + qiyz * zr;
   real qiz = qixz * xr + qiyz * yr + qizz * zr;
   real qir = qix * xr + qiy * yr + qiz * zr;
   real dkr = dkx * xr + dky * yr + dkz * zr;
   real qkx = qkxx * xr + qkxy * yr + qkxz * zr;
   real qky = qkxy * xr + qkyy * yr + qkyz * zr;
   real qkz = qkxz * xr + qkyz * yr + qkzz * zr;
   real qkr = qkx * xr + qky * yr + qkz * zr;
   real uir = uix * xr + uiy * yr + uiz * zr;
   real ukr = ukx * xr + uky * yr + ukz * zr;

   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr3 * rr2;
   real rr7 = 5 * rr5 * rr2;
   MAYBE_UNUSED real rr9;
   if CONSTEXPR (do_g) rr9 = 7 * rr7 * rr2;
   real bn[5];
   bn[1] = rr3;
   bn[2] = rr5;
   bn[3] = rr7;
   if CONSTEXPR (do_g) bn[4] = rr9;

   // if use_thole
   real ex3, ex5, ex7;
   MAYBE_UNUSED real rc31, rc32, rc33, rc51, rc52, rc53, rc71, rc72, rc73;
   if CONSTEXPR (!do_g) {
      damp_thole3(r, pdi, pti, pdk, ptk, //
         ex3, ex5, ex7);
      ex3 = 1 - ex3;
      ex5 = 1 - ex5;
      ex7 = 1 - ex7;
   } else {
      damp_thole3g(          //
         r, rr2, xr, yr, zr, //
         pdi, pti, pdk, ptk, //
         ex3, ex5, ex7,      //
         rc31, rc32, rc33,   //
         rc51, rc52, rc53,   //
         rc71, rc72, rc73);
      rc31 *= rr3;
      rc32 *= rr3;
      rc33 *= rr3;
      rc51 *= rr5;
      rc52 *= rr5;
      rc53 *= rr5;
      rc71 *= rr7;
      rc72 *= rr7;
      rc73 *= rr7;
   }
   // end if use_thole

   real sr3 = bn[1] - ex3 * rr3;
   real sr5 = bn[2] - ex5 * rr5;
   real sr7 = bn[3] - ex7 * rr7;

   if CONSTEXPR (do_e) {
      real diu = dix * ukx + diy * uky + diz * ukz;
      real qiu = qix * ukx + qiy * uky + qiz * ukz;
      real dku = dkx * uix + dky * uiy + dkz * uiz;
      real qku = qkx * uix + qky * uiy + qkz * uiz;
      real term1 = ck * uir - ci * ukr + diu + dku;
      real term2 = 2 * (qiu - qku) - uir * dkr - dir * ukr;
      real term3 = uir * qkr - ukr * qir;
      e = pscale * f * (term1 * sr3 + term2 * sr5 + term3 * sr7);
   }

   // if CONSTEXPR (do_g) {
   //    real uirp = uixp * xr + uiyp * yr + uizp * zr;
   //    real ukrp = ukxp * xr + ukyp * yr + ukzp * zr;

   //    // get the induced dipole field used for dipole torques

   //    real tuir, tukr;

   //    real tix3 = pscale * sr3 * ukx + dscale * sr3 * ukxp;
   //    real tiy3 = pscale * sr3 * uky + dscale * sr3 * ukyp;
   //    real tiz3 = pscale * sr3 * ukz + dscale * sr3 * ukzp;
   //    real tkx3 = pscale * sr3 * uix + dscale * sr3 * uixp;
   //    real tky3 = pscale * sr3 * uiy + dscale * sr3 * uiyp;
   //    real tkz3 = pscale * sr3 * uiz + dscale * sr3 * uizp;
   //    tuir = -pscale * sr5 * ukr - dscale * sr5 * ukrp;
   //    tukr = -pscale * sr5 * uir - dscale * sr5 * uirp;

   //    uf0i += f * (tix3 + xr * tuir);
   //    uf1i += f * (tiy3 + yr * tuir);
   //    uf2i += f * (tiz3 + zr * tuir);
   //    uf0k += f * (tkx3 + xr * tukr);
   //    uf1k += f * (tky3 + yr * tukr);
   //    uf2k += f * (tkz3 + zr * tukr);

   //    // get induced dipole field gradient used for quadrupole torques

   //    real tix5 = 2 * (pscale * sr5 * ukx + dscale * sr5 * ukxp);
   //    real tiy5 = 2 * (pscale * sr5 * uky + dscale * sr5 * ukyp);
   //    real tiz5 = 2 * (pscale * sr5 * ukz + dscale * sr5 * ukzp);
   //    real tkx5 = 2 * (pscale * sr5 * uix + dscale * sr5 * uixp);
   //    real tky5 = 2 * (pscale * sr5 * uiy + dscale * sr5 * uiyp);
   //    real tkz5 = 2 * (pscale * sr5 * uiz + dscale * sr5 * uizp);
   //    tuir = -pscale * sr7 * ukr - dscale * sr7 * ukrp;
   //    tukr = -pscale * sr7 * uir - dscale * sr7 * uirp;

   //    duf0i += f * (xr * tix5 + xr * xr * tuir);
   //    duf1i += f * (xr * tiy5 + yr * tix5 + 2 * xr * yr * tuir);
   //    duf2i += f * (yr * tiy5 + yr * yr * tuir);
   //    duf3i += f * (xr * tiz5 + zr * tix5 + 2 * xr * zr * tuir);
   //    duf4i += f * (yr * tiz5 + zr * tiy5 + 2 * yr * zr * tuir);
   //    duf5i += f * (zr * tiz5 + zr * zr * tuir);
   //    duf0k += f * (-xr * tkx5 - xr * xr * tukr);
   //    duf1k += f * (-xr * tky5 - yr * tkx5 - 2 * xr * yr * tukr);
   //    duf2k += f * (-yr * tky5 - yr * yr * tukr);
   //    duf3k += f * (-xr * tkz5 - zr * tkx5 - 2 * xr * zr * tukr);
   //    duf4k += f * (-yr * tkz5 - zr * tky5 - 2 * yr * zr * tukr);
   //    duf5k += f * (-zr * tkz5 - zr * zr * tukr);

   //    // get the field gradient for direct polarization force

   //    real term1, term2, term3, term4, term5, term6, term7;

   //    term1 = bn[2] - ex3 * rr5;
   //    term2 = bn[3] - ex5 * rr7;
   //    term3 = -sr3 + term1 * xr * xr - xr * rc31;
   //    term4 = rc31 - term1 * xr - sr5 * xr;
   //    term5 = term2 * xr * xr - sr5 - xr * rc51;
   //    term6 = (bn[4] - ex7 * rr9) * xr * xr - bn[3] - xr * rc71;
   //    term7 = rc51 - 2 * bn[3] * xr + (ex5 + 1.5f * ex7) * rr7 * xr;
   //    real tixx = ci * term3 + dix * term4 + dir * term5 + 2 * sr5 * qixx
   //       + (qiy * yr + qiz * zr) * ex7 * rr7 + 2 * qix * term7 + qir * term6;
   //    real tkxx = ck * term3 - dkx * term4 - dkr * term5 + 2 * sr5 * qkxx
   //       + (qky * yr + qkz * zr) * ex7 * rr7 + 2 * qkx * term7 + qkr * term6;

   //    term3 = -sr3 + term1 * yr * yr - yr * rc32;
   //    term4 = rc32 - term1 * yr - sr5 * yr;
   //    term5 = term2 * yr * yr - sr5 - yr * rc52;
   //    term6 = (bn[4] - ex7 * rr9) * yr * yr - bn[3] - yr * rc72;
   //    term7 = rc52 - 2 * bn[3] * yr + (ex5 + 1.5f * ex7) * rr7 * yr;
   //    real tiyy = ci * term3 + diy * term4 + dir * term5 + 2 * sr5 * qiyy
   //       + (qix * xr + qiz * zr) * ex7 * rr7 + 2 * qiy * term7 + qir * term6;
   //    real tkyy = ck * term3 - dky * term4 - dkr * term5 + 2 * sr5 * qkyy
   //       + (qkx * xr + qkz * zr) * ex7 * rr7 + 2 * qky * term7 + qkr * term6;

   //    term3 = -sr3 + term1 * zr * zr - zr * rc33;
   //    term4 = rc33 - term1 * zr - sr5 * zr;
   //    term5 = term2 * zr * zr - sr5 - zr * rc53;
   //    term6 = (bn[4] - ex7 * rr9) * zr * zr - bn[3] - zr * rc73;
   //    term7 = rc53 - 2 * bn[3] * zr + (ex5 + 1.5f * ex7) * rr7 * zr;
   //    real tizz = ci * term3 + diz * term4 + dir * term5 + 2 * sr5 * qizz
   //       + (qix * xr + qiy * yr) * ex7 * rr7 + 2 * qiz * term7 + qir * term6;
   //    real tkzz = ck * term3 - dkz * term4 - dkr * term5 + 2 * sr5 * qkzz
   //       + (qkx * xr + qky * yr) * ex7 * rr7 + 2 * qkz * term7 + qkr * term6;

   //    term3 = term1 * xr * yr - yr * rc31;
   //    term4 = rc31 - term1 * xr;
   //    term5 = term2 * xr * yr - yr * rc51;
   //    term6 = (bn[4] - ex7 * rr9) * xr * yr - yr * rc71;
   //    term7 = rc51 - term2 * xr;
   //    real tixy = ci * term3 - sr5 * dix * yr + diy * term4 + dir * term5
   //       + 2 * sr5 * qixy - 2 * sr7 * yr * qix + 2 * qiy * term7 + qir * term6;
   //    real tkxy = ck * term3 + sr5 * dkx * yr - dky * term4 - dkr * term5
   //       + 2 * sr5 * qkxy - 2 * sr7 * yr * qkx + 2 * qky * term7 + qkr * term6;

   //    term3 = term1 * xr * zr - zr * rc31;
   //    term5 = term2 * xr * zr - zr * rc51;
   //    term6 = (bn[4] - ex7 * rr9) * xr * zr - zr * rc71;
   //    real tixz = ci * term3 - sr5 * dix * zr + diz * term4 + dir * term5
   //       + 2 * sr5 * qixz - 2 * sr7 * zr * qix + 2 * qiz * term7 + qir * term6;
   //    real tkxz = ck * term3 + sr5 * dkx * zr - dkz * term4 - dkr * term5
   //       + 2 * sr5 * qkxz - 2 * sr7 * zr * qkx + 2 * qkz * term7 + qkr * term6;

   //    term3 = term1 * yr * zr - zr * rc32;
   //    term4 = rc32 - term1 * yr;
   //    term5 = term2 * yr * zr - zr * rc52;
   //    term6 = (bn[4] - ex7 * rr9) * yr * zr - zr * rc72;
   //    term7 = rc52 - term2 * yr;
   //    real tiyz = ci * term3 - sr5 * diy * zr + diz * term4 + dir * term5
   //       + 2 * sr5 * qiyz - 2 * sr7 * zr * qiy + 2 * qiz * term7 + qir * term6;
   //    real tkyz = ck * term3 + sr5 * dky * zr - dkz * term4 - dkr * term5
   //       + 2 * sr5 * qkyz - 2 * sr7 * zr * qky + 2 * qkz * term7 + qkr * term6;

   //    // get the dEd/dR terms for Thole direct polarization force

   //    real depx, depy, depz;
   //    real frcx, frcy, frcz;

   //    depx = tixx * ukxp + tixy * ukyp + tixz * ukzp - tkxx * uixp - tkxy * uiyp
   //       - tkxz * uizp;
   //    depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp - tkxy * uixp - tkyy * uiyp
   //       - tkyz * uizp;
   //    depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp - tkxz * uixp - tkyz * uiyp
   //       - tkzz * uizp;
   //    if CONSTEXPR (eq<ETYP, EWALD>()) {
   //       frcx = -depx;
   //       frcy = -depy;
   //       frcz = -depz;
   //    } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
   //       frcx = -depx * dscale;
   //       frcy = -depy * dscale;
   //       frcz = -depz * dscale;
   //    }

   //    // get the dEp/dR terms for Thole direct polarization force

   //    depx = tixx * ukx + tixy * uky + tixz * ukz - tkxx * uix - tkxy * uiy
   //       - tkxz * uiz;
   //    depy = tixy * ukx + tiyy * uky + tiyz * ukz - tkxy * uix - tkyy * uiy
   //       - tkyz * uiz;
   //    depz = tixz * ukx + tiyz * uky + tizz * ukz - tkxz * uix - tkyz * uiy
   //       - tkzz * uiz;
   //    if CONSTEXPR (eq<ETYP, EWALD>()) {
   //       frcx -= depx;
   //       frcy -= depy;
   //       frcz -= depz;
   //    } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
   //       frcx -= pscale * depx;
   //       frcy -= pscale * depy;
   //       frcz -= pscale * depz;
   //    }

   //    // get the dtau/dr terms used for mutual polarization force

   //    term1 = bn[2] - ex3 * rr5;
   //    term2 = bn[3] - ex5 * rr7;
   //    term3 = sr5 + term1;

   //    term5 = -xr * term3 + rc31;
   //    term6 = -sr5 + xr * xr * term2 - xr * rc51;
   //    tixx = uix * term5 + uir * term6;
   //    tkxx = ukx * term5 + ukr * term6;

   //    term5 = -yr * term3 + rc32;
   //    term6 = -sr5 + yr * yr * term2 - yr * rc52;
   //    tiyy = uiy * term5 + uir * term6;
   //    tkyy = uky * term5 + ukr * term6;

   //    term5 = -zr * term3 + rc33;
   //    term6 = -sr5 + zr * zr * term2 - zr * rc53;
   //    tizz = uiz * term5 + uir * term6;
   //    tkzz = ukz * term5 + ukr * term6;

   //    term4 = -sr5 * yr;
   //    term5 = -xr * term1 + rc31;
   //    term6 = xr * yr * term2 - yr * rc51;
   //    tixy = uix * term4 + uiy * term5 + uir * term6;
   //    tkxy = ukx * term4 + uky * term5 + ukr * term6;

   //    term4 = -sr5 * zr;
   //    term6 = xr * zr * term2 - zr * rc51;
   //    tixz = uix * term4 + uiz * term5 + uir * term6;
   //    tkxz = ukx * term4 + ukz * term5 + ukr * term6;

   //    term5 = -yr * term1 + rc32;
   //    term6 = yr * zr * term2 - zr * rc52;
   //    tiyz = uiy * term4 + uiz * term5 + uir * term6;
   //    tkyz = uky * term4 + ukz * term5 + ukr * term6;

   //    depx = tixx * ukxp + tixy * ukyp + tixz * ukzp + tkxx * uixp + tkxy * uiyp
   //       + tkxz * uizp;
   //    depy = tixy * ukxp + tiyy * ukyp + tiyz * ukzp + tkxy * uixp + tkyy * uiyp
   //       + tkyz * uizp;
   //    depz = tixz * ukxp + tiyz * ukyp + tizz * ukzp + tkxz * uixp + tkyz * uiyp
   //       + tkzz * uizp;
   //    if CONSTEXPR (eq<ETYP, EWALD>()) {
   //       frcx -= depx;
   //       frcy -= depy;
   //       frcz -= depz;
   //    } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
   //       frcx -= uscale * depx;
   //       frcy -= uscale * depy;
   //       frcz -= uscale * depz;
   //    }

   //    frcx *= f;
   //    frcy *= f;
   //    frcz *= f;
   //    frcxi += frcx;
   //    frcyi += frcy;
   //    frczi += frcz;
   //    frcxk -= frcx;
   //    frcyk -= frcy;
   //    frczk -= frcz;

   //    if CONSTEXPR (do_v) {
   //       vxx = -xr * frcx;
   //       vxy = -0.5f * (yr * frcx + xr * frcy);
   //       vxz = -0.5f * (zr * frcx + xr * frcz);
   //       vyy = -yr * frcy;
   //       vyz = -0.5f * (zr * frcy + yr * frcz);
   //       vzz = -zr * frcz;
   //    }
   // }
}
}
