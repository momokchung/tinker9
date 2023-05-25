#pragma once
#include "ff/elec.h"
#include "math/realn.h"
#include "seq/add.h"
#include "seq/damp.h"

namespace tinker {
#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_dfield(real r2, real xr, real yr, real zr, real dscale,
   real pscale, //
   real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
   real qiyy, real qiyz, real qizz, real pdi,
   real pti, //
   real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
   real qkyy, real qkyz, real qkzz, real pdk,
   real ptk, //
   real aewald, real3& restrict fid, real3& restrict fip, real3& restrict fkd,
   real3& restrict fkp)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5, scale7;
   damp_thole3(r, pdi, pti, pdk, ptk, scale3, scale5, scale7);

   real bn[4];
   if CONSTEXPR (eq<ETYP, EWALD>()) damp_ewald<4>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;
   real rr7 = 15 * rr1 * rr2 * rr2 * rr2;

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

   real3 dixyz = make_real3(dix, diy, diz);
   real3 dkxyz = make_real3(dkx, dky, dkz);
   real3 qixyz = make_real3(qix, qiy, qiz);
   real3 qkxyz = make_real3(qkx, qky, qkz);
   real3 dr = make_real3(xr, yr, zr);
   real c1;
   real3 inci, inck;

   // d-field

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
      bn[3] -= (1 - scale7) * rr7;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = dscale * scale3 * rr3;
      bn[2] = dscale * scale5 * rr5;
      bn[3] = dscale * scale7 * rr7;
   }

   c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
   inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;
   fid += inci;

   c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
   inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   fkd += inck;

   // p-field

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      fip += inci;
      fkp += inck;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      if (pscale == dscale) {
         fip += inci;
         fkp += inck;
      } else {
         bn[1] = pscale * scale3 * rr3;
         bn[2] = pscale * scale5 * rr5;
         bn[3] = pscale * scale7 * rr7;

         c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
         fip += c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;

         c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
         fkp += c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
      }
   }
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_dfield_v2(real r2, real xr, real yr, real zr, real dscale,
   real pscale, real aewald, //
   real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
   real qiyy, real qiyz, real qizz, real pdi,
   real pti, //
   real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
   real qkyy, real qkyz, real qkzz, real pdk,
   real ptk, //
   real& restrict fidx, real& restrict fidy, real& restrict fidz,
   real& restrict fipx, real& restrict fipy, real& restrict fipz,
   real& restrict fkdx, real& restrict fkdy, real& restrict fkdz,
   real& restrict fkpx, real& restrict fkpy, real& restrict fkpz)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5, scale7;
   damp_thole3(r, pdi, pti, pdk, ptk, scale3, scale5, scale7);

   real bn[4];
   if CONSTEXPR (eq<ETYP, EWALD>()) damp_ewald<4>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;
   real rr7 = 15 * rr1 * rr2 * rr2 * rr2;

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

   real3 dixyz = make_real3(dix, diy, diz);
   real3 dkxyz = make_real3(dkx, dky, dkz);
   real3 qixyz = make_real3(qix, qiy, qiz);
   real3 qkxyz = make_real3(qkx, qky, qkz);
   real3 dr = make_real3(xr, yr, zr);
   real c1;
   real3 inci, inck;

   // d-field
   real bn1, bn2, bn3;
   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn1 = bn[1];
      bn2 = bn[2];
      bn3 = bn[3];
      bn[1] = bn1 - (1 - dscale * scale3) * rr3;
      bn[2] = bn2 - (1 - dscale * scale5) * rr5;
      bn[3] = bn3 - (1 - dscale * scale7) * rr7;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = dscale * scale3 * rr3;
      bn[2] = dscale * scale5 * rr5;
      bn[3] = dscale * scale7 * rr7;
   }

   c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
   inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;
   fidx += inci.x;
   fidy += inci.y;
   fidz += inci.z;

   c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
   inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   fkdx += inck.x;
   fkdy += inck.y;
   fkdz += inck.z;

   // p-field
   if (pscale != dscale) {
      if CONSTEXPR (eq<ETYP, EWALD>()) {
         bn[1] = bn1 - (1 - pscale * scale3) * rr3;
         bn[2] = bn2 - (1 - pscale * scale5) * rr5;
         bn[3] = bn3 - (1 - pscale * scale7) * rr7;
      } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
         bn[1] = pscale * scale3 * rr3;
         bn[2] = pscale * scale5 * rr5;
         bn[3] = pscale * scale7 * rr7;
      }

      c1 = -(bn[1] * ck - bn[2] * dkr + bn[3] * qkr);
      inci = c1 * dr - bn[1] * dkxyz + 2 * bn[2] * qkxyz;

      c1 = (bn[1] * ci + bn[2] * dir + bn[3] * qir);
      inck = c1 * dr - bn[1] * dixyz - 2 * bn[2] * qixyz;
   }
   fipx += inci.x;
   fipy += inci.y;
   fipz += inci.z;
   fkpx += inck.x;
   fkpy += inck.y;
   fkpz += inck.z;
}

#pragma acc routine seq
SEQ_CUDA
void pair_dfieldgk(real r2, real xr, real yr, real zr,
   real gkc, real fc, real fd, real fq,
   real ci, real dix, real diy, real diz, real qixx, real qixy, real qixz,
   real qiyy, real qiyz, real qizz, real rbi,
   real ck, real dkx, real dky, real dkz, real qkxx, real qkxy, real qkxz,
   real qkyy, real qkyz, real qkzz, real rbk,
   real& restrict fidsx, real& restrict fidsy, real& restrict fidsz,
   real& restrict fipsx, real& restrict fipsy, real& restrict fipsz,
   real& restrict fkdsx, real& restrict fkdsy, real& restrict fkdsz,
   real& restrict fkpsx, real& restrict fkpsy, real& restrict fkpsz)
{
   real r = REAL_SQRT(r2);
   real3 inci, inck;

   real a[4][3];
   real gc[4];
   real gux[10];
   real guy[10];
   real guz[10];
   real gqxx[4];
   real gqxy[4];
   real gqxz[4];
   real gqyy[4];
   real gqyz[4];
   real gqzz[4];
   real xr2 = xr * xr;
   real yr2 = yr * yr;
   real zr2 = zr * zr;
   
   real rb2 = rbi * rbk;
   real expterm = REAL_EXP(-r2/(gkc*rb2));
   real expc = expterm / gkc;
   real dexpc = -2.0 / (gkc*rb2);
   real gf2 = 1.0 / (r2+rb2*expterm);
   real gf = REAL_SQRT(gf2);
   real gf3 = gf2 * gf;
   real gf5 = gf3 * gf2;
   real gf7 = gf5 * gf2;
   
   a[0][0] = gf;
   a[1][0] = -gf3;
   a[2][0] = 3.0 * gf5;
   a[3][0] = -15.0 * gf7;

   real expc1 = 1.0 - expc;
   a[0][1] = expc1 * a[1][0];
   a[1][1] = expc1 * a[2][0];
   a[2][1] = expc1 * a[3][0];

   real expcdexpc = -expc * dexpc;
   a[1][2] = expc1*a[2][1] + expcdexpc*a[2][0];
   a[0][1] = fc * a[0][1];
   a[1][0] = fd * a[1][0];
   a[1][1] = fd * a[1][1];
   a[1][2] = fd * a[1][2];
   a[2][0] = fq * a[2][0];
   a[2][1] = fq * a[2][1];

   gux[0] = xr * a[1][0];
   guy[0] = yr * a[1][0];
   guz[0] = zr * a[1][0];

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

   gux[4] = xr * (3.0 * a[1][1] + xr2 * a[1][2]);
   gux[5] = yr * (a[1][1] + xr2 * a[1][2]);
   gux[6] = zr * (a[1][1] + xr2 * a[1][2]);
   gux[7] = xr * (a[1][1] + yr2 * a[1][2]);
   gux[8] = zr * xr * yr * a[1][2];
   gux[9] = xr * (a[1][1] + zr2 * a[1][2]);
   guy[4] = yr * (a[1][1] + xr2 * a[1][2]);
   guy[5] = xr * (a[1][1] + yr2 * a[1][2]);
   guy[6] = gux[8];
   guy[7] = yr * (3.0 * a[1][1] + yr2 * a[1][2]);
   guy[8] = zr * (a[1][1] + yr2 * a[1][2]);
   guy[9] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[4] = zr * (a[1][1] + xr2 * a[1][2]);
   guz[5] = gux[8];
   guz[6] = xr * (a[1][1] + zr2 * a[1][2]);
   guz[7] = zr * (a[1][1] + yr2 * a[1][2]);
   guz[8] = yr * (a[1][1] + zr2 * a[1][2]);
   guz[9] = zr * (3.0 * a[1][1] + zr2 * a[1][2]);

   inci.x = dkx*gux[1] + dky*gux[2] + dkz*gux[3]
         + 0.5 * (ck*gux[0] + qkxx*gux[4]
         + qkyy*gux[7] + qkzz*gux[9]
         + 2.0*(qkxy*gux[5]+qkxz*gux[6]
         +qkyz*gux[8]))
         + 0.5 * (ck*gc[1] + qkxx*gqxx[1]
         + qkyy*gqyy[1] + qkzz*gqzz[1]
         + 2.0*(qkxy*gqxy[1]+qkxz*gqxz[1]
         +qkyz*gqyz[1]));
         
   inci.y = dkx*guy[1] + dky*guy[2] + dkz*guy[3]
         + 0.5 * (ck*guy[0] + qkxx*guy[4]
         + qkyy*guy[7] + qkzz*guy[9]
         + 2.0*(qkxy*guy[5]+qkxz*guy[6]
         +qkyz*guy[8]))
         + 0.5 * (ck*gc[2] + qkxx*gqxx[2]
         + qkyy*gqyy[2] + qkzz*gqzz[2]
         + 2.0*(qkxy*gqxy[2]+qkxz*gqxz[2]
         +qkyz*gqyz[2]));
         
   inci.z = dkx*guz[1] + dky*guz[2] + dkz*guz[3]
         + 0.5 * (ck*guz[0] + qkxx*guz[4]
         + qkyy*guz[7] + qkzz*guz[9]
         + 2.0*(qkxy*guz[5]+qkxz*guz[6]
         +qkyz*guz[8]))
         + 0.5 * (ck*gc[3] + qkxx*gqxx[3]
         + qkyy*gqyy[3] + qkzz*gqzz[3]
         + 2.0*(qkxy*gqxy[3]+qkxz*gqxz[3]
         +qkyz*gqyz[3]));

   inck.x = dix*gux[1] + diy*gux[2] + diz*gux[3]
         - 0.5 * (ci*gux[0] + qixx*gux[4]
         + qiyy*gux[7] + qizz*gux[9]
         + 2.0*(qixy*gux[5]+qixz*gux[6]
         +qiyz*gux[8]))
         - 0.5 * (ci*gc[1] + qixx*gqxx[1]
         + qiyy*gqyy[1] + qizz*gqzz[1]
         + 2.0*(qixy*gqxy[1]+qixz*gqxz[1]
         +qiyz*gqyz[1]));
         
   inck.y = dix*guy[1] + diy*guy[2] + diz*guy[3]
         - 0.5 * (ci*guy[0] + qixx*guy[4]
         + qiyy*guy[7] + qizz*guy[9]
         + 2.0*(qixy*guy[5]+qixz*guy[6]
         +qiyz*guy[8]))
         - 0.5 * (ci*gc[2] + qixx*gqxx[2]
         + qiyy*gqyy[2] + qizz*gqzz[2]
         + 2.0*(qixy*gqxy[2]+qixz*gqxz[2]
         +qiyz*gqyz[2]));
         
   inck.z = dix*guz[1] + diy*guz[2] + diz*guz[3]
         - 0.5 * (ci*guz[0] + qixx*guz[4]
         + qiyy*guz[7] + qizz*guz[9]
         + 2.0*(qixy*guz[5]+qixz*guz[6]
         +qiyz*guz[8]))
         - 0.5 * (ci*gc[3] + qixx*gqxx[3]
         + qiyy*gqyy[3] + qizz*gqzz[3]
         + 2.0*(qixy*gqxy[3]+qixz*gqxz[3]
         +qiyz*gqyz[3]));

   fidsx += inci.x;
   fidsy += inci.y;
   fidsz += inci.z;
   fkdsx += inck.x;
   fkdsy += inck.y;
   fkdsz += inck.z;

   fipsx += inci.x;
   fipsy += inci.y;
   fipsz += inci.z;
   fkpsx += inck.x;
   fkpsy += inck.y;
   fkpsz += inck.z;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_ufield(real r2, real xr, real yr, real zr, real uscale, //
   real uindi0, real uindi1, real uindi2, real uinpi0, real uinpi1, real uinpi2,
   real pdi,
   real pti, //
   real uindk0, real uindk1, real uindk2, real uinpk0, real uinpk1, real uinpk2,
   real pdk,
   real ptk, //
   real aewald, real3& restrict fid, real3& restrict fip, real3& restrict fkd,
   real3& restrict fkp)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   if CONSTEXPR (eq<ETYP, EWALD>()) damp_ewald<3>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - scale3) * rr3;
      bn[2] -= (1 - scale5) * rr5;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = uscale * scale3 * rr3;
      bn[2] = uscale * scale5 * rr5;
   }

   real coef;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 uip = make_real3(uinpi0, uinpi1, uinpi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);
   real3 ukp = make_real3(uinpk0, uinpk1, uinpk2);

   coef = bn[2] * dot3(dr, ukd);
   fid += coef * dr - bn[1] * ukd;

   coef = bn[2] * dot3(dr, ukp);
   fip += coef * dr - bn[1] * ukp;

   coef = bn[2] * dot3(dr, uid);
   fkd += coef * dr - bn[1] * uid;

   coef = bn[2] * dot3(dr, uip);
   fkp += coef * dr - bn[1] * uip;
}

#pragma acc routine seq
template <class ETYP>
SEQ_CUDA
void pair_ufield_v2(real r2, real xr, real yr, real zr, real uscale,
   real aewald, //
   real uindi0, real uindi1, real uindi2, real uinpi0, real uinpi1, real uinpi2,
   real pdi,
   real pti, //
   real uindk0, real uindk1, real uindk2, real uinpk0, real uinpk1, real uinpk2,
   real pdk,
   real ptk, //
   real& restrict fidx, real& restrict fidy, real& restrict fidz,
   real& restrict fipx, real& restrict fipy, real& restrict fipz,
   real& restrict fkdx, real& restrict fkdy, real& restrict fkdz,
   real& restrict fkpx, real& restrict fkpy, real& restrict fkpz)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   if CONSTEXPR (eq<ETYP, EWALD>()) damp_ewald<3>(bn, r, invr1, rr2, aewald);
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   if CONSTEXPR (eq<ETYP, EWALD>()) {
      bn[1] -= (1 - uscale * scale3) * rr3;
      bn[2] -= (1 - uscale * scale5) * rr5;
   } else if CONSTEXPR (eq<ETYP, NON_EWALD>()) {
      bn[1] = uscale * scale3 * rr3;
      bn[2] = uscale * scale5 * rr5;
   }

   real coef;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 uip = make_real3(uinpi0, uinpi1, uinpi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);
   real3 ukp = make_real3(uinpk0, uinpk1, uinpk2);

   coef = bn[2] * dot3(dr, ukd);
   real3 fid = coef * dr - bn[1] * ukd;
   fidx += fid.x;
   fidy += fid.y;
   fidz += fid.z;

   coef = bn[2] * dot3(dr, ukp);
   real3 fip = coef * dr - bn[1] * ukp;
   fipx += fip.x;
   fipy += fip.y;
   fipz += fip.z;

   coef = bn[2] * dot3(dr, uid);
   real3 fkd = coef * dr - bn[1] * uid;
   fkdx += fkd.x;
   fkdy += fkd.y;
   fkdz += fkd.z;

   coef = bn[2] * dot3(dr, uip);
   real3 fkp = coef * dr - bn[1] * uip;
   fkpx += fkp.x;
   fkpy += fkp.y;
   fkpz += fkp.z;
}

#pragma acc routine seq
SEQ_CUDA
void pair_ufieldgk1(real r2, real xr, real yr, real zr, real uscale,
   real uindi0, real uindi1, real uindi2, real uinpi0, real uinpi1, real uinpi2,
   real uindsi0, real uindsi1, real uindsi2, real uinpsi0, real uinpsi1, real uinpsi2,
   real pdi,
   real pti, //
   real uindk0, real uindk1, real uindk2, real uinpk0, real uinpk1, real uinpk2,
   real uindsk0, real uindsk1, real uindsk2, real uinpsk0, real uinpsk1, real uinpsk2,
   real pdk,
   real ptk, //
   real& restrict fidx, real& restrict fidy, real& restrict fidz,
   real& restrict fipx, real& restrict fipy, real& restrict fipz,
   real& restrict fidsx, real& restrict fidsy, real& restrict fidsz,
   real& restrict fipsx, real& restrict fipsy, real& restrict fipsz,
   real& restrict fkdx, real& restrict fkdy, real& restrict fkdz,
   real& restrict fkpx, real& restrict fkpy, real& restrict fkpz,
   real& restrict fkdsx, real& restrict fkdsy, real& restrict fkdsz,
   real& restrict fkpsx, real& restrict fkpsy, real& restrict fkpsz)
{
   real r = REAL_SQRT(r2);
   real invr1 = REAL_RECIP(r);
   real rr2 = invr1 * invr1;

   real scale3, scale5;
   damp_thole2(r, pdi, pti, pdk, ptk, scale3, scale5);

   real bn[3];
   real rr1 = invr1;
   real rr3 = rr1 * rr2;
   real rr5 = 3 * rr1 * rr2 * rr2;

   bn[1] = uscale * scale3 * rr3;
   bn[2] = uscale * scale5 * rr5;

   real coef;
   real3 dr = make_real3(xr, yr, zr);
   real3 uid = make_real3(uindi0, uindi1, uindi2);
   real3 uip = make_real3(uinpi0, uinpi1, uinpi2);
   real3 ukd = make_real3(uindk0, uindk1, uindk2);
   real3 ukp = make_real3(uinpk0, uinpk1, uinpk2);
   
   real3 uids = make_real3(uindsi0, uindsi1, uindsi2);
   real3 uips = make_real3(uinpsi0, uinpsi1, uinpsi2);
   real3 ukds = make_real3(uindsk0, uindsk1, uindsk2);
   real3 ukps = make_real3(uinpsk0, uinpsk1, uinpsk2);

   coef = bn[2] * dot3(dr, ukd);
   real3 fid = coef * dr - bn[1] * ukd;
   fidx += fid.x;
   fidy += fid.y;
   fidz += fid.z;

   coef = bn[2] * dot3(dr, ukp);
   real3 fip = coef * dr - bn[1] * ukp;
   fipx += fip.x;
   fipy += fip.y;
   fipz += fip.z;

   coef = bn[2] * dot3(dr, uid);
   real3 fkd = coef * dr - bn[1] * uid;
   fkdx += fkd.x;
   fkdy += fkd.y;
   fkdz += fkd.z;

   coef = bn[2] * dot3(dr, uip);
   real3 fkp = coef * dr - bn[1] * uip;
   fkpx += fkp.x;
   fkpy += fkp.y;
   fkpz += fkp.z;

   coef = bn[2] * dot3(dr, ukds);
   real3 fids = coef * dr - bn[1] * ukds;
   fidsx += fids.x;
   fidsy += fids.y;
   fidsz += fids.z;

   coef = bn[2] * dot3(dr, ukps);
   real3 fips = coef * dr - bn[1] * ukps;
   fipsx += fips.x;
   fipsy += fips.y;
   fipsz += fips.z;

   coef = bn[2] * dot3(dr, uids);
   real3 fkds = coef * dr - bn[1] * uids;
   fkdsx += fkds.x;
   fkdsy += fkds.y;
   fkdsz += fkds.z;

   coef = bn[2] * dot3(dr, uips);
   real3 fkps = coef * dr - bn[1] * uips;
   fkpsx += fkps.x;
   fkpsy += fkps.y;
   fkpsz += fkps.z;
}

#pragma acc routine seq
SEQ_CUDA
void pair_ufieldgk2(real r2, real xr, real yr, real zr,
   real gkc, real fd,
   real uidsx, real uidsy, real uidsz, real uipsx, real uipsy, real uipsz,
   real rbi, //
   real ukdsx, real ukdsy, real ukdsz, real ukpsx, real ukpsy, real ukpsz,
   real rbk, //
   real& restrict fidsx, real& restrict fidsy, real& restrict fidsz,
   real& restrict fipsx, real& restrict fipsy, real& restrict fipsz,
   real& restrict fkdsx, real& restrict fkdsy, real& restrict fkdsz,
   real& restrict fkpsx, real& restrict fkpsy, real& restrict fkpsz)
{
   real r = REAL_SQRT(r2);

   real gux[3];
   real guy[3];
   real guz[3];
   real xr2 = xr * xr;
   real yr2 = yr * yr;
   real zr2 = zr * zr;
   
   real rb2 = rbi * rbk;
   real expterm = REAL_EXP(-r2/(gkc*rb2));
   real expc = expterm / gkc;
   real gf2 = 1.0 / (r2+rb2*expterm);
   real gf = REAL_SQRT(gf2);
   real gf3 = gf2 * gf;
   real gf5 = gf3 * gf2;
   
   real a10 = -gf3;
   real a20 = 3.0 * gf5;

   real expc1 = 1.0 - expc;
   real a11 = expc1 * a20;

   gux[0] = fd * (a10 + xr2 * a11);
   gux[1] = fd * xr * yr * a11;
   gux[2] = fd * xr * zr * a11;
   guy[0] = gux[1];
   guy[1] = fd * (a10 + yr2 * a11);
   guy[2] = fd * yr * zr * a11;
   guz[0] = gux[2];
   guz[1] = guy[2];
   guz[2] = fd * (a10 + zr2 * a11);

   fidsx += ukdsx * gux[0] + ukdsy * guy[0] + ukdsz * guz[0];
   fidsy += ukdsx * gux[1] + ukdsy * guy[1] + ukdsz * guz[1];
   fidsz += ukdsx * gux[2] + ukdsy * guy[2] + ukdsz * guz[2];
   fkdsx += uidsx * gux[0] + uidsy * guy[0] + uidsz * guz[0];
   fkdsy += uidsx * gux[1] + uidsy * guy[1] + uidsz * guz[1];
   fkdsz += uidsx * gux[2] + uidsy * guy[2] + uidsz * guz[2];
   fipsx += ukpsx * gux[0] + ukpsy * guy[0] + ukpsz * guz[0];
   fipsy += ukpsx * gux[1] + ukpsy * guy[1] + ukpsz * guz[1];
   fipsz += ukpsx * gux[2] + ukpsy * guy[2] + ukpsz * guz[2];
   fkpsx += uipsx * gux[0] + uipsy * guy[0] + uipsz * guz[0];
   fkpsy += uipsx * gux[1] + uipsy * guy[1] + uipsz * guz[1];
   fkpsz += uipsx * gux[2] + uipsy * guy[2] + uipsz * guz[2];
}
}
