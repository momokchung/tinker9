#include "ff/modamoeba.h"
#include "ff/atom.h"
#include "ff/nblist.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/pair_fieldgk.h"
#include "tool/gpucard.h"

namespace tinker {
#define DFIELDGK_DPTRS x, y, z, rborn, rpole, fields, fieldps
void dfieldgk_acc(real gkc, real fc, real fd, real fq, real (*fields)[3], real (*fieldps)[3])
{
   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(DFIELDGK_DPTRS,mlst)
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
      real rbi = rborn[i];
      real fieldsi0 = 0, fieldsi1 = 0, fieldsi2 = 0;
      real fieldpsi0 = 0, fieldpsi1 = 0, fieldpsi2 = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent\
                  reduction(+:fieldsi0,fieldsi1,fieldsi2,fieldpsi0,fieldpsi1,fieldpsi2)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real fidsx = 0, fidsy = 0, fidsz = 0;
            real fipsx = 0, fipsy = 0, fipsz = 0;
            real fkdsx = 0, fkdsy = 0, fkdsz = 0;
            real fkpsx = 0, fkpsy = 0, fkpsz = 0;
            pair_dfieldgk(r2, xr, yr, zr, gkc, fc, fd, fq,
               ci, dix, diy, diz, qixx, qixy, qixz, qiyy, qiyz, qizz, rbi,
               rpole[k][MPL_PME_0], rpole[k][MPL_PME_X], rpole[k][MPL_PME_Y],
               rpole[k][MPL_PME_Z], rpole[k][MPL_PME_XX], rpole[k][MPL_PME_XY],
               rpole[k][MPL_PME_XZ], rpole[k][MPL_PME_YY], rpole[k][MPL_PME_YZ],
               rpole[k][MPL_PME_ZZ], rborn[k],
               fidsx, fidsy, fidsz, fipsx, fipsy, fipsz,
               fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);

            fieldsi0 += fidsx;
            fieldsi1 += fidsy;
            fieldsi2 += fidsz;
            fieldpsi0 += fipsx;
            fieldpsi1 += fipsy;
            fieldpsi2 += fipsz;

            atomic_add(fkdsx, &fields[k][0]);
            atomic_add(fkdsy, &fields[k][1]);
            atomic_add(fkdsz, &fields[k][2]);
            atomic_add(fkpsx, &fieldps[k][0]);
            atomic_add(fkpsy, &fieldps[k][1]);
            atomic_add(fkpsz, &fieldps[k][2]);
         }
      } // end for (int kk)

      atomic_add(fieldsi0, &fields[i][0]);
      atomic_add(fieldsi1, &fields[i][1]);
      atomic_add(fieldsi2, &fields[i][2]);
      atomic_add(fieldpsi0, &fieldps[i][0]);
      atomic_add(fieldpsi1, &fieldps[i][1]);
      atomic_add(fieldpsi2, &fieldps[i][2]);
   } // end for (int i)

   #pragma acc parallel loop independent async\
               deviceptr(rborn, rpole, fields, fieldps)
   for (int i = 0; i < n; ++i) {
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real rbi = rborn[i];

      real rb2 = rbi * rbi;
      real gf2 = 1 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real a10 = -fd * gf3;

      real fx = dix*a10;
      real fy = diy*a10;
      real fz = diz*a10;

      atomic_add(fx, &fields[i][0]);
      atomic_add(fy, &fields[i][1]);
      atomic_add(fz, &fields[i][2]);
      atomic_add(fx, &fieldps[i][0]);
      atomic_add(fy, &fieldps[i][1]);
      atomic_add(fz, &fieldps[i][2]);
   }
}

#define UFIELDGK_DPTRS1 x, y, z, thole, pdamp, jpolar, thlval, uinds, uinps, fields, fieldps
#define UFIELDGK_DPTRS2 x, y, z, rborn, uinds, uinps, fields, fieldps
void ufieldgk_acc(real gkc, real fd, const real (*uinds)[3], const real (*uinps)[3], real (*fields)[3], real (*fieldps)[3])
{
   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   darray::zero(g::q0, n, fields, fieldps);

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(UFIELDGK_DPTRS1,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uidsx = uinds[i][0];
      real uidsy = uinds[i][1];
      real uidsz = uinds[i][2];
      real uipsx = uinps[i][0];
      real uipsy = uinps[i][1];
      real uipsz = uinps[i][2];
      real pdi = pdamp[i];
      int jpi = jpolar[i];

      real fieldsi0 = 0, fieldsi1 = 0, fieldsi2 = 0;
      real fieldpsi0 = 0, fieldpsi1 = 0, fieldpsi2 = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent\
                  reduction(+:fieldsi0,fieldsi1,fieldsi2,fieldpsi0,fieldpsi1,fieldpsi2)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real ukdsx = uinds[k][0];
            real ukdsy = uinds[k][1];
            real ukdsz = uinds[k][2];
            real ukpsx = uinps[k][0];
            real ukpsy = uinps[k][1];
            real ukpsz = uinps[k][2];
            real pdk = pdamp[k];
            int jpk = jpolar[k];
            real pga = thlval[njpolar * jpi + jpk];

            real fidsx = 0, fidsy = 0, fidsz = 0;
            real fipsx = 0, fipsy = 0, fipsz = 0;
            real fkdsx = 0, fkdsy = 0, fkdsz = 0;
            real fkpsx = 0, fkpsy = 0, fkpsz = 0;
            pair_ufieldgk1(r2, xr, yr, zr, 1,
               uidsx, uidsy, uidsz, uipsx, uipsy, uipsz, pdi, pga,
               ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk, pga,
               fidsx, fidsy, fidsz, fipsx, fipsy, fipsz,
               fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);

            fieldsi0 += fidsx;
            fieldsi1 += fidsy;
            fieldsi2 += fidsz;
            fieldpsi0 += fipsx;
            fieldpsi1 += fipsy;
            fieldpsi2 += fipsz;

            atomic_add(fkdsx, &fields[k][0]);
            atomic_add(fkdsy, &fields[k][1]);
            atomic_add(fkdsz, &fields[k][2]);
            atomic_add(fkpsx, &fieldps[k][0]);
            atomic_add(fkpsy, &fieldps[k][1]);
            atomic_add(fkpsz, &fieldps[k][2]);
         }
      } // end for (int kk)

      atomic_add(fieldsi0, &fields[i][0]);
      atomic_add(fieldsi1, &fields[i][1]);
      atomic_add(fieldsi2, &fields[i][2]);
      atomic_add(fieldpsi0, &fieldps[i][0]);
      atomic_add(fieldpsi1, &fieldps[i][1]);
      atomic_add(fieldpsi2, &fieldps[i][2]);
   } // end for (int i)

   #pragma acc parallel async\
               deviceptr(UFIELDGK_DPTRS1,uexclude,uexclude_scale)
   #pragma acc loop independent
   for (int ii = 0; ii < nuexclude; ++ii) {
      int i = uexclude[ii][0];
      int k = uexclude[ii][1];
      real uscale = uexclude_scale[ii] - 1;

      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uidsx = uinds[i][0];
      real uidsy = uinds[i][1];
      real uidsz = uinds[i][2];
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
         real ukdsx = uinds[k][0];
         real ukdsy = uinds[k][1];
         real ukdsz = uinds[k][2];
         real ukpsx = uinps[k][0];
         real ukpsy = uinps[k][1];
         real ukpsz = uinps[k][2];
         real pdk = pdamp[k];
         int jpk = jpolar[k];
         real pga = thlval[njpolar * jpi + jpk];

         real fidsx = 0, fidsy = 0, fidsz = 0;
         real fipsx = 0, fipsy = 0, fipsz = 0;
         real fkdsx = 0, fkdsy = 0, fkdsz = 0;
         real fkpsx = 0, fkpsy = 0, fkpsz = 0;
         pair_ufieldgk1(r2, xr, yr, zr, uscale,
            uidsx, uidsy, uidsz, uipsx, uipsy, uipsz, pdi, pga,
            ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk, pga,
            fidsx, fidsy, fidsz, fipsx, fipsy, fipsz,
            fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);

         atomic_add(fidsx, &fields[i][0]);
         atomic_add(fidsy, &fields[i][1]);
         atomic_add(fidsz, &fields[i][2]);
         atomic_add(fipsx, &fieldps[i][0]);
         atomic_add(fipsy, &fieldps[i][1]);
         atomic_add(fipsz, &fieldps[i][2]);

         atomic_add(fkdsx, &fields[k][0]);
         atomic_add(fkdsy, &fields[k][1]);
         atomic_add(fkdsz, &fields[k][2]);
         atomic_add(fkpsx, &fieldps[k][0]);
         atomic_add(fkpsy, &fieldps[k][1]);
         atomic_add(fkpsz, &fieldps[k][2]);
      }
   }

   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(UFIELDGK_DPTRS2,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real uidsx = uinds[i][0];
      real uidsy = uinds[i][1];
      real uidsz = uinds[i][2];
      real uipsx = uinps[i][0];
      real uipsy = uinps[i][1];
      real uipsz = uinps[i][2];
      real rbi = rborn[i];

      real fieldsi0 = 0, fieldsi1 = 0, fieldsi2 = 0;
      real fieldpsi0 = 0, fieldpsi1 = 0, fieldpsi2 = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent\
                  reduction(+:fieldsi0,fieldsi1,fieldsi2,fieldpsi0,fieldpsi1,fieldpsi2)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real ukdsx = uinds[k][0];
            real ukdsy = uinds[k][1];
            real ukdsz = uinds[k][2];
            real ukpsx = uinps[k][0];
            real ukpsy = uinps[k][1];
            real ukpsz = uinps[k][2];
            real rbk = rborn[k];

            real fidsx = 0, fidsy = 0, fidsz = 0;
            real fipsx = 0, fipsy = 0, fipsz = 0;
            real fkdsx = 0, fkdsy = 0, fkdsz = 0;
            real fkpsx = 0, fkpsy = 0, fkpsz = 0;
            pair_ufieldgk2(r2, xr, yr, zr, gkc, fd,
               uidsx, uidsy, uidsz, uipsx, uipsy, uipsz, rbi,
               ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, rbk,
               fidsx, fidsy, fidsz, fipsx, fipsy, fipsz, fkdsx,
               fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);

            fieldsi0 += fidsx;
            fieldsi1 += fidsy;
            fieldsi2 += fidsz;
            fieldpsi0 += fipsx;
            fieldpsi1 += fipsy;
            fieldpsi2 += fipsz;

            atomic_add(fkdsx, &fields[k][0]);
            atomic_add(fkdsy, &fields[k][1]);
            atomic_add(fkdsz, &fields[k][2]);
            atomic_add(fkpsx, &fieldps[k][0]);
            atomic_add(fkpsy, &fieldps[k][1]);
            atomic_add(fkpsz, &fieldps[k][2]);
         }
      } // end for (int kk)

      atomic_add(fieldsi0, &fields[i][0]);
      atomic_add(fieldsi1, &fields[i][1]);
      atomic_add(fieldsi2, &fields[i][2]);
      atomic_add(fieldpsi0, &fieldps[i][0]);
      atomic_add(fieldpsi1, &fieldps[i][1]);
      atomic_add(fieldpsi2, &fieldps[i][2]);
   } // end for (int i)

   #pragma acc parallel loop independent async\
               deviceptr(rborn, uinds, uinps, fields, fieldps)
   for (int i = 0; i < n; ++i) {
      real duixs = uinds[i][0];
      real duiys = uinds[i][1];
      real duizs = uinds[i][2];
      real puixs = uinps[i][0];
      real puiys = uinps[i][1];
      real puizs = uinps[i][2];
      real rbi = rborn[i];

      real rb2 = rbi * rbi;
      real gf2 = 1 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real a10 = -gf3;
      real gu = fd * a10;
      atomic_add(duixs*gu, &fields[i][0]);
      atomic_add(duiys*gu, &fields[i][1]);
      atomic_add(duizs*gu, &fields[i][2]);
      atomic_add(puixs*gu, &fieldps[i][0]);
      atomic_add(puiys*gu, &fieldps[i][1]);
      atomic_add(puizs*gu, &fieldps[i][2]);
   }
}
}