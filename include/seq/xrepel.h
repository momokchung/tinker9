#pragma once
#include "seq/rotpole.h"

namespace tinker {
SEQ_ROUTINE
inline void scoeffAtomI(int i, const real (*restrict xrepole)[MPL_TOTAL],
   const real* restrict crpxr, real (*restrict cpxr)[4])
{
   int ind1,ind2,ind3;
   real cr,cs;
   real p2p1,p3p1;
   real pcoeff[3];
   real ppole[3];
   bool l1,l2,l3;

   // determine pseudo orbital coefficients
   for (int k = 0; k < 3; ++k) {
      pcoeff[k] = 0.;
   }
   ppole[0] = xrepole[i][1];
   ppole[1] = xrepole[i][2];
   ppole[2] = xrepole[i][3];
   cr = crpxr[i];
   l1 = (abs(ppole[0]) < 1e-10);
   l2 = (abs(ppole[1]) < 1e-10);
   l3 = (abs(ppole[2]) < 1e-10);

   // case for no dipole
   if (l1 and l2 and l3) {
      cs = 1.;
      ind1 = 0;
      ind2 = 1;
      ind3 = 2;
   }
   // case for p orbital coefficients set to 0
   else if (cr < 1e-10) {
      cs = 1.;
      ind1 = 0;
      ind2 = 1;
      ind3 = 2;
   }
   // case for anisotropic repulsion
   else {
      // determine normalized coefficients
      cs = 1. / REAL_SQRT(1. + cr);
      // determine index for largest absolute dipole component
      ind1 = 0;
      for (int k = 1; k < 3; ++k) {
         if (abs(ppole[k]) > abs(ppole[ind1])) {
            ind1 = k;
         }
      }
      ind2 = (ind1+1) % 3;
      ind3 = (ind1+2) % 3;
      p2p1 = ppole[ind2] / ppole[ind1];
      p3p1 = ppole[ind3] / ppole[ind1];
      pcoeff[ind1] = cs * REAL_SQRT(cr / (1. + p2p1*p2p1 + p3p1*p3p1));
      if (ppole[ind1] < 0.) {
         pcoeff[ind1] = -pcoeff[ind1];
      }
      pcoeff[ind2] = pcoeff[ind1] * p2p1;
      pcoeff[ind3] = pcoeff[ind1] * p3p1;
   }
   cpxr[i][0] = cs;
   cpxr[i][ind1+1] = pcoeff[ind1];
   cpxr[i][ind2+1] = pcoeff[ind2];
   cpxr[i][ind3+1] = pcoeff[ind3];
}

SEQ_ROUTINE
inline void rotcoeffAtomI(int i, const LocalFrame* restrict zaxis,
   const real* restrict x, const real* restrict y, const real* restrict z,
   const real (*restrict cpxr)[4], real (*restrict rcpxr)[4])
{
   // rotmat routine
   real xi = x[i];
   real yi = y[i];
   real zi = z[i];
   int iz = zaxis[i].zaxis;
   int ix = zaxis[i].xaxis;
   int iy = INT_ABS(zaxis[i].yaxis) - 1;
   int polaxe = zaxis[i].polaxe;
   // the default identity matrix
   real a[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};

   if (polaxe != LFRM_NONE) {
      real* restrict xx = &a[0][0];
      real* restrict yy = &a[1][0];
      real* restrict zz = &a[2][0];

      // STEP 1: PICK Z AND NORM Z
      // pick z
      zz[0] = x[iz] - xi;
      zz[1] = y[iz] - yi;
      zz[2] = z[iz] - zi;
      // norm z
      rotpoleNorm(zz);

      // STEP 2: PICK X AND NORM X
      // even if it is not needef for z then x)
      if (polaxe == LFRM_Z_ONLY) {
         // pick x
         int okay = !(REAL_ABS(zz[0]) > 0.866f);
         xx[0] = (okay ? 1 : 0);
         xx[1] = (okay ? 0 : 1);
         xx[2] = 0;
      } else {
         // pick x
         xx[0] = x[ix] - xi;
         xx[1] = y[ix] - yi;
         xx[2] = z[ix] - zi;
         rotpoleNorm(xx);
      }

      // STEP 3: PICK Y AND NORM Y
      // only for z biscector and 3 fold
      if (polaxe == LFRM_Z_BISECT || polaxe == LFRM_3_FOLD) {
         yy[0] = x[iy] - xi;
         yy[1] = y[iy] - yi;
         yy[2] = z[iy] - zi;
         rotpoleNorm(yy);
      }

      // STEP 4
      if (polaxe == LFRM_BISECTOR) {
         rotpoleAddBy(zz, xx);
         rotpoleNorm(zz);
      } else if (polaxe == LFRM_Z_BISECT) {
         rotpoleAddBy(xx, yy);
         rotpoleNorm(xx);
      } else if (polaxe == LFRM_3_FOLD) {
         rotpoleAddBy2(zz, xx, yy);
         rotpoleNorm(zz);
      }

      // STEP 5
      // x -= (x.z) z
      real dotxz = xx[0] * zz[0] + xx[1] * zz[1] + xx[2] * zz[2];
      xx[0] -= dotxz * zz[0];
      xx[1] -= dotxz * zz[1];
      xx[2] -= dotxz * zz[2];
      // norm x
      rotpoleNorm(xx);
      // y = z cross x
      a[1][0] = a[0][2] * a[2][1] - a[0][1] * a[2][2];
      a[1][1] = a[0][0] * a[2][2] - a[0][2] * a[2][0];
      a[1][2] = a[0][1] * a[2][0] - a[0][0] * a[2][1];
   } // end if (.not. LFRM_NONE)

   // rotsite for orbitals
   // s coefficients same in all frames
   rcpxr[i][0] = cpxr[i][0];

   // rotate p coefficients to global frame
   rcpxr[i][1] = 0;
   rcpxr[i][2] = 0;
   rcpxr[i][3] = 0;
#if _OPENACC
#pragma acc loop seq collapse(2)
#endif
   for (int j = 1; j < 4; ++j)
      for (int k = 1; k < 4; ++k)
         rcpxr[i][j] += cpxr[i][k] * a[k - 1][j - 1];
}
}
