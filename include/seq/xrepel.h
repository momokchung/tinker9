#pragma once
#include "math/realn.h"
#include "math/switch.h"
#include "seq/pair_vlambda.h"
#include "seq/rotpole.h"
#include <tinker/detail/units.hh>

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
      if (abs(ppole[1]) > abs(ppole[ind1])) ind1 = 1;
      if (abs(ppole[2]) > abs(ppole[ind1])) ind1 = 2;
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

struct PairRepelGrad
{
   real frcx, frcy, frcz;
   real ttqi[3];
   real ttqk[3];
};

SEQ_ROUTINE
inline void zero(PairRepelGrad& pgrad)
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

SEQ_ROUTINE
inline void computeOverlap(real a, real b, real r, bool grad,
                           real& SS, real& dSS, real& SPz, real& dSPz, real& PzS, real& dPzS,
                           real& PxPx, real& dPxPx, real& PyPy, real& dPyPy, real& PzPz, real& dPzPz)
{
   real diff = abs(a - b);
   real eps = 0.001;
   real rho,rho2,rho3,rho4;
   real exp1;
   real alpha,tau,tau2;
   real rhoA,rhoB;
   real a2,b2,kappa;
   real pre,pre1,pre2,pre3;
   real term1,term2;
   real rhoA2,rhoA3,rhoA4,rhoA5;
   real rhoB2,rhoB3,rhoB4,rhoB5;
   real kappam,kappam2;
   real kappap,kappap2;
   real taurho,taurho2,taurho3;
   real expA,expB;
   if (diff < eps) {
      rho = a * r;
      rho2 = rho * rho;
      rho3 = rho2 * rho;
      rho4 = rho3 * rho;
      exp1 = REAL_EXP(-rho);
      SS = (1. + rho + rho2 / 3.) * exp1;
      SPz = -0.5 * rho * (1. + rho + rho2 / 3.) * exp1;
      PzS = -SPz;
      PxPx = (1. + rho + 2./5. * rho2 + rho3/15.) * exp1;
      PzPz = -(-1. - rho - rho2 / 5. + 2./15. * rho3 + rho4 / 15.) * exp1;
      if CONSTEXPR (grad) {
         dSS = -1./3. * a * rho * (1. + rho) * exp1;
         dSPz = -0.5 * a * (1. + rho - rho3 / 3.) * exp1;
         dPzS = -dSPz;
         dPxPx = -0.2 * a * rho * (1. + rho + rho2 / 3.) * exp1;
         dPzPz = -0.6 * a * rho * (1. + rho + 2./9. * rho2 - rho3 / 9.) * exp1;
      }
   }
   else {
      alpha = 1. / 2. * (a + b);
      tau = (a - b) / (a + b);
      tau2 = tau * tau;
      rho = alpha * r;
      rhoA = a * r;
      rhoB = b * r;
      a2 = a * a;
      b2 = b * b;
      kappa = (a2 + b2) / (a2 - b2);
      rho2 = rho * rho;
      rho3 = rho2 * rho;
      rhoA2 = rhoA * rhoA;
      rhoA3 = rhoA2 * rhoA;
      rhoA4 = rhoA3 * rhoA;
      rhoB2 = rhoB * rhoB;
      rhoB3 = rhoB2 * rhoB;
      rhoB4 = rhoB3 * rhoB;
      kappam = 1. - kappa;
      kappap = 1. + kappa;
      kappam2 = kappam*kappam;
      kappap2 = kappap*kappap;
      taurho = tau * rho;
      taurho2 = taurho * rho;
      taurho3 = taurho2 * rho;
      expA = REAL_EXP(-rhoA);
      expB = REAL_EXP(-rhoB);
      pre1 = REAL_SQRT(1. - tau2);
      pre2 = REAL_SQRT((1. + tau) / (1. - tau));
      pre3 = REAL_SQRT((1. - tau) / (1. + tau));
      pre = pre1 / taurho;
      term1 =-kappam * (2. * kappap + rhoA) * expA;
      term2 = kappap * (2. * kappam + rhoB) * expB;
      SS = pre * (term1 + term2);
      pre = pre2 / taurho2;
      term1 =-kappam2 * (6. * kappap * (1. + rhoA) + 2. * rhoA2) * expA;
      term2 = kappap * (6. * kappam2 * (1. + rhoB) + 4. * kappam * rhoB2 + rhoB3) * expB;
      SPz = -pre * (term1 + term2);
      pre = -pre3 / taurho2;
      term1 =-kappap2 * (6. * kappam * (1. + rhoB) + 2. * rhoB2) * expB;
      term2 = kappam * (6. * kappap2 * (1. + rhoA) + 4. * kappap * rhoA2 + rhoA3) * expA;
      PzS = pre * (term1 + term2);
      pre = 1. / (pre1 * taurho3);
      term1 =-kappam2 * (24. * kappap2 * (1. + rhoA) + 12. * kappap * rhoA2 + 2. * rhoA3) * expA;
      term2 = kappap2 * (24. * kappam2 * (1. + rhoB) + 12. * kappam * rhoB2 + 2. * rhoB3) * expB;
      PxPx = pre * (term1 + term2);
      term1 =-kappam2 * (48. * kappap2 * (1. + rhoA + 0.5 * rhoA2) + 2. * (5. + 6. * kappa) * rhoA3 + 2. * rhoA4) * expA;
      term2 = kappap2 * (48. * kappam2 * (1. + rhoB + 0.5 * rhoB2) + 2. * (5. - 6. * kappa) * rhoB3 + 2. * rhoB4) * expB;
      PzPz = -pre * (term1 + term2);
      if CONSTEXPR (grad) {
         rhoA5 = rhoA4 * rhoA;
         rhoB5 = rhoB4 * rhoB;
         pre = pre1 / taurho;
         term1 = kappam * (2. * kappap * (1. + rhoA) + rhoA2) * expA;
         term2 =-kappap * (2. * kappam * (1. + rhoB) + rhoB2) * expB;
         dSS = pre / r * (term1 + term2);
         pre = pre2 / taurho2;
         term1 = 2. * kappam2 * (6. * kappap * (1. + rhoA + 0.5 * rhoA2) + rhoA3) * expA;
         term2 = kappap * (-12. * kappam2 * (1. + rhoB + 0.5 * rhoB2) + (1. - 4. * kappam) * rhoB3 - rhoB4) * expB;
         dSPz = -pre / r * (term1 + term2);
         pre = -pre3 / taurho2;
         term1 = 2. * kappap2 * (6. * kappam * (1. + rhoB + 0.5 * rhoB2) + rhoB3) * expB;
         term2 = kappam * (-12. * kappap2 * (1. + rhoA + 0.5 * rhoA2) + (1. - 4. * kappap) * rhoA3 - rhoA4) * expA;
         dPzS = pre / r * (term1 + term2);
         pre = 1. / (pre1 * taurho3);
         term1 = 2. * kappam2 * (36. * kappap2 * (1. + rhoA) + 6. * kappap * (1. + 2. * kappap) * rhoA2 + 6. * kappap * rhoA3 + rhoA4) * expA;
         term2 =-2. * kappap2 * (36. * kappam2 * (1. + rhoB) + 6. * kappam * (1. + 2. * kappam) * rhoB2 + 6. * kappam * rhoB3 + rhoB4) * expB;
         dPxPx = pre / r * (term1 + term2);
         term1 = kappam2 * (72. * kappap2 * (1. + rhoA + 0.5 * rhoA2 + rhoA3 / 6.) + 2. * (2. + 3. * kappa) * rhoA4 + rhoA5) * expA;
         term2 =-kappap2 * (72. * kappam2 * (1. + rhoB + 0.5 * rhoB2 + rhoB3 / 6.) + 2. * (2. - 3. * kappa) * rhoB4 + rhoB5) * expB;
         dPzPz = -2. * pre / r * (term1 + term2);
      }
   }
   PyPy = PxPx;
   dPyPy = dPxPx;
}

#pragma acc routine seq
template <bool do_g, int SOFTCORE>
SEQ_CUDA
void pair_xrepel(real r2, real rscale, real vlambda, real cut, real off, real xr, real yr, real zr,
                 real zxri, real dmpi, real cis, real cix, real ciy, real ciz,
                 real zxrk, real dmpk, real cks, real ckx, real cky, real ckz,
                 real& restrict e, PairRepelGrad& restrict pgrad)
{
   real vali = zxri;
   real valk = zxrk;
   real cut2 = cut * cut;
   real r = REAL_SQRT(r2);
   real r3 = r2 * r;

   // choose orthogonal 2-body coordinates / solve rotation matrix
   real bi[3],bj[3],bk[3];
   bk[0] = xr / r;
   bk[1] = yr / r;
   bk[2] = zr / r;
   int ind1 = 0;
   if (abs(bk[1]) > abs(bk[ind1])) ind1 = 1;
   if (abs(bk[2]) > abs(bk[ind1])) ind1 = 2;
   int ind2 = (ind1+1) % 3;
   int ind3 = (ind1+2) % 3;
   bi[ind1] = -bk[ind2];
   bi[ind2] = bk[ind1];
   bi[ind3] = 0.;
   real normi = REAL_SQRT(bi[0]*bi[0] + bi[1]*bi[1] + bi[2]*bi[2]);
   bi[0] = bi[0] / normi;
   bi[1] = bi[1] / normi;
   bi[2] = bi[2] / normi;
   bj[0] = bk[1]*bi[2] - bk[2]*bi[1];
   bj[1] = bk[2]*bi[0] - bk[0]*bi[2];
   bj[2] = bk[0]*bi[1] - bk[1]*bi[0];

   // rotate p orbital cofficients to 2-body (prolate spheroid) frame
   real rcix = bi[0]*cix + bi[1]*ciy + bi[2]*ciz;
   real rciy = bj[0]*cix + bj[1]*ciy + bj[2]*ciz;
   real rciz = bk[0]*cix + bk[1]*ciy + bk[2]*ciz;
   real rckx = bi[0]*ckx + bi[1]*cky + bi[2]*ckz;
   real rcky = bj[0]*ckx + bj[1]*cky + bj[2]*ckz;
   real rckz = bk[0]*ckx + bk[1]*cky + bk[2]*ckz;
   real cscs = cis * cks;
   real cxcx = rcix * rckx;
   real cycy = rciy * rcky;
   real czcz = rciz * rckz;
   real cscz = cis * rckz;
   real czcs = rciz * cks;

   // compute overlap terms
   real SS, dSS, SPz, dSPz, PzS, dPzS;
   real PxPx, dPxPx, PyPy, dPyPy, PzPz, dPzPz;
   computeOverlap(dmpi, dmpk, r, do_g, SS, dSS, SPz, dSPz,
                  PzS, dPzS, PxPx, dPxPx, PyPy, dPyPy, PzPz, dPzPz);
   real intS = cscs * SS + cxcx * PxPx + cycy * PyPy + czcz * PzPz + cscz * SPz + czcs * PzS;
   real intS2 = intS * intS;
   real pre = units::hartree * (zxri*valk + zxrk*vali) * rscale;
   e = pre * intS2 / r;

   // energy via soft core lambda scaling
   real termsc, soft;
   if CONSTEXPR (SOFTCORE) {
      real vlambda3 = vlambda * vlambda * vlambda;
      real vlambda4 = vlambda3 * vlambda;
      real vlambda5 = vlambda4 * vlambda;
      termsc = vlambda3 - vlambda4 + r2;
      soft = vlambda5 * r / REAL_SQRT(termsc);
      e *= soft;
   }

   // gradient
   if CONSTEXPR (do_g) {
      real dintS = cscs*dSS + cxcx*dPxPx + cycy*dPyPy + czcz*dPzPz + cscz*dSPz + czcs*dPzS;
      real dintSx = dintS * bk[0];
      real dintSy = dintS * bk[1];
      real dintSz = dintS * bk[2];
      real rcixr = rcix/r;
      real rciyr = rciy/r;
      real rcizr = rciz/r;
      real rckxr = rckx/r;
      real rckyr = rcky/r;
      real rckzr = rckz/r;
      real drcixdx = bi[0]*(-rcizr);
      real drcixdy = bi[1]*(-rcizr);
      real drcixdz = bi[2]*(-rcizr);
      real drciydx = bj[0]*(-rcizr);
      real drciydy = bj[1]*(-rcizr);
      real drciydz = bj[2]*(-rcizr);
      real drcizdx = bi[0]*( rcixr) + bj[0]*( rciyr);
      real drcizdy = bi[1]*( rcixr) + bj[1]*( rciyr);
      real drcizdz = bi[2]*( rcixr) + bj[2]*( rciyr);
      real drckxdx = bi[0]*(-rckzr);
      real drckxdy = bi[1]*(-rckzr);
      real drckxdz = bi[2]*(-rckzr);
      real drckydx = bj[0]*(-rckzr);
      real drckydy = bj[1]*(-rckzr);
      real drckydz = bj[2]*(-rckzr);
      real drckzdx = bi[0]*( rckxr) + bj[0]*( rckyr);
      real drckzdy = bi[1]*( rckxr) + bj[1]*( rckyr);
      real drckzdz = bi[2]*( rckxr) + bj[2]*( rckyr);
      dintSx += drcizdx*cks*PzS + drcixdx*rckx*PxPx + drciydx*rcky*PyPy + drcizdx*rckz*PzPz;
      dintSy += drcizdy*cks*PzS + drcixdy*rckx*PxPx + drciydy*rcky*PyPy + drcizdy*rckz*PzPz;
      dintSz += drcizdz*cks*PzS + drcixdz*rckx*PxPx + drciydz*rcky*PyPy + drcizdz*rckz*PzPz;
      dintSx += cis*drckzdx*SPz + rcix*drckxdx*PxPx + rciy*drckydx*PyPy + rciz*drckzdx*PzPz;
      dintSy += cis*drckzdy*SPz + rcix*drckxdy*PxPx + rciy*drckydy*PyPy + rciz*drckzdy*PzPz;
      dintSz += cis*drckzdz*SPz + rcix*drckxdz*PxPx + rciy*drckydz*PyPy + rciz*drckzdz*PzPz;
      real term1 = -intS2 / r3;
      real intSR = 2. * intS / r;
      real term2x = intSR * dintSx;
      real term2y = intSR * dintSy;
      real term2z = intSR * dintSz;

      // compute the force components for this interaction
      pgrad.frcx = -pre * (xr * term1 + term2x);
      pgrad.frcy = -pre * (yr * term1 + term2y);
      pgrad.frcz = -pre * (zr * term1 + term2z);

      // compute the torque components for this interaction
      real ncix,nciy,nciz;
      real nckx,ncky,nckz;
      real nrcix,nrciy,nrciz;
      real nrckx,nrcky,nrckz;
      real tixintS,tiyintS,tizintS;
      real tkxintS,tkyintS,tkzintS;

      nciy = -ciz;
      nciz = ciy;
      nrcix = bi[1]*nciy + bi[2]*nciz;
      nrciy = bj[1]*nciy + bj[2]*nciz;
      nrciz = bk[1]*nciy + bk[2]*nciz;
      cxcx = nrcix * rckx;
      cycy = nrciy * rcky;
      czcz = nrciz * rckz;
      czcs = nrciz * cks;
      tixintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + czcs * PzS;

      ncix = ciz;
      nciz = -cix;
      nrcix = bi[0]*ncix + bi[2]*nciz;
      nrciy = bj[0]*ncix + bj[2]*nciz;
      nrciz = bk[0]*ncix + bk[2]*nciz;
      cxcx = nrcix * rckx;
      cycy = nrciy * rcky;
      czcz = nrciz * rckz;
      czcs = nrciz * cks;
      tiyintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + czcs * PzS;

      ncix = -ciy;
      nciy = cix;
      nrcix = bi[0]*ncix + bi[1]*nciy;
      nrciy = bj[0]*ncix + bj[1]*nciy;
      nrciz = bk[0]*ncix + bk[1]*nciy;
      cxcx = nrcix * rckx;
      cycy = nrciy * rcky;
      czcz = nrciz * rckz;
      czcs = nrciz * cks;
      tizintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + czcs * PzS;

      ncky = -ckz;
      nckz = cky;
      nrckx = bi[1]*ncky + bi[2]*nckz;
      nrcky = bj[1]*ncky + bj[2]*nckz;
      nrckz = bk[1]*ncky + bk[2]*nckz;
      cxcx = rcix * nrckx;
      cycy = rciy * nrcky;
      czcz = rciz * nrckz;
      cscz = cis * nrckz;
      tkxintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + cscz * SPz;

      nckx = ckz;
      nckz = -ckx;
      nrckx = bi[0]*nckx + bi[2]*nckz;
      nrcky = bj[0]*nckx + bj[2]*nckz;
      nrckz = bk[0]*nckx + bk[2]*nckz;
      cxcx = rcix * nrckx;
      cycy = rciy * nrcky;
      czcz = rciz * nrckz;
      cscz = cis * nrckz;
      tkyintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + cscz * SPz;

      nckx = -cky;
      ncky = ckx;
      nrckx = bi[0]*nckx + bi[1]*ncky;
      nrcky = bj[0]*nckx + bj[1]*ncky;
      nrckz = bk[0]*nckx + bk[1]*ncky;
      cxcx = rcix * nrckx;
      cycy = rciy * nrcky;
      czcz = rciz * nrckz;
      cscz = cis * nrckz;
      tkzintS = cxcx * PxPx + cycy * PyPy + czcz * PzPz + cscz * SPz;
      
      real preintSR = -pre * intSR;
      pgrad.ttqi[0] = preintSR * tixintS;
      pgrad.ttqi[1] = preintSR * tiyintS;
      pgrad.ttqi[2] = preintSR * tizintS;
      pgrad.ttqk[0] = preintSR * tkxintS;
      pgrad.ttqk[1] = preintSR * tkyintS;
      pgrad.ttqk[2] = preintSR * tkzintS;

      // force via soft core lambda scaling
      if CONSTEXPR (SOFTCORE) {
         real dsoft = e * soft * (1./r2 - 1/termsc);
         pgrad.frcx = pgrad.frcx * soft - dsoft * xr;
         pgrad.frcy = pgrad.frcy * soft - dsoft * yr;
         pgrad.frcz = pgrad.frcz * soft - dsoft * zr;

         pgrad.ttqi[0] *= soft;
         pgrad.ttqi[1] *= soft;
         pgrad.ttqi[2] *= soft;
         pgrad.ttqk[0] *= soft;
         pgrad.ttqk[1] *= soft;
         pgrad.ttqk[2] *= soft;
      }
   }

   if (r2 > cut2) {
      real taper, dtaper;
      switchTaper5<do_g>(r, cut, off, taper, dtaper);
      if CONSTEXPR (do_g) {
         dtaper *= e / r;
         pgrad.frcx = pgrad.frcx * taper - dtaper * xr;
         pgrad.frcy = pgrad.frcy * taper - dtaper * yr;
         pgrad.frcz = pgrad.frcz * taper - dtaper * zr;

         pgrad.ttqi[0] *= taper;
         pgrad.ttqi[1] *= taper;
         pgrad.ttqi[2] *= taper;
         pgrad.ttqk[0] *= taper;
         pgrad.ttqk[1] *= taper;
         pgrad.ttqk[2] *= taper;
      }
      e *= taper;
   }
}
}
