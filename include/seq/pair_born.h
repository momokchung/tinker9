#pragma once
#include "ff/solv/solute.h"
#include "seq/neck.h"
#include "seq/seq.h"
#include <algorithm>
#include <cmath>

namespace tinker {
#pragma acc routine seq
SEQ_ROUTINE
inline void pair_grycuk(real r, real r2, real ri, real rdk, real sk, real mixsn, real pi43, bool useneck, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck, real& pairrborni)
{
   if (ri < r+sk) {
      real sk2 = sk * sk;
      real lik,uik;
      if (ri+r < sk) {
         lik = ri;
         uik = sk - r;
         real uik3 = uik*uik*uik;
         real lik3 = lik*lik*lik;
         pairrborni = pi43*(1./uik3-1./lik3);
      }
      uik = r + sk;
      if (ri+r < sk) {
         lik = sk - r;
      }
      else if (r < ri+sk) {
         lik = ri;
      }
      else {
         lik = r - sk;
      }
      real l2 = lik * lik;
      real l4 = l2 * l2;
      real lr = lik * r;
      real l4r = l4 * r;
      real u2 = uik * uik;
      real u4 = u2 * u2;
      real ur = uik * r;
      real u4r = u4 * r;
      real term = (3.*(r2-sk2)+6.*u2-8.*ur)/u4r - (3.*(r2-sk2)+6.*l2-8.*lr)/l4r;
      pairrborni -= pi*term/12.;
   }
   if (useneck) {
      real neckval = 0.;
      neck(r,ri,rdk,mixsn,pi43,neckval,aneck,bneck,rneck);
      pairrborni -= neckval;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void pair_dgrycuk(real r, real r2, real ri, real rdk, real sk, real mixsn, real pi43, real drbi, real drbpi, real term, bool use_gk, bool useneck, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck, real& de)
{
   if (ri < r+sk) {
      real sk2 = sk * sk;
      real uik,lik;
      if (ri+r < sk) {
         uik = sk - r;
         de = -4. * pi / REAL_POW(uik,4);
      }
      if (ri+r < sk) {
         lik = sk - r;
         de = de + 0.25*pi*(sk2-4.*sk*r+17.*r2) / (r2*REAL_POW(lik,4));
      }
      else if (r < ri+sk) {
         lik = ri;
         de = de + 0.25*pi*(2.*ri*ri-sk2-r2) / (r2*REAL_POW(lik,4));
      }
      else {
         lik = r - sk;
         de = de + 0.25*pi*(sk2-4.*sk*r+r2) / (r2*REAL_POW(lik,4));
      }
      uik = r + sk;
      de = de - 0.25*pi*(sk2+4.*sk*r+r2) / (r2*REAL_POW(uik,4));
      if (useneck) {
         real neckderi;
         neckder(r,ri,rdk,mixsn,pi43,neckderi,aneck,bneck,rneck);
         de += neckderi;
      }
      real dbr = term * de/r;
      real dborn = drbi;
      if (use_gk)  dborn = dborn + drbpi;
      de = dbr * dborn;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void tanhrsc (real& ii, real rhoi, real pi43)
{
   // recipmaxborn3 is 30^(-3)
   constexpr real recipmaxborn3 = 3.70370370370370370370370370370e-5;
   constexpr real b0 = 0.9563;
   constexpr real b1 = 0.2578;
   constexpr real b2 = 0.0810;
   real rho3 = rhoi * rhoi * rhoi;
   real rho3psi = rho3 * (-1.*ii);
   real rho6psi2 = rho3psi * rho3psi;
   real rho9psi3 = rho6psi2 * rho3psi;
   real tanhconst = pi43 * ((1./rho3)-recipmaxborn3);
   ii = -tanhconst * REAL_TANH(b0*rho3psi-b1*rho6psi2+b2*rho9psi3);
}

#pragma acc routine seq
SEQ_ROUTINE
inline void tanhrscchr (real ii, real rhoi, real& derival, real pi43)
{
   // recipmaxborn3 is 30^(-3)
   constexpr real recipmaxborn3 = 3.70370370370370370370370370370e-5;
   constexpr real b0 = 0.9563;
   constexpr real b1 = 0.2578;
   constexpr real b2 = 0.0810;
   real rho3 = rhoi * rhoi * rhoi;
   real rho3psi = rho3 * (-1.*ii);
   real rho6psi2 = rho3psi * rho3psi;
   real rho9psi3 = rho6psi2 * rho3psi;
   real rho6psi = rho3 * rho3 * (-1.*ii);
   real rho9psi2 = rho6psi2 * rho3;
   real tanhterm = REAL_TANH(b0*rho3psi-b1*rho6psi2+b2*rho9psi3);
   real tanh2 = tanhterm * tanhterm;
   real chainrule = b0*rho3 - 2.*b1*rho6psi + 3.*b2*rho9psi2 ;
   real tanhconst = pi43 * ((1./rho3)-recipmaxborn3);
   derival = tanhconst * chainrule * (1.-tanh2);
}

#pragma acc routine seq
SEQ_ROUTINE
inline void pair_hctobc(real r, real ri, real rk, real sk, real& pairrborni)
{
   real sk2 = sk * sk;
   real lik,lik2;
   real uik,uik2;
   real uik2lik2;
   real rpsk = r + sk;
   real rmsk = REAL_ABS(r - sk);
   lik = 1. / REAL_MAX(ri,rmsk);
   uik = 1. / (rpsk);
   lik2 = lik * lik;
   uik2 = uik * uik;
   if (rmsk > ri) uik2lik2 = -4. * r * sk / (REAL_POW(rmsk*rpsk,2));
   else uik2lik2 = uik2-lik2;
   pairrborni = lik - uik + 0.25f*r*(uik2lik2) + (0.5f/r)*std::log(uik/lik) + (0.25f*sk2/r)*(-uik2lik2);
   if (ri < sk-r) {
      pairrborni = pairrborni + 2.*(1./ri-lik);
   }
   pairrborni = -0.5f * pairrborni;
}
}
