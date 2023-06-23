#pragma once
#include "ff/solv/solute.h"
#include "seq/seq.h"
#include <algorithm>
#include <cmath>

namespace tinker {
#pragma acc routine seq
SEQ_ROUTINE
inline void pair_grycuk(real r, real r2, real rmi, real sk, real& pairrborni)
{
   real sk2 = sk * sk;
   real lik,uik;
   if (rmi+r < sk) {
      lik = rmi;
      uik = sk - r;
      pairrborni = 1.f/REAL_POW(uik,3)-1.f/REAL_POW(lik,3);
   }
   uik = r + sk;
   if (rmi+r < sk) {
      lik = sk - r;
   }
   else if (r < rmi+sk) {
      lik = rmi;
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
   real term = (3.f*(r2-sk2)+6.f*u2-8.f*ur)/u4r - (3.f*(r2-sk2)+6.f*l2-8.f*lr)/l4r;
   pairrborni = pairrborni - term/16.f;
}

#pragma acc routine seq
SEQ_ROUTINE
inline void pair_dgrycuk(real r, real r2, real ri, real sk, real drbi, real drbpi, real term, bool use_gk, real& de)
{
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
   real dbr = term * de/r;
   real dborn = drbi;
   if (use_gk)  dborn = dborn + drbpi;
   de = dbr * dborn;
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
   lik = 1.f / REAL_MAX(ri,rmsk);
   uik = 1.f / (rpsk);
   lik2 = lik * lik;
   uik2 = uik * uik;
   if (rmsk > ri) uik2lik2 = -4.f * r * sk / (REAL_POW(rmsk*rpsk,2));
   else uik2lik2 = uik2-lik2;
   pairrborni = lik - uik + 0.25f*r*(uik2lik2) + (0.5f/r)*std::log(uik/lik) + (0.25f*sk2/r)*(-uik2lik2);
   if (ri < sk-r) {
      pairrborni = pairrborni + 2.f*(1.f/ri-lik);
   }
   pairrborni = -0.5f * pairrborni;
}
}
