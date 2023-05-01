#pragma once
#include "ff/solv/born.h"
#include "seq/seq.h"
#include <algorithm>
#include <cmath>

namespace tinker {
#pragma acc routine seq
SEQ_ROUTINE
inline void pair_grycuk(real r, real r2, real pi43, real rsi, real rdi, real rdk, real shctk, real& pairrborni)
{
   real rmi = REAL_MAX(rsi,rdi);

   real sk = rdk * shctk;
   if (rmi < r+sk) {
      real sk2 = sk * sk;
      real lik,uik;
      if (rmi+r < sk) {
         lik = rmi;
         uik = sk - r;
         pairrborni = pi43*(1./REAL_POW(uik,3)-1./REAL_POW(lik,3));
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
      real term = (3.*(r2-sk2)+6.*u2-8.*ur)/u4r - (3.*(r2-sk2)+6.*l2-8.*lr)/l4r;
      pairrborni = pairrborni - M_PI*term/12.;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void pair_hctobc(real r, real ri, real rk, real sk, real& pairrborni)
{
   real sk2 = sk * sk;
   real lik,lik2;
   real uik,uik2;
   lik = 1.f / REAL_MAX(ri,REAL_ABS(r-sk));
   uik = 1.f / (r+sk);
   lik2 = lik * lik;
   uik2 = uik * uik;
   pairrborni = lik - uik + 0.25f*r*(uik2-lik2) + (0.5f/r)*std::log(uik/lik) + (0.25f*sk2/r)*(lik2-uik2);
   if (ri < sk-r) {
      pairrborni = pairrborni + 2.f*(1.f/ri-lik);
   }
   pairrborni = -0.5f * pairrborni;
}
}
