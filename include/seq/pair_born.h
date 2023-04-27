#pragma once
#include "ff/solv/born.h"
#include "seq/seq.h"
#include <algorithm>

namespace tinker {
#pragma acc routine seq

SEQ_ROUTINE
inline void pair_born(real r, real r2, real pi43, real rsi, real rdi, real rdk, real shctk, real& pairrborni)
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
}
