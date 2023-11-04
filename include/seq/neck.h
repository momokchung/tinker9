#pragma once
#include "ff/solv/solute.h"
#include "seq/seq.h"

namespace tinker {
#pragma acc routine seq
SEQ_ROUTINE
inline void getbounds(real rho, int& below, int& above)
{
   constexpr real minrad = 0.8;
   constexpr real space = 0.05;
   constexpr int numpoints = 45;
   real calcindex = 0.;
   calcindex = (rho-minrad) / space;
   below = static_cast<int>(REAL_FLOOR(calcindex));
   above = below + 1;
   if (above >= numpoints) {
      below = numpoints - 1;
      above = numpoints - 2;
   }
   else if (below < 0) {
      below = 0;
      above = 1;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void interp2d(real x1, real x2, real y1, real y2, real x, real y, real fx1y1, real fx2y1, real fx1y2, real fx2y2, real& val)
{
   real fxy1 = (x2-x)/(x2-x1)*fx1y1 + (x-x1)/(x2-x1)*fx2y1;
   real fxy2 = (x2-x)/(x2-x1)*fx1y2 + (x-x1)/(x2-x1)*fx2y2;
   val = (y2-y)/(y2-y1)*fxy1 + (y-y1)/(y2-y1)*fxy2;
}

#pragma acc routine seq
SEQ_ROUTINE
inline void neckcon(real rhdsd, real rhdsg, real& aloc, real& bloc, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck)
{
   constexpr int numpoints = 45;
   int lowi = 0;
   int lowj = 0;
   int highi = 1;
   int highj = 1;
   aloc = 0.;
   bloc = 0.;
   getbounds(rhdsd,lowi,highi);
   getbounds(rhdsg,lowj,highj);
   real rli = rneck[lowi];
   real rhi = rneck[highi];
   real rlj = rneck[lowj];
   real rhj = rneck[highj];
   int lilj = lowi + numpoints*lowj;
   int hilj = highi + numpoints*lowj;
   int lihj = lowi + numpoints*highj;
   int hihj = highi + numpoints*highj;
   real lla = aneck[lilj];
   real hla = aneck[hilj];
   real lha = aneck[lihj];
   real hha = aneck[hihj];
   real llb = bneck[lilj];
   real hlb = bneck[hilj];
   real lhb = bneck[lihj];
   real hhb = bneck[hihj];
   interp2d(rli,rhi,rlj,rhj,rhdsd,rhdsg,lla,hla,lha,hha,aloc);
   interp2d(rli,rhi,rlj,rhj,rhdsd,rhdsg,llb,hlb,lhb,hhb,bloc);
   if (aloc < 0.) {
      aloc = 0.;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void neck(real r, real intstarti, real desck, real mixsn, real pi43, real& neckval, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck)
{
   constexpr real rhow = 1.4;
   real usea = 0.;
   real useb = 0.;
   if (r > intstarti+desck+2.*rhow) {
      neckval = 0.;
   }
   else {
      neckcon(intstarti,desck,usea,useb,aneck,bneck,rneck);
      real rminb = r - useb;
      real rminb4 = REAL_POW(rminb,4);
      real radminr = intstarti + desck + 2.*rhow - r;
      real radminr4 = REAL_POW(radminr,4);
      neckval = pi43 * mixsn * usea * rminb4 * radminr4;
   }
}

#pragma acc routine seq
SEQ_ROUTINE
inline void neckder(real r, real intstarti, real desck, real mixsn, real pi43, real& neckderi, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck)
{
   constexpr real rhow = 1.4;
   real usea = 0.;
   real useb = 0.;
   if (r > intstarti+desck+2.*rhow) {
      neckderi = 0.;
   }
   else {
      neckcon(intstarti,desck,usea,useb,aneck,bneck,rneck);
      real rminb = r - useb;
      real rminb3 = REAL_POW(rminb,3);
      real rminb4 = rminb3 * rminb;
      real radminr = intstarti + desck + 2.*rhow - r;
      real radminr3 = REAL_POW(radminr,3);
      real radminr4 = radminr3 * radminr;
      neckderi = 4. * pi43 * (mixsn*usea*rminb3*radminr4
                            - mixsn*usea*rminb4*radminr3);
   }
}
}
