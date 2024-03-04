#include "ff/solv/alphamol.h"

namespace tinker {
void alphamol(int vers)
{
   // initialize Delaunay procedure
   initdelcx();

   // compute Delaunay triangulation
   delaunay();

   // generate alpha complex (with alpha=0.0)
   double alpha = 0;
   alfcx(alpha);

   alfcxedges();
   alfcxfaces();

   auto do_g = vers & calc::grad;
   alphavol(wsurf, wvol, wmean, wgauss, tsurf, tvol, tmean, tgauss,
      surf, vol, mean, gauss, dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz,
      dmeanx, dmeany, dmeanz, dgaussx, dgaussy, dgaussz, do_g);
}
}