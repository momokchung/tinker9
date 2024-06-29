#include "ff/atom.h"
#include "ff/nblist.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/pair_born.h"
#include "tool/externfunc.h"
#include "tool/gpucard.h"
#include <cmath>

namespace tinker {
#define GRYCUK_DPTRS                                               \
   x, y, z, rborn, rsolv, rdescr, shct, sneck, aneck, bneck, rneck
static void grycuk_acc1()
{
   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   real third = (real)0.333333333333333333;
   real pi43 = 4 * third * pi;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(GRYCUK_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real rsi = rsolv[i];
      real rdi = rdescr[i];
      real shcti = shct[i];
      real snecki = sneck[i];
      real ri = REAL_MAX(rsi, rdi) + descoff;
      real si = rdi * shcti;
      real rborni = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:rborni)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real rsk = rsolv[k];
            real rdk = rdescr[k];
            real shctk = shct[k];
            real sneckk = sneck[k];
            real rk = REAL_MAX(rsk, rdk) + descoff;
            real sk = rdk * shctk;
            real mixsn = (real)0.5 * (snecki + sneckk);
            real r = REAL_SQRT(r2);
            bool computei = (rsi > 0) and (rdk > 0) and (sk > 0);
            bool computek = (rsk > 0) and (rdi > 0) and (si > 0);
            real rbi = 0, rbk = 0;
            if (computei)
               pair_grycuk(r, r2, ri, rdk, sk, mixsn, pi43, useneck, aneck, bneck, rneck, rbi);
            if (computek)
               pair_grycuk(r, r2, rk, rdi, si, mixsn, pi43, useneck, aneck, bneck, rneck, rbk);

            rborni += rbi;

            atomic_add(rbk, rborn, k);
         }
      } // end for (int kk)

      atomic_add(rborni, rborn, i);
   } // end for (int i)

   real maxbrad = 30;
   #pragma acc parallel loop independent async\
               deviceptr(rsolv, rborn, bornint)
   for (int i = 0; i < n; ++i) {
      real ri = rsolv[i];
      if (ri > 0) {
         real rborni = rborn[i];
         if (usetanh) {
            bornint[i] = rborni;
            tanhrsc(rborni,ri,pi43);
         }
         rborni = pi43 / REAL_POW(ri,3) + rborni;
         if (rborni < 0) {
            rborn[i] = maxbrad;
         } else {
            rborni = REAL_POW((rborni/pi43),third);
            rborni = 1 / rborni;
            rborn[i] = rborni;
            if (rborni < ri) {
               rborn[i] = ri;
            } else if (rborni > maxbrad) {
               rborn[i] = maxbrad;
            } else if (std::isinf(rborni) or std::isnan(rborni)) {
               rborn[i] = ri;
            }
         }
      }
   } // end for (int i)
}

void born_acc()
{
   if (borntyp == Born::GRYCUK) {
      grycuk_acc1();
   } else {
      throwExceptionMissingFunction("born_acc", __FILE__, __LINE__);
   }
}
}

namespace tinker {
#define BORN1_DPTRS                                                                                      \
   x, y, z, desx, desy, desz, rsolv, rdescr, shct, rborn, drb, drbp, aneck, bneck, rneck, sneck, bornint
static void born1_acc1()
{
   const real off = switchOff(Switch::MPOLE);
   const real off2 = off * off;
   const int maxnlst = mlist_unit->maxnlst;
   const auto* mlst = mlist_unit.deviceptr();

   bool use_gk = false;
   if (solvtyp == Solv::GK) use_gk = true;
   real third = (real)0.333333333333333333;
   real pi43 = 4 * third * pi;
   real factor = -REAL_POW(pi,third) * REAL_POW((real)6.,2*third) / 9;

   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(BORN1_DPTRS,mlst)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      real xi = x[i];
      real yi = y[i];
      real zi = z[i];
      real rsi = rsolv[i];
      real rdi = rdescr[i];
      real shcti = shct[i];
      real rbi = rborn[i];
      real drbi = drb[i];
      real drbpi = drbp[i];
      real snecki = sneck[i];
      real borni = bornint[i];
      MAYBE_UNUSED real gxi = 0, gyi = 0, gzi = 0;

      int nmlsti = mlst->nlst[i];
      int base = i * maxnlst;
      #pragma acc loop vector independent reduction(+:gxi,gyi,gzi)
      for (int kk = 0; kk < nmlsti; ++kk) {
         int k = mlst->lst[base + kk];
         real xr = x[k] - xi;
         real yr = y[k] - yi;
         real zr = z[k] - zi;

         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off2) {
            real rsk = rsolv[k];
            real rdk = rdescr[k];
            real shctk = shct[k];
            real rbk = rborn[k];
            real drbk = drb[k];
            real drbpk = drbp[k];
            real sneckk = sneck[k];
            real bornk = bornint[k];
            real r = REAL_SQRT(r2);
            real ri = REAL_MAX(rsi, rdi) + descoff;
            real si = rdi * shcti;
            real rbir = rbi;
            real rbi3 = rbir * rbir * rbir;
            real termi = pi43 / rbi3;
            termi = factor / REAL_POW(termi, (real)4 / 3);
            real mixsn = (real)0.5 * (snecki + sneckk);
            real rk = REAL_MAX(rsk, rdk) + descoff;
            real sk = rdk * shctk;
            real rbkr = rbk;
            real rbk3 = rbkr * rbkr * rbkr;
            real termk = pi43 / rbk3;
            termk = factor / REAL_POW(termk, (real)4 / 3);
            if (usetanh) {
               real tcr;
               tanhrscchr(borni, rsi, tcr, pi43);
               termi = termi * tcr;
               tanhrscchr(bornk, rsk, tcr, pi43);
               termk = termk * tcr;
            }
            bool computei = (rsi > 0) and (rdk > 0) and (sk > 0);
            bool computek = (rsk > 0) and (rdi > 0) and (si > 0);
            real dei = 0;
            real dek = 0;
            if (computei) {
               pair_dgrycuk(r, r2, ri, rdk, sk, mixsn, pi43, drbi, drbpi, termi, use_gk, useneck, aneck, bneck, rneck, dei);
            }
            if (computek) {
               pair_dgrycuk(r, r2, rk, rdi, si, mixsn, pi43, drbk, drbpk, termk, use_gk, useneck, aneck, bneck, rneck, dek);
            }
            real de = dei + dek;
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;
            gxi += dedx;
            gyi += dedy;
            gzi += dedz;
            atomic_add(-dedx, desx, k);
            atomic_add(-dedy, desy, k);
            atomic_add(-dedz, desz, k);
         }
      } // end for (int kk)

      atomic_add(gxi, desx, i);
      atomic_add(gyi, desy, i);
      atomic_add(gzi, desz, i);
   } // end for (int i)
}

void born1_acc(int vers)
{
   auto do_g = vers & calc::grad;
   if (borntyp == Born::GRYCUK) {
      if (do_g)
         born1_acc1();
   } else {
      throwExceptionMissingFunction("born1_acc", __FILE__, __LINE__);
   }
}
}
