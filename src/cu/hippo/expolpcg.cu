#include "ff/amoeba/induce.h"
#include "ff/amoebamod.h"
#include "ff/atom.h"
#include "ff/hippo/induce.h"
#include "ff/hippomod.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "tool/darray.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
__global__
void eppcgUdirDonly(
   int n, const real* restrict polarity, real (*restrict udir)[3], const real (*restrict field)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         udir[i][j] = poli * field[i][j];
      }
   }
}

__global__
void eppcgUdirGuess(int n, const real* restrict polarity, real (*restrict uind)[3],
   const real (*restrict field)[3], const real (*restrict polinv)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] = poli *
            (polinv[i][0][j] * field[i][0] + polinv[i][1][j] * field[i][1] +
               polinv[i][2][j] * field[i][2]);
      }
   }
}

__global__
void eppcgRsd2(int n, const real* restrict polarity_inv, //
   real (*restrict rsd)[3],                              //
   const real (*restrict udir)[3], const real (*restrict uind)[3], const real (*restrict field)[3],
   const real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         rsd[i][j] = (udir[i][j] - uind[i][0] * polscale[i][0][j] - uind[i][1] * polscale[i][1][j] -
                        uind[i][2] * polscale[i][2][j]) *
               poli_inv +
            field[i][j];
      }
   }
}

__global__
void eppcgRsd1(int n, const real* restrict polarity, real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}

__global__
void eppcgP4(int n, const real* restrict polarity_inv, real (*restrict vec)[3],
   const real (*restrict conj)[3], const real (*restrict field)[3],
   const real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         vec[i][j] = poli_inv *
               (conj[i][0] * polscale[i][0][j] + conj[i][1] * polscale[i][1][j] +
                  conj[i][2] * polscale[i][2][j]) -
            field[i][j];
   }
}

__global__
void eppcgP5(int n, const real* restrict polarity, //
   const real* restrict ka,                        //
   const real* restrict ksum, real (*restrict uind)[3], const real (*restrict conj)[3],
   real (*restrict rsd)[3], const real (*restrict vec)[3])
{
   real kaval = *ka;
   real a = *ksum / kaval;
   if (kaval == 0)
      a = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += a * conj[i][j];
         rsd[i][j] -= a * vec[i][j];
      }
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}

__global__
void eppcgP6(int n, const real* restrict ksum, const real* restrict ksum1, real (*restrict conj)[3],
   real (*restrict zrsd)[3])
{
   real ksumval = *ksum;
   real b = *ksum1 / ksumval;
   if (ksumval == 0)
      b = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         conj[i][j] = zrsd[i][j] + b * conj[i][j];
   }
}

__global__
void eppcgPeek1(int n, float pcgpeek, const real* restrict polarity, real (*restrict uind)[3],
   const real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real term = pcgpeek * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         uind[i][j] += term * rsd[i][j];
   }
}

void induceMutualPcg4_cu(real (*uind)[3])
{
   auto* field = work01_;
   auto* rsd = work02_;
   auto* zrsd = work03_;
   auto* conj = work04_;
   auto* vec = work05_;

   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);
   bool dirguess = polpcg::pcgguess;
   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }

   // get the electrostatic field due to permanent multipoles
   dfieldChgpen(field);
   // direct induced dipoles
   launch_k1s(g::s0, n, eppcgUdirDonly, n, polarity, udir, field);

   alterpol(polscale, polinv);

   // initial induced dipole
   if (predict) {
      ulspredSum(uind, nullptr);
   } else if (dirguess) {
      launch_k1s(g::s0, n, eppcgUdirGuess, n, polarity, uind, field, polinv);
   } else {
      darray::zero(g::q0, n, uind);
   }

   if (predict) {
      ufieldChgpen(uind, field);
      launch_k1s(g::s0, n, eppcgRsd2, n, polarity_inv, rsd, udir, uind, field, polscale);
   } else if (dirguess) {
      // uind is used here instead of udir since without exchange polarization udir = uind
      // but with exchange polarization udir != uind (for dirguess).
      ufieldChgpen(uind, rsd);
   } else {
      darray::copy(g::q0, n, rsd, field);
   }
   launch_k1s(g::s0, n, eppcgRsd1, n, polarity, rsd);

   // initial M r(0) and p(0)
   if (sparse_prec) {
      sparsePrecondBuild2();
      sparsePrecondApply2(rsd, zrsd);
   } else {
      diagPrecond2(rsd, zrsd);
   }
   darray::copy(g::q0, n, conj, zrsd);

   // initial r(0) M r(0)
   real* sum = &((real*)dptr_buf)[0];
   darray::dot(g::q0, n, sum, rsd, zrsd);

   // conjugate gradient iteration of the mutual induced dipoles
   const bool debug = inform::debug;
   const int politer = polpot::politer;
   const real poleps = polpot::poleps;
   const real debye = units::debye;
   const real pcgpeek = polpcg::pcgpeek;
   const int maxiter = 100; // see also subroutine induce0a in induce.f

   bool done = false;
   int iter = 0;
   real eps = 100;
   real epsold;

   while (not done) {
      ++iter;

      // T p and p
      // vec = (inv_alpha + Tu) conj, field = -Tu conj
      // vec = inv_alpha * conj - field
      ufieldChgpen(conj, field);
      launch_k1s(g::s0, n, eppcgP4, n, polarity_inv, vec, conj, field, polscale);

      // a <- p T p
      real* a = &((real*)dptr_buf)[1];
      // a <- r M r / p T p; a = sum / a; ap = sump / ap
      darray::dot(g::q0, n, a, conj, vec);

      // u <- u + a p
      // r <- r - a T p
      launch_k1s(g::s0, n, eppcgP5, n, polarity, a, sum, uind, conj, rsd, vec);

      // calculate/update M r
      if (sparse_prec)
         sparsePrecondApply2(rsd, zrsd);
      else
         diagPrecond2(rsd, zrsd);

      // b = sum1 / sum; bp = sump1 / sump
      real* sum1 = &((real*)dptr_buf)[2];
      darray::dot(g::q0, n, sum1, rsd, zrsd);

      // calculate/update p
      launch_k1s(g::s0, n, eppcgP6, n, sum, sum1, conj, zrsd);

      // copy sum1/p to sum/p
      darray::copy(g::q0, 2, sum, sum1);

      real* epsd = &((real*)dptr_buf)[3];
      darray::dot(g::q0, n, epsd, rsd, rsd);
      check_rt(
         cudaMemcpyAsync((real*)pinned_buf, epsd, sizeof(real), cudaMemcpyDeviceToHost, g::s0));
      check_rt(cudaStreamSynchronize(g::s0));
      epsold = eps;
      eps = ((real*)pinned_buf)[0];
      eps = debye * REAL_SQRT(eps / n);

      if (debug) {
         if (iter == 1) {
            print(stdout,
               "\n Determination of SCF Induced Dipole Moments\n\n"
               "    Iter    RMS Residual (Debye)\n\n");
         }
         print(stdout, " %8d       %-16.10f\n", iter, eps);
      }

      if (eps < poleps)
         done = true;
      // if (eps > epsold)
      //    done = true;
      if (iter >= politer)
         done = true;

      // apply a "peek" iteration to the mutual induced dipoles
      if (done)
         launch_k1s(g::s0, n, eppcgPeek1, n, pcgpeek, polarity, uind, rsd);
   }

   // print the results from the conjugate gradient iteration
   if (debug) {
      print(stdout,
         " Induced Dipoles :    Iterations %4d      RMS"
         " Residual %14.10f\n",
         iter, eps);
   }

   // terminate the calculation if dipoles failed to converge
   // if (iter >= maxiter || eps > epsold) {
   if (iter >= maxiter) {
      printError();
      TINKER_THROW("INDUCE  --  Warning, Induced Dipoles are not Converged");
   }
}
}
