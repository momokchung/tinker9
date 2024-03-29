#include "ff/amoeba/induce.h"
#include "ff/modamoeba.h"
#include "ff/cuinduce.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "tool/error.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void induceMutualPcg3_cu(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3])
{
   auto* field = work01_;
   auto* fieldp = work02_;
   auto* fields = work03_;
   auto* fieldps = work04_;
   auto* rsd = work05_;
   auto* rsdp = work06_;
   auto* rsds = work07_;
   auto* rsdps = work08_;
   auto* zrsd = work09_;
   auto* zrsdp = work10_;
   auto* zrsds = work11_;
   auto* zrsdps = work12_;
   auto* conj = work13_;
   auto* conjp = work14_;
   auto* conjs = work15_;
   auto* conjps = work16_;
   auto* vec = work17_;
   auto* vecp = work18_;
   auto* vecs = work19_;
   auto* vecps = work20_;

   const bool sparse_prec = polpcg::pcgprec and (switchOff(Switch::USOLVE) > 0);
   bool dirguess = polpcg::pcgguess;
   bool predict = polpred != UPred::NONE;
   if (predict and nualt < maxualt) {
      predict = false;
      dirguess = true;
   }

   // get the electrostatic field due to permanent multipoles
   dfieldsolv(field, fieldp);
   real dwater = 78.3;
   real fc = 1.0 * (1.0-dwater) / (1.0*dwater);
   real fd = 2.0 * (1.0-dwater) / (1.0+2.0*dwater);
   real fq = 3.0 * (1.0-dwater) / (2.0+3.0*dwater);
   dfieldgk(gkc, fc, fd, fq, field, fieldp, fields, fieldps);

   // direct induced dipoles
   launch_k1s(g::s0, n, pcgUdirV4, n, polarity, udir, udirp, udirs, udirps, field, fieldp, fields, fieldps);

   // // initial induced dipole TODO_Moses
   // if (predict) {
   //    ulspredSum(uind, uinp);
   // } else if (dirguess) {
   //    darray::copy(g::q0, n, uind, udir);
   //    darray::copy(g::q0, n, uinp, udirp);
   //    darray::copy(g::q0, n, uinds, udirs);
   //    darray::copy(g::q0, n, uinps, udirps);
   // } else {
   //    darray::zero(g::q0, n, uind, uinp, uinds, uinps);
   // }
   darray::copy(g::q0, n, uind, udir);    // Temporary
   darray::copy(g::q0, n, uinp, udirp);   // Temporary
   darray::copy(g::q0, n, uinds, udirs);  // Temporary
   darray::copy(g::q0, n, uinps, udirps); // Temporary

   // initial residual r(0) TODO_Moses
   //
   // if use pcgguess, r(0) = E - (inv_alpha + Tu) alpha E
   //                       = E - E -Tu udir
   //                       = -Tu udir
   //
   // in general, r(0) = E - (inv_alpha + Tu) u(0)
   //                  = -Tu u(0) + E - inv_alpha u(0)
   //                  = -Tu u(0) + inv_alpha (udir - u(0))
   //
   // if do not use pcgguess, r(0) = E - T Zero = E
   // if (predict) {
   //    ufield(uind, uinp, field, fieldp);
   //    launch_k1s(g::s0, n, pcgRsd0V2, n, polarity_inv, rsd, rsdp, udir, udirp, uind, uinp, field, fieldp);
   // } else if (dirguess) {
   //    ufieldgk(gkc, fd, uind, uinp, uinds, uinps, rsd, rsdp, rsds, rsdps);
   // } else {
   //    darray::copy(g::q0, n, rsd, field);
   //    darray::copy(g::q0, n, rsdp, fieldp);
   //    darray::copy(g::q0, n, rsd, fields);
   //    darray::copy(g::q0, n, rsdp, fieldps);
   // }
   ufieldgk(gkc, fd, uind, uinp, uinds, uinps, rsd, rsdp, rsds, rsdps); // Temporary
   launch_k1s(g::s0, n, pcgRsd0gk, n, polarity, rsd, rsdp, rsds, rsdps);

   // // initial M r(0) and p(0) TODO_Moses
   // if (sparse_prec) {
   //    sparsePrecondBuild();
   //    sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
   // } else {
   //    diagPrecond(rsd, rsdp, zrsd, zrsdp);
   // }
   diagPrecondgk(rsd, rsdp, rsds, rsdps, zrsd, zrsdp, zrsds, zrsdps); // Temporary
   darray::copy(g::q0, n, conj, zrsd);
   darray::copy(g::q0, n, conjp, zrsdp);
   darray::copy(g::q0, n, conjs, zrsds);
   darray::copy(g::q0, n, conjps, zrsdps);

   // initial r(0) M r(0)
   real* sum = &((real*)dptr_buf)[0];
   real* sump = &((real*)dptr_buf)[1];
   real* sums = &((real*)dptr_buf)[2];
   real* sumps = &((real*)dptr_buf)[3];
   darray::dot(g::q0, n, sum, rsd, zrsd);
   darray::dot(g::q0, n, sump, rsdp, zrsdp);
   darray::dot(g::q0, n, sums, rsds, zrsds);
   darray::dot(g::q0, n, sumps, rsdps, zrsdps);

   // conjugate gradient iteration of the mutual induced dipoles
   const bool debug = inform::debug;
   const int politer = polpot::politer;
   const real poleps = polpot::poleps;
   const real debye = units::debye;
   const real pcgpeek = polpcg::pcgpeek;
   const int maxiter = 100; // see also subroutine induce0a in induce.f
   const int miniter = std::min(3, n);

   bool done = false;
   int iter = 0;
   real eps = 100;
   // real epsold;

   while (not done) {
      ++iter;

      // T p and p
      // vec = (inv_alpha + Tu) conj, field = -Tu conj
      // vec = inv_alpha * conj - field
      ufieldgk(gkc, fd, conj, conjp, conjs, conjps, field, fieldp, fields, fieldps);
      launch_k1s(g::s0, n, pcgP1gk, n, polarity_inv, vec, vecp, conj, conjp, field, fieldp, vecs, vecps, conjs, conjps, fields, fieldps);

      // a <- p T p
      real* a = &((real*)dptr_buf)[4];
      real* ap = &((real*)dptr_buf)[5];
      real* as = &((real*)dptr_buf)[6];
      real* aps = &((real*)dptr_buf)[7];
      // a <- r M r / p T p; a = sum / a; ap = sump / ap
      darray::dot(g::q0, n, a, conj, vec);
      darray::dot(g::q0, n, ap, conjp, vecp);
      darray::dot(g::q0, n, as, conjs, vecs);
      darray::dot(g::q0, n, aps, conjps, vecps);

      // u <- u + a p
      // r <- r - a T p
      launch_k1s(g::s0, n, pcgP2gk, n, polarity, a, ap, sum, sump, uind, uinp, conj, conjp, rsd, rsdp, vec, vecp,
         as, aps, sums, sumps, uinds, uinps, conjs, conjps, rsds, rsdps, vecs, vecps);

      // // calculate/update M r TODO_Moses
      // if (sparse_prec)
      //    sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
      // else
      //    diagPrecond(rsd, rsdp, zrsd, zrsdp);
      diagPrecondgk(rsd, rsdp, rsds, rsdps, zrsd, zrsdp, zrsds, zrsdps); // Temporary

      // b = sum1 / sum; bp = sump1 / sump
      real* sum1 = &((real*)dptr_buf)[8];
      real* sump1 = &((real*)dptr_buf)[9];
      real* sum1s = &((real*)dptr_buf)[10];
      real* sump1s = &((real*)dptr_buf)[11];
      darray::dot(g::q0, n, sum1, rsd, zrsd);
      darray::dot(g::q0, n, sump1, rsdp, zrsdp);
      darray::dot(g::q0, n, sum1s, rsds, zrsds);
      darray::dot(g::q0, n, sump1s, rsdps, zrsdps);

      // calculate/update p
      launch_k1s(g::s0, n, pcgP3gk, n, sum, sump, sum1, sump1, conj, conjp, zrsd, zrsdp,
         sums, sumps, sum1s, sump1s, conjs, conjps, zrsds, zrsdps);

      // copy sum1/p to sum/p
      darray::copy(g::q0, 2, sum, sum1);
      darray::copy(g::q0, 2, sums, sum1s);

      real* epsd = &((real*)dptr_buf)[12];
      real* epsp = &((real*)dptr_buf)[13];
      real* epsds = &((real*)dptr_buf)[14];
      real* epsps = &((real*)dptr_buf)[15];
      darray::dot(g::q0, n, epsd, rsd, rsd);
      darray::dot(g::q0, n, epsp, rsdp, rsdp);
      darray::dot(g::q0, n, epsds, rsds, rsds);
      darray::dot(g::q0, n, epsps, rsdps, rsdps);
      check_rt(cudaMemcpyAsync((real*)pinned_buf, epsd, 4 * sizeof(real), cudaMemcpyDeviceToHost, g::s0));
      check_rt(cudaStreamSynchronize(g::s0));
      // epsold = eps;
      eps = REAL_MAX(((real*)pinned_buf)[0], ((real*)pinned_buf)[1]);
      real epss = REAL_MAX(((real*)pinned_buf)[2], ((real*)pinned_buf)[3]);
      eps = REAL_MAX(eps, epss);
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
      // if (eps > epsold) done = true;
      if (iter < miniter)
         done = false;
      if (iter >= politer)
         done = true;

      // apply a "peek" iteration to the mutual induced dipoles
      if (done)
         launch_k1s(g::s0, n, pcgPeekgk, n, pcgpeek, polarity, uind, uinp, rsd, rsdp, uinds, uinps, rsds, rsdps);
   }

   // print the results from the conjugate gradient iteration
   if (debug) {
      print(stdout,
         " Induced Dipoles :    Iterations %4d      RMS "
         "Residual %14.10f\n",
         iter, eps);
   }

   // terminate the calculation if dipoles failed to converge
   if (iter >= maxiter) {
      printError();
      TINKER_THROW("INDUCE  --  Warning, Induced Dipoles are not Converged");
   }
}
}
