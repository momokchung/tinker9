#include "ff/amoeba/induce.h"
#include "ff/modamoeba.h"
#include "ff/atom.h"
#include "ff/nblist.h"
#include "ff/solv/inducegk.h"
#include "ff/solv/solute.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "tool/error.h"
#include "tool/gpucard.h"
#include "tool/ioprint.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/polpcg.hh>
#include <tinker/detail/polpot.hh>
#include <tinker/detail/units.hh>

#include <iostream>

namespace tinker {
static void dfieldgkSum_acc1(int n, const real (*field)[3], const real (*fieldp)[3], real (*fields)[3], real (*fieldps)[3])
{
   MAYBE_UNUSED int GRID_DIM = gpuGridSize(BLOCK_DIM);
   #pragma acc parallel async num_gangs(GRID_DIM) vector_length(BLOCK_DIM)\
               deviceptr(field, fieldp, fields, fieldps)
   #pragma acc loop gang independent
   for (int i = 0; i < n; ++i) {
      atomic_add(field[i][0], &fields[i][0]);
      atomic_add(field[i][1], &fields[i][1]);
      atomic_add(field[i][2], &fields[i][2]);
      atomic_add(fieldp[i][0], &fieldps[i][0]);
      atomic_add(fieldp[i][1], &fieldps[i][1]);
      atomic_add(fieldp[i][2], &fieldps[i][2]);
   }
}

void inducegk_print(int n, real (*array)[3])
{
   #pragma acc parallel loop independent async\
            deviceptr(array)
   for (int i = 0; i < n; i++) {
      real arrayi0 = array[i][0];
      real arrayi1 = array[i][1];
      real arrayi2 = array[i][2];
      printf("d %d %15.6e %15.6e %15.6e \n", i, arrayi0, arrayi1, arrayi2);
   }
}

void induceMutualPcg3_acc(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3])
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
   // bool dirguess = polpcg::pcgguess;
   // bool predict = polpred != UPred::NONE;
   // if (predict and nualt < maxualt) {
   //    predict = false;
   //    dirguess = true;
   // }

   // get the electrostatic field due to permanent multipoles
   dfieldsolv(field, fieldp);
   real dwater = 78.3;
   real fc = 1 * (1-dwater) / (1*dwater);
   real fd = 2 * (1-dwater) / (1+2*dwater);
   real fq = 3 * (1-dwater) / (2+3*dwater);
   darray::zero(g::q0, n, fields, fieldps);
   dfieldgk(gkc, fc, fd, fq, fields, fieldps);
   dfieldgkSum_acc1(n, field, fieldp, fields, fieldps);

   // direct induced dipoles
   #pragma acc parallel loop independent async\
               deviceptr(polarity,udir,udirp,udirs,udirps,field,fieldp,fields,fieldps)
   for (int i = 0; i < n; ++i) {
      real poli = polarity[i];
      #pragma acc loop seq
      for (int j = 0; j < 3; ++j) {
         udir[i][j] = poli * field[i][j];
         udirp[i][j] = poli * fieldp[i][j];
         udirs[i][j] = poli * fields[i][j];
         udirps[i][j] = poli * fieldps[i][j];
      }
   }

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
   //    ufieldsolv(uind, uinp, rsd, rsdp);
   //    ufieldgk(gkc, fd, uinds, uinps, rsds, rsdps);
   // } else {
   //    darray::copy(g::q0, n, rsd, field);
   //    darray::copy(g::q0, n, rsdp, fieldp);
   //    darray::copy(g::q0, n, rsd, fields);
   //    darray::copy(g::q0, n, rsdp, fieldps);
   // }
   ufieldsolv(uind, uinp, rsd, rsdp);
   ufieldgk(gkc, fd, uinds, uinps, rsds, rsdps); // Temporary
   #pragma acc parallel loop independent async deviceptr(polarity,rsd,rsdp,rsds,rsdps)
   for (int i = 0; i < n; ++i) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
         rsdp[i][0] = 0;
         rsdp[i][1] = 0;
         rsdp[i][2] = 0;
         rsds[i][0] = 0;
         rsds[i][1] = 0;
         rsds[i][2] = 0;
         rsdps[i][0] = 0;
         rsdps[i][1] = 0;
         rsdps[i][2] = 0;
      }
   }

   // // initial M r(0) and p(0) TODO_Moses
   // if (sparse_prec) {
   //    sparsePrecondBuild();
   //    sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
   // } else {
   //    diagPrecond(rsd, rsdp, zrsd, zrsdp);
   // }
   diagPrecond(rsd, rsdp, zrsd, zrsdp); // Temporary
   diagPrecond(rsds, rsdps, zrsds, zrsdps); // Temporary
   darray::copy(g::q0, n, conj, zrsd);
   darray::copy(g::q0, n, conjp, zrsdp);
   darray::copy(g::q0, n, conjs, zrsds);
   darray::copy(g::q0, n, conjps, zrsdps);

   // initial r(0) M r(0)
   real sum,sump,sums,sumps;
   sum = darray::dotThenReturn(g::q0, n, rsd, zrsd);
   sump = darray::dotThenReturn(g::q0, n, rsdp, zrsdp);
   sums = darray::dotThenReturn(g::q0, n, rsds, zrsds);
   sumps = darray::dotThenReturn(g::q0, n, rsdps, zrsdps);

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
      ufieldsolv(conj, conjp, field, fieldp);
      ufieldgk(gkc, fd, conjs, conjps, fields, fieldps);
      #pragma acc parallel loop independent async\
                  deviceptr(polarity_inv,vec,vecp,conj,conjp,field,fieldp,\
                            vecs,vecps,conjs,conjps,fields,fieldps)
      for (int i = 0; i < n; ++i) {
         real poli_inv = polarity_inv[i];
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j) {
            vec[i][j] = poli_inv * conj[i][j] - field[i][j];
            vecp[i][j] = poli_inv * conjp[i][j] - fieldp[i][j];
            vecs[i][j] = poli_inv * conjs[i][j] - fields[i][j];
            vecps[i][j] = poli_inv * conjps[i][j] - fieldps[i][j];
         }
      }

      // a <- p T p
      real a,ap,as,aps;
      a = darray::dotThenReturn(g::q0, n, conj, vec);
      ap = darray::dotThenReturn(g::q0, n, conjp, vecp);
      as = darray::dotThenReturn(g::q0, n, conjs, vecs);
      aps = darray::dotThenReturn(g::q0, n, conjps, vecps);
      // a <- r M r / p T p
      if (a != 0) a = sum / a;
      if (ap != 0) ap = sump / ap;
      if (as != 0) as = sums / as;
      if (aps != 0) aps = sumps / aps;

      // u <- u + a p
      // r <- r - a T p
      #pragma acc parallel loop independent async\
                  deviceptr(polarity,uind,uinp,conj,conjp,rsd,rsdp,vec,vecp,\
                            uinds,uinps,conjs,conjps,rsds,rsdps,vecs,vecps)
      for (int i = 0; i < n; ++i) {
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j) {
            uind[i][j] += a * conj[i][j];
            uinp[i][j] += ap * conjp[i][j];
            uinds[i][j] += as * conjs[i][j];
            uinps[i][j] += aps * conjps[i][j];
            rsd[i][j] -= a * vec[i][j];
            rsdp[i][j] -= ap * vecp[i][j];
            rsds[i][j] -= as * vecs[i][j];
            rsdps[i][j] -= aps * vecps[i][j];
         }
         if (polarity[i] == 0) {
            rsd[i][0] = 0;
            rsd[i][1] = 0;
            rsd[i][2] = 0;
            rsdp[i][0] = 0;
            rsdp[i][1] = 0;
            rsdp[i][2] = 0;
            rsds[i][0] = 0;
            rsds[i][1] = 0;
            rsds[i][2] = 0;
            rsdps[i][0] = 0;
            rsdps[i][1] = 0;
            rsdps[i][2] = 0;
         }
      }

      // // calculate/update M r TODO_Moses
      // if (sparse_prec)
      //    sparsePrecondApply(rsd, rsdp, zrsd, zrsdp);
      // else
      //    diagPrecond(rsd, rsdp, zrsd, zrsdp);
      diagPrecond(rsd, rsdp, zrsd, zrsdp); // Temporary
      diagPrecond(rsds, rsdps, zrsds, zrsdps); // Temporary

      real b,bp,bs,bps;
      real sum1,sump1,sum1s,sump1s;
      sum1 = darray::dotThenReturn(g::q0, n, rsd, zrsd);
      sump1 = darray::dotThenReturn(g::q0, n, rsdp, zrsdp);
      sum1s = darray::dotThenReturn(g::q0, n, rsds, zrsds);
      sump1s = darray::dotThenReturn(g::q0, n, rsdps, zrsdps);
      b = 0;
      bp = 0;
      bs = 0;
      bps = 0;
      if (sum != 0) b = sum1 / sum;
      if (sump != 0) bp = sump1 / sump;
      if (sums != 0) bs = sum1s / sums;
      if (sumps != 0) bps = sump1s / sumps;

      // calculate/update p
      #pragma acc parallel loop independent async\
                  deviceptr(conj,conjp,conjs,conjps,zrsd,zrsdp,zrsds,zrsdps)
      for (int i = 0; i < n; ++i) {
         #pragma acc loop seq
         for (int j = 0; j < 3; ++j) {
            conj[i][j] = zrsd[i][j] + b * conj[i][j];
            conjp[i][j] = zrsdp[i][j] + bp * conjp[i][j];
            conjs[i][j] = zrsds[i][j] + bs * conjs[i][j];
            conjps[i][j] = zrsdps[i][j] + bps * conjps[i][j];
         }
      }

      sum = sum1;
      sump = sump1;
      sums = sum1s;
      sumps = sump1s;

      real epsd;
      real epsp;
      real epsds;
      real epsps;
      epsd = darray::dotThenReturn(g::q0, n, rsd, rsd);
      epsp = darray::dotThenReturn(g::q0, n, rsdp, rsdp);
      epsds = darray::dotThenReturn(g::q0, n, rsds, rsds);
      epsps = darray::dotThenReturn(g::q0, n, rsdps, rsdps);

      // epsold = eps;
      eps = REAL_MAX(epsd, epsp);
      real epss = REAL_MAX(epsds, epsps);
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

      if (eps < poleps) done = true;
      // if (eps > epsold) done = true;
      if (iter < miniter) done = false;
      if (iter >= politer) done = true;

      // apply a "peek" iteration to the mutual induced dipoles
      if (done) {
         #pragma acc parallel loop independent async\
                     deviceptr(polarity,uind,uinp,rsd,rsdp,\
                               uinds,uinps,rsds,rsdps)
         for (int i = 0; i < n; ++i) {
            real term = pcgpeek * polarity[i];
            #pragma acc loop seq
            for (int j = 0; j < 3; ++j) {
               uind[i][j] += term * rsd[i][j];
               uinp[i][j] += term * rsdp[i][j];
               uinds[i][j] += term * rsds[i][j];
               uinps[i][j] += term * rsdps[i][j];
            }
         }
      }
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
   // inducegk_print(n, uinds);
}
}