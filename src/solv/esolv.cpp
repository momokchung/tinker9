#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "ff/precision.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/solute.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/nonpol.hh>
#include <tinker/detail/solute.hh>
#include <tinker/routines.h>


namespace tinker {
void esolvData(RcOp op)
{
   if (not use(Potent::SOLV))
   return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(raddsp, epsdsp, cdsp);
      darray::deallocate(drb, drbp);
      if (rc_a) {
         bufferDeallocate(rc_flag, nes);
         bufferDeallocate(rc_flag, es, vir_es, desx, desy, desz);
      }
      nes = nullptr;
      es = nullptr;
      vir_es = nullptr;
      desx = nullptr;
      desy = nullptr;
      desz = nullptr;
      delete[] radii;
      delete[] coefS;
      delete[] coefV;
      delete[] coefM;
      delete[] coefG;
      delete[] surf;
      delete[] vol;
      delete[] mean;
      delete[] gauss;
      delete[] dsurf;
      delete[] dvol;
      delete[] dmean;
      delete[] dgauss;
      delete[] dcav;
   }

   if (op & RcOp::ALLOC) {
      nes = nullptr;
      es = eng_buf_elec;
      vir_es = vir_buf_elec;
      desx = gx_elec;
      desy = gy_elec;
      desz = gz_elec;
      darray::allocate(n, &raddsp, &epsdsp, &cdsp);
      darray::allocate(n, &drb, &drbp);
      if (rc_a) {
         bufferAllocate(rc_flag, &nes);
         bufferAllocate(rc_flag, &es, &vir_es, &desx, &desy, &desz);
      }
      radii = new double[n];
      coefS = new double[n];
      coefV = new double[n];
      coefM = new double[n];
      coefG = new double[n];
      int fudge = 8;
      surf = new double[n+fudge];
      vol = new double[n+fudge];
      mean = new double[n+fudge];
      gauss = new double[n+fudge];
      dsurf = new double[3*(n+fudge)];
      dvol = new double[3*(n+fudge)];
      dmean = new double[3*(n+fudge)];
      dgauss = new double[3*(n+fudge)];
      dcav = new double[3*n];
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, raddsp, nonpol::raddsp);
      darray::copyin(g::q0, n, epsdsp, nonpol::epsdsp);
      darray::copyin(g::q0, n, cdsp, nonpol::cdsp);
      epso = nonpol::epso;
      epsh = nonpol::epsh;
      rmino = nonpol::rmino;
      rminh = nonpol::rminh;
      awater = nonpol::awater;
      slevy = nonpol::slevy;
      shctd = nonpol::shctd;
      cavoff = nonpol::cavoff;
      dspoff = nonpol::dspoff;
      surften = nonpol::surften;
      solvprs = nonpol::solvprs;
      spcut = nonpol::spcut;
      spoff = nonpol::spoff;
      stcut = nonpol::stcut;
      stoff = nonpol::stoff;
      for (int i = 0; i < n; ++i) {
         double exclude = 1.4;
         if (solvtyp == Solv::GK) exclude = 0;
         radii[i] = nonpol::radcav[i] + exclude;
         coefS[i] = solute::asolv[i];
         coefV[i] = 1.0;
         coefM[i] = 1.0;
         coefG[i] = 1.0;
      }
   }
}

void esolv(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_es, virial_es);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nes);
      if (do_e)
         darray::zero(g::q0, bsize, es);
      if (do_v)
         darray::zero(g::q0, bsize, vir_es);
      if (do_g)
         darray::zero(g::q0, n, desx, desy, desz);
   }

   darray::zero(g::q0, n, drb, drbp);

   esolvInit(vers);

   if (solvtyp == Solv::GK) {
      enp(vers);
   }

   if (solvtyp == Solv::GK) {
      if ((not use(Potent::MPOLE)) and (not use(Potent::POLAR))) {
         inducegk(uind, uinp, uinds, uinps);
      }
      egk(vers);
   }

   torque(vers, desx, desy, desz);
   // if (do_v) {
   //    VirialBuffer u2 = vir_trq;
   //    virial_prec v2[9];
   //    virialReduce(v2, u2);
   //    for (int iv = 0; iv < 9; ++iv) {
   //       virial_es[iv] += v2[iv];
   //       virial_elec[iv] += v2[iv];
   //    }
   // }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = es;
         energy_prec e = energyReduce(u);
         energy_es += e;
         energy_elec += e;
      }
      // if (do_v) {
      //    VirialBuffer u = vir_es;
      //    virial_prec v[9];
      //    virialReduce(v, u);
      //    for (int iv = 0; iv < 9; ++iv) {
      //       virial_es[iv] += v[iv];
      //       virial_elec[iv] += v[iv];
      //    }
      // }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, desx, desy, desz);
   }
}
}

namespace tinker {
void esolvInit(int vers)
{
   mpoleInit(vers);
}

void enp(int vers)
{
   auto do_g = vers & calc::grad;
   ecav = 0;
   double evol = 0;
   double esurf = 0;
   if (do_g) {
      for (int i = 0; i < 3*n; i++) {
         dcav[i] = 0;
      }
   }

   // ecav energy
   alphamol(vers);
   esurf = wsurf;
   double reff = 0.5 * std::sqrt(esurf/(pi*surften));
   double reff2 = reff * reff;
   double reff3 = reff2 * reff;
   double reff4 = reff3 * reff;
   double reff5 = reff4 * reff;
   double dreff = reff / (2.*esurf);
   if (do_g) {
      for (int i = 0; i < 3*n; i++) {
         dsurf[i] *= surften;
      }
   }

   // TODO compare with Mike's code to see if the conditionals are correct

   // compute solvent excluded volume needed for small solutes
   if (reff < spoff) {
      evol = wvol;
      evol *= solvprs;
      if (do_g) {
         for (int i = 0; i < 3*n; i++) {
            dvol[i] *= solvprs;
         }
      }
   }

   // include a full solvent excluded volume cavity term
   if (reff <= spcut) {
      ecav = evol;
      printf("ecav1 %10.6e\n", ecav);
      if (do_g) {
         for (int i = 0; i < 3*n; i++) {
            dcav[i] += dvol[i];
         }
      }
   }
   // include a tapered solvent excluded volume cavity term
   else if (reff <= spoff) {
      double cut = nonpol::spcut;
      double off = nonpol::spoff;
      double c0,c1,c2,c3,c4,c5;
      tswitch(cut, off, c0, c1, c2, c3, c4, c5);
      double taper = c5*reff5 + c4*reff4 + c3*reff3 + c2*reff2 + c1*reff + c0;
      double dtaper = (5*c5*reff4+4*c4*reff3+3*c3*reff2+2*c2*reff+c1) * dreff;
      ecav = evol * taper;
      printf("ecav2 %10.6e\n", ecav);
      if (do_g) {
         for (int i = 0; i < 3*n; i++) {
            dcav[i] += taper*dvol[i] + evol*dtaper*dsurf[i];
         }
      }
   }

   // include a full solvent accessible surface area term
   if (reff > stcut) {
      ecav += esurf;
      printf("ecav3 %10.6e\n", ecav);
      if (do_g) {
         for (int i = 0; i < 3*n; i++) {
            dcav[i] += dsurf[i];
         }
      }
   }
   // include a tapered solvent accessible surface area term
   else if (reff > stoff) {
      double cut = nonpol::stoff;
      double off = nonpol::stcut;
      double c0,c1,c2,c3,c4,c5;
      tswitch(cut, off, c0, c1, c2, c3, c4, c5);
      double taper = c5*reff5 + c4*reff4 + c3*reff3 + c2*reff2 + c1*reff + c0;
      taper = 1 - taper;
      double dtaper = (5*c5*reff4+4*c4*reff3+3*c3*reff2+2*c2*reff+c1) * dreff;
      dtaper = -dtaper;
      ecav += taper*esurf;
      printf("ecav4 %10.6e\n", ecav);
      if (do_g) {
         for (int i = 0; i < 3*n; i++) {
            dcav[i] += (taper+esurf*dtaper)*dsurf[i];
         }
      }
   }

   printf("ecav %10.6e\n", ecav);

   // edisp energy
   ewca(vers);
}

void egk(int vers)
{
   egka(vers);

   born1(vers);
   
   if (use(Potent::POLAR)) {
      ediff(vers);
   } else if ((not use(Potent::MPOLE)) and (not use(Potent::POLAR))) {
      ediff(vers);
   }
}

TINKER_FVOID2(acc0, cu1, ewca, int);
void ewca(int vers)
{
   TINKER_FCALL2(acc0, cu1, ewca, vers);
}

TINKER_FVOID2(acc0, cu1, egka, int);
void egka(int vers)
{
   TINKER_FCALL2(acc0, cu1, egka, vers);
}

TINKER_FVOID2(acc0, cu1, born1, int);
void born1(int vers)
{
   TINKER_FCALL2(acc0, cu1, born1, vers);
}

TINKER_FVOID2(acc0, cu1, ediff, int);
void ediff(int vers)
{
   TINKER_FCALL2(acc0, cu1, ediff, vers);
}

void tswitch(double cut, double off, double c0, double c1, double c2, double c3, double c4, double c5)
{
   if (cut >= off) return;

   c0 = 0;
   c1 = 0;
   c2 = 0;
   c3 = 0;
   c4 = 0;
   c5 = 0;

   double off2 = off * off;
   double off3 = off2 * off;
   double cut2 = cut * cut;
   
   double denom = std::pow((off-cut),5);
   c0 = off*off2 * (off2-5*off*cut+10*cut2) / denom;
   c1 = -30 * off2*cut2 / denom;
   c2 = 30 * (off2*cut+off*cut2) / denom;
   c3 = -10 * (off2+4*off*cut+cut2) / denom;
   c4 = 15 * (off+cut) / denom;
   c5 = -6 / denom;
}
}
