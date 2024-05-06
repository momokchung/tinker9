#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/atom.h"
#include "ff/energy.h"
#include "ff/evdw.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "ff/precision.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/inducegk.h"
#include "ff/solv/solute.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/nonpol.hh>
#include <tinker/detail/openmp.hh>
#include <tinker/detail/solute.hh>
#include <tinker/routines.h>


namespace tinker {
void esolvData(RcOp op)
{
   if (not use(Potent::SOLV)) return;

   auto rc_a = rc_flag & calc::analyz;
   auto do_g = rc_flag & calc::grad;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(raddsp, epsdsp, cdsp);
      darray::deallocate(drb, drbp);
      if (do_g) {
         darray::deallocate(decvx, decvy, decvz);
      }
      if (rc_a) {
         bufferDeallocate(rc_flag, nes);
         bufferDeallocate(rc_flag, es, desx, desy, desz);
      }
      nes = nullptr;
      es = nullptr;
      desx = nullptr;
      desy = nullptr;
      desz = nullptr;
      decvx = nullptr;
      decvy = nullptr;
      decvz = nullptr;
      delete[] radii;
      delete[] coefS;
      delete[] coefV;
      delete[] surf;
      delete[] vol;
      delete[] dsurfx;
      delete[] dsurfy;
      delete[] dsurfz;
      delete[] dvolx;
      delete[] dvoly;
      delete[] dvolz;
      delete[] dcavx;
      delete[] dcavy;
      delete[] dcavz;
   }

   if (op & RcOp::ALLOC) {
      nes = nullptr;
      es = eng_buf_elec;
      desx = gx_elec;
      desy = gy_elec;
      desz = gz_elec;
      darray::allocate(n, &raddsp, &epsdsp, &cdsp);
      darray::allocate(n, &drb, &drbp);
      if (do_g) {
         darray::allocate(n, &decvx, &decvy, &decvz);
      }
      if (rc_a) {
         bufferAllocate(rc_flag, &nes);
         bufferAllocate(rc_flag, &es, &desx, &desy, &desz);
      }
      radii = new double[n];
      coefS = new double[n];
      coefV = new double[n];
      surf = new double[n];
      vol = new double[n];
      dsurfx = new double[n];
      dsurfy = new double[n];
      dsurfz = new double[n];
      dvolx = new double[n];
      dvoly = new double[n];
      dvolz = new double[n];
      dcavx = new double[n];
      dcavy = new double[n];
      dcavz = new double[n];
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
      // TODO_MOSES read from Tinker8
      alfmeth = AlfMethod::AlphaMol2;
      alfsort = AlfSort::KDTree;
      alfh = true;
      alfdebug = false;
      // alfnthd = openmp::nthread;
      alfnthd = 8;
      for (int i = 0; i < n; ++i) {
         coefS[i] = solute::asolv[i];
         coefV[i] = 1.0;
         double exclude = 1.4;
         if (!alfh and atomid::atomic[i] == 1) radii[i] = 0;
         else radii[i] = nonpol::radcav[i] + exclude;
      }

      if (alfmeth == AlfMethod::AlphaMol2) initHilbert(3);
   }
}

void esolv(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   if (do_v) throwExceptionMissingFunction("esolv virial", __FILE__, __LINE__);

   zeroOnHost(energy_es);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nes);
      if (do_e)
         darray::zero(g::q0, bsize, es);
      if (do_g)
         darray::zero(g::q0, n, desx, desy, desz);
   }

   darray::zero(g::q0, n, drb, drbp);

   if (do_g)
      darray::zero(g::q0, n, decvx, decvy, decvz);

   esolvInit(vers);

   if (solvtyp == Solv::GK) {
      if ((not use(Potent::MPOLE)) and (not use(Potent::POLAR))) {
         inducegk(uind, uinp, uinds, uinps, vers);
      }
      egk(vers);
   }

   if (solvtyp == Solv::GK) {
      enp(vers);
   }

   torque(vers, desx, desy, desz);

   if (do_e)
      addToEnrgy();

   if (do_g) {
      darray::copyin(g::q0, n, decvx, dcavx);
      darray::copyin(g::q0, n, decvy, dcavy);
      darray::copyin(g::q0, n, decvz, dcavz);
      addToGrad();
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = es;
         energy_prec e = energyReduce(u);
         energy_es += e;
         energy_elec += e;
      }
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
   // edisp energy
   ewca(vers);

   // cavitation energy
   ecav(vers);
}

void egk(int vers)
{
   egka(vers);

   born1(vers);

   bool comp_ediff = (not use(Potent::MPOLE)) and (not use(Potent::POLAR));

   if ((vers == calc::v3) or comp_ediff) {
      if (use(Potent::POLAR) or comp_ediff) {
         ediff(vers);
      }
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

TINKER_FVOID2(acc0, cu1, addToEnrgy);
void addToEnrgy()
{
   TINKER_FCALL2(acc0, cu1, addToEnrgy);
}

TINKER_FVOID2(acc0, cu1, addToGrad);
void addToGrad()
{
   TINKER_FCALL2(acc0, cu1, addToGrad);
}

void tswitch(double cut, double off, double& c0, double& c1, double& c2, double& c3, double& c4, double& c5)
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

void ecav(int vers)
{
   auto do_g = vers & calc::grad;
   cave = 0;
   double evol = 0;
   double esurf = 0;
   if (do_g) {
      for (int i = 0; i < n; i++) {
         dcavx[i] = 0;
         dcavy[i] = 0;
         dcavz[i] = 0;
      }
   }

   // cavitation energy
   alfmol(vers);
   esurf = wsurf;
   double reff = 0.5 * std::sqrt(esurf/(pi*surften));
   double reff2 = reff * reff;
   double reff3 = reff2 * reff;
   double reff4 = reff3 * reff;
   double reff5 = reff4 * reff;
   double dreff = reff / (2.*esurf);

   // TODO_MOSES compare with Mike's code to see if the conditionals are correct

   // compute solvent excluded volume needed for small solutes
   if (reff < spoff) {
      evol = wvol;
      evol *= solvprs;
      if (do_g) {
         // TODO_MOSES can actually get rid of this part by giving solvprs to alphamol
         for (int i = 0; i < n; i++) {
            dvolx[i] *= solvprs;
            dvoly[i] *= solvprs;
            dvolz[i] *= solvprs;
         }
      }
   }

   // include a full solvent excluded volume cavity term
   if (reff <= spcut) {
      cave = evol;
      if (do_g) {
         for (int i = 0; i < n; i++) {
            dcavx[i] += dvolx[i];
            dcavy[i] += dvoly[i];
            dcavz[i] += dvolz[i];
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
      cave = evol * taper;
      if (do_g) {
         double dtaper = (5*c5*reff4+4*c4*reff3+3*c3*reff2+2*c2*reff+c1) * dreff;
         for (int i = 0; i < n; i++) {
            double evolxdtaper = evol*dtaper;
            dcavx[i] += taper*dvolx[i] + evolxdtaper*dsurfx[i];
            dcavy[i] += taper*dvoly[i] + evolxdtaper*dsurfy[i];
            dcavz[i] += taper*dvolz[i] + evolxdtaper*dsurfz[i];
         }
      }
   }

   // include a full solvent accessible surface area term
   if (reff > stcut) {
      cave += esurf;
      if (do_g) {
         for (int i = 0; i < n; i++) {
            dcavx[i] += dsurfx[i];
            dcavy[i] += dsurfy[i];
            dcavz[i] += dsurfz[i];
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
      cave += taper*esurf;
      if (do_g) {
         double tesurfdtaper = taper+esurf*dtaper;
         for (int i = 0; i < n; i++) {
            dcavx[i] += tesurfdtaper*dsurfx[i];
            dcavy[i] += tesurfdtaper*dsurfy[i];
            dcavz[i] += tesurfdtaper*dsurfz[i];
         }
      }
   }
}
}
