#include "ff/amoeba/empole.h"
#include "ff/amoeba/induce.h"
#include "ff/modamoeba.h"
#include "ff/evdw.h"
#include "ff/potent.h"
#include "ff/energy.h"
#include "ff/solv/solute.h"
#include "ff/atom.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/nonpol.hh>
#include <tinker/detail/solpot.hh>
#include <tinker/detail/solute.hh>
#include <tinker/detail/vdw.hh>
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
   }
}

void esolvInit(int vers)
{
   mpoleInit(vers);
}

TINKER_FVOID2(acc0, cu1, ewca, int);
void ewca(int vers)
{
   TINKER_FCALL2(acc0, cu1, ewca, vers);
}

void enp(int vers)
{
   // ecav energy
   // egaussvol(vers);
   // surface(vers);
   // // do stuff
   // if (reff < spoff) {
   //    volume (vers);
   //    // do stuff
   // }
   // if (reff <= spcut) {
   //    do stuff
   // } else if (reff <= spoff) {
   //    do stuff
   // }
   // if (reff > stcut) {
   //    do stuff
   // } else if (reff > stoff) {
   //    do stuff
   // }
   // // do stuff, watch out for switch call

   // edisp energy
   ewca(vers);
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

   if (solvtyp == Solv::GK or solvtyp == Solv::PB) {
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
