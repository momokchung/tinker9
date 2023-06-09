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
      darray::deallocate(radvdw, epsvdw, cdisp);
      if (rc_a) {
         bufferDeallocate(rc_flag, nes);
         bufferDeallocate(rc_flag, es, desx, desy, desz);
      }
      nes = nullptr;
      es = nullptr;
      desx = nullptr;
      desy = nullptr;
      desz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      nes = nullptr;
      es = eng_buf_elec;
      desx = gx_elec;
      desy = gy_elec;
      desz = gz_elec;
      darray::allocate(n, &radvdw, &epsvdw, &cdisp);
      if (rc_a) {
         bufferAllocate(rc_flag, &nes);
         bufferAllocate(rc_flag, &es, &desx, &desy, &desz);
      }
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, radvdw, vdw::radvdw);
      darray::copyin(g::q0, n, epsvdw, vdw::epsvdw);
      darray::copyin(g::q0, n, cdisp, nonpol::cdisp);
      epso = nonpol::epso;
      epsh = nonpol::epsh;
      rmino = nonpol::rmino;
      rminh = nonpol::rminh;
      awater = nonpol::awater;
      slevy = nonpol::slevy;
      shctd = nonpol::shctd;
      cavoff = nonpol::cavoff;
      dispoff = nonpol::dispoff;
   }
}

void esolvInit(int vers)
{
   if (vers & calc::grad)
      darray::zero(g::q0, n, trqx, trqy, trqz);

   if ((not use(Potent::MPOLE)) and (not use(Potent::POLAR))) {
      mpoleInit(vers);
   }
}

TINKER_FVOID2(acc0, cu1, ewca, int);
void ewca(int vers)
{
   TINKER_FCALL2(acc0, cu1, ewca, vers);
}

void enp(int vers)
{
   // ecav energy
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

TINKER_FVOID2(acc0, cu1, ediff, int);
void ediff(int vers)
{
   TINKER_FCALL2(acc0, cu1, ediff, vers);
}

void egk(int vers)
{
   egka(vers);
   
   if (use(Potent::POLAR)) {
      ediff(vers);
   }
}

void esolv(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_g = vers & calc::grad;

   printf("rc_a: %s\n", rc_a ? "true" : "false");
   printf("do_a: %s\n", do_a ? "true" : "false");
   printf("do_e: %s\n", do_e ? "true" : "false");
   printf("do_g: %s\n", do_g ? "true" : "false");

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

   esolvInit(vers);

   if (solvtyp == Solv::GK or solvtyp == Solv::PB) {
      enp(vers);
   }

   // if (solvtyp == Solv::GK) {
   //    if (not use(Potent::POLAR)) {
   //       inducegk(uind, uinp, uinds, uinps);
   //    }

   //    egk(vers);
   // }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = es;
         energy_prec e = energyReduce(u);
         energy_es += e;
      }
      if (do_g)
         sumGradient(gx_elec, gy_elec, gz_elec, desx, desy, desz);
   }

   printf("esolv %15.6e\n", energy_es);
}
}
