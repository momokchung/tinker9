#include "ff/potent.h"
#include "ff/energy.h"
#include "ff/solv/born.h"
#include "ff/atom.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/solpot.hh>
#include <tinker/detail/solute.hh>
#include <iostream>
#include <tinker/routines.h>


namespace tinker {
void esolvData(RcOp op)
{
   if (not use(Potent::SOLV))
   return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(aecav, aedisp);

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
      darray::allocate(n, &aecav, &aedisp);

      nes = nullptr;
      es = eng_buf_elec;
      desx = gx_elec;
      desy = gy_elec;
      desz = gz_elec;
      if (rc_a) {
         bufferAllocate(rc_flag, &nes);
         bufferAllocate(rc_flag, &es, &desx, &desy, &desz);
      }
   }

   if (op & RcOp::INIT) {
      // darray::copyin(g::q0, n, rsolv, solute::rsolv);
      // waitFor(g::q0);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc0, cu1, esolv, int);
void esolv(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_g = vers & calc::grad;

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

   TINKER_FCALL2(acc0, cu1, esolv, vers);
}
}
