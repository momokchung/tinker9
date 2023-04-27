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
void bornData(RcOp op)
{
   if (not use(Potent::BORN))
   return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(rsolv);
      darray::deallocate(rdescr);
      darray::deallocate(rborn);
      darray::deallocate(shct);

      // if (rc_a) {
      //    bufferDeallocate(rc_flag, esolv, desolvx, desolvy, desolvz);
      // }
      // esolv = nullptr;
      // desolvx = nullptr;
      // desolvy = nullptr;
      // desolvz = nullptr;
      
      borntyp = Born::NONE;
      solvtyp = Solv::NONE;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &rsolv);
      darray::allocate(n, &rdescr);
      darray::allocate(n, &rborn);
      darray::allocate(n, &shct);

      // esolv = eng_buf_elec;
      // desolvx = gx_elec;
      // desolvy = gy_elec;
      // desolvz = gz_elec;
      // if (rc_a) {
      //    bufferAllocate(rc_flag, &esolv, &desolvx, &desolvy, &desolvz);
      // }
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, rsolv, solute::rsolv);
      darray::copyin(g::q0, n, rdescr, solute::rdescr);
      darray::copyin(g::q0, n, rborn, solute::rborn);
      darray::copyin(g::q0, n, shct, solute::shct);
      waitFor(g::q0);

      FstrView bornview = solpot::borntyp;
      if (bornview == "GRYCUK")
         borntyp = Born::GRYCUK;
      else
         borntyp = Born::NONE;

      FstrView solvview = solpot::solvtyp;
      if (solvview == "GK")
         solvtyp = Solv::GK;
      else
         solvtyp = Solv::NONE;
   }
}
}

namespace tinker {
TINKER_FVOID2(acc0, cu1, born, int);
void born(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;

   TINKER_FCALL2(acc0, cu1, born, vers);

   // zeroOnHost(energy_esolv);
   // size_t bsize = bufferSize();
   // if (rc_a) {
   //    if (do_a)
   //       darray::zero(g::q0, bsize, esolv);
   // }
}
}
