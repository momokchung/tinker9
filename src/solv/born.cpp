#include "ff/potent.h"
#include "ff/energy.h"
#include "ff/solv/solute.h"
#include "ff/atom.h"
#include "math/zero.h"
#include "tool/darray.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/gkstuf.hh>
#include <tinker/detail/solpot.hh>
#include <tinker/detail/solute.hh>
#include <tinker/routines.h>

namespace tinker {
void ebornData(RcOp op)
{
   if (not use(Potent::BORN))
   return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(rsolv, rdescr, shct, rborn);
      darray::deallocate(rneck, aneck, bneck, sneck, bornint);
      
      borntyp = Born::NONE;
      solvtyp = Solv::NONE;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &rsolv, &rdescr, &shct, &rborn);
      darray::allocate(n, &sneck, &bornint);
      maxneck = solute::maxneck;
      darray::allocate(maxneck, &rneck);
      darray::allocate(maxneck*maxneck, &aneck, &bneck);
   }

   if (op & RcOp::INIT) {
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

      darray::copyin(g::q0, n, rsolv, solute::rsolv);
      darray::copyin(g::q0, n, shct, solute::shct);
      if (solvtyp == Solv::GK) {
         darray::copyin(g::q0, n, rdescr, solute::rdescr);
         darray::copyin(g::q0, n, sneck, solute::sneck);
         darray::copyin(g::q0, maxneck, rneck, solute::rneck);
         darray::copyin(g::q0, maxneck*maxneck, aneck, solute::aneck);
         darray::copyin(g::q0, maxneck*maxneck, bneck, solute::bneck);
      }
      waitFor(g::q0);

      gkc = gkstuf::gkc;
      descoff = solute::descoff;
      useneck = solute::useneck;
      usetanh = solute::usetanh;
   }
}

void eborn(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_v = vers & calc::virial;

   if (do_v) throwExceptionMissingFunction("born virial", __FILE__, __LINE__);

   darray::zero(g::q0, n, rborn, bornint);

   born(vers);
}
}

namespace tinker {
TINKER_FVOID2(acc0, cu1, born, int);
void born(int vers)
{
   TINKER_FCALL2(acc0, cu1, born, vers);
}
}
