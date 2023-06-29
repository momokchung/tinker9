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
void bornData(RcOp op)
{
   if (not use(Potent::BORN))
   return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(rsolv, rdescr, shct, drobc, rborn);
      darray::deallocate(aobc, bobc, gobc);
      darray::deallocate(roff);
      
      borntyp = Born::NONE;
      solvtyp = Solv::NONE;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &rsolv, &rdescr, &shct, &drobc, &rborn);
      darray::allocate(n, &aobc, &bobc, &gobc);
      darray::allocate(n, &roff);
   }

   if (op & RcOp::INIT) {
      FstrView bornview = solpot::borntyp;
      if (bornview == "GRYCUK")
         borntyp = Born::GRYCUK;
      else if (bornview == "HCT")
         borntyp = Born::HCT;
      else if (bornview == "OBC")
         borntyp = Born::OBC;
      else
         borntyp = Born::NONE;

      FstrView solvview = solpot::solvtyp;
      if (solvview == "GK")
         solvtyp = Solv::GK;
      else if (solvview == "GB")
         solvtyp = Solv::GB;
      else
         solvtyp = Solv::NONE;

      darray::copyin(g::q0, n, rsolv, solute::rsolv);
      darray::copyin(g::q0, n, shct, solute::shct);
      if (solvtyp == Solv::GK or solvtyp == Solv::GKHPMF) {
         darray::copyin(g::q0, n, rdescr, solute::rdescr);
      }
      if (borntyp == Born::OBC) {
         darray::copyin(g::q0, n, aobc, solute::aobc);
         darray::copyin(g::q0, n, bobc, solute::bobc);
         darray::copyin(g::q0, n, gobc, solute::gobc);
      }
      waitFor(g::q0);

      doffset = solute::doffset;
      gkc = gkstuf::gkc;
   }
}

TINKER_FVOID2(acc0, cu1, bornInit, int);
TINKER_FVOID2(acc0, cu1, born, int);
TINKER_FVOID2(acc0, cu1, bornFinal, int);

void born(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;

   darray::zero(g::q0, n, rborn);

   TINKER_FCALL2(acc0, cu1, bornInit, vers);

   TINKER_FCALL2(acc0, cu1, born, vers);

   TINKER_FCALL2(acc0, cu1, bornFinal, vers);
}
}
