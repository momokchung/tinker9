#include "ff/amoeba/empole.h"
#include "ff/modamoeba.h"
#include "ff/energy.h"
#include "ff/potent.h"
#include "math/zero.h"
#include "tool/externfunc.h"
#include <tinker/detail/limits.hh>

namespace tinker {
TINKER_FVOID2(acc0, cu1, emplar, int);
TINKER_FVOID2(acc0, cu1, emplargkN2, int);
static void emplargk(int vers)
{
   if (limits::use_mlist)
      TINKER_FCALL2(acc0, cu1, emplar, vers);
   else
      TINKER_FCALL2(acc0, cu1, emplargkN2, vers);
}
void emplar(int vers)
{
   auto do_v = vers & calc::virial;

   zeroOnHost(energy_em, virial_em);

   mpoleInit(vers);
   if (use(Potent::SOLV)) emplargk(vers);
   else TINKER_FCALL2(acc0, cu1, emplar, vers);
   torque(vers, demx, demy, demz);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv)
         virial_elec[iv] += v2[iv];
   }
}
}
