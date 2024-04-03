#include "ff/nblist.h"
#include "ff/potent.h"
#include "ff/amoeba/induce.h"

namespace tinker {
TINKER_FVOID2(acc0, cu1, diagPrecondgk, const real (*)[3], const real (*)[3], const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3], real (*)[3], real (*)[3]);
void diagPrecondgk(const real (*rsd)[3], const real (*rsdp)[3], const real (*rsds)[3], const real (*rsdps)[3], real (*zrsd)[3], real (*zrsdp)[3], real (*zrsds)[3], real (*zrsdps)[3])
{
   TINKER_FCALL2(acc0, cu1, diagPrecondgk, rsd, rsdp, rsds, rsdps, zrsd, zrsdp, zrsds, zrsdps);
}

TINKER_FVOID2(acc0, cu1, induceMutualPcg3, real (*)[3], real (*)[3], real (*)[3], real (*)[3]);
static void induceMutualPcg3(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3])
{
   TINKER_FCALL2(acc0, cu1, induceMutualPcg3, uind, uinp, uinds, uinps);
}

TINKER_FVOID2(acc0, cu1, induceMutualPcg5, real (*)[3], real (*)[3]);
static void induceMutualPcg5(real (*uinds)[3], real (*uinps)[3])
{
   TINKER_FCALL2(acc0, cu1, induceMutualPcg5, uinds, uinps);
}

void inducegk(real (*ud)[3], real (*up)[3], real (*uds)[3], real (*ups)[3], int vers)
{
   if (vers == calc::v3) {
      induceMutualPcg3(ud, up, uds, ups);
   } else if ((not use(Potent::MPOLE)) and (not use(Potent::POLAR))) {
      induceMutualPcg3(ud, up, uds, ups);
   } else {
      induceMutualPcg5(uds, ups);
   }
   // ulspredSave(ud, up); TODO_Moses
   // inducePrint(ud); TODO_Moses
}
}
