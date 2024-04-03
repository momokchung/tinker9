#include "ff/amoeba/induce.h"
#include "tool/externfunc.h"
#include <tinker/detail/limits.hh>

namespace tinker {
TINKER_FVOID2(acc0, cu1, dfieldNonEwaldN2, real (*)[3], real (*)[3]);
void dfieldNonEwaldN2(real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc0, cu1, dfieldNonEwaldN2, field, fieldp);
}

TINKER_FVOID2(acc0, cu1, dfieldgk, real, real, real, real, real (*)[3], real (*)[3]);
void dfieldgk(real gkc, real fc, real fd, real fq, real (*fields)[3], real (*fieldps)[3])
{
   TINKER_FCALL2(acc0, cu1, dfieldgk, gkc, fc, fd, fq, fields, fieldps);
}

TINKER_FVOID2(acc0, cu1, ufieldNonEwaldN2, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void ufieldNonEwaldN2(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3])
{
   TINKER_FCALL2(acc0, cu1, ufieldNonEwaldN2, uind, uinp, field, fieldp);
}

TINKER_FVOID2(acc0, cu1, ufieldgk, real, real, const real (*)[3], const real (*)[3], //
   real (*)[3], real (*)[3]);
void ufieldgk(real gkc, real fd, const real (*uinds)[3], const real (*uinps)[3], //
   real (*fields)[3], real (*fieldps)[3])
{
   TINKER_FCALL2(acc0, cu1, ufieldgk, gkc, fd, uinds, uinps, fields, fieldps);
}

void dfieldsolv(real (*field)[3], real (*fieldp)[3])
{
   if (limits::use_mlist)
      dfieldNonEwald(field, fieldp);
   else
      dfieldNonEwaldN2(field, fieldp);
}

void ufieldsolv(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
    if (limits::use_mlist)
      ufieldNonEwald(uind, uinp, field, fieldp);
   else
      ufieldNonEwaldN2(uind, uinp, field, fieldp);
}
}
