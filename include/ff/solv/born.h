#pragma once
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup solv
enum class Born
{
   NONE,
   ONION,
   STILL,
   HCT,
   OBC,
   ACE,
   GRYCUK,
   GONION,
   PERFECT,
};

/// \ingroup solv
enum class Solv
{
   NONE,
   ASP,
   SASA,
   GB,
   GBHPMF,
   GK,
   GKHPMF,
   PB,
   PBHPMF,
};

/// \ingroup solv
void bornData(RcOp);
/// \ingroup solv
void born(int vers);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup solv
TINKER_EXTERN real doffset;
TINKER_EXTERN real onipr;
TINKER_EXTERN real stillp1;
TINKER_EXTERN real stillp2;
TINKER_EXTERN real stillp3;
TINKER_EXTERN real stillp4;
TINKER_EXTERN real stillp5;
TINKER_EXTERN real* rsolv;
TINKER_EXTERN real* rdescr;
TINKER_EXTERN real* asolv;
TINKER_EXTERN real* rborn;
TINKER_EXTERN real* drb;
TINKER_EXTERN real* drbp;
TINKER_EXTERN real* drobc;
TINKER_EXTERN real* gpol;
TINKER_EXTERN real* shct;
TINKER_EXTERN real* aobc;
TINKER_EXTERN real* bobc;
TINKER_EXTERN real* gobc;
TINKER_EXTERN real* vsolv;

TINKER_EXTERN Born borntyp;
TINKER_EXTERN Solv solvtyp;

TINKER_EXTERN EnergyBuffer esolv;
TINKER_EXTERN grad_prec* desolvx;
TINKER_EXTERN grad_prec* desolvy;
TINKER_EXTERN grad_prec* desolvz;
TINKER_EXTERN energy_prec energy_esolv;

TINKER_EXTERN real* roff;
}
