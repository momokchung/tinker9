#pragma once
#include "ff/amoeba/mpole.h"
#include "ff/energybuffer.h"
#include "ff/evdw.h"
#include "tool/rcman.h"

namespace tinker {
/// \ingroup repel
void erepelData(RcOp);
/// \ingroup repel
void erepel(int vers);
/// \ingroup repel
void repoleInit(int vers);
/// \ingroup xrepel
void exrepelData(RcOp);
/// \ingroup xrepel
void exrepel(int vers);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
TINKER_EXTERN real (*repole)[MPL_TOTAL];
TINKER_EXTERN real (*rrepole)[MPL_TOTAL];
TINKER_EXTERN real* sizpr;
TINKER_EXTERN real* dmppr;
TINKER_EXTERN real* elepr;

TINKER_EXTERN real (*xrepole)[MPL_TOTAL];
TINKER_EXTERN real* zpxr;
TINKER_EXTERN real* dmppxr;
TINKER_EXTERN real* crpxr;
TINKER_EXTERN real (*cpxr)[4];
TINKER_EXTERN real (*rcpxr)[4];

TINKER_EXTERN int nrepexclude;
TINKER_EXTERN int (*repexclude)[2];
TINKER_EXTERN real* repexclude_scale;

TINKER_EXTERN CountBuffer nrep;
TINKER_EXTERN EnergyBuffer er;
TINKER_EXTERN VirialBuffer vir_er;
TINKER_EXTERN grad_prec* derx;
TINKER_EXTERN grad_prec* dery;
TINKER_EXTERN grad_prec* derz;
TINKER_EXTERN energy_prec energy_er;
TINKER_EXTERN virial_prec virial_er[9];

TINKER_EXTERN real r2scale;
TINKER_EXTERN real r3scale;
TINKER_EXTERN real r4scale;
TINKER_EXTERN real r5scale;
}
