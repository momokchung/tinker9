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
void ebornData(RcOp);
void esolvData(RcOp);
/// \ingroup solv
void eborn(int vers);
void born(int vers);
void born1(int vers);
void esolv(int vers);
void esolvInit(int vers);
void enp(int vers);
void ewca(int vers);
void egk(int vers);
void egka(int vers);
void ediff(int vers);
void addToEnrgy();
void addToGrad();
void tswitch(double cut, double off, double& c0, double& c1, double& c2, double& c3, double& c4, double& c5);
}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
/// \ingroup solv
TINKER_EXTERN int maxneck;
TINKER_EXTERN real doffset;
TINKER_EXTERN real onipr;
TINKER_EXTERN real stillp1;
TINKER_EXTERN real stillp2;
TINKER_EXTERN real stillp3;
TINKER_EXTERN real stillp4;
TINKER_EXTERN real stillp5;
TINKER_EXTERN real descoff;
TINKER_EXTERN real* rneck;
/// for aneck and bneck pair. Element `[j1][j2]` is accessed by `[maxneck*j1 + j2]`.
TINKER_EXTERN real* aneck;
TINKER_EXTERN real* bneck;
TINKER_EXTERN real* rsolv;
TINKER_EXTERN real* rdescr;
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
TINKER_EXTERN real* sneck;
TINKER_EXTERN real* bornint;
TINKER_EXTERN bool useneck;
TINKER_EXTERN bool usetanh;

TINKER_EXTERN Born borntyp;
TINKER_EXTERN Solv solvtyp;

TINKER_EXTERN CountBuffer nes;
TINKER_EXTERN EnergyBuffer es;
TINKER_EXTERN grad_prec* desx;
TINKER_EXTERN grad_prec* desy;
TINKER_EXTERN grad_prec* desz;
TINKER_EXTERN energy_prec energy_es;

// BORN
TINKER_EXTERN real* roff;

// GKSTUF
TINKER_EXTERN real gkc;

// NONPOL
TINKER_EXTERN double ecav;
TINKER_EXTERN real epso;
TINKER_EXTERN real epsh;
TINKER_EXTERN real rmino;
TINKER_EXTERN real rminh;
TINKER_EXTERN real awater;
TINKER_EXTERN real slevy;
TINKER_EXTERN real shctd;
TINKER_EXTERN real cavoff;
TINKER_EXTERN real dspoff;
TINKER_EXTERN double solvprs;
TINKER_EXTERN double surften;
TINKER_EXTERN double spcut;
TINKER_EXTERN double spoff;
TINKER_EXTERN double stcut;
TINKER_EXTERN double stoff;
TINKER_EXTERN real* cdsp;
TINKER_EXTERN real* decvx;
TINKER_EXTERN real* decvy;
TINKER_EXTERN real* decvz;
TINKER_EXTERN double* dcavx;
TINKER_EXTERN double* dcavy;
TINKER_EXTERN double* dcavz;
}
