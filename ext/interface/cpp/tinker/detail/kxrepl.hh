#pragma once

#include "macro.hh"

namespace tinker { namespace kxrepl {
extern double*& pxrz;
extern double*& pxrdmp;
extern double*& pxrcr;

#ifdef TINKER_FORTRAN_MODULE_CPP
extern "C" double* TINKER_MOD(kxrepl, pxrz);
extern "C" double* TINKER_MOD(kxrepl, pxrdmp);
extern "C" double* TINKER_MOD(kxrepl, pxrcr);

double*& pxrz = TINKER_MOD(kxrepl, pxrz);
double*& pxrdmp = TINKER_MOD(kxrepl, pxrdmp);
double*& pxrcr = TINKER_MOD(kxrepl, pxrcr);
#endif
} }
