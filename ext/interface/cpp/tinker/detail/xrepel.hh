#pragma once

#include "macro.hh"

namespace tinker { namespace xrepel {
extern int& nxrep;
extern int*& ixrep;
extern int*& xreplist;
extern double*& zpxr;
extern double*& dmppxr;
extern double*& crpxr;
extern double*& cpxr;
extern double*& rcpxr;
extern double*& xrepole;

#ifdef TINKER_FORTRAN_MODULE_CPP
extern "C" int TINKER_MOD(xrepel, nxrep);
extern "C" int* TINKER_MOD(xrepel, ixrep);
extern "C" int* TINKER_MOD(xrepel, xreplist);
extern "C" double* TINKER_MOD(xrepel, zpxr);
extern "C" double* TINKER_MOD(xrepel, dmppxr);
extern "C" double* TINKER_MOD(xrepel, crpxr);
extern "C" double* TINKER_MOD(xrepel, cpxr);
extern "C" double* TINKER_MOD(xrepel, rcpxr);
extern "C" double* TINKER_MOD(xrepel, xrepole);

int& nxrep = TINKER_MOD(xrepel, nxrep);
int*& ixrep = TINKER_MOD(xrepel, ixrep);
int*& xreplist = TINKER_MOD(xrepel, xreplist);
double*& zpxr = TINKER_MOD(xrepel, zpxr);
double*& dmppxr = TINKER_MOD(xrepel, dmppxr);
double*& crpxr = TINKER_MOD(xrepel, crpxr);
double*& cpxr = TINKER_MOD(xrepel, cpxr);
double*& rcpxr = TINKER_MOD(xrepel, rcpxr);
double*& xrepole = TINKER_MOD(xrepel, xrepole);
#endif
} }
