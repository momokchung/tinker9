#pragma once
#include "ff/precision.h"

namespace tinker {
/// \ingroup polar
/// \{
// electrostatic field due to permanent multipoles
void dfield(real (*field)[3], real (*fieldp)[3]);
void dfieldsolv(real (*field)[3], real (*fieldp)[3]);
void dfieldgk(real gkc, real fc, real fd, real fq, real (*field)[3], real (*fieldp)[3], real (*fields)[3], real (*fieldps)[3]);
void dfieldNonEwald(real (*field)[3], real (*fieldp)[3]);
void dfieldEwald(real (*field)[3], real (*fieldp)[3]);
void dfieldEwaldRecipSelfP1(real (*field)[3]);

// mutual electrostatic field due to induced dipole moments
// -Tu operator
void ufield(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldgk(real gkc, real fd, const real (*uind)[3], const real (*uinp)[3], const real (*uinds)[3], const real (*uinps)[3], real (*field)[3], real (*fieldp)[3], real (*fields)[3], real (*fieldps)[3]);
void ufieldNonEwald(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3]);
void ufieldEwald(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);

void diagPrecond(const real (*rsd)[3], const real (*rsdp)[3], real (*zrsd)[3], real (*zrsdp)[3]);
void diagPrecondgk(const real (*rsd)[3], const real (*rsdp)[3], const real (*rsds)[3], const real (*rsdps)[3], real (*zrsd)[3], real (*zrsdp)[3], real (*zrsds)[3], real (*zrsdps)[3]);

void sparsePrecondBuild();
void sparsePrecondApply(const real (*rsd)[3], const real (*rsdp)[3], //
   real (*zrsd)[3], real (*zrsdp)[3]);

void ulspredSave(const real (*uind)[3], const real (*uinp)[3]);
void ulspredSum(real (*uind)[3], real (*uinp)[3]);

void inducePrint(const real (*ud)[3]);
void induce(real (*uind)[3], real (*uinp)[3]);
void inducegk(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3]);
/// \}
}

//          | h        | cpp        | acc            | cu
// field    | induce.h | field.cpp  | acc/field*.cpp | field.cu
// pcg      | --       | --         | acc/induce.cpp | pcg.cu
// precond  | induce.h | induce.cpp | acc/induce.cpp | precond.cu
// upredict | induce.h | induce.cpp | acc/induce.cpp | --
