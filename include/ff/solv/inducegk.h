#pragma once
#include "ff/precision.h"

namespace tinker {
void dfieldsolv(real (*field)[3], real (*fieldp)[3]);
void dfieldgk(real gkc, real fc, real fd, real fq, real (*fields)[3], real (*fieldps)[3]);

void ufieldsolv(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldNonEwaldN2(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldgk(real gkc, real fd, const real (*uinds)[3], const real (*uinps)[3], real (*fields)[3], real (*fieldps)[3]);

void diagPrecondgk(const real (*rsd)[3], const real (*rsdp)[3], const real (*rsds)[3], const real (*rsdps)[3], real (*zrsd)[3], real (*zrsdp)[3], real (*zrsds)[3], real (*zrsdps)[3]);

void inducegk(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3], int vers);
}
