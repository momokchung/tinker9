#pragma once
#include "ff/precision.h"

namespace tinker {
void dfieldsolv(real (*field)[3], real (*fieldp)[3]);
void dfieldgk(real gkc, real fc, real fd, real fq, real (*fields)[3], real (*fieldps)[3]);

void ufieldsolv(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldNonEwaldN2(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3]);
void ufieldgk(real gkc, real fd, const real (*uinds)[3], const real (*uinps)[3], real (*fields)[3], real (*fieldps)[3]);

void inducegk(real (*uind)[3], real (*uinp)[3], real (*uinds)[3], real (*uinps)[3], int vers);
}
