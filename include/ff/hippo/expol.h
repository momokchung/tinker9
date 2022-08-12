#pragma once
#include "ff/energybuffer.h"
#include "tool/rcman.h"

namespace tinker {
void expolData(RcOp);

void alterpol(real (*polscale)[3][3], real (*polinv)[3][3]);
void dexpol(const int vers, const real (*uind)[3], grad_prec* depx, grad_prec* depy,
   grad_prec* depz, VirialBuffer restrict vir_ep);

enum class ExpolScr
{
   NONE,
   S2U, // S2 "unified" screening
   S2,  // S2 screening
   G    // Gaussian screening
};
}
