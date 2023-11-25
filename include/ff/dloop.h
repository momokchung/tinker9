#pragma once
#include "ff/precision.h"
#include "tool/genunit.h"

namespace tinker {
struct N2
{
   int nak;
   int nakp;
   int* iakp;

   ~N2();
};
typedef GenericUnit<N2, GenericUnitVersion::ENABLE_ON_DEVICE> N2Unit;

void dloopData(RcOp);
void n2Alloc(N2Unit& n2u);
void n2DataInit(N2Unit& n2u);

TINKER_EXTERN N2Unit mn2_unit;
}
