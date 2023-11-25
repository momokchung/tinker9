#include "ff/dloop.h"
#include "seq/launch.h"

namespace tinker {
__global__
void n2DataInit_cu1(int nakp, int* restrict iakp)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nakp; i += blockDim.x * gridDim.x) {
      iakp[i] = i;
   }
}


void n2DataInit_cu(N2Unit& n2u)
{
   auto& st = *n2u;
   int nakp = st.nakp;
   launch_k1s(g::s0, nakp, n2DataInit_cu1, nakp, st.iakp);
}
}
