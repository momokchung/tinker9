#include "ff/hippo/erepel.h"
#include "ff/image.h"
#include "ff/modamoeba.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/xrepel.h"
#include "seq/triangle.h"

namespace tinker {
__global__
static void solvcoeff_cu1(int n, const real (*restrict xrepole)[MPL_TOTAL], const real* restrict crpxr, real (*restrict cpxr)[4])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      scoeffAtomI(i, xrepole, crpxr, cpxr);
   }
}

__global__
static void rotcoeff_cu1(int n, const LocalFrame* restrict zaxis,
   const real* restrict x, const real* restrict y, const real* restrict z,
   const real (*restrict cpxr)[4], real (*restrict rcpxr)[4])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      rotcoeffAtomI(i, zaxis, x, y, z, cpxr, rcpxr);
   }
}

// __global__
// static void print1_cu1(int n, real (*restrict cpxr)[4])
// {
//    for (int i = ITHREAD; i < n; i += STRIDE) {
//       real cpxr0 = cpxr[i][0];
//       real cpxr1 = cpxr[i][1];
//       real cpxr2 = cpxr[i][2];
//       real cpxr3 = cpxr[i][3];
//       # if __CUDA_ARCH__>=200
//       printf("rcpxr %d %10.6f %10.6f %10.6f %10.6f \n", i, cpxr0, cpxr1, cpxr2, cpxr3);
//       #endif  
//    }
// }

// __global__
// static void print2_cu1(int n, real *restrict array)
// {
//    for (int i = ITHREAD; i < n; i += STRIDE) {
//       real arrayi = array[i];
//       # if __CUDA_ARCH__>=200
//       printf("array %d %10.6f\n", i, arrayi);
//       #endif  
//    }
// }

void solvcoeff_cu()
{
   launch_k1s(g::s0, n, solvcoeff_cu1, n, xrepole, crpxr, cpxr);
}

void rotcoeff_cu()
{
   launch_k1s(g::s0, n, rotcoeff_cu1, n, zaxis, x, y, z, cpxr, rcpxr);
}

void exrepel_cu(int vers)
{

   // if (vers == calc::v0)
   //    exrepel_cu2<calc::V0>();
   // else if (vers == calc::v1)
   //    exrepel_cu2<calc::V1>();
   // else if (vers == calc::v3)
   //    exrepel_cu2<calc::V3>();
   // else if (vers == calc::v4)
   //    exrepel_cu2<calc::V4>();
   // else if (vers == calc::v5)
   //    exrepel_cu2<calc::V5>();
   // else if (vers == calc::v6)
   //    exrepel_cu2<calc::V6>();
}
}
