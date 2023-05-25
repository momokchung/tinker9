#include "ff/precision.h"
#include "seq/launch.h"

namespace tinker {
__global__
void pcgUdirV1(int n, const real* restrict polarity, real (*restrict udir)[3], const real (*restrict field)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         udir[i][j] = poli * field[i][j];
      }
   }
}

__global__
void pcgUdirV2(int n, const real* restrict polarity, real (*restrict udir)[3], real (*restrict udirp)[3],
   const real (*restrict field)[3], const real (*restrict fieldp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         udir[i][j] = poli * field[i][j];
         udirp[i][j] = poli * fieldp[i][j];
      }
   }
}

__global__
void pcgUdirV4(int n, const real* restrict polarity, real (*restrict udir)[3], real (*restrict udirp)[3], real (*restrict udirs)[3], real (*restrict udirps)[3],
   const real (*restrict field)[3], const real (*restrict fieldp)[3], const real (*restrict fields)[3], const real (*restrict fieldps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli = polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         udir[i][j] = poli * field[i][j];
         udirp[i][j] = poli * fieldp[i][j];
         udirs[i][j] = poli * fields[i][j];
         udirps[i][j] = poli * fieldps[i][j];
      }
   }
}
}

namespace tinker {
__global__
void pcgRsd0V1(int n, const real* restrict polarity_inv, real (*restrict rsd)[3], const real (*restrict udir)[3],
   const real (*restrict uind)[3], const real (*restrict field)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         rsd[i][j] = (udir[i][j] - uind[i][j]) * poli_inv + field[i][j];
   }
}

__global__
void pcgRsd0V2(int n, const real* restrict polarity_inv, real (*restrict rsd)[3], real (*restrict rsp)[3],
   const real (*restrict udir)[3], const real (*restrict udip)[3], const real (*restrict uind)[3],
   const real (*restrict uinp)[3], const real (*restrict field)[3], const real (*restrict fielp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         rsd[i][j] = (udir[i][j] - uind[i][j]) * poli_inv + field[i][j];
         rsp[i][j] = (udip[i][j] - uinp[i][j]) * poli_inv + fielp[i][j];
      }
   }
}

__global__
void pcgRsd0V3(int n, const real* restrict polarity_inv, real (*restrict rsd)[3], const real (*restrict udir)[3],
   const real (*restrict uind)[3], const real (*restrict field)[3], const real (*restrict polscale)[3][3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         rsd[i][j] = (udir[i][j] - uind[i][0] * polscale[i][0][j] - uind[i][1] * polscale[i][1][j]
                        - uind[i][2] * polscale[i][2][j])
               * poli_inv
            + field[i][j];
      }
   }
}

__global__
void pcgRsd0(int n, const real* restrict polarity, real (*restrict rsd)[3], real (*restrict rsdp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
         rsdp[i][0] = 0;
         rsdp[i][1] = 0;
         rsdp[i][2] = 0;
      }
   }
}

__global__
void pcgRsd0gk(int n, const real* restrict polarity, real (*restrict rsd)[3], real (*restrict rsdp)[3], real (*restrict rsds)[3], real (*restrict rsdps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
         rsdp[i][0] = 0;
         rsdp[i][1] = 0;
         rsdp[i][2] = 0;
         rsds[i][0] = 0;
         rsds[i][1] = 0;
         rsds[i][2] = 0;
         rsdps[i][0] = 0;
         rsdps[i][1] = 0;
         rsdps[i][2] = 0;
      }
   }
}

__global__
void pcgRsd1(int n, const real* restrict polarity, real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}
}

namespace tinker {
__global__
void pcgP1(int n, const real* restrict polarity_inv, real (*restrict vec)[3], real (*restrict vecp)[3],
   const real (*restrict conj)[3], const real (*restrict conjp)[3], const real (*restrict field)[3],
   const real (*restrict fieldp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         vec[i][j] = poli_inv * conj[i][j] - field[i][j];
         vecp[i][j] = poli_inv * conjp[i][j] - fieldp[i][j];
      }
   }
}

__global__
void pcgP1gk(int n, const real* restrict polarity_inv,
   real (*restrict vec)[3], real (*restrict vecp)[3],
   const real (*restrict conj)[3], const real (*restrict conjp)[3],
   const real (*restrict field)[3], const real (*restrict fieldp)[3],
   real (*restrict vecs)[3], real (*restrict vecps)[3],
   const real (*restrict conjs)[3], const real (*restrict conjps)[3],
   const real (*restrict fields)[3], const real (*restrict fieldps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         vec[i][j] = poli_inv * conj[i][j] - field[i][j];
         vecp[i][j] = poli_inv * conjp[i][j] - fieldp[i][j];
         vecs[i][j] = poli_inv * conjs[i][j] - fields[i][j];
         vecps[i][j] = poli_inv * conjps[i][j] - fieldps[i][j];
      }
   }
}


__global__
void pcgP4(int n, const real* restrict polarity_inv, real (*restrict vec)[3], const real (*restrict conj)[3],
   const real (*restrict field)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real poli_inv = polarity_inv[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         vec[i][j] = poli_inv * conj[i][j] - field[i][j];
   }
}

__global__
void pcgP2(int n, const real* restrict polarity, const real* restrict ka, const real* restrict kap,
   const real* restrict ksum, const real* restrict ksump, real (*restrict uind)[3], real (*restrict uinp)[3],
   const real (*restrict conj)[3], const real (*restrict conjp)[3], real (*restrict rsd)[3], real (*restrict rsdp)[3],
   const real (*restrict vec)[3], const real (*restrict vecp)[3])
{
   real kaval = *ka, kapval = *kap;
   real a = *ksum / kaval, ap = *ksump / kapval;
   if (kaval == 0)
      a = 0;
   if (kapval == 0)
      ap = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += a * conj[i][j];
         uinp[i][j] += ap * conjp[i][j];
         rsd[i][j] -= a * vec[i][j];
         rsdp[i][j] -= ap * vecp[i][j];
      }
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
         rsdp[i][0] = 0;
         rsdp[i][1] = 0;
         rsdp[i][2] = 0;
      }
   }
}

__global__
void pcgP2gk(int n, const real* restrict polarity,
   const real* restrict ka, const real* restrict kap,
   const real* restrict ksum, const real* restrict ksump, real (*restrict uind)[3], real (*restrict uinp)[3],
   const real (*restrict conj)[3], const real (*restrict conjp)[3], real (*restrict rsd)[3], real (*restrict rsdp)[3],
   const real (*restrict vec)[3], const real (*restrict vecp)[3],
   const real* restrict kas, const real* restrict kaps,
   const real* restrict ksums, const real* restrict ksumps, real (*restrict uinds)[3], real (*restrict uinps)[3],
   const real (*restrict conjs)[3], const real (*restrict conjps)[3], real (*restrict rsds)[3], real (*restrict rsdps)[3],
   const real (*restrict vecs)[3], const real (*restrict vecps)[3])
{
   real kaval = *ka, kapval = *kap;
   real a = *ksum / kaval, ap = *ksump / kapval;
   real kavals = *kas, kapvals = *kaps;
   real as = *ksums / kavals, aps = *ksumps / kapvals;
   if (kaval == 0)
      a = 0;
   if (kapval == 0)
      ap = 0;
   if (kavals == 0)
      as = 0;
   if (kapvals == 0)
      aps = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += a * conj[i][j];
         uinp[i][j] += ap * conjp[i][j];
         rsd[i][j] -= a * vec[i][j];
         rsdp[i][j] -= ap * vecp[i][j];
         uinds[i][j] += as * conjs[i][j];
         uinps[i][j] += aps * conjps[i][j];
         rsds[i][j] -= as * vecs[i][j];
         rsdps[i][j] -= aps * vecps[i][j];
      }
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
         rsdp[i][0] = 0;
         rsdp[i][1] = 0;
         rsdp[i][2] = 0;
         rsds[i][0] = 0;
         rsds[i][1] = 0;
         rsds[i][2] = 0;
         rsdps[i][0] = 0;
         rsdps[i][1] = 0;
         rsdps[i][2] = 0;
      }
   }
}

__global__
void pcgP5(int n, const real* restrict polarity, const real* restrict ka, const real* restrict ksum,
   real (*restrict uind)[3], const real (*restrict conj)[3], real (*restrict rsd)[3], const real (*restrict vec)[3])
{
   real kaval = *ka;
   real a = *ksum / kaval;
   if (kaval == 0)
      a = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += a * conj[i][j];
         rsd[i][j] -= a * vec[i][j];
      }
      if (polarity[i] == 0) {
         rsd[i][0] = 0;
         rsd[i][1] = 0;
         rsd[i][2] = 0;
      }
   }
}

__global__
void pcgP3(int n, const real* restrict ksum, const real* restrict ksump, const real* restrict ksum1,
   const real* restrict ksump1, real (*restrict conj)[3], real (*restrict conjp)[3], const real (*restrict zrsd)[3],
   const real (*restrict zrsdp)[3])
{
   real kaval = *ksum, kapval = *ksump;
   real b = *ksum1 / kaval, bp = *ksump1 / kapval;
   if (kaval == 0)
      b = 0;
   if (kapval == 0)
      bp = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         conj[i][j] = zrsd[i][j] + b * conj[i][j];
         conjp[i][j] = zrsdp[i][j] + bp * conjp[i][j];
      }
   }
}

__global__
void pcgP3gk(int n,
   const real* restrict ksum, const real* restrict ksump, const real* restrict ksum1,
   const real* restrict ksump1, real (*restrict conj)[3], real (*restrict conjp)[3], const real (*restrict zrsd)[3],
   const real (*restrict zrsdp)[3],
   const real* restrict ksums, const real* restrict ksumps, const real* restrict ksum1s,
   const real* restrict ksump1s, real (*restrict conjs)[3], real (*restrict conjps)[3], const real (*restrict zrsds)[3],
   const real (*restrict zrsdps)[3])
{
   real kaval = *ksum, kapval = *ksump;
   real b = *ksum1 / kaval, bp = *ksump1 / kapval;
   real kavals = *ksums, kapvals = *ksumps;
   real bs = *ksum1s / kavals, bps = *ksump1s / kapvals;
   if (kaval == 0)
      b = 0;
   if (kapval == 0)
      bp = 0;
   if (kavals == 0)
      bs = 0;
   if (kapvals == 0)
      bps = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         conj[i][j] = zrsd[i][j] + b * conj[i][j];
         conjp[i][j] = zrsdp[i][j] + bp * conjp[i][j];
         conjs[i][j] = zrsds[i][j] + bs * conjs[i][j];
         conjps[i][j] = zrsdps[i][j] + bps * conjps[i][j];
      }
   }
}

__global__
void pcgP6(int n, const real* restrict ksum, const real* restrict ksum1, real (*restrict conj)[3],
   const real (*restrict zrsd)[3])
{
   real ksumval = *ksum;
   real b = *ksum1 / ksumval;
   if (ksumval == 0)
      b = 0;
   for (int i = ITHREAD; i < n; i += STRIDE) {
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         conj[i][j] = zrsd[i][j] + b * conj[i][j];
   }
}

__global__
void pcgPeek(int n, float pcgpeek, const real* restrict polarity, real (*restrict uind)[3], real (*restrict uinp)[3],
   const real (*restrict rsd)[3], const real (*restrict rsdp)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real term = pcgpeek * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += term * rsd[i][j];
         uinp[i][j] += term * rsdp[i][j];
      }
   }
}

__global__
void pcgPeekgk(int n, float pcgpeek, const real* restrict polarity,
   real (*restrict uind)[3], real (*restrict uinp)[3],
   const real (*restrict rsd)[3], const real (*restrict rsdp)[3],
   real (*restrict uinds)[3], real (*restrict uinps)[3],
   const real (*restrict rsds)[3], const real (*restrict rsdps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real term = pcgpeek * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j) {
         uind[i][j] += term * rsd[i][j];
         uinp[i][j] += term * rsdp[i][j];
         uinds[i][j] += term * rsds[i][j];
         uinps[i][j] += term * rsdps[i][j];
      }
   }
}

__global__
void pcgPeek1(int n, float pcgpeek, const real* restrict polarity, real (*restrict uind)[3],
   const real (*restrict rsd)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real term = pcgpeek * polarity[i];
      #pragma unroll
      for (int j = 0; j < 3; ++j)
         uind[i][j] += term * rsd[i][j];
   }
}
}
