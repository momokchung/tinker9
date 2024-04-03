#include "ff/modamoeba.h"
#include "ff/cumodamoeba.h"
#include "ff/dloop.h"
#include "ff/image.h"
#include "ff/pme.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "seq/pair_field.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>

namespace tinker {
__global__
void dfieldEwaldRecipSelfP2_cu1(int n, real (*restrict field)[3], real term, const real (*restrict rpole)[MPL_TOTAL],
   const real (*restrict cphi)[10])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      field[i][0] += (-cphi[i][1] + term * dix);
      field[i][1] += (-cphi[i][2] + term * diy);
      field[i][2] += (-cphi[i][3] + term * diz);
   }
}

void dfieldEwaldRecipSelfP2_cu(real (*field)[3])
{
   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;
   const real term = aewald * aewald * aewald * 4 / 3 / sqrtpi;

   launch_k1s(g::s0, n, dfieldEwaldRecipSelfP2_cu1, n, field, term, rpole, cphi);
}

#include "dfield_cu1.cc"

void dfieldEwaldReal_cu(real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::EWALD);

   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ngrid *= BLOCK_DIM;
   int nparallel = std::max(st.niak, st.nakpl) * WARP_SIZE;
   nparallel = std::max(nparallel, ngrid);
   launch_k1s(g::s0, nparallel, dfield_cu1<EWALD>, //
      st.n, TINKER_IMAGE_ARGS, off, st.si3.bit0, ndpexclude, dpexclude, dpexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niak, st.iak, st.lst, field, fieldp, aewald);
}

void dfieldNonEwald_cu(real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);
   ngrid *= BLOCK_DIM;
   int nparallel = std::max(st.niak, st.nakpl) * WARP_SIZE;
   nparallel = std::max(nparallel, ngrid);
   launch_k1s(g::s0, nparallel, dfield_cu1<NON_EWALD>, //
      st.n, TINKER_IMAGE_ARGS, off, st.si3.bit0, ndpexclude, dpexclude, dpexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niak, st.iak, st.lst, field, fieldp, 0);
}

#include "dfieldN2_cu1.cc"

void dfieldNonEwaldN2_cu(real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);

   dfieldN2_cu1<NON_EWALD><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, off, st.si3.bit0, ndpexclude, dpexclude, dpexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niakp, st.iakp, field, fieldp, 0);
}
}

namespace tinker {
#include "dfieldgk_cu1.cc"
#include "dfieldgkN2_cu1.cc"

__global__
static void dfieldgkSelf_cu1(int n, real fd, const real* restrict rborn, real (*restrict fields)[3], real (*restrict fieldps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      using d::rpole;
      real dix = rpole[i][MPL_PME_X];
      real diy = rpole[i][MPL_PME_Y];
      real diz = rpole[i][MPL_PME_Z];
      real rbi = rborn[i];

      real rb2 = rbi * rbi;
      real gf2 = 1 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real a10 = -fd * gf3;

      real fx = dix*a10;
      real fy = diy*a10;
      real fz = diz*a10;

      fields[i][0] += fx;
      fields[i][1] += fy;
      fields[i][2] += fz;
      fieldps[i][0] += fx;
      fieldps[i][1] += fy;
      fieldps[i][2] += fz;
   }
}

void dfieldgk_cu(real gkc, real fc, real fd, real fq, real (*fields)[3], real (*fieldps)[3])
{
   const real off = switchOff(Switch::MPOLE);

   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      const auto& st = *mspatial_v2_unit;
      dfieldgk_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.x, st.y, st.z, st.sorted,
         st.nakpl, st.iakpl, st.niak, st.iak, st.lst, fields, fieldps, rborn, gkc, fc, fd, fq);
   } else {
      const auto& st = *mn2_unit;
      dfieldgkN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, off, x, y, z, st.nakp, st.iakp, fields, fieldps, rborn, gkc, fc, fd, fq);
   }

   launch_k1s(g::s0, n, dfieldgkSelf_cu1, n, fd, rborn, fields, fieldps);
}
}

namespace tinker {
__global__
void ufieldEwaldRecipSelfP1_cu1(int n, const real (*restrict uind)[3], const real (*restrict uinp)[3],
   real (*restrict field)[3], real (*restrict fieldp)[3], const real (*restrict fdip_phi1)[10],
   const real (*restrict fdip_phi2)[10], real term, int nfft1, int nfft2, int nfft3, TINKER_IMAGE_PARAMS)
{
   real a[3][3];
   a[0][0] = nfft1 * recipa.x;
   a[1][0] = nfft2 * recipb.x;
   a[2][0] = nfft3 * recipc.x;
   a[0][1] = nfft1 * recipa.y;
   a[1][1] = nfft2 * recipb.y;
   a[2][1] = nfft3 * recipc.y;
   a[0][2] = nfft1 * recipa.z;
   a[1][2] = nfft2 * recipb.z;
   a[2][2] = nfft3 * recipc.z;

   if (uinp) {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         for (int j = 0; j < 3; ++j) {
            real df1 = a[0][j] * fdip_phi1[i][1] + a[1][j] * fdip_phi1[i][2] + a[2][j] * fdip_phi1[i][3];
            real df2 = a[0][j] * fdip_phi2[i][1] + a[1][j] * fdip_phi2[i][2] + a[2][j] * fdip_phi2[i][3];
            field[i][j] += (term * uind[i][j] - df1);
            fieldp[i][j] += (term * uinp[i][j] - df2);
         }
      }
   } else {
      for (int i = ITHREAD; i < n; i += STRIDE) {
         for (int j = 0; j < 3; ++j) {
            real df1 = a[0][j] * fdip_phi1[i][1] + a[1][j] * fdip_phi1[i][2] + a[2][j] * fdip_phi1[i][3];
            field[i][j] += (term * uind[i][j] - df1);
         }
      }
   }
}

void ufieldEwaldRecipSelfP1_cu(const real (*uind)[3], const real (*uinp)[3], //
   real (*field)[3], real (*fieldp)[3])
{
   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;
   const real term = aewald * aewald * aewald * 4 / 3 / sqrtpi;
   const int nfft1 = pu->nfft1;
   const int nfft2 = pu->nfft2;
   const int nfft3 = pu->nfft3;

   launch_k1s(g::s0, n, ufieldEwaldRecipSelfP1_cu1, n, uind, uinp, field, fieldp, fdip_phi1, fdip_phi2, term, nfft1,
      nfft2, nfft3, TINKER_IMAGE_ARGS);
}

#include "ufield_cu1.cc"

void ufieldEwaldReal_cu(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::EWALD);

   const PMEUnit pu = ppme_unit;
   const real aewald = pu->aewald;

   int ngrid = gpuGridSize(BLOCK_DIM);
   ngrid *= BLOCK_DIM;
   int nparallel = std::max(st.niak, st.nakpl) * WARP_SIZE;
   nparallel = std::max(nparallel, ngrid);
   launch_k1s(g::s0, nparallel, ufield_cu1<EWALD>, //
      st.n, TINKER_IMAGE_ARGS, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uind, uinp, field, fieldp, aewald);
}

void ufieldNonEwald_cu(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);
   ngrid *= BLOCK_DIM;
   int nparallel = std::max(st.niak, st.nakpl) * WARP_SIZE;
   nparallel = std::max(nparallel, ngrid);
   launch_k1s(g::s0, nparallel, ufield_cu1<NON_EWALD>, //
      st.n, TINKER_IMAGE_ARGS, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uind, uinp, field, fieldp, 0);
}
}

namespace tinker {

#include "ufieldN2_cu1.cc"
#include "ufieldgk1_cu1.cc"
#include "ufieldgk2_cu1.cc"
#include "ufieldgk1N2_cu1.cc"
#include "ufieldgk2N2_cu1.cc"

void ufieldN2_cu(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);

   ufieldN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
      st.nakpl, st.iakpl, st.niakp, st.iakp, uind, uinp, field, fieldp);
}

__global__
static void ufieldgkSelf_cu1(int n, real gkc, real fd, const real (*restrict uinds)[3], const real (*restrict uinps)[3], const real* restrict rborn, real (*restrict fields)[3], real (*restrict fieldps)[3])
{
   for (int i = ITHREAD; i < n; i += STRIDE) {
      real duixs = uinds[i][0];
      real duiys = uinds[i][1];
      real duizs = uinds[i][2];
      real puixs = uinps[i][0];
      real puiys = uinps[i][1];
      real puizs = uinps[i][2];
      real rbi = rborn[i];

      real rb2 = rbi * rbi;
      real gf2 = 1 / rb2;
      real gf = REAL_SQRT(gf2);
      real gf3 = gf2 * gf;
      real a10 = -gf3;
      real gu = fd * a10;
      fields[i][0] += duixs*gu;
      fields[i][1] += duiys*gu;
      fields[i][2] += duizs*gu;
      fieldps[i][0] += puixs*gu;
      fieldps[i][1] += puiys*gu;
      fieldps[i][2] += puizs*gu;
   }
}

void ufieldgk_cu(real gkc, real fd, const real (*uinds)[3], const real (*uinps)[3], real (*fields)[3], real (*fieldps)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, fields, fieldps);
   int ngrid = gpuGridSize(BLOCK_DIM);

   if (limits::use_mlist) {
      ufieldgk1_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
         st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uinds, uinps, fields, fieldps);

      ufieldgk2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, TINKER_IMAGE_ARGS, off, st.x, st.y, st.z, st.sorted,
         st.nakpl, st.iakpl, st.niak, st.iak, st.lst, uinds, uinps, rborn, gkc, fd, fields, fieldps);
   } else {
      ufieldgk1N2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
         st.nakpl, st.iakpl, st.niakp, st.iakp,  uinds, uinps, fields, fieldps);

      const auto& st2 = *mn2_unit;
      ufieldgk2N2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, off, x, y, z, st2.nakp, st2.iakp, uinds, uinps, rborn, gkc, fd, fields, fieldps);
   }

   launch_k1s(g::s0, n, ufieldgkSelf_cu1, n, gkc, fd, uinds, uinps, rborn, fields, fieldps);
}
}
