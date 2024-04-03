#include "ff/modamoeba.h"
#include "ff/cumodamoeba.h"
#include "ff/dloop.h"
#include "ff/solv/solute.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "seq/launch.h"
#include "seq/pair_field.h"
#include "seq/pair_fieldgk.h"
#include "seq/triangle.h"
#include <tinker/detail/limits.hh>

namespace tinker {
#include "dfieldNonEwaldN2_cu1.cc"

void dfieldNonEwaldN2_cu(real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);

   dfieldNonEwaldN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, off, st.si3.bit0, ndpexclude, dpexclude, dpexclude_scale,
      st.x, st.y, st.z, st.sorted,st.nakpl, st.iakpl, st.niakp, st.iakp, field, fieldp);
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
#include "ufieldNonEwaldN2_cu1.cc"
#include "ufieldgk1_cu1.cc"
#include "ufieldgk2_cu1.cc"
#include "ufieldgk1N2_cu1.cc"
#include "ufieldgk2N2_cu1.cc"

void ufieldNonEwaldN2_cu(const real (*uind)[3], const real (*uinp)[3], real (*field)[3], real (*fieldp)[3])
{
   const auto& st = *mspatial_v2_unit;
   const real off = switchOff(Switch::MPOLE);

   darray::zero(g::q0, n, field, fieldp);
   int ngrid = gpuGridSize(BLOCK_DIM);

   ufieldNonEwaldN2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, off, st.si4.bit0, nuexclude, uexclude, uexclude_scale, st.x, st.y, st.z, st.sorted,
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
         st.nakpl, st.iakpl, st.niakp, st.iakp, uinds, uinps, fields, fieldps);

      const auto& st2 = *mn2_unit;
      ufieldgk2N2_cu1<<<ngrid, BLOCK_DIM, 0, g::s0>>>(n, off, x, y, z, st2.nakp, st2.iakp, uinds, uinps, rborn, gkc, fd, fields, fieldps);
   }

   launch_k1s(g::s0, n, ufieldgkSelf_cu1, n, gkc, fd, uinds, uinps, rborn, fields, fieldps);
}
}
