#include "ff/evdw.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "math/switch.h"
#include "seq/add.h"
#include "seq/launch.h"
#include "seq/pair_hal.h"
#include "seq/triangle.h"

namespace tinker {
#if 1
#define GHAL    (real)0.12
#define DHAL    (real)0.07
#define SCEXP   5
#define SCALPHA (real)0.7
#elif 0
#define GHAL    ghal
#define DHAL    dhal
#define SCEXP   scexp
#define SCALPHA scalpha
#endif
#include "ehalN2_cu1.cc"

template <class Ver>
static void ehalN2_cu1()
{
   constexpr bool do_g = Ver::g;

   const auto& st = *vspatial_v2_unit;
   const real cut = switchCut(Switch::VDW);
   const real off = switchOff(Switch::VDW);

   if CONSTEXPR (do_g)
      darray::zero(g::q0, n, gxred, gyred, gzred);

   int ngrid = gpuGridSize(BLOCK_DIM);

   ehalN2_cu1<Ver><<<ngrid, BLOCK_DIM, 0, g::s0>>>(st.n, nev, ev, vir_ev, gxred, gyred, gzred, cut, off,
      st.si1.bit0, nvexclude, vexclude, vexclude_scale, st.x, st.y, st.z, st.sorted, st.nakpl, st.iakpl, st.niakp,
      st.iakp, njvdw, vlam, vcouple, radmin, epsilon, jvdw, mut);

   if CONSTEXPR (do_g) {
      ehalResolveGradient();
   }
}

void ehalN2_cu(int vers)
{
   if (vers == calc::v0)
      ehalN2_cu1<calc::V0>();
   else if (vers == calc::v1)
      ehalN2_cu1<calc::V1>();
   else if (vers == calc::v3)
      ehalN2_cu1<calc::V3>();
   else if (vers == calc::v4)
      ehalN2_cu1<calc::V4>();
   else if (vers == calc::v5)
      ehalN2_cu1<calc::V5>();
   else if (vers == calc::v6)
      ehalN2_cu1<calc::V6>();
}
}
