#include "ff/nblist.h"
#include "ff/solv/nblistgk.h"
#include "ff/modamoeba.h"
#include "ff/atom.h"
#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/elec.h"
#include "ff/evdw.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/modhippo.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include "ff/switch.h"
#include "tool/externfunc.h"
#include "tool/thrustcache.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/limits.hh>
#include <tinker/detail/mplpot.hh>
#include <tinker/detail/neigh.hh>
#include <tinker/detail/polpot.hh>

namespace tinker {
NblGK vlistVersiongk()
{
   NblGK u;
   if (not use(Potent::VDW)) {
      u = NblGK::UNDEFINED;
   } else if (vdwtyp != Vdw::HAL) {
      u = NblGK::UNDEFINED;
   } else if (limits::use_vlist) {
      u = NblGK::NEIGHBOR_LIST;
   } else {
      u = NblGK::DOUBLE_LOOP;
   }
   return u;
}

NblGK mlistVersiongk()
{
   NblGK u;
   if (not use(Potent::MPOLE) and not use(Potent::POLAR) and not use(Potent::CHGTRN) and
      not use(Potent::REPULS) and not use(Potent::SOLV)) {
      u = NblGK::UNDEFINED;
   } else if (limits::use_mlist) {
      u = NblGK::NEIGHBOR_LIST;
   } else {
      u = NblGK::DOUBLE_LOOP;
   }
   return u;
}

NblGK ulistVersiongk()
{
   NblGK u;
   if (not use(Potent::POLAR)) {
      u = NblGK::UNDEFINED;
   } else if (limits::use_ulist) {
      u = NblGK::NEIGHBOR_LIST;
   } else {
      u = NblGK::DOUBLE_LOOP;
   }
   return u;
}
}

namespace tinker {
#if TINKER_CUDART
static bool alloc_thrust_cache;
#endif

void nblistgkData(RcOp op)
{
   if (op & RcOp::DEALLOC) {
      NBListUnit::clear();
      vlist_unit.close();
      mlist_unit.close();
      ulist_unit.close();

      DLoopUnit::clear();
      vdloop_unit.close();
      mdloop_unit.close();
      udloop_unit.close();

#if TINKER_CUDART
      SpatialUnit::clear();
      vspatial_v2_unit.close();
      mspatial_v2_unit.close();
      uspatial_v2_unit.close();

      ThrustCache::deallocate();
#endif
   }

   if (op & RcOp::ALLOC) {
      assert(NBListUnit::size() == 0);
      assert(SpatialUnit::size() == 0);
      assert(DLoopUnit::size() == 0);
   }

#if TINKER_CUDART
   alloc_thrust_cache = false;
#endif
   NblGK u = NblGK::UNDEFINED;
   Nbl unb = Nbl::UNDEFINED;
   double cut = 0;
   double buf = 0;
   bool option1,option2,option3;

   // vlist
   u = vlistVersiongk();
   cut = switchOff(Switch::VDW);
   buf = neigh::lbuffer;
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option2 = (pltfm_config == Platform::CUDA) and (u == NblGK::DOUBLE_LOOP);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option3) {
      if (u == NblGK::NEIGHBOR_LIST) unb = Nbl::VERLET;
      else if (u == NblGK::DOUBLE_LOOP) unb = Nbl::DOUBLE_LOOP;
   }
   if (option1) {
      auto& un2 = vspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         spatialAlloc(un2, n, cut, buf, xred, yred, zred, 1, nvexclude, vexclude);
      }
      if (op & RcOp::INIT) {
         ehalReduceXyz();
         spatialBuild(un2);
      }
   } else if (option2) {
      auto& un2 = vdloop_unit;
      if (op & RcOp::ALLOC) {
         DLoop::dataAlloc(un2, n, 1, nvexclude, vexclude);
      }
      if (op & RcOp::INIT) {
         ehalReduceXyz();
         DLoop::dataInit(un2);
      }
   } else if (option3) {
      auto& unt = vlist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(unb, unt, 2500, cut, buf, xred, yred, zred);
      }
      if (op & RcOp::INIT) {
         ehalReduceXyz();
         nblistBuild(unt);
      }
   }

   // mlist
   u = mlistVersiongk();
   cut = useEwald() ? switchOff(Switch::EWALD) : switchOff(Switch::MPOLE);
   buf = neigh::lbuffer;
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option2 = (pltfm_config == Platform::CUDA) and (u == NblGK::DOUBLE_LOOP);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option3) {
      if (u == NblGK::NEIGHBOR_LIST) unb = Nbl::VERLET;
      else if (u == NblGK::DOUBLE_LOOP) unb = Nbl::DOUBLE_LOOP;
   }
   if (option1) {
      auto& un2 = mspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            spatialAlloc(
               un2, n, cut, buf, x, y, z, 2, nmdwexclude, mdwexclude, nrepexclude, repexclude);
         } else if (mplpot::use_chgpen and polpot::use_tholed) { // AMOEBA Plus
            spatialAlloc(un2, n, cut, buf, x, y, z, 3, nmdwexclude, mdwexclude, nmdpuexclude,
               mdpuexclude, nuexclude, uexclude);
         } else { // AMOEBA
            spatialAlloc(un2, n, cut, buf, x, y, z, 4, nmdpuexclude, mdpuexclude, nmexclude,
               mexclude, ndpexclude, dpexclude, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   } else if (option2) {
      auto& un2 = mdloop_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            DLoop::dataAlloc(un2, n, 2, nmdwexclude, mdwexclude, nrepexclude, repexclude);
         } else if (mplpot::use_chgpen and polpot::use_tholed) { // AMOEBA Plus
            DLoop::dataAlloc(un2, n, 3, nmdwexclude, mdwexclude, nmdpuexclude,
               mdpuexclude, nuexclude, uexclude);
         } else { // AMOEBA
            DLoop::dataAlloc(un2, n, 4, nmdpuexclude, mdpuexclude, nmexclude,
               mexclude, ndpexclude, dpexclude, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         DLoop::dataInit(un2);
      }
   } else if (option3) {
      auto& unt = mlist_unit;
      if (op & RcOp::ALLOC) {
         nblistAlloc(unb, unt, 2500, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }

   // ulist
   u = ulistVersiongk();
   cut = switchOff(Switch::USOLVE);
   buf = neigh::pbuffer;
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option2 = (pltfm_config == Platform::CUDA) and (u == NblGK::DOUBLE_LOOP);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option3) {
      if (u == NblGK::NEIGHBOR_LIST) unb = Nbl::VERLET;
      else if (u == NblGK::DOUBLE_LOOP) unb = Nbl::DOUBLE_LOOP;
   }
   if (option1) {
      auto& un2 = uspatial_v2_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            spatialAlloc(un2, n, cut, buf, x, y, z, 1, nwexclude, wexclude);
         } else { // AMOEBA and AMOEBA Plus
            spatialAlloc(un2, n, cut, buf, x, y, z, 1, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         spatialBuild(un2);
      }
   } else if (option2) {
      auto& un2 = udloop_unit;
      if (op & RcOp::ALLOC) {
         if (mplpot::use_chgpen and not polpot::use_tholed) { // HIPPO
            DLoop::dataAlloc(un2, n, 1, nwexclude, wexclude);
         } else { // AMOEBA and AMOEBA Plus
            DLoop::dataAlloc(un2, n, 1, nuexclude, uexclude);
         }
      }
      if (op & RcOp::INIT) {
         DLoop::dataInit(un2);
      }
   } else if (option3) {
      auto& unt = ulist_unit;
      if (op & RcOp::ALLOC) {
         const int maxnlst = 500;
         nblistAlloc(unb, unt, maxnlst, cut, buf, x, y, z);
      }
      if (op & RcOp::INIT) {
         nblistBuild(unt);
      }
   }

#if TINKER_CUDART
   if (alloc_thrust_cache)
      ThrustCache::allocate();
#endif
}

void nblistgkRefresh()
{
   NblGK u = NblGK::UNDEFINED;
   Nbl unb = Nbl::UNDEFINED;
   bool option1,option2,option3;
   // vlist
   u = vlistVersiongk();
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option2 = (pltfm_config == Platform::CUDA) and (u == NblGK::DOUBLE_LOOP);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option1) {
      auto& un2 = vspatial_v2_unit;
      ehalReduceXyz();
      spatialUpdate(un2);
   } else if (option2) {
      ehalReduceXyz();
   } else if (option3) {
      auto& unt = vlist_unit;
      ehalReduceXyz();
      nblistUpdate(unt);
   }

   // mlist
   u = mlistVersiongk();
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option1) {
      auto& un2 = mspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   } else if (option3) {
      auto& unt = mlist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }

   // ulist
   u = ulistVersiongk();
   option1 = (pltfm_config == Platform::CUDA) and (u == NblGK::NEIGHBOR_LIST);
   option3 = (pltfm_config == Platform::ACC) and (u != NblGK::UNDEFINED);
   if (option1) {
      auto& un2 = uspatial_v2_unit;
      if (rc_flag & calc::traj) {
         un2->x = x;
         un2->y = y;
         un2->z = z;
      }
      spatialUpdate(un2);
   } else if (option3) {
      auto& unt = ulist_unit;
      if (rc_flag & calc::traj) {
         unt->x = x;
         unt->y = y;
         unt->z = z;
         unt.deviceptrUpdate(*unt, g::q0);
      }
      nblistUpdate(unt);
   }
}
}

namespace tinker {
DLoop::~DLoop()
{
   darray::deallocate(iakp, iakpl, iakpa, worker);
   darray::deallocate(iakpl_rev, akpf);
   darray::deallocate(si1.bit0, si2.bit0, si3.bit0, si4.bit0);
}

void DLoop::dataAlloc(DLoopUnit& u, int n, int nstype, //
   int ns1, int (*js1)[2], int ns2, int (*js2)[2],     //
   int ns3, int (*js3)[2], int ns4, int (*js4)[2])
{
   u = DLoopUnit::open();
   auto& st = *u;

   // output
   st.nakpl = 0;
   st.nakpa = 0;
   st.iakp = nullptr;
   st.iakpl = nullptr;
   st.iakpa = nullptr;

   // internal
   st.n = n;
   st.nak = (n + DLoop::BLOCK - 1) / DLoop::BLOCK;
   st.nakp = (st.nak + 1) * st.nak / 2;
   st.nakpk = (st.nakp + DLoop::BLOCK - 1) / DLoop::BLOCK;
   st.cap_nakpl = 32 + 8 * st.nak;
   st.cap_nakpa = 32 + 8 * st.nak;

   darray::allocate(st.nakp, &st.iakp);
   darray::allocate(st.cap_nakpl, &st.iakpl);
   darray::allocate(st.cap_nakpa, &st.iakpa);

   darray::allocate(st.nakp, &st.iakpl_rev);
   darray::allocate(st.nakpk, &st.akpf);

   // darray::allocate(std::max(128, st.n * 2), &st.worker);
   darray::allocate(2, &st.worker);

   st.nstype = nstype;
   st.si1.init();
   st.si2.init();
   st.si3.init();
   st.si4.init();
   if (nstype >= 1) {
      st.si1.set(ns1, js1);
      darray::allocate(32 * st.cap_nakpl, &st.si1.bit0);
   }
   if (nstype >= 2) {
      st.si2.set(ns2, js2);
      darray::allocate(32 * st.cap_nakpl, &st.si2.bit0);
   }
   if (nstype >= 3) {
      st.si3.set(ns3, js3);
      darray::allocate(32 * st.cap_nakpl, &st.si3.bit0);
   }
   if (nstype >= 4) {
      st.si4.set(ns4, js4);
      darray::allocate(32 * st.cap_nakpl, &st.si4.bit0);
   }
}

void DLoop::ScaleInfo::init()
{
   js = nullptr;
   bit0 = nullptr;
   ns = 0;
}

void DLoop::ScaleInfo::set(int nns, int (*jjs)[2])
{
   ns = nns;
   js = jjs;
}

TINKER_FVOID2(acc0, cu1, dloopDataInit, DLoopUnit);
void DLoop::dataInit(DLoopUnit u)
{
   TINKER_FCALL2(acc0, cu1, dloopDataInit, u);
}
}
