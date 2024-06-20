#include "ff/solv/nblistgk.h"
#include "seq/launch.h"
#include "seq/triangle.h"
#include <algorithm>

namespace tinker {
__global__
void dloopStep0(int nakp, int* restrict iakp)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nakp; i += blockDim.x * gridDim.x) {
      iakp[i] = i;
   }
}

__device__
inline void dloopStep3AtomicOr(int x0, int y0, int* akpf, int* sum_nakpl)
{
   int x = max(x0, y0);
   int y = min(x0, y0);
   int f = xy_to_tri(x, y);
   int j = f / WARP_SIZE;
   int k = f & (WARP_SIZE - 1); // f % 32
   int mask = 1 << k;
   int oldflag = atomicOr(&akpf[j], mask);
   int oldkbit = oldflag & mask;
   if (oldkbit == 0) {
      atomicAdd(sum_nakpl, 1);
   }
}

__global__
void dloopStep3(int nak, int* restrict akpf, int* nakpl_ptr0, int nstype, //
   int ns1, int (*restrict js1)[2], int ns2, int (*restrict js2)[2],      //
   int ns3, int (*restrict js3)[2], int ns4, int (*restrict js4)[2])
{
   // D.1 Pairwise flag for (block i - block i) is always set.
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nak; i += blockDim.x * gridDim.x) {
      dloopStep3AtomicOr(i, i, akpf, nakpl_ptr0);
   }

   // pairwise flag
   int maxns = -1;
   maxns = max(maxns, ns1);
   maxns = max(maxns, ns2);
   maxns = max(maxns, ns3);
   maxns = max(maxns, ns4);
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < maxns; i += blockDim.x * gridDim.x) {
      int x0, y0;
      if (nstype >= 1 and i < ns1) {
         x0 = js1[i][0] / WARP_SIZE;
         y0 = js1[i][1] / WARP_SIZE;
         dloopStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 2 and i < ns2) {
         x0 = js2[i][0] / WARP_SIZE;
         y0 = js2[i][1] / WARP_SIZE;
         dloopStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 3 and i < ns3) {
         x0 = js3[i][0] / WARP_SIZE;
         y0 = js3[i][1] / WARP_SIZE;
         dloopStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
      if (nstype >= 4 and i < ns4) {
         x0 = js4[i][0] / WARP_SIZE;
         y0 = js4[i][1] / WARP_SIZE;
         dloopStep3AtomicOr(x0, y0, akpf, nakpl_ptr0);
      }
   }
}

__global__
void dloopStep4(int nakpk, int* restrict nakpl_ptr1, const int* restrict akpf,
   int* restrict iakpl, int* restrict iakpl_rev)
{
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < nakpk; i += blockDim.x * gridDim.x) {
      int flag = akpf[i];
      int count = __popc(flag);
      int base = atomicAdd(nakpl_ptr1, count);
      int c = 0;
      while (flag) {
         int j = __ffs(flag) - 1;
         flag &= (flag - 1);
         int tri = WARP_SIZE * i + j;
         iakpl[base + c] = tri;
         iakpl_rev[tri] = base + c;
         ++c;
      }
   }
}

__device__
void dloopStep5Bits(int x0, int y0, unsigned int* bit0, const int* iakpl_rev)
{
   int x, y;
   int bx, by, ax, ay;
   int f, fshort, pos;
   x = max(x0, y0);
   y = min(x0, y0);
   bx = x / WARP_SIZE;
   ax = x & (WARP_SIZE - 1);
   by = y / WARP_SIZE;
   ay = y & (WARP_SIZE - 1);
   f = xy_to_tri(bx, by);
   fshort = iakpl_rev[f];
   pos = WARP_SIZE * fshort + ax;
   atomicOr(&bit0[pos], 1 << ay);
   if (bx == by) {
      pos = WARP_SIZE * fshort + ay;
      atomicOr(&bit0[pos], 1 << ax);
   }
}

__global__
void dloopStep5(const int* restrict iakpl_rev, int nstype,
   DLoop::ScaleInfo si1, DLoop::ScaleInfo si2, DLoop::ScaleInfo si3, DLoop::ScaleInfo si4)
{
   int maxns = -1;
   maxns = max(maxns, si1.ns);
   maxns = max(maxns, si2.ns);
   maxns = max(maxns, si3.ns);
   maxns = max(maxns, si4.ns);
   for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < maxns; i += blockDim.x * gridDim.x) {
      int x0, y0;
      if (nstype >= 1 and i < si1.ns) {
         auto& si = si1;
         x0 = si.js[i][0];
         y0 = si.js[i][1];
         dloopStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 2 and i < si2.ns) {
         auto& si = si2;
         x0 = si.js[i][0];
         y0 = si.js[i][1];
         dloopStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 3 and i < si3.ns) {
         auto& si = si3;
         x0 = si.js[i][0];
         y0 = si.js[i][1];
         dloopStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
      if (nstype >= 4 and i < si4.ns) {
         auto& si = si4;
         x0 = si.js[i][0];
         y0 = si.js[i][1];
         dloopStep5Bits(x0, y0, si.bit0, iakpl_rev);
      }
   }
}
}

namespace tinker {
void dloopDataInit_cu(DLoopUnit u)
{
   const int n = u->n;
   auto& si1 = u->si1;
   auto& si2 = u->si2;
   auto& si3 = u->si3;
   auto& si4 = u->si4;

   darray::zero(g::q0, u->nakp, u->iakp);
   darray::zero(g::q0, u->cap_nakpl, u->iakpl);
   darray::zero(g::q0, u->cap_nakpa, u->iakpa);
   darray::zero(g::q0, u->nakp, u->iakpl_rev);
   darray::zero(g::q0, u->nakpk, u->akpf);
   darray::zero(g::q0, 2, u->worker);
   if (u->nstype >= 1) {
      auto& si = si1;
      darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
   }
   if (u->nstype >= 2) {
      auto& si = si2;
      darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
   }
   if (u->nstype >= 3) {
      auto& si = si3;
      darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
   }
   if (u->nstype >= 4) {
      auto& si = si4;
      darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
   }

   launch_k1s(g::s0, u->nakp, dloopStep0, u->nakp, u->iakp);  

   int* nakpl_ptr0 = &u->worker[0];
   launch_k1s(g::s0, n, dloopStep3,   //
      u->nak, u->akpf, nakpl_ptr0,    //
      u->nstype,                      //
      si1.ns, si1.js, si2.ns, si2.js, //
      si3.ns, si3.js, si4.ns, si4.js);
   darray::copyout(g::q0, 1, &u->nakpl, nakpl_ptr0);
   waitFor(g::q0);
   if (WARP_SIZE + u->nakpl > (unsigned)u->cap_nakpl) {
      u->cap_nakpl = WARP_SIZE + u->nakpl;
      darray::deallocate(u->iakpl);
      darray::allocate(u->cap_nakpl, &u->iakpl);
      darray::zero(g::q0, u->cap_nakpl, u->iakpl);
      if (u->nstype >= 1) {
         auto& si = si1;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
         darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
      }
      if (u->nstype >= 2) {
         auto& si = si2;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
         darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
      }
      if (u->nstype >= 3) {
         auto& si = si3;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
         darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
      }
      if (u->nstype >= 4) {
         auto& si = si4;
         darray::deallocate(si.bit0);
         darray::allocate(32 * u->cap_nakpl, &si.bit0);
         darray::zero(g::q0, 32 * u->cap_nakpl, si.bit0);
      }
   }

   int* nakpl_ptr1 = &u->worker[1];
   launch_k1s(g::s0, u->nakpk, dloopStep4,
      u->nakpk, nakpl_ptr1, u->akpf, u->iakpl, u->iakpl_rev);

   std::vector<int> skip_pair(u->nakpl);
   darray::copyout(g::q0, u->nakpl, skip_pair.data(), u->iakpl);
   waitFor(g::q0);
   std::sort(skip_pair.begin(), skip_pair.end());
   std::vector<int> all_pair;
   for (int i = 0; i < u->nakp; i++) {
      all_pair.push_back(i);
   }
   std::vector<int> iakpa;
   std::set_difference(all_pair.begin(), all_pair.end(), skip_pair.begin(), skip_pair.end(), std::back_inserter(iakpa));
   u->nakpa = iakpa.size();
   if (WARP_SIZE + u->nakpa > (unsigned)u->cap_nakpa) {
      u->cap_nakpa = WARP_SIZE + u->nakpa;
      darray::deallocate(u->iakpa);
      darray::allocate(u->cap_nakpa, &u->iakpa);
      darray::zero(g::q0, u->cap_nakpa, u->iakpa);
   }
   darray::copyin(g::q0, u->nakpa, u->iakpa, iakpa.data());
   waitFor(g::q0);

   if (box_shape == BoxShape::UNBOUND) {
      launch_k1s(g::s0, u->nakp, dloopStep5, u->iakpl_rev, //
         u->nstype, u->si1, u->si2, u->si3, u->si4);
   } else {
      assert(false);
   }
}
}
