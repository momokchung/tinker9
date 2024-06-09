#include "md/misc.h"
#include "md/pq.h"
#include "seq/launch.h"
#include "seq/reduce.h"
#include "tool/externfunc.h"
#include <tinker/detail/bound.hh>
#include <tinker/detail/inform.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/molcul.hh>
#include <tinker/detail/units.hh>

namespace tinker {
template <unsigned int B>
__global__
void mdrestSumP_cu(int n, vel_prec* restrict odata, const double* restrict mass,
   const vel_prec* restrict vx, const vel_prec* restrict vy, const vel_prec* restrict vz)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   vel_prec x = 0, y = 0, z = 0;
   for (int i = ithread; i < n; i += stride) {
      auto m = mass[i];
      x += m * vx[i];
      y += m * vy[i];
      z += m * vz[i];
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = x; ty[t] = y; tz[t] = z;                                          __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   if (t == 0) {
      const int b = blockIdx.x;
      odata[3 * b + 0] = tx[t] + tx[t + 1];
      odata[3 * b + 1] = ty[t] + ty[t + 1];
      odata[3 * b + 2] = tz[t] + tz[t + 1];
   }
}

template <int B>
__global__
void mdrestRemoveP_cu(int n, double invtotmass, const vel_prec* restrict idata,
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz, vel_prec* restrict xout)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   vel_prec x = 0, y = 0, z = 0;
   for (int i = t; i < gridDim.x; i += B) {
      x += idata[3 * i + 0];
      y += idata[3 * i + 1];
      z += idata[3 * i + 2];
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = x; ty[t] = y; tz[t] = z;                                          __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   x = (tx[0] + tx[1]) * invtotmass;
   y = (ty[0] + ty[1]) * invtotmass;
   z = (tz[0] + tz[1]) * invtotmass;
   xout[0] = x;
   xout[1] = y;
   xout[2] = z;
   for (int i = ithread; i < n; i += stride) {
      vx[i] -= x;
      vy[i] -= y;
      vz[i] -= z;
   }
}

template <unsigned int B>
__global__
void mdrestSumX1_cu(int n, pos_prec* restrict odata, const double* restrict mass,
   const pos_prec* restrict x, const pos_prec* restrict y, const pos_prec* restrict z)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   pos_prec xsum = 0, ysum = 0, zsum = 0;
   for (int i = ithread; i < n; i += stride) {
      auto m = mass[i];
      xsum += m * x[i];
      ysum += m * y[i];
      zsum += m * z[i];
   }

   __shared__ pos_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = xsum; ty[t] = ysum; tz[t] = zsum;                                 __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   if (t == 0) {
      const int b = blockIdx.x;
      odata[3 * b + 0] = tx[t] + tx[t + 1];
      odata[3 * b + 1] = ty[t] + ty[t + 1];
      odata[3 * b + 2] = tz[t] + tz[t + 1];
   }
}

template <int B>
__global__
void mdrestSumX2_cu(int n, double invtotmass, const pos_prec* restrict idata, pos_prec* restrict xout)
{
   static_assert(B == 64, "");
   const int t = threadIdx.x;

   pos_prec xsum = 0, ysum = 0, zsum = 0;
   for (int i = t; i < gridDim.x; i += B) {
      xsum += idata[3 * i + 0];
      ysum += idata[3 * i + 1];
      zsum += idata[3 * i + 2];
   }

   __shared__ pos_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = xsum; ty[t] = ysum; tz[t] = zsum;                                 __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   xsum = (tx[0] + tx[1]) * invtotmass;
   ysum = (ty[0] + ty[1]) * invtotmass;
   zsum = (tz[0] + tz[1]) * invtotmass;
   xout[0] = xsum;
   xout[1] = ysum;
   xout[2] = zsum;
}

template <unsigned int B>
__global__
void mdrestSumA1_cu(int n, vel_prec* restrict odata, const double* restrict mass,
   const pos_prec* restrict x, const pos_prec* restrict y, const pos_prec* restrict z,
   const vel_prec* restrict vx, const vel_prec* restrict vy, const vel_prec* restrict vz)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   vel_prec xsum = 0, ysum = 0, zsum = 0;
   for (int i = ithread; i < n; i += stride) {
      auto m = mass[i];
      xsum += m * (y[i] * vz[i] - z[i] * vy[i]);
      ysum += m * (z[i] * vx[i] - x[i] * vz[i]);
      zsum += m * (x[i] * vy[i] - y[i] * vx[i]);
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = xsum; ty[t] = ysum; tz[t] = zsum;                                 __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   if (t == 0) {
      const int b = blockIdx.x;
      odata[3 * b + 0] = tx[t] + tx[t + 1];
      odata[3 * b + 1] = ty[t] + ty[t + 1];
      odata[3 * b + 2] = tz[t] + tz[t + 1];
   }
}

template <int B>
__global__
void mdrestSumA2_cu(int n, const vel_prec* restrict idata, vel_prec* restrict xout)
{
   static_assert(B == 64, "");
   const int t = threadIdx.x;

   vel_prec xsum = 0, ysum = 0, zsum = 0;
   for (int i = t; i < gridDim.x; i += B) {
      xsum += idata[3 * i + 0];
      ysum += idata[3 * i + 1];
      zsum += idata[3 * i + 2];
   }

   __shared__ vel_prec tx[B], ty[B], tz[B];
   // clang-format off
   tx[t] = xsum; ty[t] = ysum; tz[t] = zsum;                                 __syncthreads();
   if (t < 32) { tx[t] += tx[t+32]; ty[t] += ty[t+32]; tz[t] += tz[t+32]; }  __syncthreads();
   if (t < 16) { tx[t] += tx[t+16]; ty[t] += ty[t+16]; tz[t] += tz[t+16]; }  __syncthreads();
   if (t <  8) { tx[t] += tx[t+ 8]; ty[t] += ty[t+ 8]; tz[t] += tz[t+ 8]; }  __syncthreads();
   if (t <  4) { tx[t] += tx[t+ 4]; ty[t] += ty[t+ 4]; tz[t] += tz[t+ 4]; }  __syncthreads();
   if (t <  2) { tx[t] += tx[t+ 2]; ty[t] += ty[t+ 2]; tz[t] += tz[t+ 2]; }  __syncthreads();
   // clang-format on
   xsum = tx[0] + tx[1];
   ysum = ty[0] + ty[1];
   zsum = tz[0] + tz[1];
   xout[0] = xsum;
   xout[1] = ysum;
   xout[2] = zsum;
}

template <unsigned int B>
__global__
void mdrestSumT1_cu(int n, pos_prec* restrict odata, const double* restrict mass,
   const pos_prec xtot, const pos_prec ytot, const pos_prec ztot,
   const pos_prec* restrict x, const pos_prec* restrict y, const pos_prec* restrict z)
{
   static_assert(B == 64, "");
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;
   const int t = threadIdx.x;

   pos_prec xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
   for (int i = ithread; i < n; i += stride) {
      auto m = mass[i];
      auto xdel = x[i] - xtot;
      auto ydel = y[i] - ytot;
      auto zdel = z[i] - ztot;
      xx += m * xdel * xdel;
      xy += m * xdel * ydel;
      xz += m * xdel * zdel;
      yy += m * ydel * ydel;
      yz += m * ydel * zdel;
      zz += m * zdel * zdel;
   }

   __shared__ pos_prec txx[B], txy[B], txz[B], tyy[B], tyz[B], tzz[B];
   // clang-format off
   txx[t] = xx; txy[t] = xy; txz[t] = xz; tyy[t] = yy; tyz[t] = yz; tzz[t] = zz;   __syncthreads();
   if (t < 32) { txx[t] += txx[t+32]; txy[t] += txy[t+32]; txz[t] += txz[t+32];
                 tyy[t] += tyy[t+32]; tyz[t] += tyz[t+32]; tzz[t] += tzz[t+32]; }  __syncthreads();
   if (t < 16) { txx[t] += txx[t+16]; txy[t] += txy[t+16]; txz[t] += txz[t+16];
                 tyy[t] += tyy[t+16]; tyz[t] += tyz[t+16]; tzz[t] += tzz[t+16]; }  __syncthreads();
   if (t <  8) { txx[t] += txx[t+ 8]; txy[t] += txy[t+ 8]; txz[t] += txz[t+ 8];
                 tyy[t] += tyy[t+ 8]; tyz[t] += tyz[t+ 8]; tzz[t] += tzz[t+ 8]; }  __syncthreads();
   if (t <  4) { txx[t] += txx[t+ 4]; txy[t] += txy[t+ 4]; txz[t] += txz[t+ 4];
                 tyy[t] += tyy[t+ 4]; tyz[t] += tyz[t+ 4]; tzz[t] += tzz[t+ 4]; }  __syncthreads();
   if (t <  2) { txx[t] += txx[t+ 2]; txy[t] += txy[t+ 2]; txz[t] += txz[t+ 2];
                 tyy[t] += tyy[t+ 2]; tyz[t] += tyz[t+ 2]; tzz[t] += tzz[t+ 2]; }  __syncthreads();
   // clang-format on
   if (t == 0) {
      const int b = blockIdx.x;
      odata[6 * b + 0] = txx[t] + txx[t + 1];
      odata[6 * b + 1] = txy[t] + txy[t + 1];
      odata[6 * b + 2] = txz[t] + txz[t + 1];
      odata[6 * b + 3] = tyy[t] + tyy[t + 1];
      odata[6 * b + 4] = tyz[t] + tyz[t + 1];
      odata[6 * b + 5] = tzz[t] + tzz[t + 1];
   }
}

template <int B>
__global__
void mdrestSumT2_cu(int n, const pos_prec* restrict idata, pos_prec* restrict xout)
{
   static_assert(B == 64, "");
   const int t = threadIdx.x;

   pos_prec xx = 0, xy = 0, xz = 0, yy = 0, yz = 0, zz = 0;
   for (int i = t; i < gridDim.x; i += B) {
      xx += idata[6 * i + 0];
      xy += idata[6 * i + 1];
      xz += idata[6 * i + 2];
      yy += idata[6 * i + 3];
      yz += idata[6 * i + 4];
      zz += idata[6 * i + 5];
   }

   __shared__ pos_prec txx[B], txy[B], txz[B], tyy[B], tyz[B], tzz[B];
   // clang-format off
   txx[t] = xx; txy[t] = xy; txz[t] = xz; tyy[t] = yy; tyz[t] = yz; tzz[t] = zz;   __syncthreads();
   if (t < 32) { txx[t] += txx[t+32]; txy[t] += txy[t+32]; txz[t] += txz[t+32];
                 tyy[t] += tyy[t+32]; tyz[t] += tyz[t+32]; tzz[t] += tzz[t+32]; }  __syncthreads();
   if (t < 16) { txx[t] += txx[t+16]; txy[t] += txy[t+16]; txz[t] += txz[t+16];
                 tyy[t] += tyy[t+16]; tyz[t] += tyz[t+16]; tzz[t] += tzz[t+16]; }  __syncthreads();
   if (t <  8) { txx[t] += txx[t+ 8]; txy[t] += txy[t+ 8]; txz[t] += txz[t+ 8];
                 tyy[t] += tyy[t+ 8]; tyz[t] += tyz[t+ 8]; tzz[t] += tzz[t+ 8]; }  __syncthreads();
   if (t <  4) { txx[t] += txx[t+ 4]; txy[t] += txy[t+ 4]; txz[t] += txz[t+ 4];
                 tyy[t] += tyy[t+ 4]; tyz[t] += tyz[t+ 4]; tzz[t] += tzz[t+ 4]; }  __syncthreads();
   if (t <  2) { txx[t] += txx[t+ 2]; txy[t] += txy[t+ 2]; txz[t] += txz[t+ 2];
                 tyy[t] += tyy[t+ 2]; tyz[t] += tyz[t+ 2]; tzz[t] += tzz[t+ 2]; }  __syncthreads();
   // clang-format on
   xx = txx[0] + txx[1];
   xy = txy[0] + txy[1];
   xz = txz[0] + txz[1];
   yy = tyy[0] + tyy[1];
   yz = tyz[0] + tyz[1];
   zz = tzz[0] + tzz[1];
   xout[0] = xx;
   xout[1] = xy;
   xout[2] = xz;
   xout[3] = yy;
   xout[4] = yz;
   xout[5] = zz;
}

__global__
void mdrestRemoveA_cu(int n, const pos_prec xtot, const pos_prec ytot, const pos_prec ztot,
   const pos_prec* restrict x, const pos_prec* restrict y, const pos_prec* restrict z,
   const vel_prec vang1, const vel_prec vang2, const vel_prec vang3,
   vel_prec* restrict vx, vel_prec* restrict vy, vel_prec* restrict vz)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int stride = blockDim.x * gridDim.x;

   for (int i = ithread; i < n; i += stride) {
      pos_prec xdel = x[i] - xtot;
      pos_prec ydel = y[i] - ytot;
      pos_prec zdel = z[i] - ztot;
      vx[i] += -vang2*zdel + vang3*ydel;
      vy[i] += -vang3*xdel + vang1*zdel;
      vz[i] += -vang1*ydel + vang2*xdel;
   }
}

void mdrestRemovePbcMomentum_cu(bool copyout, vel_prec& vtot1, vel_prec& vtot2, vel_prec& vtot3)
{
   vel_prec* xout;
   xout = (vel_prec*)dptr_buf;
   auto invtotmass = 1 / molcul::totmass;

   constexpr int HN = 3;
   constexpr int B = 64;
   vel_prec* ptr = &xout[4];
   int grid_siz1 = -4 + gpuGridSize(BLOCK_DIM);
   grid_siz1 /= HN;
   int grid_siz2 = (n + B - 1) / B;
   int ngrid = std::min(grid_siz1, grid_siz2);

   mdrestSumP_cu<B><<<ngrid, B, 0, g::s0>>>(n, ptr, mass, vx, vy, vz);
   mdrestRemoveP_cu<B><<<ngrid, B, 0, g::s0>>>(n, invtotmass, ptr, vx, vy, vz, xout);

   if (copyout) {
      vel_prec v[3];
      darray::copyout(g::q0, 3, v, xout);
      waitFor(g::q0);
      vtot1 = v[0];
      vtot2 = v[1];
      vtot3 = v[2];
   }
}

void mdrestRemoveAngularMomentum_cu()
{
   pos_prec* xout;
   xout = (pos_prec*)dptr_buf;
   vel_prec* vout;
   vout = (vel_prec*)dptr_buf;
   auto invtotmass = 1 / molcul::totmass;

   constexpr int HN = 3;
   constexpr int HN_b = 6;
   constexpr int B = 64;
   pos_prec* ptrx = &xout[4];
   vel_prec* ptrv = &vout[4];
   int grid_siz1 = -4 + gpuGridSize(BLOCK_DIM);
   int grid_siz1_b = grid_siz1;
   grid_siz1 /= HN;
   grid_siz1_b /= HN_b;
   int grid_siz2 = (n + B - 1) / B;
   int grid_siz2_b = grid_siz2;
   int ngrid = std::min(grid_siz1, grid_siz2);
   int ngrid_b = std::min(grid_siz1_b, grid_siz2_b);

   // calculate the center of mass
   mdrestSumX1_cu<B><<<ngrid, B, 0, g::s0>>>(n, ptrx, mass, xpos, ypos, zpos);
   mdrestSumX2_cu<B><<<ngrid, B, 0, g::s0>>>(n, invtotmass, ptrx, xout);
   pos_prec cm[3];
   darray::copyout(g::q0, 3, cm, xout);
   waitFor(g::q0);

   // calculate the angular momentum of system
   mdrestSumA1_cu<B><<<ngrid, B, 0, g::s0>>>(n, ptrv, mass, xpos, ypos, zpos, vx, vy, vz);
   mdrestSumA2_cu<B><<<ngrid, B, 0, g::s0>>>(n, ptrv, vout);
   vel_prec mang[3];
   darray::copyout(g::q0, 3, mang, vout);
   waitFor(g::q0);

   // calculate the moment of inertia tensor
   mdrestSumT1_cu<B><<<ngrid_b, B, 0, g::s0>>>(n, ptrx, mass, cm[0], cm[1], cm[2], xpos, ypos, zpos);
   mdrestSumT2_cu<B><<<ngrid_b, B, 0, g::s0>>>(n, ptrx, xout);
   pos_prec ts[6];
   darray::copyout(g::q0, 6, ts, xout);
   waitFor(g::q0);

   // invert tensor
   double tnsr[3][3];
   double xx = ts[0];
   double xy = ts[1];
   double xz = ts[2];
   double yy = ts[3];
   double yz = ts[4];
   double zz = ts[5];
   double eps = (n <= 2 ? 0.000001 : 0);
   tnsr[0][0] = yy + zz + eps;
   tnsr[0][1] = -xy;
   tnsr[0][2] = -xz;
   tnsr[1][0] = -xy;
   tnsr[1][1] = xx + zz + eps;
   tnsr[1][2] = -yz;
   tnsr[2][0] = -xz;
   tnsr[2][1] = -yz;
   tnsr[2][2] = xx + yy + eps;
   // diagonalize the moment of inertia tensor
   double tinv[3][3];
   double det = tnsr[0][0] * (tnsr[1][1] *tnsr[2][2] - tnsr[1][2] * tnsr[2][1])
              - tnsr[1][0] * (tnsr[0][1] *tnsr[2][2] - tnsr[2][1] * tnsr[0][2])
              + tnsr[2][0] * (tnsr[0][1] *tnsr[1][2] - tnsr[1][1] * tnsr[0][2]);
   tinv[0][0] = (tnsr[1][1] * tnsr[2][2] - tnsr[1][2] * tnsr[2][1]) / det;
   tinv[1][0] = (tnsr[2][0] * tnsr[1][2] - tnsr[1][0] * tnsr[2][2]) / det;
   tinv[2][0] = (tnsr[1][0] * tnsr[2][1] - tnsr[2][0] * tnsr[1][1]) / det;
   tinv[0][1] = (tnsr[2][1] * tnsr[0][2] - tnsr[0][1] * tnsr[2][2]) / det;
   tinv[1][1] = (tnsr[0][0] * tnsr[2][2] - tnsr[2][0] * tnsr[0][2]) / det;
   tinv[2][1] = (tnsr[0][1] * tnsr[2][0] - tnsr[0][0] * tnsr[2][1]) / det;
   tinv[0][2] = (tnsr[0][1] * tnsr[1][2] - tnsr[0][2] * tnsr[1][1]) / det;
   tinv[1][2] = (tnsr[0][2] * tnsr[1][0] - tnsr[0][0] * tnsr[1][2]) / det;
   tinv[2][2] = (tnsr[0][0] * tnsr[1][1] - tnsr[0][1] * tnsr[1][0]) / det;

   // calculate the angular velocity
   vel_prec vang[3];
   for (int i = 0; i < 3; i++) {
      vang[i] = 0;
      for (int j = 0; j < 3; j++) {
         vang[i] += tinv[i][j] * mang[j];
      }
   }

   // eliminate any rotation about the system center of mass
   mdrestRemoveA_cu<<<ngrid, B, 0, g::s0>>>(n, cm[0], cm[1], cm[2], xpos, ypos, zpos, vang[0], vang[1], vang[2], vx, vy, vz);
}

void mdrest_cu(int istep)
{
   if (not mdstuf::dorest)
      return;
   if ((istep % mdstuf::irest) != 0)
      return;

   // const energy_prec ekcal = units::ekcal;

   // zero out the total mass and overall linear velocity

   auto totmass = molcul::totmass;
   vel_prec vtot1 = 0, vtot2 = 0, vtot3 = 0;

   bool copyout = static_cast<bool>(inform::debug);
   mdrestRemovePbcMomentum_cu(copyout, vtot1, vtot2, vtot3);

   // print the translational velocity of the overall system

   mdrestPrintP1(static_cast<bool>(inform::debug), vtot1, vtot2, vtot3, totmass);

   if (not bound::use_bounds) {
      mdrestRemoveAngularMomentum_cu();
   }
}
}
