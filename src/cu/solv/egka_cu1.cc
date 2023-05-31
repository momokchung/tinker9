// ck.py Version 3.1.0
template <class Ver>
__global__
void egka_cu1(int n, TINKER_IMAGE_PARAMS, EnergyBuffer restrict es, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, real off, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, const real* restrict rborn, const real (*restrict rpole)[10],
   const real (*restrict uinds)[3], real gkc, real fc, real fd, real fq)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_g = Ver::g;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec estl;
   if CONSTEXPR (do_e) {
      estl = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], rbi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM],
      ukdx[BLOCK_DIM], ukdy[BLOCK_DIM], ukdz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rbk;
   real frcxi, frcyi, frczi;
   real frcxk, frcyk, frczk;

   /* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
       const int klane = threadIdx.x;    if CONSTEXPR (do_g) {frcxi = 0;frcyi = 0;frczi = 0;frcxk = 0;frcyk = 0;frczk =
0;}

       int i = exclude[ii][0];
       int k = exclude[ii][1];


       xi[klane] = x[i];yi[klane] = y[i];zi[klane] = z[i];ci[klane] = rpole[i][MPL_PME_0];dix[klane] =
rpole[i][MPL_PME_X];diy[klane] = rpole[i][MPL_PME_Y];diz[klane] = rpole[i][MPL_PME_Z];qixx[klane] =
rpole[i][MPL_PME_XX];qixy[klane] = rpole[i][MPL_PME_XY];qixz[klane] = rpole[i][MPL_PME_XZ];qiyy[klane] =
rpole[i][MPL_PME_YY];qiyz[klane] = rpole[i][MPL_PME_YZ];qizz[klane] = rpole[i][MPL_PME_ZZ];uidx[klane] =
uinds[i][0];uidy[klane] = uinds[i][1];uidz[klane] = uinds[i][2];rbi[klane] = rborn[i];xk[threadIdx.x] =
x[k];yk[threadIdx.x] = y[k];zk[threadIdx.x] = z[k];dkx[threadIdx.x] = rpole[k][MPL_PME_X];dky[threadIdx.x] =
rpole[k][MPL_PME_Y];dkz[threadIdx.x] = rpole[k][MPL_PME_Z];ukdx[threadIdx.x] = uinds[k][0];ukdy[threadIdx.x] =
uinds[k][1];ukdz[threadIdx.x] = uinds[k][2];ck = rpole[k][MPL_PME_0];qkxx = rpole[k][MPL_PME_XX];qkxy =
rpole[k][MPL_PME_XY];qkxz = rpole[k][MPL_PME_XZ];qkyy = rpole[k][MPL_PME_YY];qkyz = rpole[k][MPL_PME_YZ];qkzz =
rpole[k][MPL_PME_ZZ];rbk = rborn[k];

       constexpr bool incl = true;
       real xr = xk[threadIdx.x] - xi[klane];
real yr = yk[threadIdx.x] - yi[klane];
real zr = zk[threadIdx.x] - zi[klane];
real xr2 = xr*xr;
real yr2 = yr*yr;
real zr2 = zr*zr;
real r2 = xr2 + yr2 + zr2;
if (r2 <= off * off and incl) {
 real e;
 pair_egka<do_g>(
   r2, xr, yr, zr, xr2, yr2, zr2,
   ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane], qiyz[klane],
qizz[klane], uidx[klane], uidy[klane], uidz[klane], rbi[klane], ck, dkx[threadIdx.x], dky[threadIdx.x],
dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], rbk, gkc,
fc, fd, fq, e); if CONSTEXPR (do_e) { estl += floatTo<ebuf_prec>(e);
 }
} // end if (include)


       if CONSTEXPR (do_g) {atomic_add(frcxi, gx, i);atomic_add(frcyi, gy, i);atomic_add(frczi, gz, i);atomic_add(frcxk,
gx, k);atomic_add(frcyk, gy, k);atomic_add(frczk, gz, k);}
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
      }

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      ci[threadIdx.x] = rpole[i][MPL_PME_0];
      dix[threadIdx.x] = rpole[i][MPL_PME_X];
      diy[threadIdx.x] = rpole[i][MPL_PME_Y];
      diz[threadIdx.x] = rpole[i][MPL_PME_Z];
      qixx[threadIdx.x] = rpole[i][MPL_PME_XX];
      qixy[threadIdx.x] = rpole[i][MPL_PME_XY];
      qixz[threadIdx.x] = rpole[i][MPL_PME_XZ];
      qiyy[threadIdx.x] = rpole[i][MPL_PME_YY];
      qiyz[threadIdx.x] = rpole[i][MPL_PME_YZ];
      qizz[threadIdx.x] = rpole[i][MPL_PME_ZZ];
      uidx[threadIdx.x] = uinds[i][0];
      uidy[threadIdx.x] = uinds[i][1];
      uidz[threadIdx.x] = uinds[i][2];
      rbi[threadIdx.x] = rborn[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uinds[k][0];
      ukdy[threadIdx.x] = uinds[k][1];
      ukdz[threadIdx.x] = uinds[k][2];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      rbk = rborn[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real xr2 = xr * xr;
         real yr2 = yr * yr;
         real zr2 = zr * zr;
         real r2 = xr2 + yr2 + zr2;
         if (r2 <= off * off and incl) {
            real e;
            pair_egka<do_g>(r2, xr, yr, zr, xr2, yr2, zr2, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane],
               rbi[klane], ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], rbk, gkc, fc, fd, fq, e);
            if CONSTEXPR (do_e) {
               estl += floatTo<ebuf_prec>(e);
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      ci[threadIdx.x] = rpole[i][MPL_PME_0];
      dix[threadIdx.x] = rpole[i][MPL_PME_X];
      diy[threadIdx.x] = rpole[i][MPL_PME_Y];
      diz[threadIdx.x] = rpole[i][MPL_PME_Z];
      qixx[threadIdx.x] = rpole[i][MPL_PME_XX];
      qixy[threadIdx.x] = rpole[i][MPL_PME_XY];
      qixz[threadIdx.x] = rpole[i][MPL_PME_XZ];
      qiyy[threadIdx.x] = rpole[i][MPL_PME_YY];
      qiyz[threadIdx.x] = rpole[i][MPL_PME_YZ];
      qizz[threadIdx.x] = rpole[i][MPL_PME_ZZ];
      uidx[threadIdx.x] = uinds[i][0];
      uidy[threadIdx.x] = uinds[i][1];
      uidz[threadIdx.x] = uinds[i][2];
      rbi[threadIdx.x] = rborn[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uinds[k][0];
      ukdy[threadIdx.x] = uinds[k][1];
      ukdz[threadIdx.x] = uinds[k][2];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      rbk = rborn[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real xr2 = xr * xr;
         real yr2 = yr * yr;
         real zr2 = zr * zr;
         real r2 = xr2 + yr2 + zr2;
         if (r2 <= off * off and incl) {
            real e;
            pair_egka<do_g>(r2, xr, yr, zr, xr2, yr2, zr2, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane],
               rbi[klane], ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], rbk, gkc, fc, fd, fq, e);
            if CONSTEXPR (do_e) {
               estl += floatTo<ebuf_prec>(e);
            }
         } // end if (include)

         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_e) {
      atomic_add(estl, es, ithread);
   }
}
