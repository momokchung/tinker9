// ck.py Version 3.1.0
__global__
void dfieldgkN2_cu1(int n, real off, const real* restrict x, const real* restrict y, const real* restrict z, int nakp,
   const int* restrict iakp, real (*restrict fields)[3], real (*restrict fieldps)[3], const real* restrict rborn,
   real gkc, real fc, real fd, real fq)
{
   using d::rpole;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM], diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM],
      qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM], qizz[BLOCK_DIM], rbi[BLOCK_DIM];
   real xi, yi, zi;
   real xk, yk, zk, ck, dkx, dky, dkz, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, rbk;
   real fidsx, fidsy, fidsz, fipsx, fipsy, fipsz;
   real fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz;

   for (int iw = iwarp; iw < nakp; iw += nwarp) {
      fidsx = 0;
      fidsy = 0;
      fidsz = 0;
      fipsx = 0;
      fipsy = 0;
      fipsz = 0;
      fkdsx = 0;
      fkdsy = 0;
      fkdsz = 0;
      fkpsx = 0;
      fkpsy = 0;
      fkpsz = 0;

      int tri, tx, ty;
      tri = iakp[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
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
      rbi[threadIdx.x] = rborn[i];
      xi = x[i];
      yi = y[i];
      zi = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ck = rpole[k][MPL_PME_0];
      dkx = rpole[k][MPL_PME_X];
      dky = rpole[k][MPL_PME_Y];
      dkz = rpole[k][MPL_PME_Z];
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
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            pair_dfieldgk(r2, xr, yr, zr, gkc, fc, fd, fq, ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane],
               qixy[klane], qixz[klane], qiyy[klane], qiyz[klane], qizz[klane], rbi[klane], ck, dkx, dky, dkz, qkxx,
               qkxy, qkxz, qkyy, qkyz, qkzz, rbk, fidsx, fidsy, fidsz, fipsx, fipsy, fipsz, fkdsx, fkdsy, fkdsz, fkpsx,
               fkpsy, fkpsz);
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidsx = __shfl_sync(ALL_LANES, fidsx, ilane + 1);
         fidsy = __shfl_sync(ALL_LANES, fidsy, ilane + 1);
         fidsz = __shfl_sync(ALL_LANES, fidsz, ilane + 1);
         fipsx = __shfl_sync(ALL_LANES, fipsx, ilane + 1);
         fipsy = __shfl_sync(ALL_LANES, fipsy, ilane + 1);
         fipsz = __shfl_sync(ALL_LANES, fipsz, ilane + 1);
      }

      atomic_add(fidsx, &fields[i][0]);
      atomic_add(fidsy, &fields[i][1]);
      atomic_add(fidsz, &fields[i][2]);
      atomic_add(fipsx, &fieldps[i][0]);
      atomic_add(fipsy, &fieldps[i][1]);
      atomic_add(fipsz, &fieldps[i][2]);
      atomic_add(fkdsx, &fields[k][0]);
      atomic_add(fkdsy, &fields[k][1]);
      atomic_add(fkdsz, &fields[k][2]);
      atomic_add(fkpsx, &fieldps[k][0]);
      atomic_add(fkpsy, &fieldps[k][1]);
      atomic_add(fkpsz, &fieldps[k][2]);
      __syncwarp();
   }
}
