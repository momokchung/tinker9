// ck.py Version 3.1.0
__global__
void ufieldgk2N2_cu1(int n, real off, const real* restrict x, const real* restrict y, const real* restrict z, int nakp,
   const int* restrict iakp, const real (*restrict uinds)[3], const real (*restrict uinps)[3],
   const real* restrict rborn, real gkc, real fd, real (*restrict fields)[3], real (*restrict fieldps)[3])
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real uidsx[BLOCK_DIM], uidsy[BLOCK_DIM], uidsz[BLOCK_DIM], uipsx[BLOCK_DIM], uipsy[BLOCK_DIM],
      uipsz[BLOCK_DIM], rbi[BLOCK_DIM];
   real xi, yi, zi;
   real xk, yk, zk, ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, rbk;
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
      uidsx[threadIdx.x] = uinds[i][0];
      uidsy[threadIdx.x] = uinds[i][1];
      uidsz[threadIdx.x] = uinds[i][2];
      uipsx[threadIdx.x] = uinps[i][0];
      uipsy[threadIdx.x] = uinps[i][1];
      uipsz[threadIdx.x] = uinps[i][2];
      rbi[threadIdx.x] = rborn[i];
      xi = x[i];
      yi = y[i];
      zi = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ukdsx = uinds[k][0];
      ukdsy = uinds[k][1];
      ukdsz = uinds[k][2];
      ukpsx = uinps[k][0];
      ukpsy = uinps[k][1];
      ukpsz = uinps[k][2];
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
            pair_ufieldgk2(r2, xr, yr, zr, gkc, fd, uidsx[klane], uidsy[klane], uidsz[klane], uipsx[klane],
               uipsy[klane], uipsz[klane], rbi[klane], ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, rbk, fidsx, fidsy,
               fidsz, fipsx, fipsy, fipsz, fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);
         } // end if (include)

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
