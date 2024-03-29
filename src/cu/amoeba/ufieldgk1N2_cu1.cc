// ck.py Version 3.1.0
__global__
void ufieldgk1N2_cu1(int n, real off, const unsigned* restrict uinfo, int nexclude, const int (*restrict exclude)[2],
   const real* restrict exclude_scale, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niakp,
   const int* restrict iakp, const real (*restrict uind)[3], const real (*restrict uinp)[3],
   const real (*restrict uinds)[3], const real (*restrict uinps)[3], real (*restrict field)[3],
   real (*restrict fieldp)[3], real (*restrict fields)[3], real (*restrict fieldps)[3])
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM], uipz[BLOCK_DIM],
      uidsx[BLOCK_DIM], uidsy[BLOCK_DIM], uidsz[BLOCK_DIM], uipsx[BLOCK_DIM], uipsy[BLOCK_DIM], uipsz[BLOCK_DIM],
      pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   real xi, yi, zi;
   real xk, yk, zk, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk;
   int jpk;
   real fidx, fidy, fidz, fipx, fipy, fipz, fidsx, fidsy, fidsz, fipsx, fipsy, fipsz;
   real fkdx, fkdy, fkdz, fkpx, fkpy, fkpz, fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fidsx = 0;
      fidsy = 0;
      fidsz = 0;
      fipsx = 0;
      fipsy = 0;
      fipsz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;
      fkdsx = 0;
      fkdsy = 0;
      fkdsz = 0;
      fkpsx = 0;
      fkpsy = 0;
      fkpsz = 0;

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
      uidsx[klane] = uinds[i][0];
      uidsy[klane] = uinds[i][1];
      uidsz[klane] = uinds[i][2];
      uipsx[klane] = uinps[i][0];
      uipsy[klane] = uinps[i][1];
      uipsz[klane] = uinps[i][2];
      pdi[klane] = pdamp[i];
      jpi[klane] = jpolar[i];
      xi = x[i];
      yi = y[i];
      zi = z[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      ukdsx = uinds[k][0];
      ukdsy = uinds[k][1];
      ukdsz = uinds[k][2];
      ukpsx = uinps[k][0];
      ukpsy = uinps[k][1];
      ukpsz = uinps[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];

      constexpr bool incl = true;
      real xr = xk - xi;
      real yr = yk - yi;
      real zr = zk - zi;
      real r2 = xr * xr + yr * yr + zr * zr;
      if (r2 <= off * off and incl) {
         real pga = thlval[njpolar * jpi[klane] + jpk];
         pair_ufieldgk1(r2, xr, yr, zr, scalea, uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane],
            uipz[klane], uidsx[klane], uidsy[klane], uidsz[klane], uipsx[klane], uipsy[klane], uipsz[klane], pdi[klane],
            pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk, pga, fidx, fidy,
            fidz, fipx, fipy, fipz, fidsx, fidsy, fidsz, fipsx, fipsy, fipsz, fkdx, fkdy, fkdz, fkpx, fkpy, fkpz, fkdsx,
            fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);
      } // end if (include)

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fidsx, &fields[i][0]);
      atomic_add(fidsy, &fields[i][1]);
      atomic_add(fidsz, &fields[i][2]);
      atomic_add(fipsx, &fieldps[i][0]);
      atomic_add(fipsy, &fieldps[i][1]);
      atomic_add(fipsz, &fieldps[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
      atomic_add(fkdsx, &fields[k][0]);
      atomic_add(fkdsy, &fields[k][1]);
      atomic_add(fkdsz, &fields[k][2]);
      atomic_add(fkpsx, &fieldps[k][0]);
      atomic_add(fkpsy, &fieldps[k][1]);
      atomic_add(fkpsz, &fieldps[k][2]);
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fidsx = 0;
      fidsy = 0;
      fidsz = 0;
      fipsx = 0;
      fipsy = 0;
      fipsz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;
      fkdsx = 0;
      fkdsy = 0;
      fkdsz = 0;
      fkpsx = 0;
      fkpsy = 0;
      fkpsz = 0;

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      uidsx[threadIdx.x] = uinds[i][0];
      uidsy[threadIdx.x] = uinds[i][1];
      uidsz[threadIdx.x] = uinds[i][2];
      uipsx[threadIdx.x] = uinps[i][0];
      uipsy[threadIdx.x] = uinps[i][1];
      uipsz[threadIdx.x] = uinps[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      ukdsx = uinds[k][0];
      ukdsy = uinds[k][1];
      ukdsz = uinds[k][2];
      ukpsx = uinps[k][0];
      ukpsy = uinps[k][1];
      ukpsz = uinps[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      __syncwarp();

      unsigned int uinfo0 = uinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (uinfo0 & srcmask) == 0;
         real scalea = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            pair_ufieldgk1(r2, xr, yr, zr, scalea, uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane],
               uipz[klane], uidsx[klane], uidsy[klane], uidsz[klane], uipsx[klane], uipsy[klane], uipsz[klane],
               pdi[klane], pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk, pga,
               fidx, fidy, fidz, fipx, fipy, fipz, fidsx, fidsy, fidsz, fipsx, fipsy, fipsz, fkdx, fkdy, fkdz, fkpx,
               fkpy, fkpz, fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
         fipx = __shfl_sync(ALL_LANES, fipx, ilane + 1);
         fipy = __shfl_sync(ALL_LANES, fipy, ilane + 1);
         fipz = __shfl_sync(ALL_LANES, fipz, ilane + 1);
         fidsx = __shfl_sync(ALL_LANES, fidsx, ilane + 1);
         fidsy = __shfl_sync(ALL_LANES, fidsy, ilane + 1);
         fidsz = __shfl_sync(ALL_LANES, fidsz, ilane + 1);
         fipsx = __shfl_sync(ALL_LANES, fipsx, ilane + 1);
         fipsy = __shfl_sync(ALL_LANES, fipsy, ilane + 1);
         fipsz = __shfl_sync(ALL_LANES, fipsz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fidsx, &fields[i][0]);
      atomic_add(fidsy, &fields[i][1]);
      atomic_add(fidsz, &fields[i][2]);
      atomic_add(fipsx, &fieldps[i][0]);
      atomic_add(fipsy, &fieldps[i][1]);
      atomic_add(fipsz, &fieldps[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
      atomic_add(fkdsx, &fields[k][0]);
      atomic_add(fkdsy, &fields[k][1]);
      atomic_add(fkdsz, &fields[k][2]);
      atomic_add(fkpsx, &fieldps[k][0]);
      atomic_add(fkpsy, &fieldps[k][1]);
      atomic_add(fkpsz, &fieldps[k][2]);
      __syncwarp();
   }

   for (int iw = iwarp; iw < niakp; iw += nwarp) {
      fidx = 0;
      fidy = 0;
      fidz = 0;
      fipx = 0;
      fipy = 0;
      fipz = 0;
      fidsx = 0;
      fidsy = 0;
      fidsz = 0;
      fipsx = 0;
      fipsy = 0;
      fipsz = 0;
      fkdx = 0;
      fkdy = 0;
      fkdz = 0;
      fkpx = 0;
      fkpy = 0;
      fkpz = 0;
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
      int atomi = min(iid, n - 1);
      int i = sorted[atomi].unsorted;
      int kid = tx * WARP_SIZE + ilane;
      int atomk = min(kid, n - 1);
      int k = sorted[atomk].unsorted;
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      uidsx[threadIdx.x] = uinds[i][0];
      uidsy[threadIdx.x] = uinds[i][1];
      uidsz[threadIdx.x] = uinds[i][2];
      uipsx[threadIdx.x] = uinps[i][0];
      uipsy[threadIdx.x] = uinps[i][1];
      uipsz[threadIdx.x] = uinps[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xi = sorted[atomi].x;
      yi = sorted[atomi].y;
      zi = sorted[atomi].z;
      xk = sorted[atomk].x;
      yk = sorted[atomk].y;
      zk = sorted[atomk].z;
      ukdx = uind[k][0];
      ukdy = uind[k][1];
      ukdz = uind[k][2];
      ukpx = uinp[k][0];
      ukpy = uinp[k][1];
      ukpz = uinp[k][2];
      ukdsx = uinds[k][0];
      ukdsy = uinds[k][1];
      ukdsz = uinds[k][2];
      ukpsx = uinps[k][0];
      ukpsy = uinps[k][1];
      ukpsz = uinps[k][2];
      pdk = pdamp[k];
      jpk = jpolar[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real scalea = 1;
         real xr = xk - xi;
         real yr = yk - yi;
         real zr = zk - zi;
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            pair_ufieldgk1(r2, xr, yr, zr, scalea, uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane],
               uipz[klane], uidsx[klane], uidsy[klane], uidsz[klane], uipsx[klane], uipsy[klane], uipsz[klane],
               pdi[klane], pga, ukdx, ukdy, ukdz, ukpx, ukpy, ukpz, ukdsx, ukdsy, ukdsz, ukpsx, ukpsy, ukpsz, pdk, pga,
               fidx, fidy, fidz, fipx, fipy, fipz, fidsx, fidsy, fidsz, fipsx, fipsy, fipsz, fkdx, fkdy, fkdz, fkpx,
               fkpy, fkpz, fkdsx, fkdsy, fkdsz, fkpsx, fkpsy, fkpsz);
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         fidx = __shfl_sync(ALL_LANES, fidx, ilane + 1);
         fidy = __shfl_sync(ALL_LANES, fidy, ilane + 1);
         fidz = __shfl_sync(ALL_LANES, fidz, ilane + 1);
         fipx = __shfl_sync(ALL_LANES, fipx, ilane + 1);
         fipy = __shfl_sync(ALL_LANES, fipy, ilane + 1);
         fipz = __shfl_sync(ALL_LANES, fipz, ilane + 1);
         fidsx = __shfl_sync(ALL_LANES, fidsx, ilane + 1);
         fidsy = __shfl_sync(ALL_LANES, fidsy, ilane + 1);
         fidsz = __shfl_sync(ALL_LANES, fidsz, ilane + 1);
         fipsx = __shfl_sync(ALL_LANES, fipsx, ilane + 1);
         fipsy = __shfl_sync(ALL_LANES, fipsy, ilane + 1);
         fipsz = __shfl_sync(ALL_LANES, fipsz, ilane + 1);
      }

      atomic_add(fidx, &field[i][0]);
      atomic_add(fidy, &field[i][1]);
      atomic_add(fidz, &field[i][2]);
      atomic_add(fipx, &fieldp[i][0]);
      atomic_add(fipy, &fieldp[i][1]);
      atomic_add(fipz, &fieldp[i][2]);
      atomic_add(fidsx, &fields[i][0]);
      atomic_add(fidsy, &fields[i][1]);
      atomic_add(fidsz, &fields[i][2]);
      atomic_add(fipsx, &fieldps[i][0]);
      atomic_add(fipsy, &fieldps[i][1]);
      atomic_add(fipsz, &fieldps[i][2]);
      atomic_add(fkdx, &field[k][0]);
      atomic_add(fkdy, &field[k][1]);
      atomic_add(fkdz, &field[k][2]);
      atomic_add(fkpx, &fieldp[k][0]);
      atomic_add(fkpy, &fieldp[k][1]);
      atomic_add(fkpz, &fieldp[k][2]);
      atomic_add(fkdsx, &fields[k][0]);
      atomic_add(fkdsy, &fields[k][1]);
      atomic_add(fkdsz, &fields[k][2]);
      atomic_add(fkpsx, &fieldps[k][0]);
      atomic_add(fkpsy, &fieldps[k][1]);
      atomic_add(fkpsz, &fieldps[k][2]);
      __syncwarp();
   }
}
