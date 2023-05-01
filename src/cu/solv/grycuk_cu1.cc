// ck.py Version 3.1.0
__global__
void grycuk_cu1(int n, TINKER_IMAGE_PARAMS, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, real* restrict rborn, const real* restrict rsolv, const real* restrict rdescr,
   const real* restrict shct, real pi43)
{
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], rsi[BLOCK_DIM], rdi[BLOCK_DIM], shcti[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM];
   real rsk, rdk, shctk;
   real rborni;
   real rbornk;

   /* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
       const int klane = threadIdx.x;    rborni = 0;rbornk = 0;

       int i = exclude[ii][0];
       int k = exclude[ii][1];


       xi[klane] = x[i];yi[klane] = y[i];zi[klane] = z[i];rsi[klane] = rsolv[i];rdi[klane] = rdescr[i];shcti[klane] =
shct[i];xk[threadIdx.x] = x[k];yk[threadIdx.x] = y[k];zk[threadIdx.x] = z[k];rsk = rsolv[k];rdk = rdescr[k];shctk =
shct[k];

       constexpr bool incl = true;
       real xr = xk[threadIdx.x] - xi[klane];
real yr = yk[threadIdx.x] - yi[klane];
real zr = zk[threadIdx.x] - zi[klane];
real r2 = xr*xr + yr*yr + zr*zr;
if (incl) {
 real r = REAL_SQRT(r2);
 bool computei = (rsi[klane] > 0.) and (rdk > 0.);
 bool computek = (rdi[klane] > 0.) and (rsk > 0.);
 real pairrborni = 0.;
 real pairrbornk = 0.;
 if (computei) {
   pair_grycuk(r, r2, pi43, rsi[klane], rdi[klane], rdk, shctk, pairrborni);
 }
 if (computek) {
   pair_grycuk(r, r2, pi43, rsk, rdk, rdi[klane], shcti[klane], pairrbornk);
 }

 rborni += pairrborni;
 rbornk += pairrbornk;
}


       atomic_add(rborni, rborn, i);atomic_add(rbornk, rborn, k);
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      rborni = 0;
      rbornk = 0;

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
      rsi[threadIdx.x] = rsolv[i];
      rdi[threadIdx.x] = rdescr[i];
      shcti[threadIdx.x] = shct[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      rsk = rsolv[k];
      rdk = rdescr[k];
      shctk = shct[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         if (incl) {
            real r = REAL_SQRT(r2);
            bool computei = (rsi[klane] > 0.) and (rdk > 0.);
            bool computek = (rdi[klane] > 0.) and (rsk > 0.);
            real pairrborni = 0.;
            real pairrbornk = 0.;
            if (computei) {
               pair_grycuk(r, r2, pi43, rsi[klane], rdi[klane], rdk, shctk, pairrborni);
            }
            if (computek) {
               pair_grycuk(r, r2, pi43, rsk, rdk, rdi[klane], shcti[klane], pairrbornk);
            }

            rborni += pairrborni;
            rbornk += pairrbornk;
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         rborni = __shfl_sync(ALL_LANES, rborni, ilane + 1);
      }

      atomic_add(rborni, rborn, i);
      atomic_add(rbornk, rborn, k);
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      rborni = 0;
      rbornk = 0;

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      rsi[threadIdx.x] = rsolv[i];
      rdi[threadIdx.x] = rdescr[i];
      shcti[threadIdx.x] = shct[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      rsk = rsolv[k];
      rdk = rdescr[k];
      shctk = shct[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         if (incl) {
            real r = REAL_SQRT(r2);
            bool computei = (rsi[klane] > 0.) and (rdk > 0.);
            bool computek = (rdi[klane] > 0.) and (rsk > 0.);
            real pairrborni = 0.;
            real pairrbornk = 0.;
            if (computei) {
               pair_grycuk(r, r2, pi43, rsi[klane], rdi[klane], rdk, shctk, pairrborni);
            }
            if (computek) {
               pair_grycuk(r, r2, pi43, rsk, rdk, rdi[klane], shcti[klane], pairrbornk);
            }

            rborni += pairrborni;
            rbornk += pairrbornk;
         }

         rborni = __shfl_sync(ALL_LANES, rborni, ilane + 1);
      }

      atomic_add(rborni, rborn, i);
      atomic_add(rbornk, rborn, k);
      __syncwarp();
   }
}
