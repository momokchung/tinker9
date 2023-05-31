// ck.py Version 3.1.0
template <class Ver>
__global__
void ewca_cu1(int n, TINKER_IMAGE_PARAMS, EnergyBuffer restrict es, grad_prec* restrict gx, grad_prec* restrict gy,
   grad_prec* restrict gz, const real* restrict x, const real* restrict y, const real* restrict z,
   const Spatial::SortedAtom* restrict sorted, int nakpl, const int* restrict iakpl, int niak, const int* restrict iak,
   const int* restrict lst, const real* restrict epsvdw, const real* restrict radvdw, real epso, real epsh, real rmino,
   real rminh, real shctd, real dispoff, real slevy, real awater)
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
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], epsli[BLOCK_DIM], rmini[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM];
   real epslk, rmink;
   real gxi, gyi, gzi;
   real gxk, gyk, gzk;

   /* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
       const int klane = threadIdx.x;    if CONSTEXPR (do_g) {gxi = 0;gyi = 0;gzi = 0;gxk = 0;gyk = 0;gzk = 0;}

       int i = exclude[ii][0];
       int k = exclude[ii][1];


       xi[klane] = x[i];yi[klane] = y[i];zi[klane] = z[i];epsli[klane] = epsvdw[i];rmini[klane] =
radvdw[i];xk[threadIdx.x] = x[k];yk[threadIdx.x] = y[k];zk[threadIdx.x] = z[k];epslk = epsvdw[k];rmink = radvdw[k];

       constexpr bool incl = true;
       real xr = xk[threadIdx.x] - xi[klane];
real yr = yk[threadIdx.x] - yi[klane];
real zr = zk[threadIdx.x] - zi[klane];
real r2 = xr*xr + yr*yr + zr*zr;
real e;

if (incl) {
 real r = REAL_SQRT(r2);
 real epsi = epsli[klane];
 real rmin = rmini[klane];
 real emixo = 4. * epso * epsi / REAL_POW((REAL_SQRT(epso)+REAL_SQRT(epsi)),2);
 real rmixo = 2. * (REAL_POW(rmino,3)+REAL_POW(rmin,3)) / (REAL_POW(rmino,2)+REAL_POW(rmin,2));
 real rmixo7 = REAL_POW(rmixo,7);
 real aoi = emixo * rmixo7;
 real emixh = 4. * epsh * epsi / REAL_POW((REAL_SQRT(epsh)+REAL_SQRT(epsi)),2);
 real rmixh = 2. * (REAL_POW(rminh,3)+REAL_POW(rmin,3)) / (REAL_POW(rminh,2)+REAL_POW(rmin,2));
 real rmixh7 = REAL_POW(rmixh,7);
 real ahi = emixh * rmixh7;
 real rio = rmixo / 2. + dispoff;
 real rih = rmixh / 2. + dispoff;
 real si = rmin * shctd;
 real si2 = si * si;

 real epsk = epslk;
 real rmkn = rmink;
 real emkxo = 4. * epso * epsk / REAL_POW((REAL_SQRT(epso)+REAL_SQRT(epsk)),2);
 real rmkxo = 2. * (REAL_POW(rmino,3)+REAL_POW(rmkn,3)) / (REAL_POW(rmino,2)+REAL_POW(rmkn,2));
 real rmkxo7 = REAL_POW(rmkxo,7);
 real aok = emkxo * rmkxo7;
 real emkxh = 4. * epsh * epsk / REAL_POW((REAL_SQRT(epsh)+REAL_SQRT(epsk)),2);
 real rmkxh = 2. * (REAL_POW(rminh,3)+REAL_POW(rmkn,3)) / (REAL_POW(rminh,2)+REAL_POW(rmkn,2));
 real rmkxh7 = REAL_POW(rmkxh,7);
 real ahk = emkxh * rmkxh7;
 real rko = rmkxo / 2. + dispoff;
 real rkh = rmkxh / 2. + dispoff;
 real sk = rmkn * shctd;
 real sk2 = sk * sk;

 real sum1, sum2;
 pair_ewca<do_g>(
   r, r2, rio, rmixo, rmixo7,
   sk, sk2, aoi, emixo, sum1, true);
 pair_ewca<do_g>(
   r, r2, rih, rmixh, rmixh7,
   sk, sk2, ahi, emixh, sum2, false);
 e = sum1 + sum2;
 pair_ewca<do_g>(
   r, r2, rko, rmkxo, rmkxo7,
   si, si2, aok, emkxo, sum1, true);
 pair_ewca<do_g>(
   r, r2, rkh, rmkxh, rmkxh7,
   si, si2, ahk, emkxh, sum2, false);
 e += sum1 + sum2;
 e *= -slevy * awater;

 if CONSTEXPR (do_e)
     estl += floatTo<ebuf_prec>(e);
} // end if (include)


       if CONSTEXPR (do_g) {atomic_add(gxi, gx, i);atomic_add(gyi, gy, i);atomic_add(gzi, gz, i);atomic_add(gxk, gx,
k);atomic_add(gyk, gy, k);atomic_add(gzk, gz, k);}
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
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
      epsli[threadIdx.x] = epsvdw[i];
      rmini[threadIdx.x] = radvdw[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      epslk = epsvdw[k];
      rmink = radvdw[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         real e;

         if (incl) {
            real r = REAL_SQRT(r2);
            real epsi = epsli[klane];
            real rmin = rmini[klane];
            real emixo = 4. * epso * epsi / REAL_POW((REAL_SQRT(epso) + REAL_SQRT(epsi)), 2);
            real rmixo = 2. * (REAL_POW(rmino, 3) + REAL_POW(rmin, 3)) / (REAL_POW(rmino, 2) + REAL_POW(rmin, 2));
            real rmixo7 = REAL_POW(rmixo, 7);
            real aoi = emixo * rmixo7;
            real emixh = 4. * epsh * epsi / REAL_POW((REAL_SQRT(epsh) + REAL_SQRT(epsi)), 2);
            real rmixh = 2. * (REAL_POW(rminh, 3) + REAL_POW(rmin, 3)) / (REAL_POW(rminh, 2) + REAL_POW(rmin, 2));
            real rmixh7 = REAL_POW(rmixh, 7);
            real ahi = emixh * rmixh7;
            real rio = rmixo / 2. + dispoff;
            real rih = rmixh / 2. + dispoff;
            real si = rmin * shctd;
            real si2 = si * si;

            real epsk = epslk;
            real rmkn = rmink;
            real emkxo = 4. * epso * epsk / REAL_POW((REAL_SQRT(epso) + REAL_SQRT(epsk)), 2);
            real rmkxo = 2. * (REAL_POW(rmino, 3) + REAL_POW(rmkn, 3)) / (REAL_POW(rmino, 2) + REAL_POW(rmkn, 2));
            real rmkxo7 = REAL_POW(rmkxo, 7);
            real aok = emkxo * rmkxo7;
            real emkxh = 4. * epsh * epsk / REAL_POW((REAL_SQRT(epsh) + REAL_SQRT(epsk)), 2);
            real rmkxh = 2. * (REAL_POW(rminh, 3) + REAL_POW(rmkn, 3)) / (REAL_POW(rminh, 2) + REAL_POW(rmkn, 2));
            real rmkxh7 = REAL_POW(rmkxh, 7);
            real ahk = emkxh * rmkxh7;
            real rko = rmkxo / 2. + dispoff;
            real rkh = rmkxh / 2. + dispoff;
            real sk = rmkn * shctd;
            real sk2 = sk * sk;

            real sum1, sum2;
            pair_ewca<do_g>(r, r2, rio, rmixo, rmixo7, sk, sk2, aoi, emixo, sum1, true);
            pair_ewca<do_g>(r, r2, rih, rmixh, rmixh7, sk, sk2, ahi, emixh, sum2, false);
            e = sum1 + sum2;
            pair_ewca<do_g>(r, r2, rko, rmkxo, rmkxo7, si, si2, aok, emkxo, sum1, true);
            pair_ewca<do_g>(r, r2, rkh, rmkxh, rmkxh7, si, si2, ahk, emkxh, sum2, false);
            e += sum1 + sum2;
            e *= -slevy * awater;

            if CONSTEXPR (do_e)
               estl += floatTo<ebuf_prec>(e);
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niak; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
      }

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      epsli[threadIdx.x] = epsvdw[i];
      rmini[threadIdx.x] = radvdw[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      epslk = epsvdw[k];
      rmink = radvdw[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         real e;

         if (incl) {
            real r = REAL_SQRT(r2);
            real epsi = epsli[klane];
            real rmin = rmini[klane];
            real emixo = 4. * epso * epsi / REAL_POW((REAL_SQRT(epso) + REAL_SQRT(epsi)), 2);
            real rmixo = 2. * (REAL_POW(rmino, 3) + REAL_POW(rmin, 3)) / (REAL_POW(rmino, 2) + REAL_POW(rmin, 2));
            real rmixo7 = REAL_POW(rmixo, 7);
            real aoi = emixo * rmixo7;
            real emixh = 4. * epsh * epsi / REAL_POW((REAL_SQRT(epsh) + REAL_SQRT(epsi)), 2);
            real rmixh = 2. * (REAL_POW(rminh, 3) + REAL_POW(rmin, 3)) / (REAL_POW(rminh, 2) + REAL_POW(rmin, 2));
            real rmixh7 = REAL_POW(rmixh, 7);
            real ahi = emixh * rmixh7;
            real rio = rmixo / 2. + dispoff;
            real rih = rmixh / 2. + dispoff;
            real si = rmin * shctd;
            real si2 = si * si;

            real epsk = epslk;
            real rmkn = rmink;
            real emkxo = 4. * epso * epsk / REAL_POW((REAL_SQRT(epso) + REAL_SQRT(epsk)), 2);
            real rmkxo = 2. * (REAL_POW(rmino, 3) + REAL_POW(rmkn, 3)) / (REAL_POW(rmino, 2) + REAL_POW(rmkn, 2));
            real rmkxo7 = REAL_POW(rmkxo, 7);
            real aok = emkxo * rmkxo7;
            real emkxh = 4. * epsh * epsk / REAL_POW((REAL_SQRT(epsh) + REAL_SQRT(epsk)), 2);
            real rmkxh = 2. * (REAL_POW(rminh, 3) + REAL_POW(rmkn, 3)) / (REAL_POW(rminh, 2) + REAL_POW(rmkn, 2));
            real rmkxh7 = REAL_POW(rmkxh, 7);
            real ahk = emkxh * rmkxh7;
            real rko = rmkxo / 2. + dispoff;
            real rkh = rmkxh / 2. + dispoff;
            real sk = rmkn * shctd;
            real sk2 = sk * sk;

            real sum1, sum2;
            pair_ewca<do_g>(r, r2, rio, rmixo, rmixo7, sk, sk2, aoi, emixo, sum1, true);
            pair_ewca<do_g>(r, r2, rih, rmixh, rmixh7, sk, sk2, ahi, emixh, sum2, false);
            e = sum1 + sum2;
            pair_ewca<do_g>(r, r2, rko, rmkxo, rmkxo7, si, si2, aok, emkxo, sum1, true);
            pair_ewca<do_g>(r, r2, rkh, rmkxh, rmkxh7, si, si2, ahk, emkxh, sum2, false);
            e += sum1 + sum2;
            e *= -slevy * awater;

            if CONSTEXPR (do_e)
               estl += floatTo<ebuf_prec>(e);
         } // end if (include)

         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_e) {
      atomic_add(estl, es, ithread);
   }
}
