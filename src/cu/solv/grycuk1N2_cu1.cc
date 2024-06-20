// ck.py Version 3.1.0
template <class Ver>
__global__
void grycuk1N2_cu1(int n, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off,
   const real* restrict x, const real* restrict y, const real* restrict z, int nakp, const int* restrict iakp,
   real descoff, real pi43, real factor, bool useneck, bool usetanh, const real* restrict rsolv,
   const real* restrict rdescr, const real* restrict shct, const real* restrict rborn, const real* restrict drb,
   const real* restrict drbp, const real* restrict aneck, const real* restrict bneck, const real* restrict rneck,
   const real* restrict sneck, const real* restrict bornint, bool use_gk)
{
   constexpr bool do_g = Ver::g;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], rsi[BLOCK_DIM], rdi[BLOCK_DIM], shcti[BLOCK_DIM],
      rbi[BLOCK_DIM], drbi[BLOCK_DIM], drbpi[BLOCK_DIM], snecki[BLOCK_DIM], borni[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM];
   real rsk, rdk, shctk, rbk, drbk, drbpk, sneckk, bornk;
   real gxi, gyi, gzi;
   real gxk, gyk, gzk;

   for (int iw = iwarp; iw < nakp; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
      }

      int tri, tx, ty;
      tri = iakp[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
      xi[threadIdx.x] = x[i];
      yi[threadIdx.x] = y[i];
      zi[threadIdx.x] = z[i];
      rsi[threadIdx.x] = rsolv[i];
      rdi[threadIdx.x] = rdescr[i];
      shcti[threadIdx.x] = shct[i];
      rbi[threadIdx.x] = rborn[i];
      drbi[threadIdx.x] = drb[i];
      drbpi[threadIdx.x] = drbp[i];
      snecki[threadIdx.x] = sneck[i];
      borni[threadIdx.x] = bornint[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      rsk = rsolv[k];
      rdk = rdescr[k];
      shctk = shct[k];
      rbk = rborn[k];
      drbk = drb[k];
      drbpk = drbp[k];
      sneckk = sneck[k];
      bornk = bornint[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real ri = REAL_MAX(rsi[klane], rdi[klane]) + descoff;
            real si = rdi[klane] * shcti[klane];
            real rbir = rbi[klane];
            real rbi3 = rbir * rbir * rbir;
            real termi = pi43 / rbi3;
            termi = factor / REAL_POW(termi, (real)4 / 3);
            real mixsn = (real)0.5 * (snecki[klane] + sneckk);
            real rk = REAL_MAX(rsk, rdk) + descoff;
            real sk = rdk * shctk;
            real rbkr = rbk;
            real rbk3 = rbkr * rbkr * rbkr;
            real termk = pi43 / rbk3;
            termk = factor / REAL_POW(termk, (real)4 / 3);
            if (usetanh) {
               real tcr;
               tanhrscchr(borni[klane], rsi[klane], tcr, pi43);
               termi = termi * tcr;
               tanhrscchr(bornk, rsk, tcr, pi43);
               termk = termk * tcr;
            }
            bool computei = (rsi[klane] > 0) and (rdk > 0) and (sk > 0);
            bool computek = (rsk > 0) and (rdi[klane] > 0) and (si > 0);
            real dei = 0;
            real dek = 0;
            if (computei) {
               pair_dgrycuk(r, r2, ri, rdk, sk, mixsn, pi43, drbi[klane], drbpi[klane], termi, use_gk, useneck, aneck,
                  bneck, rneck, dei);
            }
            if (computek) {
               pair_dgrycuk(r, r2, rk, rdi[klane], si, mixsn, pi43, drbk, drbpk, termk, use_gk, useneck, aneck, bneck,
                  rneck, dek);
            }
            real de = dei + dek;
            real dedx = de * xr;
            real dedy = de * yr;
            real dedz = de * zr;
            if CONSTEXPR (do_g) {
               gxi += dedx;
               gyi += dedy;
               gzi += dedz;
               gxk -= dedx;
               gyk -= dedy;
               gzk -= dedz;
            }
         }

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
}
