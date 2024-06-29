// ck.py Version 3.1.0
template <class Ver>
__global__
void ewcaN2_cu1(int n, EnergyBuffer restrict es, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz,
   real off, const real* restrict x, const real* restrict y, const real* restrict z, int nakp, const int* restrict iakp,
   const real* restrict epsdsp, const real* restrict raddsp, real epso, real epsosqrt, real epsh, real epshsqrt,
   real rmino2, real rmino3, real rminh2, real rminh3, real shctd, real dspoff, real slwater)
{
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
      epsli[threadIdx.x] = epsdsp[i];
      rmini[threadIdx.x] = raddsp[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      epslk = epsdsp[k];
      rmink = raddsp[k];
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

         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real r3 = r2 * r;
            real epsi = epsli[klane];
            real rmin = rmini[klane];
            real epsisqrt = REAL_SQRT(epsi);
            real term1 = epsosqrt + epsisqrt;
            real term12 = term1 * term1;
            real rmin2 = rmin * rmin;
            real rmin3 = rmin2 * rmin;
            real emixo = 4 * epso * epsi / term12;
            real rmixo = 2 * (rmino3 + rmin3) / (rmino2 + rmin2);
            real rmixo7 = REAL_POW(rmixo, 7);
            real aoi = emixo * rmixo7;
            real term2 = epshsqrt + epsisqrt;
            real term22 = term2 * term2;
            real emixh = 4 * epsh * epsi / term22;
            real rmixh = 2 * (rminh3 + rmin3) / (rminh2 + rmin2);
            real rmixh7 = REAL_POW(rmixh, 7);
            real ahi = emixh * rmixh7;
            real rio = rmixo / 2 + dspoff;
            real rih = rmixh / 2 + dspoff;
            real si = rmin * shctd;
            real si2 = si * si;

            real epsk = epslk;
            real rmkn = rmink;
            real epsksqrt = REAL_SQRT(epsk);
            real term3 = epsosqrt + epsksqrt;
            real term32 = term3 * term3;
            real emkxo = 4 * epso * epsk / term32;
            real rmkn2 = rmkn * rmkn;
            real rmkn3 = rmkn2 * rmkn;
            real rmkxo = 2 * (rmino3 + rmkn3) / (rmino2 + rmkn2);
            real rmkxo7 = REAL_POW(rmkxo, 7);
            real aok = emkxo * rmkxo7;
            real term4 = epshsqrt + epsksqrt;
            real term42 = term4 * term4;
            real emkxh = 4 * epsh * epsk / term42;
            real rmkxh = 2 * (rminh3 + rmkn3) / (rminh2 + rmkn2);
            real rmkxh7 = REAL_POW(rmkxh, 7);
            real ahk = emkxh * rmkxh7;
            real rko = rmkxo / 2 + dspoff;
            real rkh = rmkxh / 2 + dspoff;
            real sk = rmkn * shctd;
            real sk2 = sk * sk;

            real sum1, sum2;
            real de, de1, de2;
            real de11, de12, de21, de22;

            pair_ewca<Ver>(r, r2, r3, rio, rmixo, rmixo7, sk, sk2, aoi, emixo, sum1, de11, true);
            pair_ewca<Ver>(r, r2, r3, rih, rmixh, rmixh7, sk, sk2, ahi, emixh, sum2, de12, false);
            e = sum1 + sum2;

            pair_ewca<Ver>(r, r2, r3, rko, rmkxo, rmkxo7, si, si2, aok, emkxo, sum1, de21, true);
            pair_ewca<Ver>(r, r2, r3, rkh, rmkxh, rmkxh7, si, si2, ahk, emkxh, sum2, de22, false);
            e += sum1 + sum2;

            e *= -slwater;

            if CONSTEXPR (do_e)
               estl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_g) {
               de1 = de11 + de12;
               de2 = de21 + de22;
               de = de1 + de2;
               de *= slwater / r;
               real dedx = de * xr;
               real dedy = de * yr;
               real dedz = de * zr;
               gxi += dedx;
               gyi += dedy;
               gzi += dedz;
               gxk -= dedx;
               gyk -= dedy;
               gzk -= dedz;
            }
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

   if CONSTEXPR (do_e) {
      atomic_add(estl, es, ithread);
   }
}
