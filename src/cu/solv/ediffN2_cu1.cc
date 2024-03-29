// ck.py Version 3.1.0
template <class Ver>
__global__
void ediffN2_cu1(int n, CountBuffer restrict nes, EnergyBuffer restrict es, grad_prec* restrict gx,
   grad_prec* restrict gy, grad_prec* restrict gz, real off, const unsigned* restrict mdpuinfo, int nexclude,
   const int (*restrict exclude)[2], const real (*restrict exclude_scale)[4], const real* restrict x,
   const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted, int nakpl,
   const int* restrict iakpl, int niakp, const int* restrict iakp, real* restrict trqx, real* restrict trqy,
   real* restrict trqz, const real (*restrict rpole)[10], const real (*restrict uind)[3],
   const real (*restrict uinds)[3], const real (*restrict uinp)[3], const real (*restrict uinps)[3], real f)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nestl;
   if CONSTEXPR (do_a) {
      nestl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec estl;
   if CONSTEXPR (do_e) {
      estl = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uidsx[BLOCK_DIM], uidsy[BLOCK_DIM],
      uidsz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM], uipz[BLOCK_DIM], uipsx[BLOCK_DIM], uipsy[BLOCK_DIM],
      uipsz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM],
      ukdx[BLOCK_DIM], ukdy[BLOCK_DIM], ukdz[BLOCK_DIM], ukdsx[BLOCK_DIM], ukdsy[BLOCK_DIM], ukdsz[BLOCK_DIM],
      ukpx[BLOCK_DIM], ukpy[BLOCK_DIM], ukpz[BLOCK_DIM], ukpsx[BLOCK_DIM], ukpsy[BLOCK_DIM], ukpsz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, pdk;
   int jpk;
   real gxi, gyi, gzi, txi, tyi, tzi;
   real gxk, gyk, gzk, txk, tyk, tzk;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scaleb = exclude_scale[ii][1];
      real scalec = exclude_scale[ii][2];
      real scaled = exclude_scale[ii][3];

      xi[klane] = x[i];
      yi[klane] = y[i];
      zi[klane] = z[i];
      ci[klane] = rpole[i][MPL_PME_0];
      dix[klane] = rpole[i][MPL_PME_X];
      diy[klane] = rpole[i][MPL_PME_Y];
      diz[klane] = rpole[i][MPL_PME_Z];
      qixx[klane] = rpole[i][MPL_PME_XX];
      qixy[klane] = rpole[i][MPL_PME_XY];
      qixz[klane] = rpole[i][MPL_PME_XZ];
      qiyy[klane] = rpole[i][MPL_PME_YY];
      qiyz[klane] = rpole[i][MPL_PME_YZ];
      qizz[klane] = rpole[i][MPL_PME_ZZ];
      uidx[klane] = uind[i][0];
      uidy[klane] = uind[i][1];
      uidz[klane] = uind[i][2];
      uidsx[klane] = uinds[i][0];
      uidsy[klane] = uinds[i][1];
      uidsz[klane] = uinds[i][2];
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
      uipsx[klane] = uinps[i][0];
      uipsy[klane] = uinps[i][1];
      uipsz[klane] = uinps[i][2];
      pdi[klane] = pdamp[i];
      jpi[klane] = jpolar[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uind[k][0];
      ukdy[threadIdx.x] = uind[k][1];
      ukdz[threadIdx.x] = uind[k][2];
      ukdsx[threadIdx.x] = uinds[k][0];
      ukdsy[threadIdx.x] = uinds[k][1];
      ukdsz[threadIdx.x] = uinds[k][2];
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
      ukpsx[threadIdx.x] = uinps[k][0];
      ukpsy[threadIdx.x] = uinps[k][1];
      ukpsz[threadIdx.x] = uinps[k][2];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      pdk = pdamp[k];
      jpk = jpolar[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];
      real r2 = xr * xr + yr * yr + zr * zr;
      if (r2 <= off * off and incl) {
         real pga = thlval[njpolar * jpi[klane] + jpk];
         real e;
         PairSolvGrad pgrad;
         zero(pgrad);
         pair_ediff<Ver>(r2, xr, yr, zr, scaleb, scalec, scaled, //
            ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
            qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uidsx[klane], uidsy[klane], uidsz[klane],
            uipx[klane], uipy[klane], uipz[klane], uipsx[klane], uipsy[klane], uipsz[klane], pdi[klane], pga, //
            ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
            ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukdsx[threadIdx.x], ukdsy[threadIdx.x],
            ukdsz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x], ukpz[threadIdx.x], ukpsx[threadIdx.x],
            ukpsy[threadIdx.x], ukpsz[threadIdx.x], pdk, pga, //
            f, e, pgrad);
         if CONSTEXPR (do_e) {
            estl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               nestl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            gxi += pgrad.frcx;
            gyi += pgrad.frcy;
            gzi += pgrad.frcz;
            gxk -= pgrad.frcx;
            gyk -= pgrad.frcy;
            gzk -= pgrad.frcz;

            txi += pgrad.ttqi[0];
            tyi += pgrad.ttqi[1];
            tzi += pgrad.ttqi[2];
            txk += pgrad.ttqk[0];
            tyk += pgrad.ttqk[1];
            tzk += pgrad.ttqk[2];
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
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
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uidsx[threadIdx.x] = uinds[i][0];
      uidsy[threadIdx.x] = uinds[i][1];
      uidsz[threadIdx.x] = uinds[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      uipsx[threadIdx.x] = uinps[i][0];
      uipsy[threadIdx.x] = uinps[i][1];
      uipsz[threadIdx.x] = uinps[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uind[k][0];
      ukdy[threadIdx.x] = uind[k][1];
      ukdz[threadIdx.x] = uind[k][2];
      ukdsx[threadIdx.x] = uinds[k][0];
      ukdsy[threadIdx.x] = uinds[k][1];
      ukdsz[threadIdx.x] = uinds[k][2];
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
      ukpsx[threadIdx.x] = uinps[k][0];
      ukpsy[threadIdx.x] = uinps[k][1];
      ukpsz[threadIdx.x] = uinps[k][2];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      pdk = pdamp[k];
      jpk = jpolar[k];
      __syncwarp();

      unsigned int mdpuinfo0 = mdpuinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (mdpuinfo0 & srcmask) == 0;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real pga = thlval[njpolar * jpi[klane] + jpk];
            real e;
            PairSolvGrad pgrad;
            zero(pgrad);
            pair_ediff<Ver>(r2, xr, yr, zr, 1, 1, 1, //
               ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uidsx[klane], uidsy[klane],
               uidsz[klane], uipx[klane], uipy[klane], uipz[klane], uipsx[klane], uipsy[klane], uipsz[klane],
               pdi[klane], pga, //
               ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukdsx[threadIdx.x], ukdsy[threadIdx.x],
               ukdsz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x], ukpz[threadIdx.x], ukpsx[threadIdx.x],
               ukpsy[threadIdx.x], ukpsz[threadIdx.x], pdk, pga, //
               f, e, pgrad);
            if CONSTEXPR (do_e) {
               estl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  nestl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               txk += pgrad.ttqk[0];
               tyk += pgrad.ttqk[1];
               tzk += pgrad.ttqk[2];
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
            txi = __shfl_sync(ALL_LANES, txi, ilane + 1);
            tyi = __shfl_sync(ALL_LANES, tyi, ilane + 1);
            tzi = __shfl_sync(ALL_LANES, tzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < niakp; iw += nwarp) {
      if CONSTEXPR (do_g) {
         gxi = 0;
         gyi = 0;
         gzi = 0;
         txi = 0;
         tyi = 0;
         tzi = 0;
         gxk = 0;
         gyk = 0;
         gzk = 0;
         txk = 0;
         tyk = 0;
         tzk = 0;
      }

      int tri, tx, ty;
      tri = iakp[iw];
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
      uidx[threadIdx.x] = uind[i][0];
      uidy[threadIdx.x] = uind[i][1];
      uidz[threadIdx.x] = uind[i][2];
      uidsx[threadIdx.x] = uinds[i][0];
      uidsy[threadIdx.x] = uinds[i][1];
      uidsz[threadIdx.x] = uinds[i][2];
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      uipsx[threadIdx.x] = uinps[i][0];
      uipsy[threadIdx.x] = uinps[i][1];
      uipsz[threadIdx.x] = uinps[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uind[k][0];
      ukdy[threadIdx.x] = uind[k][1];
      ukdz[threadIdx.x] = uind[k][2];
      ukdsx[threadIdx.x] = uinds[k][0];
      ukdsy[threadIdx.x] = uinds[k][1];
      ukdsz[threadIdx.x] = uinds[k][2];
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
      ukpsx[threadIdx.x] = uinps[k][0];
      ukpsy[threadIdx.x] = uinps[k][1];
      ukpsz[threadIdx.x] = uinps[k][2];
      ck = rpole[k][MPL_PME_0];
      qkxx = rpole[k][MPL_PME_XX];
      qkxy = rpole[k][MPL_PME_XY];
      qkxz = rpole[k][MPL_PME_XZ];
      qkyy = rpole[k][MPL_PME_YY];
      qkyz = rpole[k][MPL_PME_YZ];
      qkzz = rpole[k][MPL_PME_ZZ];
      pdk = pdamp[k];
      jpk = jpolar[k];
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
            real pga = thlval[njpolar * jpi[klane] + jpk];
            real e;
            PairSolvGrad pgrad;
            zero(pgrad);
            pair_ediff<Ver>(r2, xr, yr, zr, 1, 1, 1, //
               ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uidsx[klane], uidsy[klane],
               uidsz[klane], uipx[klane], uipy[klane], uipz[klane], uipsx[klane], uipsy[klane], uipsz[klane],
               pdi[klane], pga, //
               ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukdsx[threadIdx.x], ukdsy[threadIdx.x],
               ukdsz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x], ukpz[threadIdx.x], ukpsx[threadIdx.x],
               ukpsy[threadIdx.x], ukpsz[threadIdx.x], pdk, pga, //
               f, e, pgrad);
            if CONSTEXPR (do_e) {
               estl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  nestl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               gxi += pgrad.frcx;
               gyi += pgrad.frcy;
               gzi += pgrad.frcz;
               gxk -= pgrad.frcx;
               gyk -= pgrad.frcy;
               gzk -= pgrad.frcz;

               txi += pgrad.ttqi[0];
               tyi += pgrad.ttqi[1];
               tzi += pgrad.ttqi[2];
               txk += pgrad.ttqk[0];
               tyk += pgrad.ttqk[1];
               tzk += pgrad.ttqk[2];
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            gxi = __shfl_sync(ALL_LANES, gxi, ilane + 1);
            gyi = __shfl_sync(ALL_LANES, gyi, ilane + 1);
            gzi = __shfl_sync(ALL_LANES, gzi, ilane + 1);
            txi = __shfl_sync(ALL_LANES, txi, ilane + 1);
            tyi = __shfl_sync(ALL_LANES, tyi, ilane + 1);
            tzi = __shfl_sync(ALL_LANES, tzi, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(gxi, gx, i);
         atomic_add(gyi, gy, i);
         atomic_add(gzi, gz, i);
         atomic_add(txi, trqx, i);
         atomic_add(tyi, trqy, i);
         atomic_add(tzi, trqz, i);
         atomic_add(gxk, gx, k);
         atomic_add(gyk, gy, k);
         atomic_add(gzk, gz, k);
         atomic_add(txk, trqx, k);
         atomic_add(tyk, trqy, k);
         atomic_add(tzk, trqz, k);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(nestl, nes, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(estl, es, ithread);
   }
}
