// ck.py Version 3.1.0
template <class Ver>
__global__
void epolarNonEwaldN2_cu1(int n, CountBuffer restrict nep, EnergyBuffer restrict ep, VirialBuffer restrict vep,
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real off, const unsigned* restrict mdpuinfo,
   int nexclude, const int (*restrict exclude)[2], const real (*restrict exclude_scale)[4], const real* restrict x,
   const real* restrict y, const real* restrict z, int nakpl, const int* restrict iakpl, int nakpa,
   const int* restrict iakpa, real (*restrict ufld)[3], real (*restrict dufld)[6], const real (*restrict rpole)[10],
   const real (*restrict uind)[3], const real (*restrict uinp)[3], real f)
{
   using d::jpolar;
   using d::njpolar;
   using d::pdamp;
   using d::thlval;
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int neptl;
   if CONSTEXPR (do_a) {
      neptl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec eptl;
   if CONSTEXPR (do_e) {
      eptl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec veptlxx, veptlyx, veptlzx, veptlyy, veptlzy, veptlzz;
   if CONSTEXPR (do_v) {
      veptlxx = 0;
      veptlyx = 0;
      veptlzx = 0;
      veptlyy = 0;
      veptlzy = 0;
      veptlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], ci[BLOCK_DIM], dix[BLOCK_DIM], diy[BLOCK_DIM],
      diz[BLOCK_DIM], qixx[BLOCK_DIM], qixy[BLOCK_DIM], qixz[BLOCK_DIM], qiyy[BLOCK_DIM], qiyz[BLOCK_DIM],
      qizz[BLOCK_DIM], uidx[BLOCK_DIM], uidy[BLOCK_DIM], uidz[BLOCK_DIM], uipx[BLOCK_DIM], uipy[BLOCK_DIM],
      uipz[BLOCK_DIM], pdi[BLOCK_DIM];
   __shared__ int jpi[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], dkx[BLOCK_DIM], dky[BLOCK_DIM], dkz[BLOCK_DIM],
      ukdx[BLOCK_DIM], ukdy[BLOCK_DIM], ukdz[BLOCK_DIM], ukpx[BLOCK_DIM], ukpy[BLOCK_DIM], ukpz[BLOCK_DIM];
   real ck, qkxx, qkxy, qkxz, qkyy, qkyz, qkzz, pdk;
   int jpk;
   real frcxi, frcyi, frczi, ufld0i, ufld1i, ufld2i, dufld0i, dufld1i, dufld2i, dufld3i, dufld4i, dufld5i;
   real frcxk, frcyk, frczk, ufld0k, ufld1k, ufld2k, dufld0k, dufld1k, dufld2k, dufld3k, dufld4k, dufld5k;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      const int klane = threadIdx.x;
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         ufld0i = 0;
         ufld1i = 0;
         ufld2i = 0;
         dufld0i = 0;
         dufld1i = 0;
         dufld2i = 0;
         dufld3i = 0;
         dufld4i = 0;
         dufld5i = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         ufld0k = 0;
         ufld1k = 0;
         ufld2k = 0;
         dufld0k = 0;
         dufld1k = 0;
         dufld2k = 0;
         dufld3k = 0;
         dufld4k = 0;
         dufld5k = 0;
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
      uipx[klane] = uinp[i][0];
      uipy[klane] = uinp[i][1];
      uipz[klane] = uinp[i][2];
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
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
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
         real e, vxx, vyx, vzx, vyy, vzy, vzz;
         real aewald = 0;
         pair_polar_v2<Ver, NON_EWALD>(r2, xr, yr, zr, scaleb, scalec, scaled, //
            ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
            qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane], uipz[klane],
            pdi[klane], pga, //
            ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
            ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x],
            ukpz[threadIdx.x], pdk, pga, //
            f, aewald,                   //
            frcxi, frcyi, frczi, frcxk, frcyk, frczk, ufld0i, ufld1i, ufld2i, ufld0k, ufld1k, ufld2k, dufld0i, dufld1i,
            dufld2i, dufld3i, dufld4i, dufld5i, dufld0k, dufld1k, dufld2k, dufld3k, dufld4k, dufld5k, //
            e, vxx, vyx, vzx, vyy, vzy, vzz);
         if CONSTEXPR (do_e) {
            eptl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalec != 0 and e != 0) // pscale != 0
                  neptl += 1;
            }
         }
         if CONSTEXPR (do_v) {
            veptlxx += floatTo<vbuf_prec>(vxx);
            veptlyx += floatTo<vbuf_prec>(vyx);
            veptlzx += floatTo<vbuf_prec>(vzx);
            veptlyy += floatTo<vbuf_prec>(vyy);
            veptlzy += floatTo<vbuf_prec>(vzy);
            veptlzz += floatTo<vbuf_prec>(vzz);
         }
      } // end if (include)

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(ufld0i, &ufld[i][0]);
         atomic_add(ufld1i, &ufld[i][1]);
         atomic_add(ufld2i, &ufld[i][2]);
         atomic_add(dufld0i, &dufld[i][0]);
         atomic_add(dufld1i, &dufld[i][1]);
         atomic_add(dufld2i, &dufld[i][2]);
         atomic_add(dufld3i, &dufld[i][3]);
         atomic_add(dufld4i, &dufld[i][4]);
         atomic_add(dufld5i, &dufld[i][5]);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(ufld0k, &ufld[k][0]);
         atomic_add(ufld1k, &ufld[k][1]);
         atomic_add(ufld2k, &ufld[k][2]);
         atomic_add(dufld0k, &dufld[k][0]);
         atomic_add(dufld1k, &dufld[k][1]);
         atomic_add(dufld2k, &dufld[k][2]);
         atomic_add(dufld3k, &dufld[k][3]);
         atomic_add(dufld4k, &dufld[k][4]);
         atomic_add(dufld5k, &dufld[k][5]);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         ufld0i = 0;
         ufld1i = 0;
         ufld2i = 0;
         dufld0i = 0;
         dufld1i = 0;
         dufld2i = 0;
         dufld3i = 0;
         dufld4i = 0;
         dufld5i = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         ufld0k = 0;
         ufld1k = 0;
         ufld2k = 0;
         dufld0k = 0;
         dufld1k = 0;
         dufld2k = 0;
         dufld3k = 0;
         dufld4k = 0;
         dufld5k = 0;
      }

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
      xi[threadIdx.x] = x[i];
      yi[threadIdx.x] = y[i];
      zi[threadIdx.x] = z[i];
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
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uind[k][0];
      ukdy[threadIdx.x] = uind[k][1];
      ukdz[threadIdx.x] = uind[k][2];
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
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
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            real aewald = 0;
            pair_polar_v2<Ver, NON_EWALD>(r2, xr, yr, zr, 1, 1, 1, //
               ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane], uipz[klane],
               pdi[klane], pga, //
               ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x],
               ukpz[threadIdx.x], pdk, pga, //
               f, aewald,                   //
               frcxi, frcyi, frczi, frcxk, frcyk, frczk, ufld0i, ufld1i, ufld2i, ufld0k, ufld1k, ufld2k, dufld0i,
               dufld1i, dufld2i, dufld3i, dufld4i, dufld5i, dufld0k, dufld1k, dufld2k, dufld3k, dufld4k, dufld5k, //
               e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               eptl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     neptl += 1;
               }
            }
            if CONSTEXPR (do_v) {
               veptlxx += floatTo<vbuf_prec>(vxx);
               veptlyx += floatTo<vbuf_prec>(vyx);
               veptlzx += floatTo<vbuf_prec>(vzx);
               veptlyy += floatTo<vbuf_prec>(vyy);
               veptlzy += floatTo<vbuf_prec>(vzy);
               veptlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            ufld0i = __shfl_sync(ALL_LANES, ufld0i, ilane + 1);
            ufld1i = __shfl_sync(ALL_LANES, ufld1i, ilane + 1);
            ufld2i = __shfl_sync(ALL_LANES, ufld2i, ilane + 1);
            dufld0i = __shfl_sync(ALL_LANES, dufld0i, ilane + 1);
            dufld1i = __shfl_sync(ALL_LANES, dufld1i, ilane + 1);
            dufld2i = __shfl_sync(ALL_LANES, dufld2i, ilane + 1);
            dufld3i = __shfl_sync(ALL_LANES, dufld3i, ilane + 1);
            dufld4i = __shfl_sync(ALL_LANES, dufld4i, ilane + 1);
            dufld5i = __shfl_sync(ALL_LANES, dufld5i, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(ufld0i, &ufld[i][0]);
         atomic_add(ufld1i, &ufld[i][1]);
         atomic_add(ufld2i, &ufld[i][2]);
         atomic_add(dufld0i, &dufld[i][0]);
         atomic_add(dufld1i, &dufld[i][1]);
         atomic_add(dufld2i, &dufld[i][2]);
         atomic_add(dufld3i, &dufld[i][3]);
         atomic_add(dufld4i, &dufld[i][4]);
         atomic_add(dufld5i, &dufld[i][5]);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(ufld0k, &ufld[k][0]);
         atomic_add(ufld1k, &ufld[k][1]);
         atomic_add(ufld2k, &ufld[k][2]);
         atomic_add(dufld0k, &dufld[k][0]);
         atomic_add(dufld1k, &dufld[k][1]);
         atomic_add(dufld2k, &dufld[k][2]);
         atomic_add(dufld3k, &dufld[k][3]);
         atomic_add(dufld4k, &dufld[k][4]);
         atomic_add(dufld5k, &dufld[k][5]);
      }
      __syncwarp();
   }

   for (int iw = iwarp; iw < nakpa; iw += nwarp) {
      if CONSTEXPR (do_g) {
         frcxi = 0;
         frcyi = 0;
         frczi = 0;
         ufld0i = 0;
         ufld1i = 0;
         ufld2i = 0;
         dufld0i = 0;
         dufld1i = 0;
         dufld2i = 0;
         dufld3i = 0;
         dufld4i = 0;
         dufld5i = 0;
         frcxk = 0;
         frcyk = 0;
         frczk = 0;
         ufld0k = 0;
         ufld1k = 0;
         ufld2k = 0;
         dufld0k = 0;
         dufld1k = 0;
         dufld2k = 0;
         dufld3k = 0;
         dufld4k = 0;
         dufld5k = 0;
      }

      int tri, tx, ty;
      tri = iakpa[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
      xi[threadIdx.x] = x[i];
      yi[threadIdx.x] = y[i];
      zi[threadIdx.x] = z[i];
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
      uipx[threadIdx.x] = uinp[i][0];
      uipy[threadIdx.x] = uinp[i][1];
      uipz[threadIdx.x] = uinp[i][2];
      pdi[threadIdx.x] = pdamp[i];
      jpi[threadIdx.x] = jpolar[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      dkx[threadIdx.x] = rpole[k][MPL_PME_X];
      dky[threadIdx.x] = rpole[k][MPL_PME_Y];
      dkz[threadIdx.x] = rpole[k][MPL_PME_Z];
      ukdx[threadIdx.x] = uind[k][0];
      ukdy[threadIdx.x] = uind[k][1];
      ukdz[threadIdx.x] = uind[k][2];
      ukpx[threadIdx.x] = uinp[k][0];
      ukpy[threadIdx.x] = uinp[k][1];
      ukpz[threadIdx.x] = uinp[k][2];
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
            real e, vxx, vyx, vzx, vyy, vzy, vzz;
            real aewald = 0;
            pair_polar_v2<Ver, NON_EWALD>(r2, xr, yr, zr, 1, 1, 1, //
               ci[klane], dix[klane], diy[klane], diz[klane], qixx[klane], qixy[klane], qixz[klane], qiyy[klane],
               qiyz[klane], qizz[klane], uidx[klane], uidy[klane], uidz[klane], uipx[klane], uipy[klane], uipz[klane],
               pdi[klane], pga, //
               ck, dkx[threadIdx.x], dky[threadIdx.x], dkz[threadIdx.x], qkxx, qkxy, qkxz, qkyy, qkyz, qkzz,
               ukdx[threadIdx.x], ukdy[threadIdx.x], ukdz[threadIdx.x], ukpx[threadIdx.x], ukpy[threadIdx.x],
               ukpz[threadIdx.x], pdk, pga, //
               f, aewald,                   //
               frcxi, frcyi, frczi, frcxk, frcyk, frczk, ufld0i, ufld1i, ufld2i, ufld0k, ufld1k, ufld2k, dufld0i,
               dufld1i, dufld2i, dufld3i, dufld4i, dufld5i, dufld0k, dufld1k, dufld2k, dufld3k, dufld4k, dufld5k, //
               e, vxx, vyx, vzx, vyy, vzy, vzz);
            if CONSTEXPR (do_e) {
               eptl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     neptl += 1;
               }
            }
            if CONSTEXPR (do_v) {
               veptlxx += floatTo<vbuf_prec>(vxx);
               veptlyx += floatTo<vbuf_prec>(vyx);
               veptlzx += floatTo<vbuf_prec>(vzx);
               veptlyy += floatTo<vbuf_prec>(vyy);
               veptlzy += floatTo<vbuf_prec>(vzy);
               veptlzz += floatTo<vbuf_prec>(vzz);
            }
         } // end if (include)

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         if CONSTEXPR (do_g) {
            frcxi = __shfl_sync(ALL_LANES, frcxi, ilane + 1);
            frcyi = __shfl_sync(ALL_LANES, frcyi, ilane + 1);
            frczi = __shfl_sync(ALL_LANES, frczi, ilane + 1);
            ufld0i = __shfl_sync(ALL_LANES, ufld0i, ilane + 1);
            ufld1i = __shfl_sync(ALL_LANES, ufld1i, ilane + 1);
            ufld2i = __shfl_sync(ALL_LANES, ufld2i, ilane + 1);
            dufld0i = __shfl_sync(ALL_LANES, dufld0i, ilane + 1);
            dufld1i = __shfl_sync(ALL_LANES, dufld1i, ilane + 1);
            dufld2i = __shfl_sync(ALL_LANES, dufld2i, ilane + 1);
            dufld3i = __shfl_sync(ALL_LANES, dufld3i, ilane + 1);
            dufld4i = __shfl_sync(ALL_LANES, dufld4i, ilane + 1);
            dufld5i = __shfl_sync(ALL_LANES, dufld5i, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(frcxi, gx, i);
         atomic_add(frcyi, gy, i);
         atomic_add(frczi, gz, i);
         atomic_add(ufld0i, &ufld[i][0]);
         atomic_add(ufld1i, &ufld[i][1]);
         atomic_add(ufld2i, &ufld[i][2]);
         atomic_add(dufld0i, &dufld[i][0]);
         atomic_add(dufld1i, &dufld[i][1]);
         atomic_add(dufld2i, &dufld[i][2]);
         atomic_add(dufld3i, &dufld[i][3]);
         atomic_add(dufld4i, &dufld[i][4]);
         atomic_add(dufld5i, &dufld[i][5]);
         atomic_add(frcxk, gx, k);
         atomic_add(frcyk, gy, k);
         atomic_add(frczk, gz, k);
         atomic_add(ufld0k, &ufld[k][0]);
         atomic_add(ufld1k, &ufld[k][1]);
         atomic_add(ufld2k, &ufld[k][2]);
         atomic_add(dufld0k, &dufld[k][0]);
         atomic_add(dufld1k, &dufld[k][1]);
         atomic_add(dufld2k, &dufld[k][2]);
         atomic_add(dufld3k, &dufld[k][3]);
         atomic_add(dufld4k, &dufld[k][4]);
         atomic_add(dufld5k, &dufld[k][5]);
      }
      __syncwarp();
   }

   if CONSTEXPR (do_a) {
      atomic_add(neptl, nep, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(eptl, ep, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(veptlxx, veptlyx, veptlzx, veptlyy, veptlzy, veptlzz, vep, ithread);
   }
}
