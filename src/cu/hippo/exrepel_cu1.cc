// ck.py Version 3.1.0
template <class Ver>
__global__
void exrepel_cu1(int n, TINKER_IMAGE_PARAMS, CountBuffer restrict nr, EnergyBuffer restrict er,
   VirialBuffer restrict vr, grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real cut, real off,
   const unsigned* restrict rinfo, int nexclude, const int (*restrict exclude)[2], const real* restrict exclude_scale,
   const real* restrict x, const real* restrict y, const real* restrict z, const Spatial::SortedAtom* restrict sorted,
   int nakpl, const int* restrict iakpl, int niak, const int* restrict iak, const int* restrict lst,
   real* restrict trqx, real* restrict trqy, real* restrict trqz, const real* restrict zpxr,
   const real* restrict dmppxr, const real (*rcpxr)[4], const int* restrict mut, real vlam, Vdw vcouple)
{
   constexpr bool do_a = Ver::a;
   constexpr bool do_e = Ver::e;
   constexpr bool do_v = Ver::v;
   constexpr bool do_g = Ver::g;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nrtl;
   if CONSTEXPR (do_a) {
      nrtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec ertl;
   if CONSTEXPR (do_e) {
      ertl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vrtlxx, vrtlyx, vrtlzx, vrtlyy, vrtlzy, vrtlzz;
   if CONSTEXPR (do_v) {
      vrtlxx = 0;
      vrtlyx = 0;
      vrtlzx = 0;
      vrtlyy = 0;
      vrtlzy = 0;
      vrtlzz = 0;
   }
   __shared__ real xi[BLOCK_DIM], yi[BLOCK_DIM], zi[BLOCK_DIM], cis[BLOCK_DIM], cix[BLOCK_DIM], ciy[BLOCK_DIM],
      ciz[BLOCK_DIM], zxri[BLOCK_DIM], dmpi[BLOCK_DIM];
   __shared__ int imut[BLOCK_DIM];
   __shared__ real xk[BLOCK_DIM], yk[BLOCK_DIM], zk[BLOCK_DIM], ckx[BLOCK_DIM], cky[BLOCK_DIM], ckz[BLOCK_DIM];
   real cks, zxrk, dmpk;
   int kmut;
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
      real scalea = exclude_scale[ii];

      xi[klane] = x[i];
      yi[klane] = y[i];
      zi[klane] = z[i];
      cis[klane] = rcpxr[i][0];
      cix[klane] = rcpxr[i][1];
      ciy[klane] = rcpxr[i][2];
      ciz[klane] = rcpxr[i][3];
      zxri[klane] = zpxr[i];
      dmpi[klane] = dmppxr[i];
      imut[klane] = mut[i];
      xk[threadIdx.x] = x[k];
      yk[threadIdx.x] = y[k];
      zk[threadIdx.x] = z[k];
      ckx[threadIdx.x] = rcpxr[k][1];
      cky[threadIdx.x] = rcpxr[k][2];
      ckz[threadIdx.x] = rcpxr[k][3];
      cks = rcpxr[k][0];
      zxrk = zpxr[k];
      dmpk = dmppxr[k];
      kmut = mut[k];

      constexpr bool incl = true;
      real xr = xk[threadIdx.x] - xi[klane];
      real yr = yk[threadIdx.x] - yi[klane];
      real zr = zk[threadIdx.x] - zi[klane];

      real e = 0.;
      PairRepelGrad pgrad;
      zero(pgrad);

      real r2 = image2(xr, yr, zr);
      if (r2 <= off * off and incl) {
         real vlambda = pair_vlambda(vlam, vcouple, imut[klane], kmut);
         pair_xrepel<do_g, 1>( //
            r2, scalea, vlambda, cut, off, xr, yr, zr, zxri[klane], dmpi[klane], cis[klane], cix[klane], ciy[klane],
            ciz[klane], zxrk, dmpk, cks, ckx[threadIdx.x], cky[threadIdx.x], ckz[threadIdx.x], e, pgrad);

         if CONSTEXPR (do_a)
            if (e != 0)
               nrtl += 1;
         if CONSTEXPR (do_e)
            ertl += floatTo<ebuf_prec>(e);
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
         if CONSTEXPR (do_v) {
            vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
            vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
            vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
            vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
            vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
            vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
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
      cis[threadIdx.x] = rcpxr[i][0];
      cix[threadIdx.x] = rcpxr[i][1];
      ciy[threadIdx.x] = rcpxr[i][2];
      ciz[threadIdx.x] = rcpxr[i][3];
      zxri[threadIdx.x] = zpxr[i];
      dmpi[threadIdx.x] = dmppxr[i];
      imut[threadIdx.x] = mut[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ckx[threadIdx.x] = rcpxr[k][1];
      cky[threadIdx.x] = rcpxr[k][2];
      ckz[threadIdx.x] = rcpxr[k][3];
      cks = rcpxr[k][0];
      zxrk = zpxr[k];
      dmpk = dmppxr[k];
      kmut = mut[k];
      __syncwarp();

      unsigned int rinfo0 = rinfo[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (rinfo0 & srcmask) == 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e = 0.;
         PairRepelGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real vlambda = pair_vlambda(vlam, vcouple, imut[klane], kmut);
            pair_xrepel<do_g, 1>( //
               r2, scalea, vlambda, cut, off, xr, yr, zr, zxri[klane], dmpi[klane], cis[klane], cix[klane], ciy[klane],
               ciz[klane], zxrk, dmpk, cks, ckx[threadIdx.x], cky[threadIdx.x], ckz[threadIdx.x], e, pgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nrtl += 1;
            if CONSTEXPR (do_e)
               ertl += floatTo<ebuf_prec>(e);
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
            if CONSTEXPR (do_v) {
               vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
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

   for (int iw = iwarp; iw < niak; iw += nwarp) {
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

      int ty = iak[iw];
      int atomi = ty * WARP_SIZE + ilane;
      int i = sorted[atomi].unsorted;
      int atomk = lst[iw * WARP_SIZE + ilane];
      int k = sorted[atomk].unsorted;
      xi[threadIdx.x] = sorted[atomi].x;
      yi[threadIdx.x] = sorted[atomi].y;
      zi[threadIdx.x] = sorted[atomi].z;
      cis[threadIdx.x] = rcpxr[i][0];
      cix[threadIdx.x] = rcpxr[i][1];
      ciy[threadIdx.x] = rcpxr[i][2];
      ciz[threadIdx.x] = rcpxr[i][3];
      zxri[threadIdx.x] = zpxr[i];
      dmpi[threadIdx.x] = dmppxr[i];
      imut[threadIdx.x] = mut[i];
      xk[threadIdx.x] = sorted[atomk].x;
      yk[threadIdx.x] = sorted[atomk].y;
      zk[threadIdx.x] = sorted[atomk].z;
      ckx[threadIdx.x] = rcpxr[k][1];
      cky[threadIdx.x] = rcpxr[k][2];
      ckz[threadIdx.x] = rcpxr[k][3];
      cks = rcpxr[k][0];
      zxrk = zpxr[k];
      dmpk = dmppxr[k];
      kmut = mut[k];
      __syncwarp();

      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         int klane = srclane + threadIdx.x - ilane;
         bool incl = atomk > 0;
         real scalea = 1;
         real xr = xk[threadIdx.x] - xi[klane];
         real yr = yk[threadIdx.x] - yi[klane];
         real zr = zk[threadIdx.x] - zi[klane];

         real e = 0.;
         PairRepelGrad pgrad;
         zero(pgrad);

         real r2 = image2(xr, yr, zr);
         if (r2 <= off * off and incl) {
            real vlambda = pair_vlambda(vlam, vcouple, imut[klane], kmut);
            pair_xrepel<do_g, 1>( //
               r2, scalea, vlambda, cut, off, xr, yr, zr, zxri[klane], dmpi[klane], cis[klane], cix[klane], ciy[klane],
               ciz[klane], zxrk, dmpk, cks, ckx[threadIdx.x], cky[threadIdx.x], ckz[threadIdx.x], e, pgrad);

            if CONSTEXPR (do_a)
               if (e != 0)
                  nrtl += 1;
            if CONSTEXPR (do_e)
               ertl += floatTo<ebuf_prec>(e);
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
            if CONSTEXPR (do_v) {
               vrtlxx += floatTo<vbuf_prec>(-xr * pgrad.frcx);
               vrtlyx += floatTo<vbuf_prec>(-0.5f * (yr * pgrad.frcx + xr * pgrad.frcy));
               vrtlzx += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcx + xr * pgrad.frcz));
               vrtlyy += floatTo<vbuf_prec>(-yr * pgrad.frcy);
               vrtlzy += floatTo<vbuf_prec>(-0.5f * (zr * pgrad.frcy + yr * pgrad.frcz));
               vrtlzz += floatTo<vbuf_prec>(-zr * pgrad.frcz);
            }
         } // end if (include)

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
      atomic_add(nrtl, nr, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(ertl, er, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vrtlxx, vrtlyx, vrtlzx, vrtlyy, vrtlzy, vrtlzz, vr, ithread);
   }
}
