// ck.py Version 3.1.0
template <class Ver>
__global__
void ehalN2_cu1(int n, CountBuffer restrict nev, EnergyBuffer restrict ev, VirialBuffer restrict vev,
   grad_prec* restrict gx, grad_prec* restrict gy, grad_prec* restrict gz, real cut, real off,
   const unsigned* restrict info, int nexclude, const int (*restrict exclude)[2], const real* restrict exclude_scale,
   const real* restrict x, const real* restrict y, const real* restrict z, int nakpl, const int* restrict iakpl,
   int nakpa, const int* restrict iakpa, int njvdw, real vlam, Vdw vcouple, const real* restrict radmin,
   const real* restrict epsilon, const int* restrict jvdw, const int* restrict mut)
{
   constexpr bool do_e = Ver::e;
   constexpr bool do_a = Ver::a;
   constexpr bool do_g = Ver::g;
   constexpr bool do_v = Ver::v;
   const int ithread = threadIdx.x + blockIdx.x * blockDim.x;
   const int iwarp = ithread / WARP_SIZE;
   const int nwarp = blockDim.x * gridDim.x / WARP_SIZE;
   const int ilane = threadIdx.x & (WARP_SIZE - 1);

   int nevtl;
   if CONSTEXPR (do_a) {
      nevtl = 0;
   }
   using ebuf_prec = EnergyBufferTraits::type;
   ebuf_prec evtl;
   if CONSTEXPR (do_e) {
      evtl = 0;
   }
   using vbuf_prec = VirialBufferTraits::type;
   vbuf_prec vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz;
   if CONSTEXPR (do_v) {
      vevtlxx = 0;
      vevtlyx = 0;
      vevtlzx = 0;
      vevtlyy = 0;
      vevtlzy = 0;
      vevtlzz = 0;
   }
   real xi, yi, zi;
   int ijvdw, imut;
   real xk, yk, zk;
   int kjvdw, kmut;
   real fix, fiy, fiz;
   real fkx, fky, fkz;

   //* /
   for (int ii = ithread; ii < nexclude; ii += blockDim.x * gridDim.x) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }

      int i = exclude[ii][0];
      int k = exclude[ii][1];
      real scalea = exclude_scale[ii];

      xi = x[i];
      yi = y[i];
      zi = z[i];
      ijvdw = jvdw[i];
      imut = mut[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kjvdw = jvdw[k];
      kmut = mut[k];

      constexpr bool incl = true;
      real xr = xi - xk;
      real yr = yi - yk;
      real zr = zi - zk;
      real r2 = xr * xr + yr * yr + zr * zr;
      if (r2 <= off * off and incl) {
         real r = REAL_SQRT(r2);
         real rv = radmin[ijvdw * njvdw + kjvdw];
         real eps = epsilon[ijvdw * njvdw + kjvdw];
         real vlambda = 1;
         if (vcouple == Vdw::DECOUPLE) {
            vlambda = (imut == kmut ? 1 : vlam);
         } else if (vcouple == Vdw::ANNIHILATE) {
            vlambda = (imut || kmut ? vlam : 1);
         }
         real e, de;
         pair_hal_v2<do_g, 0>(r, scalea, rv, eps, cut, off, vlambda, GHAL, DHAL, SCEXP, SCALPHA, e, de);
         if CONSTEXPR (do_e) {
            evtl += floatTo<ebuf_prec>(e);
            if CONSTEXPR (do_a) {
               if (scalea != 0 and e != 0)
                  nevtl += 1;
            }
         }
         if CONSTEXPR (do_g) {
            real dedx, dedy, dedz;
            de = de * REAL_RECIP(r);
            dedx = de * xr;
            dedy = de * yr;
            dedz = de * zr;
            fix += dedx;
            fiy += dedy;
            fiz += dedz;
            fkx -= dedx;
            fky -= dedy;
            fkz -= dedz;
            if CONSTEXPR (do_v) {
               vevtlxx += floatTo<vbuf_prec>(xr * dedx);
               vevtlyx += floatTo<vbuf_prec>(yr * dedx);
               vevtlzx += floatTo<vbuf_prec>(zr * dedx);
               vevtlyy += floatTo<vbuf_prec>(yr * dedy);
               vevtlzy += floatTo<vbuf_prec>(zr * dedy);
               vevtlzz += floatTo<vbuf_prec>(zr * dedz);
            }
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
   }
   // */

   for (int iw = iwarp; iw < nakpl; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }

      int tri, tx, ty;
      tri = iakpl[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
      xi = x[i];
      yi = y[i];
      zi = z[i];
      ijvdw = jvdw[i];
      imut = mut[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kjvdw = jvdw[k];
      kmut = mut[k];

      unsigned int info0 = info[iw * WARP_SIZE + ilane];
      for (int j = 0; j < WARP_SIZE; ++j) {
         int srclane = (ilane + j) & (WARP_SIZE - 1);
         bool incl = iid < kid and kid < n;
         int srcmask = 1 << srclane;
         incl = incl and (info0 & srcmask) == 0;
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real rv = radmin[ijvdw * njvdw + kjvdw];
            real eps = epsilon[ijvdw * njvdw + kjvdw];
            real vlambda = 1;
            if (vcouple == Vdw::DECOUPLE) {
               vlambda = (imut == kmut ? 1 : vlam);
            } else if (vcouple == Vdw::ANNIHILATE) {
               vlambda = (imut || kmut ? vlam : 1);
            }
            real e, de;
            pair_hal_v2<do_g, 1>(r, 1, rv, eps, cut, off, vlambda, GHAL, DHAL, SCEXP, SCALPHA, e, de);
            if CONSTEXPR (do_e) {
               evtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nevtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               real dedx, dedy, dedz;
               de = de * REAL_RECIP(r);
               dedx = de * xr;
               dedy = de * yr;
               dedz = de * zr;
               fix += dedx;
               fiy += dedy;
               fiz += dedz;
               fkx -= dedx;
               fky -= dedy;
               fkz -= dedz;
               if CONSTEXPR (do_v) {
                  vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                  vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                  vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                  vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                  vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                  vevtlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
         imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
   }

   for (int iw = iwarp; iw < nakpa; iw += nwarp) {
      if CONSTEXPR (do_g) {
         fix = 0;
         fiy = 0;
         fiz = 0;
         fkx = 0;
         fky = 0;
         fkz = 0;
      }

      int tri, tx, ty;
      tri = iakpa[iw];
      tri_to_xy(tri, tx, ty);

      int iid = ty * WARP_SIZE + ilane;
      int i = min(iid, n - 1);
      int kid = tx * WARP_SIZE + ilane;
      int k = min(kid, n - 1);
      xi = x[i];
      yi = y[i];
      zi = z[i];
      ijvdw = jvdw[i];
      imut = mut[i];
      xk = x[k];
      yk = y[k];
      zk = z[k];
      kjvdw = jvdw[k];
      kmut = mut[k];

      for (int j = 0; j < WARP_SIZE; ++j) {
         bool incl = iid < kid and kid < n;
         real xr = xi - xk;
         real yr = yi - yk;
         real zr = zi - zk;
         real r2 = xr * xr + yr * yr + zr * zr;
         if (r2 <= off * off and incl) {
            real r = REAL_SQRT(r2);
            real rv = radmin[ijvdw * njvdw + kjvdw];
            real eps = epsilon[ijvdw * njvdw + kjvdw];
            real vlambda = 1;
            if (vcouple == Vdw::DECOUPLE) {
               vlambda = (imut == kmut ? 1 : vlam);
            } else if (vcouple == Vdw::ANNIHILATE) {
               vlambda = (imut || kmut ? vlam : 1);
            }
            real e, de;
            pair_hal_v2<do_g, 1>(r, 1, rv, eps, cut, off, vlambda, GHAL, DHAL, SCEXP, SCALPHA, e, de);
            if CONSTEXPR (do_e) {
               evtl += floatTo<ebuf_prec>(e);
               if CONSTEXPR (do_a) {
                  if (e != 0)
                     nevtl += 1;
               }
            }
            if CONSTEXPR (do_g) {
               real dedx, dedy, dedz;
               de = de * REAL_RECIP(r);
               dedx = de * xr;
               dedy = de * yr;
               dedz = de * zr;
               fix += dedx;
               fiy += dedy;
               fiz += dedz;
               fkx -= dedx;
               fky -= dedy;
               fkz -= dedz;
               if CONSTEXPR (do_v) {
                  vevtlxx += floatTo<vbuf_prec>(xr * dedx);
                  vevtlyx += floatTo<vbuf_prec>(yr * dedx);
                  vevtlzx += floatTo<vbuf_prec>(zr * dedx);
                  vevtlyy += floatTo<vbuf_prec>(yr * dedy);
                  vevtlzy += floatTo<vbuf_prec>(zr * dedy);
                  vevtlzz += floatTo<vbuf_prec>(zr * dedz);
               }
            }
         }

         iid = __shfl_sync(ALL_LANES, iid, ilane + 1);
         xi = __shfl_sync(ALL_LANES, xi, ilane + 1);
         yi = __shfl_sync(ALL_LANES, yi, ilane + 1);
         zi = __shfl_sync(ALL_LANES, zi, ilane + 1);
         ijvdw = __shfl_sync(ALL_LANES, ijvdw, ilane + 1);
         imut = __shfl_sync(ALL_LANES, imut, ilane + 1);
         if CONSTEXPR (do_g) {
            fix = __shfl_sync(ALL_LANES, fix, ilane + 1);
            fiy = __shfl_sync(ALL_LANES, fiy, ilane + 1);
            fiz = __shfl_sync(ALL_LANES, fiz, ilane + 1);
         }
      }

      if CONSTEXPR (do_g) {
         atomic_add(fix, gx, i);
         atomic_add(fiy, gy, i);
         atomic_add(fiz, gz, i);
         atomic_add(fkx, gx, k);
         atomic_add(fky, gy, k);
         atomic_add(fkz, gz, k);
      }
   }

   if CONSTEXPR (do_a) {
      atomic_add(nevtl, nev, ithread);
   }
   if CONSTEXPR (do_e) {
      atomic_add(evtl, ev, ithread);
   }
   if CONSTEXPR (do_v) {
      atomic_add(vevtlxx, vevtlyx, vevtlzx, vevtlyy, vevtlzy, vevtlzz, vev, ithread);
   }
}
