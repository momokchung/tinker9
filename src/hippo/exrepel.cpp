#include "ff/hippo/erepel.h"
#include "ff/amoeba/empole.h"
#include "ff/energy.h"
#include "ff/modamoeba.h"
#include "ff/potent.h"
#include "math/zero.h"
#include <tinker/detail/couple.hh>
#include <tinker/detail/mpole.hh>
#include <tinker/detail/xrepel.hh>
#include <tinker/detail/reppot.hh>


namespace tinker {
void exrepelData(RcOp op)
{
   if (not use(Potent::EXREP))
      return;

   auto rc_a = rc_flag & calc::analyz;

   if (op & RcOp::DEALLOC) {
      darray::deallocate(zpxr, dmppxr, crpxr, cpxr, rcpxr);
      nrepexclude = 0;
      darray::deallocate(repexclude, repexclude_scale);
      darray::deallocate(xrepole);

      if (rc_a) {
         bufferDeallocate(rc_flag, nrep);
         bufferDeallocate(rc_flag, er, vir_er, derx, dery, derz);
      }
      nrep = nullptr;
      er = nullptr;
      vir_er = nullptr;
      derx = nullptr;
      dery = nullptr;
      derz = nullptr;
   }

   if (op & RcOp::ALLOC) {
      darray::allocate(n, &zpxr, &dmppxr, &crpxr, &cpxr, &rcpxr);
      darray::allocate(n, &xrepole);

      nrep = nullptr;
      er = eng_buf_vdw;
      vir_er = vir_buf_vdw;
      derx = gx_vdw;
      dery = gy_vdw;
      derz = gz_vdw;
      if (rc_a) {
         bufferAllocate(rc_flag, &nrep);
         bufferAllocate(rc_flag, &er, &vir_er, &derx, &dery, &derz);
      }

      r2scale = reppot::r2scale;
      r3scale = reppot::r3scale;
      r4scale = reppot::r4scale;
      r5scale = reppot::r5scale;
      std::vector<int> exclik;
      std::vector<real> excls;
      // see also attach.f
      const int maxn13 = 3 * sizes::maxval;
      const int maxn14 = 9 * sizes::maxval;
      const int maxn15 = 27 * sizes::maxval;
      for (int i = 0; i < n; ++i) {
         int nn;
         int bask;

         if (r2scale != 1) {
            nn = couple::n12[i];
            for (int j = 0; j < nn; ++j) {
               int k = couple::i12[i][j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(r2scale);
               }
            }
         }

         if (r3scale != 1) {
            nn = couple::n13[i];
            bask = i * maxn13;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i13[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(r3scale);
               }
            }
         }

         if (r4scale != 1) {
            nn = couple::n14[i];
            bask = i * maxn14;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i14[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(r4scale);
               }
            }
         }

         if (r5scale != 1) {
            nn = couple::n15[i];
            bask = i * maxn15;
            for (int j = 0; j < nn; ++j) {
               int k = couple::i15[bask + j];
               k -= 1;
               if (k > i) {
                  exclik.push_back(i);
                  exclik.push_back(k);
                  excls.push_back(r5scale);
               }
            }
         }
      }
      nrepexclude = excls.size();
      darray::allocate(nrepexclude, &repexclude, &repexclude_scale);
      darray::copyin(g::q0, nrepexclude, repexclude, exclik.data());
      darray::copyin(g::q0, nrepexclude, repexclude_scale, excls.data());
      waitFor(g::q0);
   }

   if (op & RcOp::INIT) {
      darray::copyin(g::q0, n, zpxr, xrepel::zpxr);
      darray::copyin(g::q0, n, dmppxr, xrepel::dmppxr);
      darray::copyin(g::q0, n, crpxr, xrepel::crpxr);
      waitFor(g::q0);

      std::vector<double> polebuf(MPL_TOTAL * n);
      for (int i = 0; i < n; ++i) {
         int b1 = MPL_TOTAL * i;
         int b2 = mpole::maxpole * i;
         // Tinker c = 0, dx = 1, dy = 2, dz = 3
         // Tinker qxx = 4, qxy = 5, qxz = 6
         //        qyx    , qyy = 8, qyz = 9
         //        qzx    , qzy    , qzz = 12
         polebuf[b1 + MPL_PME_0] = xrepel::xrepole[b2 + 0];
         polebuf[b1 + MPL_PME_X] = xrepel::xrepole[b2 + 1];
         polebuf[b1 + MPL_PME_Y] = xrepel::xrepole[b2 + 2];
         polebuf[b1 + MPL_PME_Z] = xrepel::xrepole[b2 + 3];
         polebuf[b1 + MPL_PME_XX] = xrepel::xrepole[b2 + 4];
         polebuf[b1 + MPL_PME_XY] = xrepel::xrepole[b2 + 5];
         polebuf[b1 + MPL_PME_XZ] = xrepel::xrepole[b2 + 6];
         polebuf[b1 + MPL_PME_YY] = xrepel::xrepole[b2 + 8];
         polebuf[b1 + MPL_PME_YZ] = xrepel::xrepole[b2 + 9];
         polebuf[b1 + MPL_PME_ZZ] = xrepel::xrepole[b2 + 12];
      }
      darray::copyin(g::q0, n, xrepole, polebuf.data());
      waitFor(g::q0);
   }
}

TINKER_FVOID2(acc0, cu1, chkxrepole);
static void chkxrepole()
{
   TINKER_FCALL2(acc0, cu1, chkxrepole);
}

TINKER_FVOID2(acc0, cu1, solvcoeff);
static void solvcoeff()
{
   TINKER_FCALL2(acc0, cu1, solvcoeff);
}

TINKER_FVOID2(acc0, cu1, rotcoeff);
static void rotcoeff()
{
   TINKER_FCALL2(acc0, cu1, rotcoeff);
}

void xrepoleInit(int vers)
{
   if (vers & calc::grad)
      darray::zero(g::q0, n, trqx, trqy, trqz);
   if (vers & calc::virial)
      darray::zero(g::q0, bufferSize(), vir_trq);

   chkxrepole();
   solvcoeff();
   rotcoeff();
}

TINKER_FVOID2(acc0, cu1, exrepel, int);
void exrepel(int vers)
{
   auto rc_a = rc_flag & calc::analyz;
   auto do_a = vers & calc::analyz;
   auto do_e = vers & calc::energy;
   auto do_v = vers & calc::virial;
   auto do_g = vers & calc::grad;

   zeroOnHost(energy_er, virial_er);
   size_t bsize = bufferSize();
   if (rc_a) {
      if (do_a)
         darray::zero(g::q0, bsize, nrep);
      if (do_e)
         darray::zero(g::q0, bsize, er);
      if (do_v)
         darray::zero(g::q0, bsize, vir_er);
      if (do_g)
         darray::zero(g::q0, n, derx, dery, derz);
   }

   xrepoleInit(vers);

   TINKER_FCALL2(acc0, cu1, exrepel, vers);

   torque(vers, derx, dery, derz);
   if (do_v) {
      VirialBuffer u2 = vir_trq;
      virial_prec v2[9];
      virialReduce(v2, u2);
      for (int iv = 0; iv < 9; ++iv) {
         virial_er[iv] += v2[iv];
         virial_vdw[iv] += v2[iv];
      }
   }

   if (rc_a) {
      if (do_e) {
         EnergyBuffer u = er;
         energy_prec e = energyReduce(u);
         energy_er += e;
         energy_vdw += e;
      }
      if (do_v) {
         VirialBuffer u = vir_er;
         virial_prec v[9];
         virialReduce(v, u);
         for (int iv = 0; iv < 9; ++iv) {
            virial_er[iv] += v[iv];
            virial_vdw[iv] += v[iv];
         }
      }
      if (do_g)
         sumGradient(gx_vdw, gy_vdw, gz_vdw, derx, dery, derz);
   }
}
}