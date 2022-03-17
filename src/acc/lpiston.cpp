#include "lpiston.h"
#include "add.h"
#include "md.h"
#include "rattle.h"
#include <tinker/detail/units.hh>

namespace tinker {
#pragma acc routine seq
static inline vel_prec
#if TINKER_DETERMINISTIC_FORCE
cvt_grad(fixed val)
{
   return to_flt_acc<vel_prec>(val);
}
#else
cvt_grad(grad_prec val)
{
   return val;
}
#endif

void velAvbfIso_acc(int nrespa, vel_prec a, vel_prec b, vel_prec* vx, vel_prec* vy, vel_prec* vz,
   const grad_prec* gx1, const grad_prec* gy1, const grad_prec* gz1, const grad_prec* gx2,
   const grad_prec* gy2, const grad_prec* gz2)
{
   auto ekcal = units::ekcal;
   if (nrespa == 1)
      goto label_nrespa1;
   else
      goto label_nrespa2;

label_nrespa1:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = cvt_grad(gx1[i]);
      auto gry = cvt_grad(gy1[i]);
      auto grz = cvt_grad(gz1[i]);
      vx[i] = a * v0x + coef * b * grx;
      vy[i] = a * v0y + coef * b * gry;
      vz[i] = a * v0z + coef * b * grz;
   }
   return;

label_nrespa2:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1,gx2,gy2,gz2)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = cvt_grad(gx1[i]) / nrespa + cvt_grad(gx2[i]);
      auto gry = cvt_grad(gy1[i]) / nrespa + cvt_grad(gy2[i]);
      auto grz = cvt_grad(gz1[i]) / nrespa + cvt_grad(gz2[i]);
      vx[i] = a * v0x + coef * b * grx;
      vy[i] = a * v0y + coef * b * gry;
      vz[i] = a * v0z + coef * b * grz;
   }
   return;
}

void velAvbfAni_acc(int nrespa, vel_prec a[3][3], vel_prec b[3][3], vel_prec* vx, vel_prec* vy,
   vel_prec* vz, const grad_prec* gx1, const grad_prec* gy1, const grad_prec* gz1,
   const grad_prec* gx2, const grad_prec* gy2, const grad_prec* gz2)
{
   auto ekcal = units::ekcal;
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];

   if (nrespa == 1)
      goto label_nrespa1;
   else
      goto label_nrespa2;

label_nrespa1:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = cvt_grad(gx1[i]);
      auto gry = cvt_grad(gy1[i]);
      auto grz = cvt_grad(gz1[i]);
      // clang-format off
      vx[i] = a00*v0x + a01*v0y + a02*v0z + coef*(b00*grx+b01*gry+b02*grz);
      vy[i] = a10*v0x + a11*v0y + a12*v0z + coef*(b10*grx+b11*gry+b12*grz);
      vz[i] = a20*v0x + a21*v0y + a22*v0z + coef*(b20*grx+b21*gry+b22*grz);
      // clang-foramt on
   }
   return;

label_nrespa2:
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,gx1,gy1,gz1,gx2,gy2,gz2)
   for (int i = 0; i < n; ++i) {
      auto coef = -ekcal * massinv[i];
      auto v0x = vx[i];
      auto v0y = vy[i];
      auto v0z = vz[i];
      auto grx = cvt_grad(gx1[i]) / nrespa + cvt_grad(gx2[i]);
      auto gry = cvt_grad(gy1[i]) / nrespa + cvt_grad(gy2[i]);
      auto grz = cvt_grad(gz1[i]) / nrespa + cvt_grad(gz2[i]);
      // clang-format off
      vx[i] = a00*v0x + a01*v0y + a02*v0z + coef*(b00*grx+b01*gry+b02*grz);
      vy[i] = a10*v0x + a11*v0y + a12*v0z + coef*(b10*grx+b11*gry+b12*grz);
      vz[i] = a20*v0x + a21*v0y + a22*v0z + coef*(b20*grx+b21*gry+b22*grz);
      // clang-foramt on
   }
   return;
}
}


namespace tinker {
void lp_matvec_acc(int len, char transpose, double mat[3][3],

   pos_prec* ax, pos_prec* ay, pos_prec* az)
{
   double m00 = mat[0][0], m01 = mat[0][1], m02 = mat[0][2];
   double m10 = mat[1][0], m11 = mat[1][1], m12 = mat[1][2];
   double m20 = mat[2][0], m21 = mat[2][1], m22 = mat[2][2];
   if (transpose == 't' or transpose == 'T' or transpose == 'y' or transpose == 'Y') {
      m01 = mat[1][0], m02 = mat[2][0];
      m10 = mat[0][1], m12 = mat[2][1];
      m20 = mat[0][2], m21 = mat[1][2];
   }
   #pragma acc parallel loop independent async\
               deviceptr(ax,ay,az)
   for (int i = 0; i < len; ++i) {
      auto x = ax[i], y = ay[i], z = az[i];
      ax[i] = m00 * x + m01 * y + m02 * z;
      ay[i] = m10 * x + m11 * y + m12 * z;
      az[i] = m20 * x + m21 * y + m22 * z;
   }
}

void propagate_pos_raxbv_acc(

   pos_prec* r1, pos_prec* r2, pos_prec* r3,

   const int* molec,

   double a, pos_prec* x1, pos_prec* x2, pos_prec* x3,

   double b, pos_prec* y1, pos_prec* y2, pos_prec* y3)
{
   #pragma acc parallel loop independent async\
               deviceptr(molec,r1,r2,r3,x1,x2,x3,y1,y2,y3)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      r1[i] = r1[i] + a * x1[im] + b * y1[i];
      r2[i] = r2[i] + a * x2[im] + b * y2[i];
      r3[i] = r3[i] + a * x3[im] + b * y3[i];
   }
}

void propagate_pos_raxbv_aniso_acc(

   pos_prec* r1, pos_prec* r2, pos_prec* r3,

   const int* molec,

   double a[3][3], pos_prec* x1, pos_prec* x2, pos_prec* x3,

   double b[3][3], pos_prec* y1, pos_prec* y2, pos_prec* y3)
{
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];
   #pragma acc parallel loop independent async\
               deviceptr(molec,r1,r2,r3,x1,x2,x3,y1,y2,y3)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      auto x1im = x1[im], x2im = x2[im], x3im = x3[im];
      auto y1i = y1[i], y2i = y2[i], y3i = y3[i];
      // clang-format off
      r1[i] += (a00*x1im+a01*x2im+a02*x3im) + (b00*y1i+b01*y2i+b02*y3i);
      r2[i] += (a10*x1im+a11*x2im+a12*x3im) + (b10*y1i+b11*y2i+b12*y3i);
      r3[i] += (a20*x1im+a21*x2im+a22*x3im) + (b20*y1i+b21*y2i+b22*y3i);
      // clang-format on
   }
}

void propagate_pos_axbv_aniso_acc(double a[3][3], double b[3][3])
{
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];
   #pragma acc parallel loop independent async\
               deviceptr(xpos,ypos,zpos,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      auto xi = xpos[i], yi = ypos[i], zi = zpos[i];
      auto vxi = vx[i], vyi = vy[i], vzi = vz[i];
      // clang-format off
      xpos[i] = (a00*xi+a01*yi+a02*zi) + (b00*vxi+b01*vyi+b02*vzi);
      ypos[i] = (a10*xi+a11*yi+a12*zi) + (b10*vxi+b11*vyi+b12*vzi);
      zpos[i] = (a20*xi+a21*yi+a22*zi) + (b20*vxi+b21*vyi+b22*vzi);
      // clang-format on
   }
}

void lp_propagate_mol_vel_acc(vel_prec scal)
{
   auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz,ratcom_vx,ratcom_vy,ratcom_vz,molec)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      vx[i] = vx[i] + scal * ratcom_vx[im];
      vy[i] = vy[i] + scal * ratcom_vy[im];
      vz[i] = vz[i] + scal * ratcom_vz[im];
   }
}

void lp_propagate_mol_vel_aniso_acc(vel_prec scal[3][3])
{
   auto s00 = scal[0][0], s01 = scal[0][1], s02 = scal[0][2];
   auto s10 = scal[1][0], s11 = scal[1][1], s12 = scal[1][2];
   auto s20 = scal[2][0], s21 = scal[2][1], s22 = scal[2][2];
   auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz,ratcom_vx,ratcom_vy,ratcom_vz,molec)
   for (int i = 0; i < n; ++i) {
      int im = molec[i];
      auto xm = ratcom_vx[im], ym = ratcom_vy[im], zm = ratcom_vz[im];
      vx[i] += s00 * xm + s01 * ym + s02 * zm;
      vy[i] += s10 * xm + s11 * ym + s12 * zm;
      vz[i] += s20 * xm + s21 * ym + s22 * zm;
   }
}

void lp_mol_virial_acc()
{
   int nmol = rattle_dmol.nmol;
   auto* molmass = rattle_dmol.molmass;
   auto* imol = rattle_dmol.imol;
   auto* kmol = rattle_dmol.kmol;

   double mvxx = 0, mvyy = 0, mvzz = 0, mvxy = 0, mvxz = 0, mvyz = 0;
   #pragma acc parallel loop independent async\
               copy(mvxx,mvyy,mvzz,mvxy,mvxz,mvyz)\
               reduction(+:mvxx,mvyy,mvzz,mvxy,mvxz,mvyz)\
               deviceptr(molmass,imol,kmol,mass,xpos,ypos,zpos,gx,gy,gz)
   for (int im = 0; im < nmol; ++im) {
      double vxx = 0, vyy = 0, vzz = 0, vxy = 0, vxz = 0, vyz = 0;
      double igx, igy, igz;             // atomic gradients
      pos_prec irx, iry, irz;           // atomic positions
      double mgx = 0, mgy = 0, mgz = 0; // molecular gradients
      pos_prec rx = 0, ry = 0, rz = 0;  // molecular positions
      int start = imol[im][0];
      int end = imol[im][1];
      #pragma acc loop seq
      for (int i = start; i < end; ++i) {
         int k = kmol[i];
#if TINKER_DETERMINISTIC_FORCE
         igx = to_flt_acc<double>(gx[k]);
         igy = to_flt_acc<double>(gy[k]);
         igz = to_flt_acc<double>(gz[k]);
#else
         igx = gx[k];
         igy = gy[k];
         igz = gz[k];
#endif
         irx = xpos[k];
         iry = ypos[k];
         irz = zpos[k];
         vxx -= igx * irx;
         vyy -= igy * iry;
         vzz -= igz * irz;
         vxy -= 0.5 * (igx * iry + igy * irx);
         vxz -= 0.5 * (igx * irz + igz * irx);
         vyz -= 0.5 * (igy * irz + igz * iry);

         mgx += igx;
         mgy += igy;
         mgz += igz;
         auto massk = mass[k];
         rx += massk * irx;
         ry += massk * iry;
         rz += massk * irz;
      }
      auto mmassinv = 1 / molmass[im];
      vxx += mgx * rx * mmassinv;
      vyy += mgy * ry * mmassinv;
      vzz += mgz * rz * mmassinv;
      vxy += 0.5 * (mgx * ry + mgy * rx) * mmassinv;
      vxz += 0.5 * (mgx * rz + mgz * rx) * mmassinv;
      vyz += 0.5 * (mgy * rz + mgz * ry) * mmassinv;
      mvxx += vxx;
      mvyy += vyy;
      mvzz += vzz;
      mvxy += vxy;
      mvxz += vxz;
      mvyz += vyz;
   }
   #pragma acc wait

   lp_vir[0] = mvxx + vir[0];
   lp_vir[1] = mvxy + vir[1];
   lp_vir[2] = mvxz + vir[2];
   lp_vir[3] = mvxy + vir[3];
   lp_vir[4] = mvyy + vir[4];
   lp_vir[5] = mvyz + vir[5];
   lp_vir[6] = mvxz + vir[6];
   lp_vir[7] = mvyz + vir[7];
   lp_vir[8] = mvzz + vir[8];
}

void lp_center_of_mass_acc(const pos_prec* ax, const pos_prec* ay, const pos_prec* az, pos_prec* mx,
   pos_prec* my, pos_prec* mz)
{
   const int nmol = rattle_dmol.nmol;
   const auto* imol = rattle_dmol.imol;
   const auto* kmol = rattle_dmol.kmol;
   const auto* mfrac = ratcom_massfrac;
   #pragma acc parallel loop independent async\
               deviceptr(ax,ay,az,mx,my,mz,mfrac,imol,kmol)
   for (int im = 0; im < nmol; ++im) {
      int start = imol[im][0];
      int end = imol[im][1];
      pos_prec tx = 0, ty = 0, tz = 0;
      #pragma acc loop seq
      for (int i = start; i < end; ++i) {
         int k = kmol[i];
         auto frk = mfrac[k];
         tx += frk * ax[k];
         ty += frk * ay[k];
         tz += frk * az[k];
      }
      mx[im] = tx;
      my[im] = ty;
      mz[im] = tz;
   }
}

void propagate_velocity_06_acc(double vbar, time_prec dt, const grad_prec* grx,
   const grad_prec* gry, const grad_prec* grz, time_prec dt2, const grad_prec* grx2,
   const grad_prec* gry2, const grad_prec* grz2)
{
   const vel_prec ekcal = units::ekcal;
   if (dt != 0 and dt2 != 0 and dt >= dt2) {
      std::swap(dt, dt2);
      std::swap(grx, grx2);
      std::swap(gry, gry2);
      std::swap(grz, grz2);
   }
   // dt < dt2
   const auto t2 = dt / 2;
   const double e1 = std::exp(-vbar * dt);
   const double e2 = std::exp(-vbar * t2) * sinhc(vbar * t2);

   if (dt2 == 0) {
      #pragma acc parallel loop independent async\
              deviceptr(massinv,vx,vy,vz,grx,gry,grz)
      for (int i = 0; i < n; ++i) {
#if TINKER_DETERMINISTIC_FORCE
         auto gx = to_flt_acc<vel_prec>(grx[i]);
         auto gy = to_flt_acc<vel_prec>(gry[i]);
         auto gz = to_flt_acc<vel_prec>(grz[i]);
#else
         auto gx = grx[i];
         auto gy = gry[i];
         auto gz = grz[i];
#endif
         auto vlx = vx[i];
         auto vly = vy[i];
         auto vlz = vz[i];
         auto coef = -ekcal * massinv[i] * dt;
         vx[i] = vlx * e1 + coef * gx * e2;
         vy[i] = vly * e1 + coef * gy * e2;
         vz[i] = vlz * e1 + coef * gz * e2;
      }
   } else {
      const auto drespa = dt2 / dt;
      #pragma acc parallel loop independent async\
              deviceptr(massinv,vx,vy,vz,grx,gry,grz,grx2,gry2,grz2)
      for (int i = 0; i < n; ++i) {
#if TINKER_DETERMINISTIC_FORCE
         auto gx = to_flt_acc<vel_prec>(grx[i]);
         auto gy = to_flt_acc<vel_prec>(gry[i]);
         auto gz = to_flt_acc<vel_prec>(grz[i]);
         auto gx2 = to_flt_acc<vel_prec>(grx2[i]);
         auto gy2 = to_flt_acc<vel_prec>(gry2[i]);
         auto gz2 = to_flt_acc<vel_prec>(grz2[i]);
#else
         auto gx = grx[i];
         auto gy = gry[i];
         auto gz = grz[i];
         auto gx2 = grx2[i];
         auto gy2 = gry2[i];
         auto gz2 = grz2[i];
#endif
         auto vlx = vx[i];
         auto vly = vy[i];
         auto vlz = vz[i];
         auto coef = -ekcal * massinv[i] * dt;
         vx[i] = vlx * e1 + coef * (gx + gx2 * drespa) * e2;
         vy[i] = vly * e1 + coef * (gy + gy2 * drespa) * e2;
         vz[i] = vlz * e1 + coef * (gz + gz2 * drespa) * e2;
      }
   }
}

void pLogVPosMolIso_acc(double s)
{
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
           deviceptr(xpos,ypos,zpos,ratcom_x,ratcom_y,ratcom_z,molec)
   for (int i = 0; i < n; ++i) {
      auto k = molec[i];
      xpos[i] += ratcom_x[k] * s;
      ypos[i] += ratcom_y[k] * s;
      zpos[i] += ratcom_z[k] * s;
   }
}

void pLogVPosMolAniso_acc(double (*scal)[3])
{
   double s00, s01, s02;
   double s10, s11, s12;
   double s20, s21, s22;
   s00 = scal[0][0], s01 = scal[0][1], s02 = scal[0][2];
   s10 = scal[1][0], s11 = scal[1][1], s12 = scal[1][2];
   s20 = scal[2][0], s21 = scal[2][1], s22 = scal[2][2];
   const auto* molec = rattle_dmol.molecule;
   #pragma acc parallel loop independent async\
           deviceptr(xpos,ypos,zpos,ratcom_x,ratcom_y,ratcom_z,molec)
   for (int i = 0; i < n; ++i) {
      auto k = molec[i];
      auto xc = ratcom_x[k];
      auto yc = ratcom_y[k];
      auto zc = ratcom_z[k];
      auto xd = s00 * xc + s01 * yc + s02 * zc;
      auto yd = s10 * xc + s11 * yc + s12 * zc;
      auto zd = s20 * xc + s21 * yc + s22 * zc;
      xpos[i] += xd;
      ypos[i] += yd;
      zpos[i] += zd;
   }
}

void pLogVPosAtmAniso_acc(double (*a)[3], double (*b)[3])
{
   auto a00 = a[0][0], a01 = a[0][1], a02 = a[0][2];
   auto a10 = a[1][0], a11 = a[1][1], a12 = a[1][2];
   auto a20 = a[2][0], a21 = a[2][1], a22 = a[2][2];
   auto b00 = b[0][0], b01 = b[0][1], b02 = b[0][2];
   auto b10 = b[1][0], b11 = b[1][1], b12 = b[1][2];
   auto b20 = b[2][0], b21 = b[2][1], b22 = b[2][2];
   #pragma acc parallel loop independent async\
               deviceptr(xpos,ypos,zpos,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      auto o = xpos[i], p = ypos[i], q = zpos[i];
      auto r = vx[i], s = vy[i], t = vz[i];
      xpos[i] = (a00 * o + a01 * p + a02 * q) + (b00 * r + b01 * s + b02 * t);
      ypos[i] = (a10 * o + a11 * p + a12 * q) + (b10 * r + b11 * s + b12 * t);
      zpos[i] = (a20 * o + a21 * p + a22 * q) + (b20 * r + b21 * s + b22 * t);
   }
}

void propagate_vel_avbf_aniso_acc(double sa[3][3], double sb[3][3], const grad_prec* grx,
   const grad_prec* gry, const grad_prec* grz)
{
   vel_prec ekcal = units::ekcal;
   vel_prec a00 = sa[0][0], a01 = sa[0][1], a02 = sa[0][2];
   vel_prec a10 = sa[1][0], a11 = sa[1][1], a12 = sa[1][2];
   vel_prec a20 = sa[2][0], a21 = sa[2][1], a22 = sa[2][2];
   vel_prec b00 = sb[0][0], b01 = sb[0][1], b02 = sb[0][2];
   vel_prec b10 = sb[1][0], b11 = sb[1][1], b12 = sb[1][2];
   vel_prec b20 = sb[2][0], b21 = sb[2][1], b22 = sb[2][2];
   #pragma acc parallel loop independent async\
               deviceptr(massinv,vx,vy,vz,grx,gry,grz)
   for (int i = 0; i < n; ++i) {
      vel_prec coef = -ekcal * massinv[i];
      vel_prec v0x, v0y, v0z;
      vel_prec gx, gy, gz;
      v0x = vx[i];
      v0y = vy[i];
      v0z = vz[i];
#if TINKER_DETERMINISTIC_FORCE
      gx = to_flt_acc<vel_prec>(grx[i]);
      gy = to_flt_acc<vel_prec>(gry[i]);
      gz = to_flt_acc<vel_prec>(grz[i]);
#else
      gx = grx[i];
      gy = gry[i];
      gz = grz[i];
#endif
      // clang-format off
      vx[i] = a00 * v0x + a01 * v0y + a02 * v0z + coef * (b00 * gx + b01 * gy + b02 * gz);
      vy[i] = a10 * v0x + a11 * v0y + a12 * v0z + coef * (b10 * gx + b11 * gy + b12 * gz);
      vz[i] = a20 * v0x + a21 * v0y + a22 * v0z + coef * (b20 * gx + b21 * gy + b22 * gz);
      // clang-foramt on
   }
}
}