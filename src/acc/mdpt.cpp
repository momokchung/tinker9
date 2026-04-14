#include "ff/energy.h"
#include "ff/molecule.h"
#include "md/misc.h"
#include "md/pq.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/units.hh>

namespace tinker {
void kineticEnergy_acc(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n,
   const double* mass, const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   const energy_prec ekcal_inv = 1.0 / units::ekcal;
   energy_prec exx = 0, eyy = 0, ezz = 0, exy = 0, eyz = 0, ezx = 0;
   #pragma acc parallel loop independent async\
               copy(exx,eyy,ezz,exy,eyz,ezx)\
               reduction(+:exx,eyy,ezz,exy,eyz,ezx)\
               deviceptr(mass,vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      energy_prec term = 0.5 * mass[i] * ekcal_inv;
      exx += term * vx[i] * vx[i];
      eyy += term * vy[i] * vy[i];
      ezz += term * vz[i] * vz[i];
      exy += term * vx[i] * vy[i];
      eyz += term * vy[i] * vz[i];
      ezx += term * vz[i] * vx[i];
   }
   #pragma acc wait

   ekin_out[0][0] = exx;
   ekin_out[0][1] = exy;
   ekin_out[0][2] = ezx;
   ekin_out[1][0] = exy;
   ekin_out[1][1] = eyy;
   ekin_out[1][2] = eyz;
   ekin_out[2][0] = ezx;
   ekin_out[2][1] = eyz;
   ekin_out[2][2] = ezz;
   eksum_out = exx + eyy + ezz;
}

//====================================================================//

void scaleBarostatAtomMove_acc(double scalex, double scaley, double scalez)
{
   #pragma acc parallel loop independent async deviceptr(xpos,ypos,zpos)
   for (int i = 0; i < n; ++i) {
      xpos[i] *= scalex;
      ypos[i] *= scaley;
      zpos[i] *= scalez;
   }
}

void scaleBarostatAtomMoveAniso_acc(const double ascale[3][3])
{
   #pragma acc parallel loop independent async\
               deviceptr(xpos,ypos,zpos)\
               firstprivate(ascale[0:3][0:3])
   for (int i = 0; i < n; ++i) {
      pos_prec xk = xpos[i];
      pos_prec yk = ypos[i];
      pos_prec zk = zpos[i];
      xpos[i] = xk * ascale[0][0] + yk * ascale[0][1] + zk * ascale[0][2];
      ypos[i] = xk * ascale[1][0] + yk * ascale[1][1] + zk * ascale[1][2];
      zpos[i] = xk * ascale[2][0] + yk * ascale[2][1] + zk * ascale[2][2];
   }
}

void scaleVelocity_acc(double scalex, double scaley, double scalez)
{
   #pragma acc parallel loop independent async deviceptr(vx,vy,vz)
   for (int i = 0; i < n; ++i) {
      vx[i] *= scalex;
      vy[i] *= scaley;
      vz[i] *= scalez;
   }
}

void scaleVelocityAniso_acc(const double ascale[3][3])
{
   #pragma acc parallel loop independent async\
               deviceptr(vx,vy,vz)\
               firstprivate(ascale[0:3][0:3])
   for (int i = 0; i < n; ++i) {
      vel_prec xk = vx[i];
      vel_prec yk = vy[i];
      vel_prec zk = vz[i];
      vx[i] = xk * ascale[0][0] + yk * ascale[0][1] + zk * ascale[0][2];
      vy[i] = xk * ascale[1][0] + yk * ascale[1][1] + zk * ascale[1][2];
      vz[i] = xk * ascale[2][0] + yk * ascale[2][1] + zk * ascale[2][2];
   }
}

void monteCarloMolMove_acc(double scale)
{
   int nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;
   pos_prec pos_scale = scale - 1;
   #pragma acc parallel loop independent async\
               deviceptr(imol,kmol,mass,molmass,xpos,ypos,zpos)
   for (int i = 0; i < nmol; ++i) {
      pos_prec xcm = 0, ycm = 0, zcm = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         auto weigh = mass[k];
         xcm += xpos[k] * weigh;
         ycm += ypos[k] * weigh;
         zcm += zpos[k] * weigh;
      }
      pos_prec term = pos_scale / molmass[i];
      pos_prec xmove, ymove, zmove;
      xmove = term * xcm;
      ymove = term * ycm;
      zmove = term * zcm;
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xmove;
         ypos[k] += ymove;
         zpos[k] += zmove;
      }
   }
}

void monteCarloMolMoveAniso_acc(const double ascale[3][3])
{
   int nmol = molecule.nmol;
   const auto* imol = molecule.imol;
   const auto* kmol = molecule.kmol;
   const auto* molmass = molecule.molmass;
   pos_prec a00 = ascale[0][0] - 1;
   pos_prec a01 = ascale[0][1];
   pos_prec a02 = ascale[0][2];
   pos_prec a10 = ascale[1][0];
   pos_prec a11 = ascale[1][1] - 1;
   pos_prec a12 = ascale[1][2];
   pos_prec a20 = ascale[2][0];
   pos_prec a21 = ascale[2][1];
   pos_prec a22 = ascale[2][2] - 1;
   #pragma acc parallel loop independent async\
               deviceptr(imol,kmol,mass,molmass,xpos,ypos,zpos)
   for (int i = 0; i < nmol; ++i) {
      pos_prec xcm = 0, ycm = 0, zcm = 0;
      int start = imol[i][0];
      int stop = imol[i][1];
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         auto weigh = mass[k];
         xcm += xpos[k] * weigh;
         ycm += ypos[k] * weigh;
         zcm += zpos[k] * weigh;
      }
      pos_prec inv_mass = 1 / molmass[i];
      xcm *= inv_mass;
      ycm *= inv_mass;
      zcm *= inv_mass;
      pos_prec xmove = xcm * a00 + ycm * a01 + zcm * a02;
      pos_prec ymove = xcm * a10 + ycm * a11 + zcm * a12;
      pos_prec zmove = xcm * a20 + ycm * a21 + zcm * a22;
      #pragma acc loop seq
      for (int j = start; j < stop; ++j) {
         int k = kmol[j];
         xpos[k] += xmove;
         ypos[k] += ymove;
         zpos[k] += zmove;
      }
   }
}
}
