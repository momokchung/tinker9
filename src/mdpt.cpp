#include "ff/energy.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "math/random.h"
#include "md/misc.h"
#include "md/pq.h"
#include "tool/externfunc.h"
#include "tool/iofortstr.h"
#include <tinker/detail/bath.hh>
#include <tinker/detail/bound.hh>
#include <tinker/detail/mdstuf.hh>
#include <tinker/detail/units.hh>

#include <cmath>

namespace tinker {
TINKER_FVOID2(acc1, cu1, kineticEnergy, energy_prec&, energy_prec (&)[3][3], int n, const double*,
   const vel_prec*, const vel_prec*, const vel_prec*);
void kineticEnergy(energy_prec& eksum_out, energy_prec (&ekin_out)[3][3], int n, const double* mass,
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   TINKER_FCALL2(acc1, cu1, kineticEnergy, eksum_out, ekin_out, n, mass, vx, vy, vz);
}

void kineticExplicit(T_prec& temp_out, energy_prec& eksum_out, energy_prec (&ekin_out)[3][3],
   const vel_prec* vx, const vel_prec* vy, const vel_prec* vz)
{
   kineticEnergy(eksum_out, ekin_out, n, mass, vx, vy, vz);
   temp_out = 2 * eksum_out / (mdstuf::nfree * units::gasconst);
}

void kinetic(T_prec& temp)
{
   kineticExplicit(temp, eksum, ekin, vx, vy, vz);
}
}

namespace tinker {
void bussiThermostat(time_prec dt_prec, T_prec temp_prec)
{
   double dt = dt_prec;
   double temp = temp_prec;

   double tautemp = bath::tautemp;
   double kelvin = bath::kelvin;
   int nfree = mdstuf::nfree;
   // double& eta = bath::eta;

   if (temp == 0)
      temp = 0.1;

   double c = std::exp(-dt / tautemp);
   double d = (1 - c) * (kelvin / temp) / nfree;
   double r = normal<double>();
   double s = chiSquared<double>(nfree - 1);
   double scale = c + (s + r * r) * d + 2 * r * std::sqrt(c * d);
   scale = std::sqrt(scale);
   if (r + std::sqrt(c / d) < 0)
      scale = -scale;
   // eta *= scale;

   vel_prec sc = scale;
   mdVelScale(sc, n, vx, vy, vz);
   for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
         ekin[i][j] *= scale * scale;
      }
   }
}

TINKER_FVOID2(acc1, cu1, monteCarloMolMove, double);
TINKER_FVOID2(acc1, cu1, monteCarloMolMoveAniso, const double (*)[3]);
void monteCarloBarostat(energy_prec epot, T_prec temp, bool semiiso, bool aniso)
{
   if (not bound::use_bounds)
      return;
   if (bath::isothermal)
      temp = bath::kelvin;

   FstrView volscale = bath::volscale;
   double third = 1.0 / 3.0;
   double volmove = bath::volmove;
   double kt = units::gasconst * temp;
   if (bath::isothermal)
      kt = units::gasconst * bath::kelvin;
   bool isotropic = true;
   if ((aniso || semiiso) and random<double>() > bath::isoprob)
      isotropic = false;

   bool semixy = true;
   if (semiiso and not isotropic and random<double>() > 0.5)
      semixy = false;

   // save the system state prior to trial box size change
   Box boxold;
   boxGetCurrent(boxold);
   double volold = boxVolume();
   double volnew = 0;
   double eold = epot;
   darray::copy(g::q0, n, x_pmonte, xpos);
   darray::copy(g::q0, n, y_pmonte, ypos);
   darray::copy(g::q0, n, z_pmonte, zpos);
   double step = volmove * (2 * random<double>() - 1);

   if (isotropic) {
      volnew = volold + step;
      double scale = std::pow(volnew / volold, third);

      lvec1 *= scale;
      lvec2 *= scale;
      lvec3 *= scale;
      boxSetCurrentRecip();

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMove, scale);
      }

      copyPosToXyz();

   } else if (semiiso) {
      double ascale[3][3] = {};
      ascale[0][0] = 1;
      ascale[1][1] = 1;
      ascale[2][2] = 1;

      volnew = volold + step;

      if (semixy) {
         double scale = std::sqrt(volnew / volold);
         lvec1 *= scale;
         lvec2 *= scale;
         ascale[0][0] = scale;
         ascale[1][1] = scale;
      } else {
         double scale = volnew / volold;
         lvec3 *= scale;
         ascale[2][2] = scale;
      }
      boxSetCurrentRecip();

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMoveAniso, ascale);
      }

      copyPosToXyz();

   } else if (aniso) {
      double rnd6 = 6 * random<double>();
      double voltarget = volold + step;
      bool offdiag = false;

      double ascale[3][3] = {};
      ascale[0][0] = 1;
      ascale[1][1] = 1;
      ascale[2][2] = 1;

      double scale = voltarget / volold;

      if (box_shape == BoxShape::MONO || box_shape == BoxShape::TRI) {
         if (rnd6 < 1) {
            ascale[0][0] = scale;
         } else if (rnd6 < 2) {
            ascale[1][1] = scale;
         } else if (rnd6 < 3) {
            ascale[2][2] = scale;
         } else if (rnd6 < 4) {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[0][1] = scale - 1;
            ascale[1][0] = scale - 1;
         } else if (rnd6 < 5) {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[0][2] = scale - 1;
            ascale[2][0] = scale - 1;
         } else {
            offdiag = true;
            scale = std::pow(1 + step / volold, third);
            ascale[1][2] = scale - 1;
            ascale[2][1] = scale - 1;
         }

         double h0[3][3] = {
            {lvec1.x, lvec1.y, lvec1.z},
            {lvec2.x, lvec2.y, lvec2.z},
            {lvec3.x, lvec3.y, lvec3.z},
         };
         double h1[3][3] = {};

         for (int i = 0; i < 3; ++i) {
            for (int j = 0; j < 3; ++j) {
               for (int k = 0; k < 3; ++k) {
                  h1[j][i] += ascale[j][k] * h0[i][k];
               }
            }
         }

         lvec1.x = h1[0][0];
         lvec1.y = h1[0][1];
         lvec1.z = h1[0][2];
         lvec2.x = h1[1][0];
         lvec2.y = h1[1][1];
         lvec2.z = h1[1][2];
         lvec3.x = h1[2][0];
         lvec3.y = h1[2][1];
         lvec3.z = h1[2][2];
         boxSetCurrentRecip();

         if (offdiag) {
            double volmid = boxVolume();
            double scale2 = std::pow(voltarget / volmid, third);

            lvec1 *= scale2;
            lvec2 *= scale2;
            lvec3 *= scale2;
            boxSetCurrentRecip();

            for (int i = 0; i < 3; ++i) {
               for (int j = 0; j < 3; ++j) {
                  ascale[i][j] *= scale2;
               }
            }
         }

         volnew = boxVolume();

      } else {
         if (rnd6 < 2) {
            ascale[0][0] = scale;
         } else if (rnd6 < 4) {
            ascale[1][1] = scale;
         } else {
            ascale[2][2] = scale;
         }

         lvec1 *= ascale[0][0];
         lvec2 *= ascale[1][1];
         lvec3 *= ascale[2][2];
         boxSetCurrentRecip();

         volnew = boxVolume();

      }

      if (volscale == "MOLECULAR") {
         TINKER_FCALL2(acc1, cu1, monteCarloMolMoveAniso, ascale);
      }

      copyPosToXyz();

   }

   // get the potential energy and PV work changes for trial move
   nblistRefresh();
   energy(calc::energy);
   energy_prec enew;
   copyEnergy(calc::energy, &enew);
   double dpot = enew - eold;
   double dpv = bath::atmsph * (volnew - volold) / units::prescon;

   // estimate the kinetic energy change as an ideal gas term
   double dkin = 0;
   if (volscale == "MOLECULAR") {
      dkin = molecule.nmol * kt * std::log(volold / volnew);
   }

   // acceptance ratio from Epot change, Ekin change and PV work
   double term = -(dpot + dpv + dkin) / kt;
   double expterm = std::exp(term);

   // reject the step, and restore values prior to trial change
   double exp_rdm = random<double>();
   if (exp_rdm > expterm) {
      esum = eold;
      boxSetCurrent(boxold);
      darray::copy(g::q0, n, xpos, x_pmonte);
      darray::copy(g::q0, n, ypos, y_pmonte);
      darray::copy(g::q0, n, zpos, z_pmonte);
      copyPosToXyz();
      nblistRefresh();
   }
}

TINKER_FVOID2(acc1, cu0, berendsenBarostat, time_prec, bool);
void berendsenBarostat(time_prec dt, bool aniso)
{
   TINKER_FCALL2(acc1, cu0, berendsenBarostat, dt, aniso);
}
}
