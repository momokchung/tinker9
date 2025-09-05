#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/output.hh>
#include <tinker/routines.h>

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const std::string integrator = "integrator verlet\n";

TEST_CASE("TinkerNIST-NODYN", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", integrator + "nodyn\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 1;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool dynExists = fileExistsAndDelete(fn + ".dyn");
   REQUIRE(arcExists == true);
   REQUIRE(dynExists == false);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-NOCOORD", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", integrator + "nocoord\n");
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 1;
   int old = inform::iwrite;
   inform::iwrite = 1;
   VerletIntegrator vvi(ThermostatEnum::NONE, BarostatEnum::NONE);
   for (int istep = 1; istep <= nsteps; ++istep) {
      vvi.dynamic(istep, dt_ps);

      // mdstat
      bool save = (istep % inform::iwrite == 0);
      if (save || (istep % BOUNDS_EVERY_X_STEPS) == 0)
         bounds();
      if (save) {
         T_prec temp;
         kinetic(temp);
         mdsaveAsync(istep, dt_ps);
      }
      mdrest(istep);
   }
   mdsaveSynchronize();
   inform::iwrite = old;

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool dynExists = fileExistsAndDelete(fn + ".dyn");
   REQUIRE(arcExists == false);
   REQUIRE(dynExists == true);

   finish();
   testEnd();
}