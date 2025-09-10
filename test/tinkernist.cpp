#include "md/integrator.h"
#include "md/misc.h"
#include "md/pq.h"
#include <tinker/detail/inform.hh>
#include <tinker/detail/output.hh>
#include <tinker/routines.h>

#include "test.h"
#include "testrt.h"

using namespace tinker;

static const std::string nodyn = "nodyn\n";
static const std::string nocoord = "nocoord\n";

TEST_CASE("TinkerNIST-NODYN", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", nodyn);
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
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", nocoord);
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

TEST_CASE("TinkerNIST-DCD", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string dcdkey = "save-ustatic\nsave-induced\nsave-velocity\ndcd-archive\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", dcdkey);
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
   const int nsteps = 2;
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

   bool dcdExists = fileExistsAndDelete(fn + ".dcd");
   bool dcddExists = fileExistsAndDelete(fn + ".dcdd");
   bool dcduExists = fileExistsAndDelete(fn + ".dcdu");
   bool dcdvExists = fileExistsAndDelete(fn + ".dcdv");
   REQUIRE(dcdExists == true);
   REQUIRE(dcddExists == true);
   REQUIRE(dcduExists == true);
   REQUIRE(dcdvExists == true);

   finish();
   testEnd();
}

TEST_CASE("TinkerNIST-SAVE", "[ff][tinkerNIST]")
{
   int mask = calc::xyz | calc::vel | calc::mass | calc::energy | calc::grad | calc::md | calc::virial;

   TestFile fx1(TINKER9_DIRSTR "/test/file/tinkernist/water30.xyz");
   TestFile fd1(TINKER9_DIRSTR "/test/file/tinkernist/water30.dyn");
   static const std::string savekey = "save-ustatic\nsave-induced\nsave-velocity\n";
   TestFile fk1(TINKER9_DIRSTR "/test/file/tinkernist/water30.key", "", savekey + nodyn);
   TestFile fp1(TINKER9_DIRSTR "/test/file/commit_6fe8e913/amoeba09.prm");
   const char* xn = "water30.xyz";
   static const std::string fn = "water30";

   auto arc_ref  = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.arc");
   auto vel_ref  = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.vel");
   auto ustc_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.ustc");
   auto uind_ref = readAmoebaCoordinateFile(TINKER9_DIRSTR "/test/ref/tinkernist.uind");
   const double eps_arc  = 0.0001;
   const double eps_vel  = 0.0001;
   const double eps_ustc = 0.0001;
   const double eps_uind = 0.0001;

   const char* argv[] = {"dummy", xn};
   int argc = 2;
   testBeginWithArgs(argc, argv);
   testMdInit(0.0, 0.0);

   rc_flag = mask;
   initialize();

   // NVE
   const double dt_ps = 0.001;
   const int nsteps = 2;
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

   auto arc_tst  = readAmoebaCoordinateFile(fn + ".arc");
   auto vel_tst  = readAmoebaCoordinateFile(fn + ".vel");
   auto ustc_tst = readAmoebaCoordinateFile(fn + ".ustc");
   auto uind_tst = readAmoebaCoordinateFile(fn + ".uind");

   for (int istep = 0; istep < nsteps; ++istep) {
      /// arc
      auto compare_arc = [&](int i) {
         COMPARE_REALS(arc_tst[istep][i].x, arc_ref[istep][i].x, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].y, arc_ref[istep][i].y, eps_arc);
         COMPARE_REALS(arc_tst[istep][i].z, arc_ref[istep][i].z, eps_arc);
      };
      /// vel
      auto compare_vel = [&](int i) {
         COMPARE_REALS(vel_tst[istep][i].x, vel_ref[istep][i].x, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].y, vel_ref[istep][i].y, eps_vel);
         COMPARE_REALS(vel_tst[istep][i].z, vel_ref[istep][i].z, eps_vel);
      };
      /// ustc
      auto compare_ustc = [&](int i) {
         COMPARE_REALS(ustc_tst[istep][i].x, ustc_ref[istep][i].x, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].y, ustc_ref[istep][i].y, eps_ustc);
         COMPARE_REALS(ustc_tst[istep][i].z, ustc_ref[istep][i].z, eps_ustc);
      };
      /// uind
      auto compare_uind = [&](int i) {
         COMPARE_REALS(uind_tst[istep][i].x, uind_ref[istep][i].x, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].y, uind_ref[istep][i].y, eps_uind);
         COMPARE_REALS(uind_tst[istep][i].z, uind_ref[istep][i].z, eps_uind);
      };
      // test the first j and last j atoms
      int j = 2;
      for (int i = 0; i < j; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_ustc(i);
         compare_uind(i);
      }
      for (int i = n-j; i < n; ++i) {
         compare_arc(i);
         compare_vel(i);
         compare_ustc(i);
         compare_uind(i);
      }
   }

   bool arcExists = fileExistsAndDelete(fn + ".arc");
   bool velExists = fileExistsAndDelete(fn + ".vel");
   bool ustcExists = fileExistsAndDelete(fn + ".ustc");
   bool uindExists = fileExistsAndDelete(fn + ".uind");
   REQUIRE(arcExists == true);
   REQUIRE(velExists == true);
   REQUIRE(ustcExists == true);
   REQUIRE(uindExists == true);

   finish();
   testEnd();
}
