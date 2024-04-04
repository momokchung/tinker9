#include "ff/modamoeba.h"
#include "ff/evdw.h"
#include "ff/solv/solute.h"

#include "test.h"
#include "testrt.h"

using namespace tinker;

TEST_CASE("ESolv-1-Implicit", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y-implicit.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y-implicit.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.1.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(countReduce(nes), ref_c);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-2-NoNeck", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y-noneck.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y-noneck.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.3.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-3-NoTanh", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y-notanh.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y-notanh.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.4.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-4-NoNeckNoTanh", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y-nonecknotanh.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y-nonecknotanh.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.5.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-5-Total", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-6-Total-Neigh", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/1l2y.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/1l2y-neigh.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "1l2y.xyz";
   const char* kn = "1l2y-neigh.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.2.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-7-Implicit", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/3cln.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/3cln.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "3cln.xyz";
   const char* kn = "3cln.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.6.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   double eng;
   int cnt;
   r.getEnergyCountByName("Van der Waals", eng, cnt);
   COMPARE_COUNT(nev, cnt);
   COMPARE_ENERGY(ev, eng, eps_e);
   r.getEnergyCountByName("Atomic Multipoles", eng, cnt);
   COMPARE_COUNT(nem, cnt);
   COMPARE_ENERGY(em, eng, eps_e);
   r.getEnergyCountByName("Polarization", eng, cnt);
   COMPARE_COUNT(nep, cnt);
   COMPARE_ENERGY(ep, eng, eps_e);
   r.getEnergyCountByName("Implicit Solvation", eng, cnt);
   COMPARE_COUNT(nes, cnt);
   COMPARE_ENERGY(es, eng, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-8-Implicit", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/3cln.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/3cln.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "3cln.xyz";
   const char* kn = "3cln.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.6.txt");
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::energy | calc::grad;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}

TEST_CASE("ESolv-9-Implicit", "[ff][amoeba][esolv]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/water18.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/water18.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");

   const char* xn = "water18.xyz";
   const char* kn = "water18.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   const double eps_e = testGetEps(0.0001, 0.0001);
   const double eps_g = testGetEps(0.001, 0.0001);

   TestReference r(TINKER9_DIRSTR "/test/ref/esolv.7.txt");
   auto ref_c = r.getCount();
   auto ref_e = r.getEnergy();
   auto ref_g = r.getGradient();

   rc_flag = calc::xyz | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   energy(calc::v0);
   COMPARE_REALS(esum, ref_e, eps_e);

   energy(calc::v3);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_INTS(countReduce(nes), ref_c);

   energy(calc::v4);
   COMPARE_REALS(esum, ref_e, eps_e);
   COMPARE_GRADIENT(ref_g, eps_g);

   energy(calc::v5);
   COMPARE_GRADIENT(ref_g, eps_g);

   finish();
   testEnd();
}
