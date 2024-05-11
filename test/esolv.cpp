#include "ff/modamoeba.h"
#include "ff/evdw.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/solute.h"
#include <tinker/detail/atomid.hh>
#include <tinker/detail/atoms.hh>
#include <tinker/routines.h>
#include <vector>

#include "test.h"
#include "testrt.h"

using namespace tinker;
// TODO_MOSES add test for AlphaMol and AlphaMol2
// TODO_MOSES add test for with and without hydrogen
// TODO_MOSES add test for None, Sort3D, BRIO, Split, KDTree
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

   // check alfdigit is even
   REQUIRE(alfdigit > 0);
   REQUIRE((alfdigit % 2) == 0);

   // check alfnthd is a power of 2
   REQUIRE(alfnthd > 0);
   REQUIRE((alfnthd & (alfnthd - 1)) == 0);

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

TEST_CASE("ESolv-10-Implicit", "[jacobi]")
{
   double tensor[9] = {
      4.2891378039728206e+03,
     -1.0429631761163696e+03,
      1.9360758481984683e+03,
     -1.0429631761163696e+03,
      4.6495025409919926e+03,
      1.4813885093349336e+03,
      1.9360758481984683e+03,
      1.4813885093349336e+03,
      2.4282642345195291e+03,
   };

   double moment_ref[3] = {
      4.3384479025468295e+02,
      5.3709467424811792e+03,
      5.5621130467484809e+03,
   };

   double vec_ref[9] = {
     -4.9538057532411855e-01, -3.9451567296866469e-01,  7.7392213392151210e-01,
      2.7529889437693889e-01,  7.7369673161948782e-01,  5.7061711000999349e-01,
     -8.2389841870608072e-01,  4.9573254004896988e-01, -2.7466460345340465e-01,
   };

   int dim = 3;
   double moment[3],vec[9];
   tinker_f_jacobi(&dim,tensor,moment,vec);

   double eps = 1e-10;
   for (int i = 0; i < 3; i++) {
      COMPARE_REALS(moment[i], moment_ref[i], eps);
   }
   for (int i = 0; i < 9; i++) {
      COMPARE_REALS(vec[i], vec_ref[i], eps);
   }
}

TEST_CASE("ESolv-11-Implicit", "[inertia]")
{
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/alatet.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/alatet.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");
   const char* xn = "alatet.xyz";
   const char* kn = "alatet.key";
   const char* argv[] = {"dummy", xn, "-k", kn};
   int argc = 4;

   rc_flag = calc::xyz | calc::mass | calc::vmask;
   testBeginWithArgs(argc, argv);
   initialize();

   // copy data
   int n = atoms::n;
   std::vector<double> x,y,z,mass;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }

   inertia(n,&mass[0],&x[0],&y[0],&z[0]);

   std::vector<double> coord = {
      7.0945791946319083e+00,  8.1539161896783829e-01,  3.3132534953615056e-01,
      5.9348147904818971e+00,  2.2866655010512649e-02, -2.4050061920959340e-01,
      6.1121807552249630e+00, -7.6937748683596041e-01, -1.1602856228097329e+00,
      7.5592424975650712e+00,  1.4229462926039691e+00, -4.7551071240278964e-01,
      7.8654132493988884e+00,  1.1410724069446715e-01,  7.1872266001239693e-01,
      6.7994741544724322e+00,  1.4993468584796730e+00,  1.1564144123983264e+00,
      4.7171136397612825e+00,  2.1065102530853450e-01,  3.1532600147753070e-01,
      3.4939517211911419e+00, -4.5279398997362119e-01, -1.2020460229799446e-01,
      2.2903976206378638e+00,  4.0066093580603074e-01,  2.8909522412493466e-01,
      2.4169160308776951e+00,  1.3325685867116774e+00,  1.0841053880316021e+00,
      4.5613540323639894e+00,  9.2335614974789293e-01,  1.0431120606400577e+00,
      3.4972219920497167e+00, -5.2093722351283323e-01, -1.2367126494097618e+00,
      3.4130068682555508e+00, -1.8663331751080450e+00,  5.0571026208432912e-01,
      2.4977977810884524e+00, -2.4115815523449453e+00,  1.8490308157819568e-01,
      3.4027933405769626e+00, -1.8233142842582371e+00,  1.6181366764153677e+00,
      4.2864679858259906e+00, -2.4853401542879108e+00,  1.9789149909510728e-01,
      1.0994848053013604e+00,  6.4311462069078756e-02, -2.5504641663982131e-01,
     -1.7587845153065776e-01,  7.0395439434361229e-01,  3.9857443491009587e-02,
     -1.3025118217835439e+00, -3.0576070257363158e-01, -1.9627341063022771e-01,
     -1.0835155994062620e+00, -1.3748697397083078e+00, -7.6607786566322100e-01,
      1.0164225211501969e+00, -7.6397749396762826e-01, -8.6117988764865272e-01,
     -1.9821894798979503e-01,  9.8630974042684605e-01,  1.1217333364108848e+00,
     -3.4839391843161227e-01,  1.9590114141714101e+00, -8.4903201734138745e-01,
     -1.3042297804158480e+00,  2.4888933882210096e+00, -6.4074595285517899e-01,
     -3.4154152156318662e-01,  1.6993147126443904e+00, -1.9317264619965471e+00,
      4.7400305798407411e-01,  2.6888948899560754e+00, -6.7124872502224531e-01,
     -2.5341452307681087e+00,  5.0882251570143387e-02,  2.3385403501078328e-01,
     -3.7509734597194480e+00, -7.3235174088750887e-01,  6.2627081828776743e-02,
     -4.9535118287720827e+00,  2.1487114359406817e-01,  3.1066683912524717e-02,
     -4.8502298005869253e+00,  1.3740409342483373e+00,  4.3110598644979087e-01,
     -2.6942861175219304e+00,  9.7697533684412885e-01,  6.5462182027411719e-01,
     -3.7046203860785885e+00, -1.2607651054468834e+00, -9.2189716807626709e-01,
     -3.8791847446098422e+00, -1.7541850264811811e+00,  1.2171191552366425e+00,
     -4.7899684032244201e+00, -2.3840351266573148e+00,  1.1136106465875633e+00,
     -3.9361144488534281e+00, -1.2491412759997345e+00,  2.2079223000239274e+00,
     -3.0054563320651506e+00, -2.4447699634935223e+00,  1.2393128539998965e+00,
     -6.1117202557505861e+00, -3.1400842319665723e-01, -4.3439603357646267e-01,
     -7.3613992548812748e+00,  4.0195976855276266e-01, -5.3330380097511831e-01,
     -6.1472073018608668e+00, -1.2882419322881193e+00, -7.6204466024183448e-01,
     -7.7198075027635289e+00,  4.3842489604310114e-01, -1.5853416173353883e+00,
     -7.2656828445612973e+00,  1.4516268978441627e+00, -1.7758294558762308e-01,
     -8.1519139960136222e+00, -8.5871430534087967e-02,  7.7971486186847391e-02,
   };

   double eps = 1e-10;
   for (int i = 0; i < n; i++) {
      COMPARE_REALS(x[i], coord[3*i+0], eps);
      COMPARE_REALS(y[i], coord[3*i+1], eps);
      COMPARE_REALS(z[i], coord[3*i+2], eps);
   }

   finish();
   testEnd();
}

TEST_CASE("ESolv-12-Implicit", "[chksymm]")
{
   rc_flag = calc::xyz | calc::mass | calc::vmask;
   int n;
   std::vector<double> x,y,z,mass;
   SymTyp symtyp;
   int argc = 4;

   // None
   TestFile fx1(TINKER9_DIRSTR "/test/file/esolv/alatet.xyz");
   TestFile fk1(TINKER9_DIRSTR "/test/file/esolv/alatet.key");
   TestFile fp1(TINKER9_DIRSTR "/test/file/esolv/amoebabio18.prm");
   const char* xn1 = "alatet.xyz";
   const char* kn1 = "alatet.key";
   const char* argv1[] = {"dummy", xn1, "-k", kn1};

   testBeginWithArgs(argc, argv1);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::None);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   // Single
   TestFile fx2(TINKER9_DIRSTR "/test/file/esolv/chloride.xyz");
   TestFile fk2(TINKER9_DIRSTR "/test/file/esolv/chloride.key");
   TestFile fp2(TINKER9_DIRSTR "/test/file/esolv/amoebabio09.prm");
   const char* xn2 = "chloride.xyz";
   const char* kn2 = "chloride.key";
   const char* argv2[] = {"dummy", xn2, "-k", kn2};

   testBeginWithArgs(argc, argv2);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Single);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   // Linear
   TestFile fx3a(TINKER9_DIRSTR "/test/file/esolv/lchloride2.xyz");
   TestFile fk3a(TINKER9_DIRSTR "/test/file/esolv/chloride.key");
   TestFile fp3a(TINKER9_DIRSTR "/test/file/esolv/amoebabio09.prm");
   const char* xn3a = "lchloride2.xyz";
   const char* kn3a = "chloride.key";
   const char* argv3a[] = {"dummy", xn3a, "-k", kn3a};

   testBeginWithArgs(argc, argv3a);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Linear);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   TestFile fx3b(TINKER9_DIRSTR "/test/file/esolv/lchloride3.xyz");
   TestFile fk3b(TINKER9_DIRSTR "/test/file/esolv/chloride.key");
   TestFile fp3b(TINKER9_DIRSTR "/test/file/esolv/amoebabio09.prm");
   const char* xn3b = "lchloride2.xyz";
   const char* kn3b = "chloride.key";
   const char* argv3b[] = {"dummy", xn3b, "-k", kn3b};

   testBeginWithArgs(argc, argv3b);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Linear);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   // Planar
   TestFile fx4a(TINKER9_DIRSTR "/test/file/esolv/pchloride3.xyz");
   TestFile fk4a(TINKER9_DIRSTR "/test/file/esolv/chloride.key");
   TestFile fp4a(TINKER9_DIRSTR "/test/file/esolv/amoebabio09.prm");
   const char* xn4a = "pchloride3.xyz";
   const char* kn4a = "chloride.key";
   const char* argv4a[] = {"dummy", xn4a, "-k", kn4a};

   testBeginWithArgs(argc, argv4a);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Planar);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   TestFile fx4b(TINKER9_DIRSTR "/test/file/esolv/pchloride4.xyz");
   TestFile fk4b(TINKER9_DIRSTR "/test/file/esolv/chloride.key");
   TestFile fp4b(TINKER9_DIRSTR "/test/file/esolv/amoebabio09.prm");
   const char* xn4b = "pchloride4.xyz";
   const char* kn4b = "chloride.key";
   const char* argv4b[] = {"dummy", xn4b, "-k", kn4b};

   testBeginWithArgs(argc, argv4b);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Planar);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   // Mirror
   TestFile fx5(TINKER9_DIRSTR "/test/file/esolv/g9.xyz");
   TestFile fk5(TINKER9_DIRSTR "/test/file/esolv/g9.key");
   TestFile fp5(TINKER9_DIRSTR "/test/file/esolv/g9.prm");
   const char* xn5 = "g9.xyz";
   const char* kn5 = "g9.key";
   const char* argv5[] = {"dummy", xn5, "-k", kn5};

   testBeginWithArgs(argc, argv5);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Mirror);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();

   // Center
   TestFile fx6(TINKER9_DIRSTR "/test/file/esolv/cb8.xyz");
   TestFile fk6(TINKER9_DIRSTR "/test/file/esolv/cb8.key");
   TestFile fp6(TINKER9_DIRSTR "/test/file/esolv/cb8.prm");
   const char* xn6 = "cb8.xyz";
   const char* kn6 = "cb8.key";
   const char* argv6[] = {"dummy", xn6, "-k", kn6};

   testBeginWithArgs(argc, argv6);
   initialize();
   n = atoms::n;
   for (int i = 0; i < n; i++) {
      x.push_back(atoms::x[i]);
      y.push_back(atoms::y[i]);
      z.push_back(atoms::z[i]);
      mass.push_back(atomid::mass[i]);
   }
   chksymm(n, &mass[0], &x[0], &y[0], &z[0], symtyp);
   REQUIRE(symtyp == SymTyp::Center);
   x.clear(); y.clear(); z.clear(); mass.clear();
   finish();
   testEnd();
}
