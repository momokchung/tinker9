#pragma once
#include "gmp.h"

namespace tinker
{
class SOS {
public:
   void init_sos_gmp();
   void clear_sos_gmp();

   template <bool use_sos>
   void sos_minor2(double a11, double a21, int& res);
   template <bool use_sos>
   void sos_minor3(double a11, double a12, double a21, double a22, double a31, double a32, int& res);
   template <bool use_sos>
   void sos_minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);
   template <bool use_sos>
   void sos_minor5(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
      double r3, double* coord_d, double r4, double* coord_e, double r5, int& res);
   template <bool use_sos>
   void minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);

private:
   inline void real_to_gmp(double coord, mpz_t val);
   inline void build_weight_gmp(mpz_t ax, mpz_t ay, mpz_t az, mpz_t r, mpz_t w);
   inline void deter2_gmp(mpz_t deter, mpz_t a, mpz_t b);
   inline void deter3_gmp(mpz_t deter, mpz_t a11, mpz_t a12, mpz_t a21, mpz_t a22,
      mpz_t a31, mpz_t a32);
   inline void deter4_gmp(mpz_t deter, mpz_t a11, mpz_t a12, mpz_t a13, mpz_t a21,
      mpz_t a22, mpz_t a23, mpz_t a31, mpz_t a32,
      mpz_t a33, mpz_t a41, mpz_t a42, mpz_t a43);
   inline void deter5_gmp(mpz_t deter, mpz_t a11, mpz_t a12, mpz_t a13, mpz_t a14,
      mpz_t a21, mpz_t a22, mpz_t a23, mpz_t a24,
      mpz_t a31, mpz_t a32, mpz_t a33, mpz_t a34,
      mpz_t a41, mpz_t a42, mpz_t a43, mpz_t a44,
      mpz_t a51, mpz_t a52, mpz_t a53, mpz_t a54);
   void sos_minor2_gmp(double xa, double xb, int& res);
   void sos_minor3_gmp(double xa, double ya, double xb, double yb, double xc, double yc, int& res);
   void sos_minor4_gmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);
   void sos_minor5_gmp(double* coord_a, double ra, double* coord_b, double rb,
      double* coord_c, double rc, double* coord_d, double rd, double* coord_e, double re, int& res);
   void minor4_gmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);

   inline double psub(double r1, double r2);
   inline double padd(double r1, double r2);
   inline int sgn(double d);
   inline void build_weight(double ax, double ay, double az, double r, double& w);
   inline void deter2(double& deter, double b11, double b21, double eps);
   inline void deter3(double& deter, double b11, double b12, double b21,
      double b22, double b31, double b32, double eps);
   inline void deter4(double& deter, double b11, double b12, double b13, double b21,
      double b22, double b23, double b31, double b32, double b33,
      double b41, double b42, double b43, double eps);
   inline void deter5(double& deter, double b11, double b12, double b13, double b14,
      double b21, double b22, double b23, double b24,
      double b31, double b32, double b33, double b34,
      double b41, double b42, double b43, double b44,
      double b51, double b52, double b53, double b54, double eps);
   void sos_minor2_nmp(double a11, double a21, int& res, double eps=1e-10);
   void sos_minor3_nmp(double a11, double a12, double a21, double a22, double a31, double a32, int& res, double eps=1e-10);
   void sos_minor4_nmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps=1e-10);
   void sos_minor5_nmp(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
      double r3, double* coord_d, double r4, double* coord_e, double r5, int& res, double eps=1e-10);
   void minor4_nmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps=1e-10);

   mpz_t a11_mp,a12_mp,a13_mp,a14_mp;
   mpz_t a21_mp,a22_mp,a23_mp,a24_mp;
   mpz_t a31_mp,a32_mp,a33_mp,a34_mp;
   mpz_t a41_mp,a42_mp,a43_mp,a44_mp;
   mpz_t a51_mp,a52_mp,a53_mp,a54_mp;
   mpz_t r1_mp,r2_mp, r3_mp, r4_mp, r5_mp;

   mpz_t temp1,temp2,temp3,temp4;
   mpz_t val1,val2,val3;

   mpz_t c11,c12,c13,c14,c21,c22,c23,c24,c31,c32,c33,c34,c41,c42,c43,c44;
   mpz_t d1,d2,d3,e1,e2,e3,f1,f2,f3,g1,g2,g3;

   double scale;
};
}
