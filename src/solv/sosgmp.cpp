#include "ff/solv/sosgmp.h"
#include "tool/macro.h"
#include <cmath>

namespace tinker
{
// "sosgmp" defines the class used to perform multiprecision
// simulation of simplicity calculations

#define round(x) ((x)>=0?(int)((x)+0.5):(int)((x)-0.5))

void SOS::init_sos_gmp()
{
   mpz_init(a11_mp);mpz_init(a12_mp); mpz_init(a13_mp); mpz_init(a14_mp);
   mpz_init(a21_mp);mpz_init(a22_mp); mpz_init(a23_mp); mpz_init(a24_mp);
   mpz_init(a31_mp);mpz_init(a32_mp); mpz_init(a33_mp); mpz_init(a34_mp);
   mpz_init(a41_mp);mpz_init(a42_mp); mpz_init(a43_mp); mpz_init(a44_mp);
   mpz_init(a51_mp);mpz_init(a52_mp); mpz_init(a53_mp); mpz_init(a54_mp);

   mpz_init(r1_mp);mpz_init(r2_mp); mpz_init(r3_mp); mpz_init(r4_mp); mpz_init(r5_mp);

   mpz_init(temp1); mpz_init(temp2); mpz_init(temp3); mpz_init(temp4);
   mpz_init(val1);mpz_init(val2); mpz_init(val3);

   mpz_init(c11); mpz_init(c12); mpz_init(c13); mpz_init(c14);
   mpz_init(c21); mpz_init(c22); mpz_init(c23); mpz_init(c24);
   mpz_init(c31); mpz_init(c32); mpz_init(c33); mpz_init(c34);
   mpz_init(c41); mpz_init(c42); mpz_init(c43); mpz_init(c44);
   mpz_init(d1); mpz_init(d2); mpz_init(d3);
   mpz_init(e1); mpz_init(e2); mpz_init(e3);
   mpz_init(f1); mpz_init(f2); mpz_init(f3);
   mpz_init(g1); mpz_init(g2); mpz_init(g3);

   scale = 1.e8;
}

void SOS::clear_sos_gmp()
{
   mpz_clear(a11_mp);mpz_clear(a12_mp); mpz_clear(a13_mp); mpz_clear(a14_mp);
   mpz_clear(a21_mp);mpz_clear(a22_mp); mpz_clear(a23_mp); mpz_clear(a24_mp);
   mpz_clear(a31_mp);mpz_clear(a32_mp); mpz_clear(a33_mp); mpz_clear(a34_mp);
   mpz_clear(a41_mp);mpz_clear(a42_mp); mpz_clear(a43_mp); mpz_clear(a44_mp);
   mpz_clear(a51_mp);mpz_clear(a52_mp); mpz_clear(a53_mp); mpz_clear(a54_mp);

   mpz_clear(r1_mp);mpz_clear(r2_mp); mpz_clear(r3_mp); mpz_clear(r4_mp); mpz_clear(r5_mp);

   mpz_clear(temp1); mpz_clear(temp2); mpz_clear(temp3); mpz_clear(temp4);
   mpz_clear(val1);mpz_clear(val2); mpz_clear(val3);

   mpz_clear(c11); mpz_clear(c12); mpz_clear(c13); mpz_clear(c14);
   mpz_clear(c21); mpz_clear(c22); mpz_clear(c23); mpz_clear(c24);
   mpz_clear(c31); mpz_clear(c32); mpz_clear(c33); mpz_clear(c34);
   mpz_clear(c41); mpz_clear(c42); mpz_clear(c43); mpz_clear(c44);
   mpz_clear(d1); mpz_clear(d2); mpz_clear(d3);
   mpz_clear(e1); mpz_clear(e2); mpz_clear(e3);
   mpz_clear(f1); mpz_clear(f2); mpz_clear(f3);
   mpz_clear(g1); mpz_clear(g2); mpz_clear(g3);
}

inline void SOS::real_to_gmp(double coord, mpz_t val)
{
   double x;
   int ivalue;

   mpz_set_d(temp3, scale);

   ivalue = (int) coord;
   mpz_set_si(temp1,ivalue);
   mpz_mul(temp1,temp1,temp3);
   x = (coord-ivalue)*(scale);
   ivalue = (int) round(x);
   mpz_set_si(temp2,ivalue);
   mpz_add(val,temp1,temp2);
}

inline void SOS::build_weight_gmp(mpz_t ax, mpz_t ay, mpz_t az, mpz_t r, mpz_t w)
{
   mpz_mul(temp1,r,r);
   mpz_mul(temp2,ax,ax), mpz_sub(temp1,temp2,temp1);
   mpz_mul(temp2,ay,ay), mpz_add(temp1,temp2,temp1);
   mpz_mul(temp2,az,az), mpz_add(w,temp2,temp1);
}

//  deter2:
//  This subroutine evaluates the determinant:
//  D = | b11 1 |
//      | b21 1 |
//  Input:
//  b11, b21
//  Output:
//  deter

inline void SOS::deter2_gmp(mpz_t deter, mpz_t b11, mpz_t b21)
{
   mpz_sub(deter,b11,b21);
}

// deter3:
// This subroutine evaluates the determinant:
// D = | b11 b12 1 |
//     | b21 b22 1 |
//     | b31 b32 1 |
// Input:
// b11, b12, b21, b22, b31, b32
// Output:
// deter3

inline void SOS::deter3_gmp(mpz_t deter, mpz_t b11, mpz_t b12, mpz_t b21,
   mpz_t b22, mpz_t b31, mpz_t b32)
{
   mpz_sub(temp1,b21,b11);
   mpz_sub(temp2,b22,b12);
   mpz_sub(temp3,b31,b11);
   mpz_sub(temp4,b32,b12);

   mpz_mul(val1,temp1,temp4);
   mpz_mul(val2,temp2,temp3);

   mpz_sub(deter,val1,val2);
}

// deter4:
// This subroutine evaluates the determinant:
// D = | b11 b12 b13 1 |
//     | b21 b22 b23 1 |
//     | b31 b32 b33 1 |
//     | b41 b42 b43 1 |
// Input:
// b11, b12, b13, b21, b22, b23, b31, b32, b33
// b41, b42, b43
// Output:
// deter4

inline void SOS::deter4_gmp(mpz_t deter, mpz_t b11, mpz_t b12, mpz_t b13, mpz_t b21,
   mpz_t b22, mpz_t b23, mpz_t b31, mpz_t b32, mpz_t b33,
   mpz_t b41, mpz_t b42, mpz_t b43)
{
   mpz_sub(c11,b21,b11);mpz_sub(c12,b22,b12);mpz_sub(c13,b23,b13);
   mpz_sub(c21,b31,b11);mpz_sub(c22,b32,b12);mpz_sub(c23,b33,b13);
   mpz_sub(c31,b41,b11);mpz_sub(c32,b42,b12);mpz_sub(c33,b43,b13);

   mpz_mul(temp1,c22,c33);mpz_mul(temp2,c32,c23);mpz_sub(val1,temp1,temp2);
   mpz_mul(temp1,c12,c33);mpz_mul(temp2,c32,c13);mpz_sub(val2,temp1,temp2);
   mpz_mul(temp1,c12,c23);mpz_mul(temp2,c22,c13);mpz_sub(val3,temp1,temp2);

   mpz_mul(temp1,c21,val2);mpz_mul(temp2,c11,val1);mpz_mul(temp3,c31,val3);

   mpz_add(val1,temp2,temp3);
   mpz_sub(deter,temp1,val1);
}

// deter5:
// This subroutine evaluates the determinant:
// D = | b11 b12 b13 b14 1 |
//     | b21 b22 b23 b24 1 |
//     | b31 b32 b33 b34 1 |
//     | b41 b42 b43 b44 1 |
//     | b51 b52 b53 b54 1 |
// Input:
// b11, b12, b13, b14, b21, b22, b23, b24, b31, b32, b33, b34
// b41, b42, b43, b44, b51, b52, b53, b54
// Output:
// deter5

inline void SOS::deter5_gmp(mpz_t deter, mpz_t b11, mpz_t b12, mpz_t b13, mpz_t b14,
   mpz_t b21, mpz_t b22, mpz_t b23, mpz_t b24,
   mpz_t b31, mpz_t b32, mpz_t b33, mpz_t b34,
   mpz_t b41, mpz_t b42, mpz_t b43, mpz_t b44,
   mpz_t b51, mpz_t b52, mpz_t b53, mpz_t b54)
{
   mpz_sub(c11,b21,b11); mpz_sub(c12,b22,b12); mpz_sub(c13,b23,b13);
   mpz_sub(c14,b24,b14);
   mpz_sub(c21,b31,b11); mpz_sub(c22,b32,b12); mpz_sub(c23,b33,b13);
   mpz_sub(c24,b34,b14);
   mpz_sub(c31,b41,b11); mpz_sub(c32,b42,b12); mpz_sub(c33,b43,b13);
   mpz_sub(c34,b44,b14);
   mpz_sub(c41,b51,b11); mpz_sub(c42,b52,b12); mpz_sub(c43,b53,b13);
   mpz_sub(c44,b54,b14);

   mpz_mul(temp1,c32,c43); mpz_mul(temp2,c42,c33); mpz_sub(d1,temp1,temp2);
   mpz_mul(temp1,c32,c44); mpz_mul(temp2,c42,c34); mpz_sub(d2,temp1,temp2);
   mpz_mul(temp1,c33,c44); mpz_mul(temp2,c43,c34); mpz_sub(d3,temp1,temp2);

   mpz_mul(temp1,c12,c23); mpz_mul(temp2,c22,c13); mpz_sub(e1,temp1,temp2);
   mpz_mul(temp1,c12,c24); mpz_mul(temp2,c22,c14); mpz_sub(e2,temp1,temp2);
   mpz_mul(temp1,c13,c24); mpz_mul(temp2,c23,c14); mpz_sub(e3,temp1,temp2);

   mpz_mul(temp1,c11,c24); mpz_mul(temp2,c21,c14); mpz_sub(f1,temp1,temp2);
   mpz_mul(temp1,c11,c23); mpz_mul(temp2,c21,c13); mpz_sub(f2,temp1,temp2);
   mpz_mul(temp1,c11,c22); mpz_mul(temp2,c21,c12); mpz_sub(f3,temp1,temp2);

   mpz_mul(temp1,c31,c44); mpz_mul(temp2,c41,c34); mpz_sub(g1,temp1,temp2);
   mpz_mul(temp1,c31,c43); mpz_mul(temp2,c41,c33); mpz_sub(g2,temp1,temp2);
   mpz_mul(temp1,c31,c42); mpz_mul(temp2,c41,c32); mpz_sub(g3,temp1,temp2);

   mpz_mul(temp1,e3,g3); mpz_mul(temp2,e2,g2); mpz_sub(temp3,temp1,temp2);
   mpz_mul(temp1,e1,g1); mpz_add(temp3,temp3,temp1);
   mpz_mul(temp1,d3,f3); mpz_add(temp3,temp3,temp1);
   mpz_mul(temp1,d2,f2); mpz_sub(temp3,temp3,temp1);
   mpz_mul(temp1,d1,f1); mpz_add(deter,temp3,temp1);
}

// sos_minor2_gmp:
// This subroutine tests the sign of the determinant
// D = | a11 1 |
//     | a21 1 |
// If the determinant is found to be 0, then the SoS procedure is used:
// a development of the determinant with respect to a perturbation EPS
// applied to the coordinates in the determinant is computed, and
// the sign of the first non zero term defines the sign of the
// determinant.
// In the case of a 2x2 determinant, the first term in the expansion
// is the coefficient 1 ...

void SOS::sos_minor2_gmp(double xa, double xb, int& res)
{
   int icomp;

   // get coordinates

   real_to_gmp(xa,a11_mp);
   real_to_gmp(xb,a21_mp);

   // compute determinant

   deter2_gmp(temp1,a11_mp,a21_mp);

   icomp = mpz_sgn(temp1);

   if (icomp != 0) {
      res = icomp;
   }
   else {
      res = 1;
   }
}

// sos_minor3_gmp:
// This subroutine tests the sign of the determinant
// D = | a11 a12 1 |
//     | a21 a22 1 |
//     | a31 a32 1 |
// If the determinant is found to be 0, then the SoS procedure is used:
// a development of the determinant with respect to a perturbation EPS
// applied to the coordinates in the determinant is computed, and
// the sign of the first non zero term defines the sign of the
// determinant.
// In the case of a 3x3 determinant, the maximum number of terms to be
// checked is 4 ...

void SOS::sos_minor3_gmp(double xa, double ya, double xb, double yb, double xc, double yc, int& res)
{
   int icomp;

   // transfer coordinates to GMP

   real_to_gmp(xa,a11_mp);
   real_to_gmp(ya,a12_mp);
   real_to_gmp(xb,a21_mp);
   real_to_gmp(yb,a22_mp);
   real_to_gmp(xc,a31_mp);
   real_to_gmp(yc,a32_mp);

   // compute determinant

   deter3_gmp(temp1,a11_mp,a12_mp,a21_mp,a22_mp,a31_mp,a32_mp);

   icomp = mpz_sgn(temp1);

   // if major determinant is non 0, return its sign

   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the determinant with
   // respect to EPS.
   // The initial determinant is: minor3(i,j,k,1,2,0)

   // term 1: -minor2(j,k,1,0)

   deter2_gmp(temp1,a21_mp,a31_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 2: minor2(j,k,2,0)

   deter2_gmp(temp1,a22_mp,a32_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 3: minor2(i,k,1,0)

   deter2_gmp(temp1,a11_mp,a31_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 4: 1

   res = 1;
}

// sos_minor4_gmp:
// This subroutine tests the sign of the determinant
// D = | a11 a12 a13 1 |
//     | a21 a22 a23 1 |
//     | a31 a32 a33 1 |
//     | a41 a42 a43 1 |
// If the determinant is found to be 0, then the SoS procedure is used:
// a development of the determinant with *respect to a perturbation EPS
// applied to the coordinates in the determinant is computed, and
// the sign of the first non zero term defines the sign of the
// determinant.
// In the case of a 4x4 determinant, the maximum number of terms to be
// checked is 14 ...

void SOS::sos_minor4_gmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res)
{
   int icomp;

   // transfer coordinates to gmp

   real_to_gmp(coord_a[0],a11_mp);
   real_to_gmp(coord_a[1],a12_mp);
   real_to_gmp(coord_a[2],a13_mp);
   real_to_gmp(coord_b[0],a21_mp);
   real_to_gmp(coord_b[1],a22_mp);
   real_to_gmp(coord_b[2],a23_mp);
   real_to_gmp(coord_c[0],a31_mp);
   real_to_gmp(coord_c[1],a32_mp);
   real_to_gmp(coord_c[2],a33_mp);
   real_to_gmp(coord_d[0],a41_mp);
   real_to_gmp(coord_d[1],a42_mp);
   real_to_gmp(coord_d[2],a43_mp);

   // compute determinant

   deter4_gmp(temp1,a11_mp,a12_mp,a13_mp,a21_mp,a22_mp,a23_mp,
            a31_mp,a32_mp,a33_mp,a41_mp,a42_mp,a43_mp);

   icomp = mpz_sgn(temp1);

   // if major determinant is non 0, return its sign

   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the determinant with
   // respect to EPS.
   // The initial determinant is: minor4(i,j,k,l,1,2,3,0)

   // term 1: minor3(j,k,l,1,2,0)

   deter3_gmp(temp1,a21_mp,a22_mp,a31_mp,a32_mp,a41_mp,a42_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 2: -minor3(j,k,l,1,3,0)

   deter3_gmp(temp1,a21_mp,a23_mp,a31_mp,a33_mp,a41_mp,a43_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 3: minor3(j,k,l,2,3,0)

   deter3_gmp(temp1,a22_mp,a23_mp,a32_mp,a33_mp,a42_mp,a43_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 4: -minor3(i,k,l,1,2,0)

   deter3_gmp(temp1,a11_mp,a12_mp,a31_mp,a32_mp,a41_mp,a42_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 5: minor2(k,l,1,0)

   deter2_gmp(temp1,a31_mp,a41_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 6: -minor2(k,l,2,0)

   deter2_gmp(temp1,a32_mp,a42_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 7: minor3(i,k,l,1,3,0)

   deter3_gmp(temp1,a11_mp,a13_mp,a31_mp,a33_mp,a41_mp,a43_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 8: minor2(k,l,3,0)

   deter2_gmp(temp1,a33_mp,a43_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 9: -minor3(i,k,l,2,3,0)

   deter3_gmp(temp1,a12_mp,a13_mp,a32_mp,a33_mp,a42_mp,a43_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 10: minor3(i,j,l,1,2,0)

   deter3_gmp(temp1,a11_mp,a12_mp,a21_mp,a22_mp,a41_mp,a42_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 11: -minor2(j,l,1,0)

   deter2_gmp(temp1,a21_mp,a41_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 12: minor2(j,l,2,0)

   deter2_gmp(temp1,a22_mp,a42_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 13: minor2(i,l,1,0)

   deter2_gmp(temp1,a11_mp,a41_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 14: 1

   res = 1;
}

// sos_minor5_gmp:
// This subroutine tests the sign of the determinant
// D = | a11 a12 a13 a14 1 |
//     | a21 a22 a23 a24 1 |
//     | a31 a32 a33 a34 1 |
//     | a41 a42 a43 a44 1 |
//     | a51 a52 a53 a54 1 |
// If the determinant is found to be 0, then the SoS procedure is used:
// a development of the determinant with *respect to a perturbation EPS
// applied to the coordinates in the determinant is computed, and
// the sign of the first non zero term defines the sign of the
// determinant.
// In the case of a 5x5 determinant, the maximum number of terms to be
// checked is 49 ...

void SOS::sos_minor5_gmp(double* coord_a, double ra, double* coord_b, double rb,
   double* coord_c, double rc, double* coord_d, double rd, double* coord_e, double re,
   int& res)
{
   int icomp;

   // initialize local GMP variables

   real_to_gmp(coord_a[0],a11_mp);
   real_to_gmp(coord_a[1],a12_mp);
   real_to_gmp(coord_a[2],a13_mp);
   real_to_gmp(coord_b[0],a21_mp);
   real_to_gmp(coord_b[1],a22_mp);
   real_to_gmp(coord_b[2],a23_mp);
   real_to_gmp(coord_c[0],a31_mp);
   real_to_gmp(coord_c[1],a32_mp);
   real_to_gmp(coord_c[2],a33_mp);
   real_to_gmp(coord_d[0],a41_mp);
   real_to_gmp(coord_d[1],a42_mp);
   real_to_gmp(coord_d[2],a43_mp);
   real_to_gmp(coord_e[0],a51_mp);
   real_to_gmp(coord_e[1],a52_mp);
   real_to_gmp(coord_e[2],a53_mp);

   real_to_gmp(ra,r1_mp);
   real_to_gmp(rb,r2_mp);
   real_to_gmp(rc,r3_mp);
   real_to_gmp(rd,r4_mp);
   real_to_gmp(re,r5_mp);

   build_weight_gmp(a11_mp,a12_mp,a13_mp,r1_mp,a14_mp);
   build_weight_gmp(a21_mp,a22_mp,a23_mp,r2_mp,a24_mp);
   build_weight_gmp(a31_mp,a32_mp,a33_mp,r3_mp,a34_mp);
   build_weight_gmp(a41_mp,a42_mp,a43_mp,r4_mp,a44_mp);
   build_weight_gmp(a51_mp,a52_mp,a53_mp,r5_mp,a54_mp);

   // compute determinant

   deter5_gmp(temp1,a11_mp,a12_mp,a13_mp,a14_mp,a21_mp,a22_mp,
      a23_mp,a24_mp,a31_mp,a32_mp,a33_mp,a34_mp,a41_mp,a42_mp,
      a43_mp,a44_mp,a51_mp,a52_mp,a53_mp,a54_mp);

   icomp = mpz_sgn(temp1);

   // if major determinant is non 0, return its sign

   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the determinant with
   // respect to EPS.
   // The initial determinant is: minor5(i,j,k,l,m,1,2,3,4,0)

   // term 1: -minor4(j,k,l,m,1,2,3,0)

   deter4_gmp(temp1,a21_mp,a22_mp,a23_mp,a31_mp,a32_mp,a33_mp,
      a41_mp,a42_mp,a43_mp,a51_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 2: minor4(j,k,l,m,1,2,4,0)

   deter4_gmp(temp1,a21_mp,a22_mp,a24_mp,a31_mp,a32_mp,a34_mp,
      a41_mp,a42_mp,a44_mp,a51_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 3: -minor4(j,k,l,m,1,3,4,0)

   deter4_gmp(temp1,a21_mp,a23_mp,a24_mp,a31_mp,a33_mp,a34_mp,
      a41_mp,a43_mp,a44_mp,a51_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 4: minor4(j,k,l,m,2,3,4,0)

   deter4_gmp(temp1,a22_mp,a23_mp,a24_mp,a32_mp,a33_mp,a34_mp,
      a42_mp,a43_mp,a44_mp,a52_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 5: minor4(i,k,l,m,1,2,3,0)

   deter4_gmp(temp1,a11_mp,a12_mp,a13_mp,a31_mp,a32_mp,a33_mp,
      a41_mp,a42_mp,a43_mp,a51_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 6: minor3(k,l,m,1,2,0)

   deter3_gmp(temp1,a31_mp,a32_mp,a41_mp,a42_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 7: -minor3(k,l,m,1,3,0)

   deter3_gmp(temp1,a31_mp,a33_mp,a41_mp,a43_mp,a51_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 8: minor3(k,l,m,2,3,0)

   deter3_gmp(temp1,a32_mp,a33_mp,a42_mp,a43_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 9: -minor4(i,k,l,m,1,2,4,0)

   deter4_gmp(temp1,a11_mp,a12_mp,a14_mp,a31_mp,a32_mp,a34_mp,
      a41_mp,a42_mp,a44_mp,a51_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 10: minor3(k,l,m,1,4,0)

   deter3_gmp(temp1,a31_mp,a34_mp,a41_mp,a44_mp,a51_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 11: -minor3(k,l,m,2,4,0)

   deter3_gmp(temp1,a32_mp,a34_mp,a42_mp,a44_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 12: minor4(i,k,l,m,1,3,4,0)

   deter4_gmp(temp1,a11_mp,a13_mp,a14_mp,a31_mp,a33_mp,a34_mp,
      a41_mp,a43_mp,a44_mp,a51_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 13: minor3(k,l,m,3,4,0)

   deter3_gmp(temp1,a33_mp,a34_mp,a43_mp,a44_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 14: -minor4(i,k,l,m,2,3,4,0)

   deter4_gmp(temp1,a12_mp,a13_mp,a14_mp,a32_mp,a33_mp,a34_mp,
      a42_mp,a43_mp,a44_mp,a52_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 15: -minor4(i,j,l,m,1,2,3,0)

   deter4_gmp(temp1,a11_mp,a12_mp,a13_mp,a21_mp,a22_mp,a23_mp,
      a41_mp,a42_mp,a43_mp,a51_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 16: -minor3(j,l,m,1,2,0)

   deter3_gmp(temp1,a21_mp,a22_mp,a41_mp,a42_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 17: minor3(j,l,m,1,3,0)

   deter3_gmp(temp1,a21_mp,a23_mp,a41_mp,a43_mp,a51_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 18: -minor3(j,l,m,2,3,0)

   deter3_gmp(temp1,a22_mp,a23_mp,a42_mp,a43_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 19: minor3(i,l,m,1,2,0)

   deter3_gmp(temp1,a11_mp,a12_mp,a41_mp,a42_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 20: -minor2(l,m,1,0)

   deter2_gmp(temp1,a41_mp,a51_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 21: minor2(l,m,2,0)

   deter2_gmp(temp1,a42_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 22: -minor3(i,l,m,1,3,0)

   deter3_gmp(temp1,a11_mp,a13_mp,a41_mp,a43_mp,a51_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 23: -minor2(l,m,3,0)

   deter2_gmp(temp1,a43_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 24: minor3(i,l,m,2,3,0)

   deter3_gmp(temp1,a12_mp,a13_mp,a42_mp,a43_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 25: minor4(i,j,l,m,1,2,4,0)

   deter4_gmp(temp1,a11_mp,a12_mp,a14_mp,a21_mp,a22_mp,a24_mp,
      a41_mp,a42_mp,a44_mp,a51_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 26: -minor3(j,l,m,1,4,0)

   deter3_gmp(temp1,a21_mp,a24_mp,a41_mp,a44_mp,a51_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 27: minor3(j,l,m,2,4,0)

   deter3_gmp(temp1,a22_mp,a24_mp,a42_mp,a44_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 28: minor3(i,l,m,1,4,0)

   deter3_gmp(temp1,a11_mp,a14_mp,a41_mp,a44_mp,a51_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 29: minor2(l,m,4,0)

   deter2_gmp(temp1,a44_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 30: -minor3(i,l,m,2,4,0)

   deter3_gmp(temp1,a12_mp,a14_mp,a42_mp,a44_mp,a52_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 31: -minor4(i,j,l,m,1,3,4,0)

   deter4_gmp(temp1,a11_mp,a13_mp,a14_mp,a21_mp,a23_mp,a24_mp,
      a41_mp,a43_mp,a44_mp,a51_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 32: -minor3(j,l,m,3,4,0)

   deter3_gmp(temp1,a23_mp,a24_mp,a43_mp,a44_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 33: minor3(i,l,m,3,4,0)

   deter3_gmp(temp1,a13_mp,a14_mp,a43_mp,a44_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 34: minor4(i,j,l,m,2,3,4,0)

   deter4_gmp(temp1,a12_mp,a13_mp,a14_mp,a22_mp,a23_mp,a24_mp,
      a42_mp,a43_mp,a44_mp,a52_mp,a53_mp,a54_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 35: minor4(i,j,k,m,1,2,3,0)

   deter4_gmp(temp1,a11_mp,a12_mp,a13_mp,a21_mp,a22_mp,a23_mp,
      a31_mp,a32_mp,a33_mp,a51_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 36: minor3(j,k,m,1,2,0)

   deter3_gmp(temp1,a21_mp,a22_mp,a31_mp,a32_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 37: -minor3(j,k,m,1,3,0)

   deter3_gmp(temp1,a21_mp,a23_mp,a31_mp,a33_mp,a51_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 38: minor3(j,k,m,2,3,0)

   deter3_gmp(temp1,a22_mp,a23_mp,a32_mp,a33_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 39: -minor3(i,k,m,1,2,0)

   deter3_gmp(temp1,a11_mp,a12_mp,a31_mp,a32_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 40: minor2(k,m,1,0)

   deter2_gmp(temp1,a31_mp,a51_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 41: -minor2(k,m,2,0)

   deter2_gmp(temp1,a32_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 42: minor3(i,k,m,1,3,0)

   deter3_gmp(temp1,a11_mp,a13_mp,a31_mp,a33_mp,a51_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 43: minor2(k,m,3,0)

   deter2_gmp(temp1,a33_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 44: -minor3(i,k,m,2,3,0)

   deter3_gmp(temp1,a12_mp,a13_mp,a32_mp,a33_mp,a52_mp,a53_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 45: minor3(i,j,m,1,2,0)

   deter3_gmp(temp1,a11_mp,a12_mp,a21_mp,a22_mp,a51_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 46: -minor2(j,m,1,0)

   deter2_gmp(temp1,a21_mp,a51_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  -icomp;
      return;
   }

   // term 47: minor2(j,m,2,0)

   deter2_gmp(temp1,a22_mp,a52_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 48: minor2(i,m,1,0)

   deter2_gmp(temp1,a11_mp,a51_mp);
   icomp = mpz_sgn(temp1);
   if (icomp != 0) {
      res =  icomp;
      return;
   }

   // term 49: 1

   res = 1;
}

// minor4_:
// This subroutine tests the sign of the determinant
// D = | a11 a12 a13 1 |
//     | a21 a22 a23 1 |
//     | a31 a32 a33 1 |
//     | a41 a42 a43 1 |
// and return 1 if positive, -1 if negative, 0 otherwise

void SOS::minor4_gmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res)
{
   int icomp;

   // transfer coordinates to gmp

   real_to_gmp(coord_a[0],a11_mp);
   real_to_gmp(coord_a[1],a12_mp);
   real_to_gmp(coord_a[2],a13_mp);
   real_to_gmp(coord_b[0],a21_mp);
   real_to_gmp(coord_b[1],a22_mp);
   real_to_gmp(coord_b[2],a23_mp);
   real_to_gmp(coord_c[0],a31_mp);
   real_to_gmp(coord_c[1],a32_mp);
   real_to_gmp(coord_c[2],a33_mp);
   real_to_gmp(coord_d[0],a41_mp);
   real_to_gmp(coord_d[1],a42_mp);
   real_to_gmp(coord_d[2],a43_mp);

   // compute determinant

   deter4_gmp(temp1,a11_mp,a12_mp,a13_mp,a21_mp,a22_mp,a23_mp,
            a31_mp,a32_mp,a33_mp,a41_mp,a42_mp,a43_mp);

   icomp = mpz_sgn(temp1);

   res = icomp;
}
}

namespace tinker
{

// "sos" defines the class used to perform
// simulation of simplicity calculations

inline double SOS::psub(double r1, double r2)
{
   constexpr double epsabs = 1e-14;
   constexpr double epsrel = 1e-14;
   double r = r1 - r2;
   if (std::abs(r) < epsabs) return 0.;
   double maxr = std::fmax(std::abs(r1),std::abs(r2));
   if (maxr != 0.) {
      if (std::abs(r/maxr) < epsrel) return 0.;
   }
   return r;
}

inline double SOS::padd(double r1, double r2)
{
   constexpr double epsabs = 1e-14;
   constexpr double epsrel = 1e-14;
   double r = r1 + r2;
   if (std::abs(r) < epsabs) return 0.;
   double maxr = std::fmax(std::abs(r1),std::abs(r2));
   if (maxr != 0.) {
      if (std::abs(r/maxr) < epsrel) return 0.;
   }
   return r;
}

inline int SOS::sgn(double d)
{
   if (d == 0.) return 0;
   else if (d > 0.) return 1;
   else return -1;
}

// builds the weight of a point: w = x**2 + y**2 + z**2 - ra**2
inline void SOS::build_weight(double ax, double ay, double az, double r, double& w)
{
   double temp1,temp2;
   temp1 = r * r;
   temp2 = ax * ax;
   temp1 = psub(temp2,temp1);
   temp2 = ay * ay;
   temp1 = padd(temp2,temp1);
   temp2 = az * az;

   w = padd(temp2,temp1);
}

// deter2 evaluates the determinant:
// D = | b11 1 |
//     | b21 1 |
inline void SOS::deter2(double& deter, double b11, double b21, double eps)
{
   deter = psub(b11,b21);

   if (std::abs(deter) < eps) deter = 0.;
}

// deter3 evaluates the determinant:
// D = | b11 b12 1 |
//     | b21 b22 1 |
//     | b31 b32 1 |
inline void SOS::deter3(double& deter, double b11, double b12, double b21,
   double b22, double b31, double b32, double eps)
{
   double temp1,temp2,temp3,temp4;
   double val1,val2;

   temp1 = psub(b21,b11);
   temp2 = psub(b22,b12);
   temp3 = psub(b31,b11);
   temp4 = psub(b32,b12);
   val1 = temp1*temp4;
   val2 = temp2*temp3;

   deter = psub(val1,val2);

   if (std::abs(deter) < eps) deter = 0.;
}

// deter4 evaluates the determinant:
// D = | b11 b12 b13 1 |
//     | b21 b22 b23 1 |
//     | b31 b32 b33 1 |
//     | b41 b42 b43 1 |
inline void SOS::deter4(double& deter, double b11, double b12, double b13, double b21,
   double b22, double b23, double b31, double b32, double b33,
   double b41, double b42, double b43, double eps)
{
   double c11,c12,c13;
   double c21,c22,c23;
   double c31,c32,c33;
   double temp1,temp2,temp3;
   double val1,val2,val3;

   c11 = psub(b21,b11); c12 = psub(b22,b12); c13 = psub(b23,b13);
   c21 = psub(b31,b11); c22 = psub(b32,b12); c23 = psub(b33,b13);
   c31 = psub(b41,b11); c32 = psub(b42,b12); c33 = psub(b43,b13);
   temp1 = c22*c33; temp2 = c32*c23; val1 = psub(temp1,temp2);
   temp1 = c12*c33; temp2 = c32*c13; val2 = psub(temp1,temp2);
   temp1 = c12*c23; temp2 = c22*c13; val3 = psub(temp1,temp2);
   temp1 = c21*val2; temp2 = c11*val1; temp3 = c31*val3;
   val1 = padd(temp2,temp3);

   deter = psub(temp1,val1);

   if (std::abs(deter) < eps) deter = 0.;
}

// deter5 evaluates the determinant:
// D = | b11 b12 b13 b14 1 |
//     | b21 b22 b23 b24 1 |
//     | b31 b32 b33 b34 1 |
//     | b41 b42 b43 b44 1 |
//     | b51 b52 b53 b54 1 |
inline void SOS::deter5(double& deter, double b11, double b12, double b13, double b14,
   double b21, double b22, double b23, double b24,
   double b31, double b32, double b33, double b34,
   double b41, double b42, double b43, double b44,
   double b51, double b52, double b53, double b54, double eps)
{
   double c11,c12,c13,c14;
   double c21,c22,c23,c24;
   double c31,c32,c33,c34;
   double c41,c42,c43,c44;
   double temp1,temp2,temp3;
   double d1,d2,d3;
   double e1,e2,e3;
   double f1,f2,f3;
   double g1,g2,g3;

   c11 = psub(b21,b11); c12 = psub(b22,b12); c13 = psub(b23,b13); c14 = psub(b24,b14);
   c21 = psub(b31,b11); c22 = psub(b32,b12); c23 = psub(b33,b13); c24 = psub(b34,b14);
   c31 = psub(b41,b11); c32 = psub(b42,b12); c33 = psub(b43,b13); c34 = psub(b44,b14);
   c41 = psub(b51,b11); c42 = psub(b52,b12); c43 = psub(b53,b13); c44 = psub(b54,b14);
   temp1 = c32 * c43; temp2 = c42 * c33; d1 = psub(temp1,temp2);
   temp1 = c32 * c44; temp2 = c42 * c34; d2 = psub(temp1,temp2);
   temp1 = c33 * c44; temp2 = c43 * c34; d3 = psub(temp1,temp2);
   temp1 = c12 * c23; temp2 = c22 * c13; e1 = psub(temp1,temp2);
   temp1 = c12 * c24; temp2 = c22 * c14; e2 = psub(temp1,temp2);
   temp1 = c13 * c24; temp2 = c23 * c14; e3 = psub(temp1,temp2);
   temp1 = c11 * c24; temp2 = c21 * c14; f1 = psub(temp1,temp2);
   temp1 = c11 * c23; temp2 = c21 * c13; f2 = psub(temp1,temp2);
   temp1 = c11 * c22; temp2 = c21 * c12; f3 = psub(temp1,temp2);
   temp1 = c31 * c44; temp2 = c41 * c34; g1 = psub(temp1,temp2);
   temp1 = c31 * c43; temp2 = c41 * c33; g2 = psub(temp1,temp2);
   temp1 = c31 * c42; temp2 = c41 * c32; g3 = psub(temp1,temp2);
   temp1 = e3 * g3; temp2 = e2 * g2; temp3 = psub(temp1,temp2);
   temp1 = e1 * g1; temp3 = padd(temp3,temp1);
   temp1 = d3 * f3; temp3 = padd(temp3,temp1);
   temp1 = d2 * f2; temp3 = psub(temp3,temp1);
   temp1 = d1 * f1;

   deter = padd(temp3,temp1);

   if (std::abs(deter) < eps) deter = 0.;
}

// sos_minor2 tests the sign of the determinant:
// D = | a11 1 |
//     | a21 1 |
void SOS::sos_minor2_nmp(double a11, double a21, int& res, double eps)
{
   int icomp;

   // compute determinant
   double temp1;
   deter2(temp1,a11,a21,eps);

   icomp = sgn(temp1);

   if (icomp != 0) res = icomp;
   else res = 1;
}

// sos_minor3 tests the sign of the determinant:
// D = | a11 a12 1 |
//     | a21 a22 1 |
//     | a31 a32 1 |
void SOS::sos_minor3_nmp(double a11, double a12, double a21, double a22, double a31, double a32, int& res, double eps)
{
   int icomp;

   // compute determinant
   double temp1;
   deter3(temp1,a11,a12,a21,a22,a31,a32,eps);

   icomp = sgn(temp1);

   // if major determinant is non 0, return its sign
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the
   // determinant with respect to EPS.
   // The initial determinant is: minor3(i,j,k,1,2,0)

   // term 1: -minor2(j,k,1,0)
   deter2(temp1,a21,a31,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 2: minor2(j,k,2,0)
   deter2(temp1,a22,a32,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 3: minor2(i,k,1,0)
   deter2(temp1,a11,a31,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 4: 1
   res = 1;
}

// sos_minor4 tests the sign of the determinant:
// D = | a11 a12 a13 1 |
//     | a21 a22 a23 1 |
//     | a31 a32 a33 1 |
//     | a41 a42 a43 1 |
void SOS::sos_minor4_nmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps)
{
   int icomp;

   double a11 = coord_a[0];
   double a12 = coord_a[1];
   double a13 = coord_a[2];
   double a21 = coord_b[0];
   double a22 = coord_b[1];
   double a23 = coord_b[2];
   double a31 = coord_c[0];
   double a32 = coord_c[1];
   double a33 = coord_c[2];
   double a41 = coord_d[0];
   double a42 = coord_d[1];
   double a43 = coord_d[2];

   // compute determinant
   double temp1;
   deter4(temp1,a11,a12,a13,a21,a22,a23,a31,a32,a33,a41,a42,a43,eps);

   icomp = sgn(temp1);

   // if major determinant is non 0, return its sign
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the
   // determinant with respect to EPS.
   // The initial determinant is: minor4(i,j,k,l,1,2,3,0)

   // term 1: minor3(j,k,l,1,2,0)
   deter3(temp1,a21,a22,a31,a32,a41,a42,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 2: -minor3(j,k,l,1,3,0)
   deter3(temp1,a21,a23,a31,a33,a41,a43,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 3: minor3(j,k,l,2,3,0)
   deter3(temp1,a22,a23,a32,a33,a42,a43,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 4: -minor3(i,k,l,1,2,0)
   deter3(temp1,a11,a12,a31,a32,a41,a42,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 5: minor2(k,l,1,0)
   deter2(temp1,a31,a41,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 6: -minor2(k,l,2,0)
   deter2(temp1,a32,a42,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 7: minor3(i,k,l,1,3,0)
   deter3(temp1,a11,a13,a31,a33,a41,a43,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 8: minor2(k,l,3,0)
   deter2(temp1,a33,a43,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 9: -minor3(i,k,l,2,3,0)
   deter3(temp1,a12,a13,a32,a33,a42,a43,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 10: minor3(i,j,l,1,2,0)
   deter3(temp1,a11,a12,a21,a22,a41,a42,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 11: -minor2(j,l,1,0)
   deter2(temp1,a21,a41,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 12: minor2(j,l,2,0)
   deter2(temp1,a22,a42,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 13: minor2(i,l,1,0)
   deter2(temp1,a11,a41,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 14: 1
   res = 1;
}


// sos_minor5 tests the sign of the determinant:
// D = | a11 a12 a13 a14 1 |
//     | a21 a22 a23 a24 1 |
//     | a31 a32 a33 a34 1 |
//     | a41 a42 a43 a44 1 |
//     | a51 a52 a53 a54 1 |
void SOS::sos_minor5_nmp(double* coord_a, double r1, double* coord_b, double r2,
   double* coord_c, double r3, double* coord_d, double r4, double* coord_e, double r5,
   int& res, double eps)
{
   int icomp;

   double a11 = coord_a[0];
   double a12 = coord_a[1];
   double a13 = coord_a[2];
   double a21 = coord_b[0];
   double a22 = coord_b[1];
   double a23 = coord_b[2];
   double a31 = coord_c[0];
   double a32 = coord_c[1];
   double a33 = coord_c[2];
   double a41 = coord_d[0];
   double a42 = coord_d[1];
   double a43 = coord_d[2];
   double a51 = coord_e[0];
   double a52 = coord_e[1];
   double a53 = coord_e[2];

   double a14,a24,a34,a44,a54;

   build_weight(a11,a12,a13,r1,a14);
   build_weight(a21,a22,a23,r2,a24);
   build_weight(a31,a32,a33,r3,a34);
   build_weight(a41,a42,a43,r4,a44);
   build_weight(a51,a52,a53,r5,a54);

   // compute determinant
   double temp1;
   deter5(temp1,a11,a12,a13,a14,a21,a22,
      a23,a24,a31,a32,a33,a34,a41,a42,
      a43,a44,a51,a52,a53,a54,eps);

   icomp = sgn(temp1);

   // if major determinant is non 0, return its sign
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // Look now at each term in the expansion of the
   // determinant with respect to EPS.
   // The initial determinant is: minor5(i,j,k,l,m,1,2,3,4,0)

   // term 1: -minor4(j,k,l,m,1,2,3,0)
   deter4(temp1,a21,a22,a23,a31,a32,a33,
      a41,a42,a43,a51,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 2: minor4(j,k,l,m,1,2,4,0)
   deter4(temp1,a21,a22,a24,a31,a32,a34,
      a41,a42,a44,a51,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 3: -minor4(j,k,l,m,1,3,4,0)
   deter4(temp1,a21,a23,a24,a31,a33,a34,
      a41,a43,a44,a51,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 4: minor4(j,k,l,m,2,3,4,0)
   deter4(temp1,a22,a23,a24,a32,a33,a34,
      a42,a43,a44,a52,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 5: minor4(i,k,l,m,1,2,3,0)
   deter4(temp1,a11,a12,a13,a31,a32,a33,
      a41,a42,a43,a51,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 6: minor3(k,l,m,1,2,0)
   deter3(temp1,a31,a32,a41,a42,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 7: -minor3(k,l,m,1,3,0)
   deter3(temp1,a31,a33,a41,a43,a51,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 8: minor3(k,l,m,2,3,0)
   deter3(temp1,a32,a33,a42,a43,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 9: -minor4(i,k,l,m,1,2,4,0)
   deter4(temp1,a11,a12,a14,a31,a32,a34,
      a41,a42,a44,a51,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 10: minor3(k,l,m,1,4,0)
   deter3(temp1,a31,a34,a41,a44,a51,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 11: -minor3(k,l,m,2,4,0)
   deter3(temp1,a32,a34,a42,a44,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 12: minor4(i,k,l,m,1,3,4,0)
   deter4(temp1,a11,a13,a14,a31,a33,a34,
      a41,a43,a44,a51,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 13: minor3(k,l,m,3,4,0)
   deter3(temp1,a33,a34,a43,a44,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 14: -minor4(i,k,l,m,2,3,4,0)
   deter4(temp1,a12,a13,a14,a32,a33,a34,
      a42,a43,a44,a52,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 15: -minor4(i,j,l,m,1,2,3,0)
   deter4(temp1,a11,a12,a13,a21,a22,a23,
      a41,a42,a43,a51,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 16: -minor3(j,l,m,1,2,0)
   deter3(temp1,a21,a22,a41,a42,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 17: minor3(j,l,m,1,3,0)
   deter3(temp1,a21,a23,a41,a43,a51,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 18: -minor3(j,l,m,2,3,0)
   deter3(temp1,a22,a23,a42,a43,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 19: minor3(i,l,m,1,2,0)
   deter3(temp1,a11,a12,a41,a42,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 20: -minor2(l,m,1,0)
   deter2(temp1,a41,a51,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 21: minor2(l,m,2,0)
   deter2(temp1,a42,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 22: -minor3(i,l,m,1,3,0)
   deter3(temp1,a11,a13,a41,a43,a51,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 23: -minor2(l,m,3,0)
   deter2(temp1,a43,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 24: minor3(i,l,m,2,3,0)
   deter3(temp1,a12,a13,a42,a43,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 25: minor4(i,j,l,m,1,2,4,0)
   deter4(temp1,a11,a12,a14,a21,a22,a24,
      a41,a42,a44,a51,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 26: -minor3(j,l,m,1,4,0)
   deter3(temp1,a21,a24,a41,a44,a51,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 27: minor3(j,l,m,2,4,0)
   deter3(temp1,a22,a24,a42,a44,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 28: minor3(i,l,m,1,4,0)
   deter3(temp1,a11,a14,a41,a44,a51,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 29: minor2(l,m,4,0)
   deter2(temp1,a44,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 30: -minor3(i,l,m,2,4,0)
   deter3(temp1,a12,a14,a42,a44,a52,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 31: -minor4(i,j,l,m,1,3,4,0)
   deter4(temp1,a11,a13,a14,a21,a23,a24,
      a41,a43,a44,a51,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 32: -minor3(j,l,m,3,4,0)
   deter3(temp1,a23,a24,a43,a44,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 33: minor3(i,l,m,3,4,0)
   deter3(temp1,a13,a14,a43,a44,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 34: minor4(i,j,l,m,2,3,4,0)
   deter4(temp1,a12,a13,a14,a22,a23,a24,
      a42,a43,a44,a52,a53,a54,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 35: minor4(i,j,k,m,1,2,3,0)
   deter4(temp1,a11,a12,a13,a21,a22,a23,
      a31,a32,a33,a51,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 36: minor3(j,k,m,1,2,0)
   deter3(temp1,a21,a22,a31,a32,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 37: -minor3(j,k,m,1,3,0)
   deter3(temp1,a21,a23,a31,a33,a51,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 38: minor3(j,k,m,2,3,0)
   deter3(temp1,a22,a23,a32,a33,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 39: -minor3(i,k,m,1,2,0)
   deter3(temp1,a11,a12,a31,a32,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 40: minor2(k,m,1,0)
   deter2(temp1,a31,a51,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 41: -minor2(k,m,2,0)
   deter2(temp1,a32,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 42: minor3(i,k,m,1,3,0)
   deter3(temp1,a11,a13,a31,a33,a51,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 43: minor2(k,m,3,0)
   deter2(temp1,a33,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 44: -minor3(i,k,m,2,3,0)
   deter3(temp1,a12,a13,a32,a33,a52,a53,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 45: minor3(i,j,m,1,2,0)
   deter3(temp1,a11,a12,a21,a22,a51,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 46: -minor2(j,m,1,0)
   deter2(temp1,a21,a51,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = -icomp;
      return;
   }

   // term 47: minor2(j,m,2,0)
   deter2(temp1,a22,a52,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 48: minor2(i,m,1,0)
   deter2(temp1,a11,a51,eps);
   icomp = sgn(temp1);
   if (icomp != 0) {
      res = icomp;
      return;
   }

   // term 49: 1
   res = 1;
}

// minor4:
// This subroutine tests the sign of the determinant
// D = | a11 a12 a13 1 |
//     | a21 a22 a23 1 |
//     | a31 a32 a33 1 |
//     | a41 a42 a43 1 |
// and return 1 if positive, -1 if negative, 0 otherwise
void SOS::minor4_nmp(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps)
{
   int icomp;

   double a11 = coord_a[0];
   double a12 = coord_a[1];
   double a13 = coord_a[2];
   double a21 = coord_b[0];
   double a22 = coord_b[1];
   double a23 = coord_b[2];
   double a31 = coord_c[0];
   double a32 = coord_c[1];
   double a33 = coord_c[2];
   double a41 = coord_d[0];
   double a42 = coord_d[1];
   double a43 = coord_d[2];

   // compute determinant
   double temp1;
   deter4(temp1,a11,a12,a13,a21,a22,a23,a31,a32,a33,a41,a42,a43,eps);

   icomp = sgn(temp1);

   res = icomp;
}
}

namespace tinker
{
template <bool use_sos>
void SOS::sos_minor2(double a11, double a21, int& res)
{
   if CONSTEXPR (use_sos) sos_minor2_gmp(a11, a21, res);
   else sos_minor2_nmp(a11, a21, res);
}

template <bool use_sos>
void SOS::sos_minor3(double a11, double a12, double a21, double a22, double a31, double a32, int& res)
{
   if CONSTEXPR (use_sos) sos_minor3_gmp(a11, a12, a21, a22, a31, a32, res);
   else sos_minor3_nmp(a11, a12, a21, a22, a31, a32, res);
}

template <bool use_sos>
void SOS::sos_minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res)
{
   if CONSTEXPR (use_sos) sos_minor4_gmp(coord_a, coord_b, coord_c, coord_d, res);
   else sos_minor4_nmp(coord_a, coord_b, coord_c, coord_d, res);
}

template <bool use_sos>
void SOS::sos_minor5(double* coord_a, double r1, double* coord_b, double r2, double* coord_c, double r3, double* coord_d, double r4, double* coord_e, double r5, int& res)
{
   if CONSTEXPR (use_sos) sos_minor5_gmp(coord_a, r1, coord_b, r2, coord_c, r3, coord_d, r4, coord_e, r5, res);
   else sos_minor5_nmp(coord_a, r1, coord_b, r2, coord_c, r3, coord_d, r4, coord_e, r5, res);
}

template <bool use_sos>
void SOS::minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res)
{
   if CONSTEXPR (use_sos) minor4_gmp(coord_a, coord_b, coord_c, coord_d, res);
   else minor4_nmp(coord_a, coord_b, coord_c, coord_d, res);
}

// explicit instatiation
template void SOS::sos_minor2<true>(double a11, double a21, int& res);
template void SOS::sos_minor3<true>(double a11, double a12, double a21, double a22, double a31, double a32, int& res);
template void SOS::sos_minor4<true>(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);
template void SOS::sos_minor5<true>(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
   double r3, double* coord_d, double r4, double* coord_e, double r5, int& res);
template void SOS::minor4<true>(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);

template void SOS::sos_minor2<false>(double a11, double a21, int& res);
template void SOS::sos_minor3<false>(double a11, double a12, double a21, double a22, double a31, double a32, int& res);
template void SOS::sos_minor4<false>(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);
template void SOS::sos_minor5<false>(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
   double r3, double* coord_d, double r4, double* coord_e, double r5, int& res);
template void SOS::minor4<false>(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res);
}