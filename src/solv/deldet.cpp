#include "ff/solv/alffunc.h"

namespace tinker
{
inline void deter2(double& deter, double b11, double b21, double eps)
{
   deter = psub(b11,b21);

   if (std::abs(deter) < eps) deter = 0.;
}

// deter3 evaluates the determinant:
// D = | b11 b12 1 |
//     | b21 b22 1 |
//     | b31 b32 1 |
inline void deter3(double& deter, double b11, double b12, double b21, 
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
inline void deter4(double& deter, double b11, double b12, double b13, double b21, 
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
inline void deter5(double& deter, double b11, double b12, double b13, double b14, 
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

// minor2 tests the sign of the determinant:
// D = | a11 1 |
//     | a21 1 |
void minor2(double a11, double a21, int& res, double eps)
{
   int icomp;

   // compute determinant
   double temp1;
   deter2(temp1,a11,a21,eps);

   icomp = sgn(temp1);

   if (icomp != 0) res = icomp;
   else res = 1;
}

// minor3 tests the sign of the determinant:
// D = | a11 a12 1 |
//     | a21 a22 1 |
//     | a31 a32 1 |
void minor3(double a11, double a12, double a21, double a22, double a31, double a32, int& res, double eps)
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

// minor4 tests the sign of the determinant:
// D = | a11 a12 a13 1 |
//     | a21 a22 a23 1 |
//     | a31 a32 a33 1 |
//     | a41 a42 a43 1 |
void minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps)
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


// minor5 tests the sign of the determinant:
// D = | a11 a12 a13 a14 1 |
//     | a21 a22 a23 a24 1 |
//     | a31 a32 a33 a34 1 |
//     | a41 a42 a43 a44 1 |
//     | a51 a52 a53 a54 1 |
void minor5(double* coord_a, double r1, double* coord_b, double r2, 
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

   buildweight(a11,a12,a13,r1,a14);
   buildweight(a21,a22,a23,r2,a24);
   buildweight(a31,a32,a33,r3,a34);
   buildweight(a41,a42,a43,r4,a44);
   buildweight(a51,a52,a53,r5,a54);

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
}
