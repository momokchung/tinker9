#include "ff/solv/alphamol.h"

namespace tinker
{
inline void edgeradius(double* a, double* b, double ra, double rb, double* Dab, double* Sab, double* Tab, int& testr, double alpha);
inline void edgeattach(double* a, double* b, double* c, double ra, double rb, double rc, double* Dab, double* Sab, double* Tab, int& testa);

// "alfedge" checks if the edge belongs to the alpha complex
void alfedge(double* a, double* b, double ra, double rb, 
   double* cg, std::vector<int>& listcheck, int& irad, int& iattach, double alpha)
{
   double Dab[4], Sab[3], Tab[3];

   iattach = 1;
   irad = 0;

   double val = a[3] + b[3] - 2*(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+ra*rb);
   if (val > 0) return;

   // 1. Compute all Minors Dab(i) = M(a,b,i,0) = Det | a(i) 1 |
   //                                                 | b(i) 1 |
   //    for all i in [1,4]
   for (int i = 0; i < 4; i++) {
      Dab[i] = a[i] - b[i];
   }

   // 2. Computes all Minors Sab(i,j)= M(a,b,i,j) = Det | a(i)  a(j) |
   //                                                   | b(i)  b(j) |
   for (int i = 0; i < 2; i++) {
      for (int j = i+1; j < 3; j++) {
         int k = i+j-1;
         Sab[k] = a[i]*b[j] - b[i]*a[j];
      }
   }

   // 3. Computes all Minors Tab(i)= M(a,b,i,4) = Det | a(i)  a(4) |
   //                                                 | b(i)  b(4) |
   for (int i = 0; i < 3; i++) {
      Tab[i] = a[i]*b[3] - b[i]*a[3];
   }

   // first check attachment
   int ic;
   int attach;
   int ncheck = listcheck.size();
   double c[4];

   for (int i = 0; i < ncheck; i++) {
      ic = listcheck[i];

      for (int j = 0; j < 3; j++) {
         c[j] = vertices[ic].coord[j] - cg[j];
      }
      double rc = vertices[ic].r;
      c[3] = c[0]*c[0] + c[1]*c[1] + c[2]*c[2] - rc*rc;

      edgeattach(a, b, c, ra, rb, rc, Dab, Sab, Tab, attach);

      if (attach==1) return;
   }

   iattach = 0;

   // edge is not attached; check radius

   int rad;
   edgeradius(a, b, ra, rb, Dab, Sab, Tab, rad, alpha);

   if (rad==1) irad = 1;
}

// edge_radius computes the radius of the smallest
// circumsphere to an edge, and compares it to alpha.
inline void edgeradius(double* a, double* b, double ra, double rb,
   double* Dab, double* Sab, double* Tab, int& testr, double alpha)
{
   double res[4][4]={0};
   testr = 0;

   // Formulas have been derived by projection on 4D space,
   // which requires some precaution when some coordinates are
   // equal.

   res[0][3] = Dab[3];

   if (a[0] != b[0]) {
      for (int i = 0; i < 3; i++) {
         res[0][i] = Dab[i];
         res[i+1][3] = Tab[i];
      }
      res[1][1] = Sab[0];
      res[1][2] = Sab[1];
      res[2][2] = Sab[2];
   }
   else if (a[1] != b[1]) {
      res[0][0] = Dab[1];
      res[0][1] = Dab[2];
      res[0][2] = Dab[0];
      res[1][1] = Sab[2];
      res[1][2] = -Sab[0];
      res[2][2] = -Sab[1];
      res[1][3] = Tab[1];
      res[2][3] = Tab[2];
      res[3][3] = Tab[0];
   }
   else if (a[2] != b[2]) {
      res[0][0] = Dab[2];
      res[0][1] = Dab[0];
      res[0][2] = Dab[1];
      res[1][1] = -Sab[1];
      res[1][2] = -Sab[2];
      res[2][2] = Sab[0];
      res[1][3] = Tab[2];
      res[2][3] = Tab[0];
      res[3][3] = Tab[1];
   }

   double r_11 = res[0][0]*res[0][0];
   double r_22 = res[0][1]*res[0][1];
   double r_33 = res[0][2]*res[0][2];
   double r_14 = res[0][0]*res[0][3];
   double r_313 = res[0][2]*res[1][2];
   double r_212 = res[0][1]*res[1][1];
   double diff = res[0][2]*res[1][1] - res[0][1]*res[1][2];

   // first compute radius of circumsphere

   double d0 = -2*res[0][0]*(r_11+r_22+r_33);
   double d1 = res[0][0]*(2*(r_313 + r_212)-r_14);
   double d2 = -2*res[1][1]*(r_11+r_33) - res[0][1]*(r_14-2*r_313);
   double d3 = -2*res[1][2]*(r_11+r_22) -res[0][2]*(r_14-2*r_212);
   double d4 = 2*res[0][0]*(res[0][0]*res[1][3]+res[0][1]*res[2][3]+
      res[0][2]*res[3][3]) +4*(res[2][2]*diff
      - res[0][0]*(res[1][1]*res[1][1]+res[1][2]*res[1][2]));

   double num = d1*d1 + d2*d2 + d3*d3 - d0*d4;
   double den = d0*d0;

   // For efficiency purpose, I assume that this routine is only used to compute
   // the dual complex (i.e. alpha=0), and therefore I do not consider the denominator as
   // it is always positive)

   double rho2 = num/den;
   rho2 = num;

   if (alpha > rho2) testr = 1;

   return;
}

// edge_attach checks if an edge ab of a tetrahedron
// is "attached" to a given vertex c
inline void edgeattach(double* a, double* b, double* c, double ra, double rb,
   double rc, double* Dab, double* Sab, double* Tab, int& testa)
{
   testa = 0;

   // Need to compute:
   // Sc : minor(a,b,c,i,j,0) for i=1,2 and j = i+1,3
   // Tc : minor(a,b,c,i,4,0) for i = 1,2,3

   double Sc[3], Tc[3];
   for (int i = 0; i < 2; i++) {
      for (int j = i+1; j < 3; j++) {
         int k = i+j-1;
         Sc[k] = c[i]*Dab[j] - c[j]*Dab[i] + Sab[k];
      }
   }

   for (int i = 0; i < 3; i++) {
      Tc[i] = c[i]*Dab[3] - c[3]*Dab[i] + Tab[i];
   }

   // Formulas have been derived by projection on 4D space,
   // which requires some precaution when some coordinates are
   // equal.

   double res[4][4];
   double res2_c[4][4];
   if (a[0] != b[0]) {
      for (int i = 0; i < 3; i++) {
         res[0][i] = Dab[i];
         res2_c[i+1][3] = Tc[i];
      }
      res[1][1] = Sab[0];
      res[1][2] = Sab[1];
      res[2][2] = Sab[2];
      res2_c[1][1] = Sc[0];
      res2_c[1][2] = Sc[1];
      res2_c[2][2] = Sc[2];
   }
   else if (a[1] != b[1]) {
      res[0][0] = Dab[1];
      res[0][1] = Dab[2];
      res[0][2] = Dab[0];
      res[1][1] = Sab[2];
      res[1][2] = -Sab[0];
      res[2][2] = -Sab[1];
      res2_c[1][1] = Sc[2];
      res2_c[1][2] = -Sc[0];
      res2_c[2][2] = -Sc[1];
      res2_c[1][3] = Tc[1];
      res2_c[2][3] = Tc[2];
      res2_c[3][3] = Tc[0];
   }
   else if (a[2] != b[2]) {
      res[0][0] = Dab[2];
      res[0][1] = Dab[0];
      res[0][2] = Dab[1];
      res[1][1] = -Sab[1];
      res[1][2] = -Sab[2];
      res[2][2] = Sab[0];
      res2_c[1][1] = -Sc[1];
      res2_c[1][2] = -Sc[2];
      res2_c[2][2] = Sc[0];
      res2_c[1][3] = Tc[2];
      res2_c[2][3] = Tc[0];
      res2_c[3][3] = Tc[1];
   }

   double r_11 = res[0][0]*res[0][0];
   double r_22 = res[0][1]*res[0][1];
   double r_33 = res[0][2]*res[0][2];
   double diff = res[0][2]*res[1][1] - res[0][1]*res[1][2];

   // check attachement with vertex C

   double d0 = -2*res[0][0]*(r_11+r_22+r_33);

   double d5 = res[0][0]*(res[0][0]*res2_c[1][3]+res[0][1]*res2_c[2][3]
      +res[0][2]*res2_c[3][3] - 2*(res[1][2]*res2_c[1][2]
      +res[1][1]*res2_c[1][1]))+2*res2_c[2][2]*diff;

   double dtest = d0*d5;

   // if no problem, set testa to true if t < 0

   if (dtest < 0) testa = 1;

   return;
}
}
