#pragma once
#include "ff/solv/alphamol.h"
#include "ff/solv/tetra.h"
#include "math/const.h"
#include <cmath>

namespace tinker
{
inline double psub(double r1, double r2)
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

inline double padd(double r1, double r2)
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

// builds the weight of a point: w = x**2 + y**2 + z**2 - ra**2
inline void buildweight(double ax, double ay, double az, double r, double& w)
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

inline int sgn(double d)
{
   if (d == 0.) return 0;
   else if (d > 0.) return 1;
   else return -1;
}

// sign associated with the missing infinite point
inline void missinf_sign(int i, int j, int k, int& l, int& sign)
{
   int a, b, c, d;
   l = 6 - i - j - k;
   a = i;
   b = j;
   c = k;
   sign = 1;
   if (a > b) {
      d = a;
      a = b;
      b = d;
      sign = -sign;
   }
   if (a > c) {
      d = a;
      a = c;
      c = d;
      sign = -sign;
   }
   if (b > c) {
      sign = -sign;
   }
}

inline void alftetra(double* a, double* b, double* c, double* d,
   double ra, double rb, double rc, double rd, int& iflag, double alpha)
{
   double D1, D2, D3, D4, Det;
   double Dabc, Dabd, Dacd, Dbcd;
   double num,den;
   double test,val;
   double Sab[3], Sac[3], Sad[3], Sbc[3], Sbd[3], Scd[3];
   double Sa[3], Sb[3], Sc[3], Sd[3];
   double Deter[3];

   iflag = 0;
   val = a[3]+b[3] -2*(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+ra*rb);
   if (val > 0) return;
   val = a[3]+c[3] -2*(a[0]*c[0]+a[1]*c[1]+a[2]*c[2]+ra*rc);
   if (val > 0) return;
   val = a[3]+d[3] -2*(a[0]*d[0]+a[1]*d[1]+a[2]*d[2]+ra*rd);
   if (val > 0) return;
   val = b[3]+c[3] -2*(b[0]*c[0]+b[1]*c[1]+b[2]*c[2]+rb*rc);
   if (val > 0) return;
   val = b[3]+d[3] -2*(b[0]*d[0]+b[1]*d[1]+b[2]*d[2]+rb*rd);
   if (val > 0) return;
   val = c[3]+d[3] -2*(c[0]*d[0]+c[1]*d[1]+c[2]*d[2]+rc*rd);
   if (val > 0) return;

   // Perform computation in floating points; if a problem occurs, switch precision

   // 1. Computes all Minors Smn(i+j-2)= M(m,n,i,j) = Det | m(i)  m(j) |
   //                                                     | n(i)  n(j) |
   //    for all i in [0,1] and all j in [i+1,2]

   for (int i = 0; i < 2; i++) {
      for (int j = i+1; j < 3; j++) {
         int k = i+j-1;
         Sab[k] = a[i]*b[j]-a[j]*b[i];
         Sac[k] = a[i]*c[j]-a[j]*c[i];
         Sad[k] = a[i]*d[j]-a[j]*d[i];
         Sbc[k] = b[i]*c[j]-b[j]*c[i];
         Sbd[k] = b[i]*d[j]-b[j]*d[i];
         Scd[k] = c[i]*d[j]-c[j]*d[i];
      }
   }

   // Now compute all Minors 
   //     Sq(i+j-2) = M(m,n,p,i,j,0) = Det | m(i) m(j) 1 |
   //                                      | n(i) n(j) 1 |
   //                                      | p(i) p(j) 1 |

   // and all Minors
   //     Det(i+j-2) = M(m,n,p,q,i,j,4,0) = Det | m(i) m(j) m(4) 1 |
   //                                           | n(i) n(j) n(4) 1 |
   //                                           | p(i) p(j) p(4) 1 |
   //                                           | q(i) q(j) q(4) 1 |

   // m,n,p,q are the four vertices of the tetrahedron, i and j correspond
   // to two of the coordinates of the vertices, and m(4) refers to the
   // "weight" of vertices m

   for (int i = 0; i < 3; i++) {
      Sa[i] = Scd[i] - Sbd[i] + Sbc[i];
      Sb[i] = Scd[i] - Sad[i] + Sac[i];
      Sc[i] = Sbd[i] - Sad[i] + Sab[i];
      Sd[i] = Sbc[i] - Sac[i] + Sab[i];
   }

   for (int i = 0; i < 3; i++) {
      Deter[i] = a[3]*Sa[i]-b[3]*Sb[i]+c[3]*Sc[i]-d[3]*Sd[i];
   }

   // Now compute the determinant needed to compute the radius of the
   // sphere orthogonal to the four balls that define the tetrahedron :
   //     D1 = Minor(a,b,c,d,4,2,3,0)
   //     D2 = Minor(a,b,c,d,1,3,4,0)
   //     D3 = Minor(a,b,c,d,1,2,4,0)
   //     D4 = Minor(a,b,c,d,1,2,3,0)

   D1 = Deter[2];
   D2 = Deter[1];
   D3 = Deter[0];
   D4 = a[0]*Sa[2]-b[0]*Sb[2]+c[0]*Sc[2]-d[0]*Sd[2];

   // Now compute all minors:
   //     Dmnp = Minor(m,n,p,1,2,3) = Det | m(1) m(2) m(3) |
   //                                     | n(1) n(2) n(3) |
   //                                     | p(1) p(2) p(3) |

   Dabc = a[0]*Sbc[2]-b[0]*Sac[2] + c[0]*Sab[2];
   Dabd = a[0]*Sbd[2]-b[0]*Sad[2] + d[0]*Sab[2];
   Dacd = a[0]*Scd[2]-c[0]*Sad[2] + d[0]*Sac[2];
   Dbcd = b[0]*Scd[2]-c[0]*Sbd[2] + d[0]*Sbc[2];

   // We also need :
   //     Det = Det | m(1) m(2) m(3) m(4) |
   //               | n(1) n(2) n(3) n(4) |
   //               | p(1) p(2) p(3) p(4) |
   //               | q(1) q(2) q(3) q(4) |

   Det = -a[3]*Dbcd + b[3]*Dacd -c[3]*Dabd + d[3]*Dabc;

   // the radius of the circumsphere of the weighted tetrahedron is then:

   num = D1*D1 + D2*D2 + D3*D3 + 4*D4*Det;
   den = 4*D4*D4;

   // if this radius is too close to the value of ALPHA, we switch precision
   test = alpha*den - num;
   // int itest;
   // if (REAL_ABS(test) < alfeps) {
   //     tetrad(a, b, c, d, ra, rb, rc, rd, itest, alpha);
   //     test = itest;
   // }

   // The spectrum for a tetrahedron is [R_t Infinity[. If ALPHA is in
   // that interval, the tetrahedron is part of the alpha shape, otherwise
   // it is discarded
   // If tetrahedron is part of the alpha shape, then the 4 triangles,
   // the 6 edges and the four vertices are also part of the alpha
   // complex

   iflag = 0;
   if (test > 0) iflag = 1;
}

// triattach checks if triangles are attached
inline void triattach(double* a, double* b, double* c, double* d,
   double ra, double rb, double rc, double rd, double S[3][4], double T[2][3],
   double Dabc, int& testa, int& memory)
{
   testa = 0;

   // We need to compute:
   // Det1 = Minor(a,b,c,d,2,3,4,0)
   // Det2 = Minor(a,b,c,d,1,3,4,0)
   // Det3 = Minor(a,b,c,d,1,2,4,0)
   // Deter= Minor(a,b,c,d,1,2,3,0)

   double Det1 = -d[1]*S[2][3] + d[2]*S[1][3] - d[3]*S[1][2] + T[1][2];
   double Det2 = -d[0]*S[2][3] + d[2]*S[0][3] - d[3]*S[0][2] + T[0][2];
   double Det3 = -d[0]*S[1][3] + d[1]*S[0][3] - d[3]*S[0][1] + T[0][1];
   double Deter = -d[0]*S[1][2] + d[1]*S[0][2] - d[2]*S[0][1] + Dabc;

   // Check if the face is "attached" to the fourth
   // vertex of the parent tetrahedron

   double test = Det1*S[1][2]+Det2*S[0][2]+Det3*S[0][1]-2*Deter*Dabc;

   // check for problems, in which case change precision

   // if (REAL_ABS(test) < alfeps) {
   //     test = 0.;
   //     memory = 1;
   // }

   // if no problem, set testa to true if test > 0
   if (test > 0) testa = 1;
   return;
}

// triradius computes the radius of the
// smallest circumsphere to a triangle
inline void triradius(double* a, double* b, double* c, double ra, double rb,
   double rc, double S[3][4], double T[2][3], double Dabc, int& testr, double alpha, int& memory)
{
   testr = 0;
   double sums2 = S[0][1]*S[0][1] + S[0][2]*S[0][2] + S[1][2]*S[1][2];
   double d0 = sums2;
   double d1 = S[0][2]*S[2][3] + S[0][1]*S[1][3] - 2*Dabc*S[1][2];
   double d2 = S[0][1]*S[0][3] - S[1][2]*S[2][3] - 2*Dabc*S[0][2];
   double d3 = S[1][2]*S[1][3] + S[0][2]*S[0][3] + 2*Dabc*S[0][1];
   double d4 = S[0][1]*T[0][1] + S[0][2]*T[0][2] + S[1][2]*T[1][2] - 2*Dabc*Dabc;
   double num  = 4*(d1*d1+d2*d2+d3*d3) + 16*d0*d4;

   // if (REAL_ABS(alpha-num) < alfeps) {
   //     num = alpha;
   //     memory = 1;
   // }

   if (alpha > num) testr = 1;
   return;
}

inline void alftrig(double* a, double* b, double* c, double* d, double* e,
   double ra,double rb, double rc, double rd, double re, int ie, 
   int& irad,int& iattach, double alpha)
{
   double Dabc;
   double Sab[3][4],Sac[3][4],Sbc[3][4];
   double S[3][4],T[2][3];

   iattach = 0;
   irad = 0;

   double val = a[3]+b[3] -2*(a[0]*b[0]+a[1]*b[1]+a[2]*b[2]+ra*rb);
   if (val > 0) return;
   val = a[3]+c[3] -2*(a[0]*c[0]+a[1]*c[1]+a[2]*c[2]+ra*rc);
   if (val > 0) return;
   val = b[3]+c[3] -2*(b[0]*c[0]+b[1]*c[1]+b[2]*c[2]+rb*rc);
   if (val > 0) return;

   // Perform computation in floating points; if a problem occurs,
   // switch precision

   // 1. Computes all Minors Smn(i,j) = M(m,n,i,j) = Det | m(i)  m(j) |
   //                                                    | n(i)  n(j) |
   // m,n are two vertices of the triangle, i and j correspond
   // to two of the coordinates of the vertices

   // for all i in [0,2] and all j in [i+1,3]
   for (int i = 0; i < 3; i++) {
      for (int j = i+1; j < 4; j++) {
         Sab[i][j] = a[i]*b[j]-a[j]*b[i];
         Sac[i][j] = a[i]*c[j]-a[j]*c[i];
         Sbc[i][j] = b[i]*c[j]-b[j]*c[i];
      }
   }

   // Now compute all Minors 
   //     S(i,j) = M(a,b,c,i,j,0) = Det | a(i) a(j) 1 |
   //                                   | b(i) b(j) 1 |
   //                                   | c(i) c(j) 1 |
   // a,b,c are the 3 vertices of the triangle, i and j correspond
   // to two of the coordinates of the vertices

   // for all i in [0,2] and all j in [i+1,3]
   for (int i = 0; i < 3; i++) {
      for (int j = i+1; j < 4; j++) {
         S[i][j] = Sbc[i][j] - Sac[i][j] + Sab[i][j];
      }
   }

   // Now compute all Minors
   //     T(i,j) = M(a,b,c,i,j,4) = Det | a(i) a(j) a(4) |
   //                                   | b(i) b(j) b(4) |
   //                                   | c(i) c(j) c(4) |

   // for all i in [0,1] and all j in [i+1,2]
   for (int i = 0; i < 2; i++) {
      for (int j = i+1; j < 3; j++) {
         T[i][j] = a[3]*Sbc[i][j] - b[3]*Sac[i][j] + c[3]*Sab[i][j];
      }
   }

   // Finally,  need Dabc = M(a,b,c,1,2,3) = Det | a(1) a(2) a(3) |
   //                                            | b(2) b(2) b(3) |
   //                                            | c(3) c(2) c(3) |

   Dabc = a[0]*Sbc[1][2] - b[0]*Sac[1][2] + c[0]*Sab[1][2];

   // first check if a,b,c attached to d:
   int memory = 0;
   int attach;
   triattach(a, b, c, d, ra, rb, rc, rd, S, T, Dabc, attach, memory);

   // If attached, we can stop there, the triangle will 
   // not be part of the alpha complex
   if (attach == 1) {
      iattach = 1;
      return;
   }

   // if e exists, check if a,b,c attached to e:
   if (ie >= 0) {
      triattach(a, b, c, e, ra, rb, rc, re, S, T, Dabc, attach, memory);

      // If attached, we can stop there, the triangle will
      // not be part of the alpha complex
      if (attach == 1) {
         iattach = 1;
         return;
      }
   }

   // Now check if alpha is bigger than the radius of the sphere orthogonal
   // to the three balls at A, B, C:

   int testr;
   triradius(a, b, c, ra, rb, rc, S, T, Dabc, testr, alpha, memory);

   if (testr == 1) irad = 1;
}

// "findedge" finds the index of the edge
// given two vertices of a tetrahedron

inline int findedge(Tetrahedron t, int i1, int j1)
{
   int ipair;

   if (i1==t.vertices[0]) {
      if (j1==t.vertices[1]) ipair = 5;
      else if (j1==t.vertices[2]) ipair = 4;
      else ipair = 3;
   }
   else if (i1==t.vertices[1]) {
      if (j1==t.vertices[2]) ipair = 2;
      else ipair = 1;
   }
   else {
      ipair = 0;
   }
   return ipair;
}

inline double dist2(std::vector<Vertex>& vertices, int n1, int n2)
{
   double x;
   double dist = 0;
   for(int i = 0; i < 3; i++) {
      x = vertices[n1].coord[i] - vertices[n2].coord[i];
      dist += x*x;
   }

   return dist;
}

// "twosph" calculates the surface area and volume of the
// intersection of two spheres
inline void twosph(double ra, double ra2, double rb, double rb2,
   double rab, double rab2, double& surfa, double& surfb,
   double& vola, double& volb, double& r, double& phi)
{
   double cosine,vala,valb,lambda,ha,hb;
   double Aab,sa,ca,sb,cb;
   constexpr double twopi = 2 * pi;

   // Get distance between center of sphere A and Voronoi plane
   // between A and B
   lambda = plane_dist(ra2, rb2, rab2);
   valb = lambda*rab;
   vala = rab-valb;

   // Get height of the cap of sphere A occluded by sphere B
   ha = ra - vala;

   // same for sphere B ...
   hb = rb - valb;

   // get surfaces of intersection
   surfa = twopi*ra*ha;
   surfb = twopi*rb*hb;

   // now get volume
   Aab = pi*(ra2-vala*vala);

   sa = ra*(surfa);
   ca = vala*Aab;

   vola = (sa-ca)/3;

   sb = rb*(surfb);
   cb = valb*Aab;

   volb = (sb-cb)/3;

   // get radius of the circle of intersection between the two spheres
   r = std::sqrt(ra2 - vala*vala);

   // get angle between normals of the sphere at a point on this circle
   cosine = (ra2+rb2-rab2)/(2.0*ra*rb);
   phi = std::acos(cosine);
}

// "twosphder" calculates the surface area and volume derivatives
// of the intersection of two spheres
inline void twosphder(double ra, double ra2, double rb, double rb2, double rab, double rab2,
   double& surfa, double& surfb, double& vola, double& volb, double& r, double& phi,
   double& dsurfa, double& dsurfb, double& dvola, double& dvolb, double& dr, double& dphi, bool compder)
{
   double cosine,vala,valb,lambda,ha,hb;
   double Aab,sa,ca,sb,cb;
   double dera,derb;
   constexpr double twopi = 2 * pi;

   // Get distance between center of sphere A and Voronoi plane
   // between A and B
   lambda = plane_dist(ra2, rb2, rab2);
   valb = lambda*rab;
   vala = rab-valb;

   // get height of the cap of sphere A occluded by sphere B
   ha = ra - vala;

   // same for sphere B ...
   hb = rb - valb;

   // get surfaces of intersection
   surfa = twopi*ra*ha;
   surfb = twopi*rb*hb;

   // now get volume
   Aab = pi*(ra2-vala*vala);

   sa = ra*(surfa);
   ca = vala*Aab;

   vola = (sa-ca)/3;

   sb = rb*(surfb);
   cb = valb*Aab;

   volb = (sb-cb)/3;

   // get radius of the circle of intersection between the two spheres
   r = std::sqrt(ra2 - vala*vala);

   // get angle between normals of the sphere at a point on this circle
   cosine = (ra2+rb2-rab2)/(2.0*ra*rb);
   phi = std::acos(cosine);

   if (!compder) return;

   dera = - lambda;
   derb = lambda - 1;

   dsurfa = twopi*ra*dera;
   dsurfb = twopi*rb*derb;

   dvola = -Aab*lambda;
   dvolb = -(dvola) - Aab;

   dr   = -vala*lambda/(r);
   dphi = rab/(ra*rb*std::sqrt(1-cosine*cosine));
}

// "threesphder" calculates the surface area and volume derivatives
// of the intersection of three spheres
inline void threesphder(double ra, double rb,double rc, double ra2,
   double rb2, double rc2, double rab, double rac, double rbc,
   double rab2, double rac2, double rbc2, double *angle,
   double& surfa, double& surfb, double& surfc, double& vola, double& volb, double& volc,
   double* dsurfa, double* dsurfb, double* dsurfc, double* dvola, double* dvolb, double* dvolc, bool compder)
{
   double a1,a2,a3,s2,c1,c2;
   double seg_ang_ab,seg_ang_ac,seg_ang_bc;
   double ang_dih_ap,ang_dih_bp,ang_dih_cp;
   double val1,val2,val3,l1,l2,l3;
   double val1b,val2b,val3b;
   double ang_abc,ang_acb,ang_bca;
   double cos_abc,cos_acb,cos_bca;
   double sin_abc,sin_acb,sin_bca;
   double s_abc,s_acb,s_bca;
   double rho_ab2,rho_ac2,rho_bc2;
   double drho_ab2,drho_ac2,drho_bc2;
   double val_abc,val_acb,val_bca;
   double val2_abc,val2_acb,val2_bca;
   double der_val1b,der_val1,der_val2b,der_val2,der_val3b,der_val3;
   double cosine[6],sine[6],deriv[6][3];
   constexpr double twopi = 2 * pi;

   l1 = plane_dist(ra2, rb2, rab2);
   l2 = plane_dist(ra2, rc2, rac2);
   l3 = plane_dist(rb2, rc2, rbc2);

   val1 = l1*rab; val2 = l2*rac; val3 = l3*rbc;
   val1b = rab - val1; val2b = rac - val2; val3b = rbc - val3;

   // We consider the tetrahedron (A,B,C,P) where P is the
   // point of intersection of the three spheres such that (A,B,C,P) is ccw.
   // The edge lengths in this tetrahedron are: rab, rac, rAP=ra, rbc, rBP=rb, rCP=rc

   tetdihedder3(rab2, rac2, ra2, rbc2, rb2, rc2, angle, cosine, sine, deriv, compder);

   // the seg_ang_ are the dihedral angles around the three edges AB, AC and BC

   seg_ang_ab = angle[0];
   seg_ang_ac = angle[1];
   seg_ang_bc = angle[3];

   // the ang_dih_ are the dihedral angles around the three edges AP, BP and CP
   ang_dih_ap = angle[2];
   ang_dih_bp = angle[4];
   ang_dih_cp = angle[5];

   a1 = ra*(1-2*ang_dih_ap);
   a2 = 2*seg_ang_ab*val1b;
   a3 = 2*seg_ang_ac*val2b;

   surfa = twopi*ra*(a1 - a2 - a3);

   a1 = rb*(1-2*ang_dih_bp);
   a2 = 2*seg_ang_ab*val1;
   a3 = 2*seg_ang_bc*val3b;

   surfb = twopi*rb*(a1 - a2 - a3);

   a1 = rc*(1-2*ang_dih_cp);
   a2 = 2*seg_ang_ac*val2;
   a3 = 2*seg_ang_bc*val3;

   surfc = twopi*rc*(a1 - a2 - a3);

   // compute volumes of the three caps
   ang_abc = twopi*seg_ang_ab;
   ang_acb = twopi*seg_ang_ac;
   ang_bca = twopi*seg_ang_bc;

   cos_abc = cosine[0];
   sin_abc = sine[0];
   cos_acb = cosine[1];
   sin_acb = sine[1];
   cos_bca = cosine[3];
   sin_bca = sine[3];

   rho_ab2 = ra2 - val1b*val1b;
   rho_ac2 = ra2 - val2b*val2b;
   rho_bc2 = rb2 - val3b*val3b;

   val_abc = ang_abc - sin_abc*cos_abc; s_abc = rho_ab2*val_abc;
   val_acb = ang_acb - sin_acb*cos_acb; s_acb = rho_ac2*val_acb;
   val_bca = ang_bca - sin_bca*cos_bca; s_bca = rho_bc2*val_bca;

   s2 = ra*(surfa);
   c1 = val1b*s_abc;
   c2 = val2b*s_acb;

   vola = (s2 - c1 - c2)/3;

   s2 = rb*(surfb);
   c1 = val1*s_abc;
   c2 = val3b*s_bca;

   volb = (s2 - c1 - c2)/3;

   s2 = rc*(surfc);
   c1 = val2*s_acb;
   c2 = val3*s_bca;

   volc = (s2 - c1 - c2)/3;

   if (!compder) return;

   der_val1b = l1; der_val1  = 1-l1;
   der_val2b = l2; der_val2  = 1-l2;
   der_val3b = l3; der_val3  = 1-l3;

   dsurfa[0] = -2*ra*(
      twopi*seg_ang_ab*der_val1b +
      (ra*deriv[2][0] +
      val1b*deriv[0][0] +val2b*deriv[1][0]));
   dsurfa[1] = -2*ra*(
      twopi*seg_ang_ac*der_val2b +
      (ra*deriv[2][1] +
      val1b*deriv[0][1] +val2b*deriv[1][1]));
   dsurfa[2] = -2*ra*( ra*deriv[2][2] +
      val1b*deriv[0][2]+val2b*deriv[1][2]);

   dsurfb[0] = -2*rb*(
      twopi*seg_ang_ab*der_val1
      +(rb*deriv[4][0]+
      val1*deriv[0][0]+val3b*deriv[3][0]));
   dsurfb[1] = -2*rb*(rb*deriv[4][1]+
      val1*deriv[0][1]+val3b*deriv[3][1]);
   dsurfb[2] = -2*rb*(
      twopi*seg_ang_bc*der_val3b
      +(rb*deriv[4][2]+
      val1*deriv[0][2]+val3b*deriv[3][2]));

   dsurfc[0] = -2*rc*(rc*deriv[5][0]+
         val2*deriv[1][0]+val3*deriv[3][0]);
   dsurfc[1] = -2*rc*(
      twopi*seg_ang_ac*der_val2
      +(rc*deriv[5][1]+
      val2*deriv[1][1]+val3*deriv[3][1]));
   dsurfc[2] = -2*rc*(
      twopi*seg_ang_bc*der_val3
      +(rc*deriv[5][2]+
      val2*deriv[1][2]+val3*deriv[3][2]));

   drho_ab2 = -2*der_val1b*val1b;
   drho_ac2 = -2*der_val2b*val2b;
   drho_bc2 = -2*der_val3b*val3b;

   val2_abc = rho_ab2*(1 - cos_abc*cos_abc + sin_abc*sin_abc);
   val2_acb = rho_ac2*(1 - cos_acb*cos_acb + sin_acb*sin_acb);
   val2_bca = rho_bc2*(1 - cos_bca*cos_bca + sin_bca*sin_bca);

   dvola[0] = ra*dsurfa[0] - der_val1b*s_abc - 
      (val1b*deriv[0][0]*val2_abc + val2b*deriv[1][0]*val2_acb)
      - val1b*drho_ab2*val_abc;
   dvola[0] = dvola[0]/3;
   dvola[1] = ra*dsurfa[1] - der_val2b*s_acb - 
      (val1b*deriv[0][1]*val2_abc + val2b*deriv[1][1]*val2_acb)
      - val2b*drho_ac2*val_acb;
   dvola[1] = dvola[1]/3;
   dvola[2] = ra*dsurfa[2] - 
      (val1b*deriv[0][2]*val2_abc + val2b*deriv[1][2]*val2_acb);
   dvola[2] = dvola[2]/3;

   dvolb[0] = rb*dsurfb[0] - der_val1*s_abc - 
      (val1*deriv[0][0]*val2_abc + val3b*deriv[3][0]*val2_bca)
      - val1*drho_ab2*val_abc;
   dvolb[0] = dvolb[0]/3;
   dvolb[1] = rb*dsurfb[1] - 
      (val1*deriv[0][1]*val2_abc + val3b*deriv[3][1]*val2_bca);
   dvolb[1] = dvolb[1]/3;
   dvolb[2] = rb*dsurfb[2] - der_val3b*s_bca - 
      (val1*deriv[0][2]*val2_abc + val3b*deriv[3][2]*val2_bca)
      - val3b*drho_bc2*val_bca;
   dvolb[2] = dvolb[2]/3;

   dvolc[0] = rc*dsurfc[0] - 
      (val2*deriv[1][0]*val2_acb + val3*deriv[3][0]*val2_bca);
   dvolc[0] = dvolc[0]/3;
   dvolc[1] = rc*dsurfc[1] - der_val2*s_acb - 
      (val2*deriv[1][1]*val2_acb + val3*deriv[3][1]*val2_bca)
      - val2*drho_ac2*val_acb;
   dvolc[1] = dvolc[1]/3;
   dvolc[2] = rc*dsurfc[2] - der_val3*s_bca - 
      (val2*deriv[1][2]*val2_acb + val3*deriv[3][2]*val2_bca)
      - val3*drho_bc2*val_bca;
   dvolc[2] = dvolc[2]/3;
}

inline double trig_dradius(double a, double b, double c, double *der_r, bool compder)
{
   double u = 4*a*b*c - (a+b+c-1)*(a+b+c-1);
   double sqr_u = std::sqrt(u);

   double v = (a-1)*(a-1) + (b-1)*(b-1) + (c-1)*(c-1) - (a-b)*(a-b) - (a-c)*(a-c) - (b-c)*(b-c);
   double sqr_v = std::sqrt(v);

   double r = 0.5 + 0.5* sqr_u/sqr_v;

   if (!compder) return r;

   double du_da = 4*b*c - 2*(a+b+c-1);
   double du_db = 4*a*c - 2*(a+b+c-1);
   double du_dc = 4*a*b - 2*(a+b+c-1);

   double dv_da = 2*(b+c-a-1);
   double dv_db = 2*(a+c-b-1);
   double dv_dc = 2*(a+b-c-1);

   der_r[0] = 0.5*(r-0.5)*(du_da/u -dv_da/v);
   der_r[1] = 0.5*(r-0.5)*(du_db/u -dv_db/v);
   der_r[2] = 0.5*(r-0.5)*(du_dc/u -dv_dc/v);

   return r;
}

// "trig_darea" omputes the surface area S of a
// spherical triangle and its derivatives
inline double trig_darea(double a, double b, double c, double *der_S, bool compder)
{
   double tol = 1.e-14;

   double u = 4*a*b*c - (a+b+c-1)*(a+b+c-1);
   double v = 4*a*b*c;
   if (std::fabs(u) < tol) u = 0.0;
   double w = std::sqrt(std::fabs(u));
   double t = std::sqrt(v);

   double S = 2*std::asin(w/t);

   if (!compder) return S;

   if (w>0) {
      der_S[0] = (b+c-a-1)/(a*w);
      der_S[1] = (a+c-b-1)/(b*w);
      der_S[2] = (a+b-c-1)/(c*w);
   }
   else {
      der_S[0] = 0;
      der_S[1] = 0;
      der_S[2] = 0;
   }

   return S;
}

// "sign" defines if the surface area of a
// spherical triangle is positive or negative
inline double sign(double a, double b, double c)
{
   double s = -1;
   if ( a + b <= 1 + c) {
      s = 1;
   }
   else {
      s = -1.;
   }

   return s;
}
}
