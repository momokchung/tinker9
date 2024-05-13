#pragma once
#include "ff/solv/alphamol.h"
#include "ff/solv/tetra.h"
#include "math/const.h"
#include <cmath>

namespace tinker
{
constexpr double aseps = 1e-14;
constexpr double gceps = 1e-14;

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
   if (std::abs(cosine - 1) < aseps) cosine = 1;
   else if (std::abs(cosine + 1) < aseps) cosine = -1;
   phi = std::acos(cosine);
}

// "twosphder" calculates the surface area and volume derivatives
// of the intersection of two spheres
template <bool compder>
inline void twosphder(double ra, double ra2, double rb, double rb2, double rab, double rab2,
   double& surfa, double& surfb, double& vola, double& volb, double& r, double& phi,
   double& dsurfa, double& dsurfb, double& dvola, double& dvolb, double& dr, double& dphi)
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
   if (std::abs(cosine - 1) < aseps) cosine = 1;
   else if (std::abs(cosine + 1) < aseps) cosine = -1;
   phi = std::acos(cosine);

   if CONSTEXPR (!compder) return;

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
template <bool compder>
inline void threesphder(double ra, double rb,double rc, double ra2,
   double rb2, double rc2, double rab, double rac, double rbc,
   double rab2, double rac2, double rbc2, double *angle,
   double& surfa, double& surfb, double& surfc, double& vola, double& volb, double& volc,
   double* dsurfa, double* dsurfb, double* dsurfc, double* dvola, double* dvolb, double* dvolc)
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

   tetdihedder3<compder>(rab2, rac2, ra2, rbc2, rb2, rc2, angle, cosine, sine, deriv);

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

   if CONSTEXPR (!compder) return;

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

template <bool compder>
inline double trig_dradius(double a, double b, double c, double *der_r)
{
   double u = 4*a*b*c - (a+b+c-1)*(a+b+c-1);
   double sqr_u = std::sqrt(u);

   double v = (a-1)*(a-1) + (b-1)*(b-1) + (c-1)*(c-1) - (a-b)*(a-b) - (a-c)*(a-c) - (b-c)*(b-c);
   double sqr_v = std::sqrt(v);

   double r = 0.5 + 0.5* sqr_u/sqr_v;

   if CONSTEXPR (!compder) return r;

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
template <bool compder>
inline double trig_darea(double a, double b, double c, double *der_S)
{
   double tol = 1.e-14;

   double u = 4*a*b*c - (a+b+c-1)*(a+b+c-1);
   double v = 4*a*b*c;
   if (std::fabs(u) < tol) u = 0.0;
   double w = std::sqrt(std::fabs(u));
   double t = std::sqrt(v);
   double wt = w/t;

   if (std::abs(wt - 1) < gceps) wt = 1;
   else if (std::abs(wt + 1) < gceps) wt = -1;

   double S = 2*std::asin(wt);

   if CONSTEXPR (!compder) return S;

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
