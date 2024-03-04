#include "ff/solv/alphamol.h"
#include "ff/solv/alffunc.h"
#include "math/const.h"

namespace tinker
{
void alphavol(double& WSurf, double& WVol, double& WMean, double& WGauss,
   double& Surf, double& Vol, double& Mean, double& Gauss,
   double* ballwsurf, double* ballwvol, double* ballwmean, double* ballwgauss,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz,
   double* dmeanx, double* dmeany, double* dmeanz, double* dgaussx, double* dgaussy, double* dgaussz, bool compder)
{
   int ia,ib,ic,id;
   int e1,e2,e3;
   int edge_list[6];
   double ra,ra2,rb,rb2,rc,rc2,rd,rd2;
   double rab,rac,rad,rbc,rbd,rcd;
   double rab2,rac2,rad2,rbc2,rbd2,rcd2;
   double val,val1S,val2S,val3S,val4S;
   double val1M1,val2M1,val3M1,val4M1;
   double val1G1;
   double d1,d2,d3,d4;
   double val1V,val2V,val3V,val4V;
   double coefval,coefvalS;
   double surfa,surfb,surfc;
   double gaussa,gaussb,gaussc;
   double vola,volb,volc,vold;
   double r,phi,l,dr,dphi,dl;
   double coefaS,coefbS,coefcS,coefdS;
   double coefaV,coefbV,coefcV,coefdV;
   double coefaM,coefbM,coefcM,coefdM;
   double coefaG,coefbG,coefcG,coefdG;
   double dsurfa2,dsurfb2;
   double dvola2,dvolb2;
   double u[3];
   double angle[6],cosine[6],sine[6];
   double deriv[6][6],deriv2[6][3],dg[3][3];
   double dsurfa3[3],dsurfb3[3],dsurfc3[3];
   double dvola3[3],dvolb3[3],dvolc3[3];
   double dvola[6],dvolb[6],dvolc[6],dvold[6];
   constexpr double twopi = 2 * pi;
   constexpr double coefe = 1.;
   constexpr double coeff = 2.;

   int nedges = edges.size();
   int nvertices = vertices.size();
   int nfaces = faces.size();
   int ntetra = tetra.size();

   // initialize results arrays
   WSurf = 0;
   Surf  = 0;
   WVol  = 0;
   Vol   = 0;
   WMean = 0;
   Mean  = 0;
   WGauss= 0;
   Gauss = 0;
   for (int i = 0; i < nvertices; i++) {
      ballwsurf[i] = 0.;
      ballwvol[i] = 0.;
      ballwmean[i] = 0.;
      ballwgauss[i] = 0.;
   }

   // initialize edge and vertex info
   for (int i = 0; i < nedges; i++) {
      edges[i].gamma = 1.;
      edges[i].sigma = 1.;

      ia = edges[i].vertices[0];
      ib = edges[i].vertices[1];

      if (vertices[ia].status==0 || vertices[ib].status==0) continue;

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;

      coefaS = vertices[ia].coefs; coefbS = vertices[ib].coefs;
      coefaV = vertices[ia].coefv; coefbV = vertices[ib].coefv;
      coefaM = vertices[ia].coefm; coefbM = vertices[ib].coefm;
      coefaG = vertices[ia].coefg; coefbG = vertices[ib].coefg;

      rab2 = dist2(ia, ib);
      rab = std::sqrt(rab2);

      twosph(ra, ra2, rb, rb2, rab, rab2, surfa, surfb, vola, volb, r, phi, l);

      edges[i].len = rab;
      edges[i].surf   = (coefaS*surfa + coefbS*surfb)/twopi;
      edges[i].vol    = (coefaV*vola + coefbV*volb)/twopi;
      edges[i].coefm1 = (coefaM*surfa/ra + coefbM*surfb/rb)/twopi;
      edges[i].coefm2 = coefe*(coefaM+coefbM)*r*phi/2;
      edges[i].coefg1 = (coefaG*surfa/ra2 + coefbG*surfb/rb2)/twopi;
      edges[i].coefg2 = coefe*(coefaG+coefbG)*l/2;
      edges[i].dsurf  = 0;
      edges[i].dvol   = 0;
      edges[i].dmean  = 0;
      edges[i].dgauss = 0;

   }

   for (int i = 0; i < nvertices; i++) {
      vertices[i].gamma = 1.;
   }

   // Contributions of four overlapping spheres:
   // We are using the weighted inclusion-exclusion formula:
   // Each tetrahedron in the Alpha Complex only contributes to the weight of each
   // its edges and each of its vertices

   for (int idx = 0; idx < ntetra; idx++) {
      if (tetra[idx].info[6]==0) continue;

      ia = tetra[idx].vertices[0];
      ib = tetra[idx].vertices[1];
      ic = tetra[idx].vertices[2];
      id = tetra[idx].vertices[3];

      if (vertices[ia].status==0 || vertices[ib].status==0
      || vertices[ic].status==0 || vertices[id].status==0) continue;

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;
      rc = vertices[ic].r; rc2 = rc*rc;
      rd = vertices[id].r; rd2 = rd*rd;

      coefaS = vertices[ia].coefs; coefaV = vertices[ia].coefv; 
      coefaM = vertices[ia].coefm; coefaG = vertices[ia].coefg;
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv; 
      coefbM = vertices[ib].coefm; coefbG = vertices[ib].coefg;
      coefcS = vertices[ic].coefs; coefcV = vertices[ic].coefv; 
      coefcM = vertices[ic].coefm; coefcG = vertices[ic].coefg;
      coefdS = vertices[id].coefs; coefdV = vertices[id].coefv; 
      coefdM = vertices[id].coefm; coefdG = vertices[id].coefg;

      for (int iedge = 0; iedge < 6; iedge++) {
         // iedge is the edge number in the tetrahedron idx, with:
         // iedge = 1  (c,d)
         // iedge = 2  (b,d)
         // iedge = 3  (b,c)
         // iedge = 4  (a,d)
         // iedge = 5  (a,c)
         // iedge = 6  (a,b)
         edge_list[5-iedge] = tetra[idx].info_edge[iedge];
      }

      rab = edges[edge_list[0]].len; rab2 = rab*rab;
      rac = edges[edge_list[1]].len; rac2 = rac*rac;
      rad = edges[edge_list[2]].len; rad2 = rad*rad;
      rbc = edges[edge_list[3]].len; rbc2 = rbc*rbc;
      rbd = edges[edge_list[4]].len; rbd2 = rbd*rbd;
      rcd = edges[edge_list[5]].len; rcd2 = rcd*rcd;

      // characterize tetrahedron (A,B,C,D)
      if (!compder) {
         tetdihed(rab2, rac2, rad2, rbc2, rbd2, rcd2, angle, cosine, sine);
      }
      else {
         tetdihedder(rab2, rac2, rad2, rbc2, rbd2, rcd2, angle, cosine, sine, deriv);
      }

      // add fraction of tetrahedron that "belongs" to each ball
      tetvorder(ra2, rb2, rc2, rd2, rab, rac, rad, rbc,
      rbd, rcd, rab2, rac2, rad2, rbc2, rbd2, rcd2, cosine, sine,
      deriv, vola, volb, volc, vold, dvola, dvolb, dvolc, dvold, compder);

      ballwvol[ia] += vola; ballwvol[ib] += volb;
      ballwvol[ic] += volc; ballwvol[id] += vold;

      if (compder) {
         for (int iedge = 0; iedge < 6; iedge++) {
            int i1 = edge_list[iedge];
            edges[i1].dvol += coefaV*dvola[iedge]
               +coefbV*dvolb[iedge]+coefcV*dvolc[iedge]
               +coefdV*dvold[iedge];
         }
      }

      // weights on each vertex: fraction of solid angle
      vertices[ia].gamma -= (angle[0]+angle[1]+angle[2])/2 - 0.250;
      vertices[ib].gamma -= (angle[0]+angle[3]+angle[4])/2 - 0.250;
      vertices[ic].gamma -= (angle[1]+angle[3]+angle[5])/2 - 0.250;
      vertices[id].gamma -= (angle[2]+angle[4]+angle[5])/2 - 0.250;

      // weights on each edge: fraction of dihedral angle
      for (int iedge = 0; iedge < 6; iedge++) {
         int i1 = edge_list[iedge];
         if (edges[i1].gamma != 0) {
            edges[i1].gamma -= angle[iedge];
            edges[i1].sigma -= angle[iedge];
         }
      }

      if (compder) {
         // Derivative: take into account the derivatives of the edge
         // weight in weightedinclusion-exclusion formula
         for (int iedge = 0; iedge < 6; iedge++) {
            int i1 = edge_list[iedge];
            val1S = edges[i1].surf;
            val1V = edges[i1].vol;
            val1M1 = edges[i1].coefm1 + edges[i1].coefm2;
            val1G1 = edges[i1].coefg1 + edges[i1].coefg2;
            for (int ie = 0; ie < 6; ie++) {
               int je = edge_list[ie];
               edges[je].dsurf  += val1S*deriv[iedge][ie];
               edges[je].dvol   += val1V*deriv[iedge][ie];
               edges[je].dmean  += val1M1*deriv[iedge][ie];
               edges[je].dgauss += val1G1*deriv[iedge][ie];
            }
         }

         // Derivative: take into account the derivatives of the vertex
         // weight in weighted inclusion-exclusion formula

         val1S = ra2*coefaS; val2S = rb2*coefbS;
         val3S = rc2*coefcS; val4S = rd2*coefdS;

         val1V = ra2*ra*coefaV/3; val2V = rb2*rb*coefbV/3;
         val3V = rc2*rc*coefcV/3; val4V = rd2*rd*coefdV/3;

         val1M1 = ra*coefaM; val2M1 = rb*coefbM;
         val3M1 = rc*coefcM; val4M1 = rd*coefdM;

         for (int ie = 0; ie < 6; ie++) {
            int je = edge_list[ie];
            d1 = deriv[0][ie]+deriv[1][ie]+deriv[2][ie];
            d2 = deriv[0][ie]+deriv[3][ie]+deriv[4][ie];
            d3 = deriv[1][ie]+deriv[3][ie]+deriv[5][ie];
            d4 = deriv[2][ie]+deriv[4][ie]+deriv[5][ie];

            val = val1S*d1 + val2S*d2 + val3S*d3 + val4S*d4;
            edges[je].dsurf -= val;

            val = val1V*d1 + val2V*d2 + val3V*d3 + val4V*d4;
            edges[je].dvol -= val;

            val = val1M1*d1 + val2M1*d2 + val3M1*d3 + val4M1*d4;
            edges[je].dmean -= val;

            val = coefaG*d1 + coefbG*d2 + coefcG*d3 + coefdG*d4;
            edges[je].dgauss -= val;
         }
      }
   }

   // contribution of 3-balls (i.e. triangles of the alpha complex)

   for (int idx = 0; idx < nfaces; idx++) {
      coefval = faces[idx].gamma;
      if (coefval==0) continue;

      ia = faces[idx].vertices[0];
      ib = faces[idx].vertices[1];
      ic = faces[idx].vertices[2];

      if (vertices[ia].status==0 || vertices[ib].status==0
      || vertices[ic].status==0 ) continue;


      coefaS = vertices[ia].coefs; coefaV = vertices[ia].coefv; 
      coefaM = vertices[ia].coefm; coefaG = vertices[ia].coefg;
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv; 
      coefbM = vertices[ib].coefm; coefbG = vertices[ib].coefg;
      coefcS = vertices[ic].coefs; coefcV = vertices[ic].coefv; 
      coefcM = vertices[ic].coefm; coefcG = vertices[ic].coefg;

      e1 = faces[idx].edges[0];
      e2 = faces[idx].edges[1];
      e3 = faces[idx].edges[2];

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;
      rc = vertices[ic].r; rc2 = rc*rc;

      rab = edges[e1].len; rab2=rab*rab;
      rac = edges[e2].len; rac2=rac*rac;
      rbc = edges[e3].len; rbc2=rbc*rbc;

      threesphder(ra, rb, rc, ra2, rb2, rc2, rab, rac, rbc, rab2, rac2, rbc2,
      angle, deriv2, surfa, surfb, surfc, vola, volb, volc, 
      dsurfa3, dsurfb3, dsurfc3, dvola3, dvolb3, dvolc3, compder);

      threesphgss(ra, rb, rc, ra2, rb2, rc2, rab, rac, rbc,
      rab2, rac2, rbc2, gaussa, gaussb, gaussc, dg, compder);

      ballwsurf[ia] += coefval*surfa;
      ballwsurf[ib] += coefval*surfb;
      ballwsurf[ic] += coefval*surfc;

      ballwvol[ia] += coefval*vola;
      ballwvol[ib] += coefval*volb;
      ballwvol[ic] += coefval*volc;

      ballwgauss[ia] += coeff*coefval*gaussa;
      ballwgauss[ib] += coeff*coefval*gaussb;
      ballwgauss[ic] += coeff*coefval*gaussc;

      edges[e1].sigma -= 2*coefval*angle[0];
      edges[e2].sigma -= 2*coefval*angle[1];
      edges[e3].sigma -= 2*coefval*angle[3];

      if (compder) {
         edges[e1].dsurf += coefval*(coefaS*dsurfa3[0]+coefbS*dsurfb3[0]+coefcS*dsurfc3[0]);
         edges[e2].dsurf += coefval*(coefaS*dsurfa3[1]+coefbS*dsurfb3[1]+coefcS*dsurfc3[1]);
         edges[e3].dsurf += coefval*(coefaS*dsurfa3[2]+coefbS*dsurfb3[2]+coefcS*dsurfc3[2]);

         edges[e1].dvol += coefval*(coefaV*dvola3[0]+coefbV*dvolb3[0]+coefcV*dvolc3[0]);
         edges[e2].dvol += coefval*(coefaV*dvola3[1]+coefbV*dvolb3[1]+coefcV*dvolc3[1]);
         edges[e3].dvol += coefval*(coefaV*dvola3[2]+coefbV*dvolb3[2]+coefcV*dvolc3[2]);

         edges[e1].dmean += coefval*(coefaM*dsurfa3[0]/ra+coefbM*dsurfb3[0]/rb+
            coefcM*dsurfc3[0]/rc);
         edges[e2].dmean += coefval*(coefaM*dsurfa3[1]/ra+coefbM*dsurfb3[1]/rb+
            coefcM*dsurfc3[1]/rc);
         edges[e3].dmean += coefval*(coefaM*dsurfa3[2]/ra+coefbM*dsurfb3[2]/rb+
            coefcM*dsurfc3[2]/rc);

         edges[e1].dmean += 2*coefval*(edges[e1].coefm2*deriv2[0][0]+
            edges[e2].coefm2*deriv2[1][0]+edges[e3].coefm2*deriv2[3][0]);
         edges[e2].dmean += 2*coefval*(edges[e1].coefm2*deriv2[0][1]+
            edges[e2].coefm2*deriv2[1][1]+edges[e3].coefm2*deriv2[3][1]);
         edges[e3].dmean += 2*coefval*(edges[e1].coefm2*deriv2[0][2]+
            edges[e2].coefm2*deriv2[1][2]+edges[e3].coefm2*deriv2[3][2]);

         edges[e1].dgauss += coefval*(coefaG*dsurfa3[0]/ra2+coefbG*dsurfb3[0]/rb2+
            coefcG*dsurfc3[0]/rc2);
         edges[e2].dgauss += coefval*(coefaG*dsurfa3[1]/ra2+coefbG*dsurfb3[1]/rb2+
            coefcG*dsurfc3[1]/rc2);
         edges[e3].dgauss += coefval*(coefaG*dsurfa3[2]/ra2+coefbG*dsurfb3[2]/rb2+
            coefcG*dsurfc3[2]/rc2);

         edges[e1].dgauss += 2*coefval*(edges[e1].coefg2*deriv2[0][0]+
            edges[e2].coefg2*deriv2[1][0]+edges[e3].coefg2*deriv2[3][0]);
         edges[e2].dgauss += 2*coefval*(edges[e1].coefg2*deriv2[0][1]+
            edges[e2].coefg2*deriv2[1][1]+edges[e3].coefg2*deriv2[3][1]);
         edges[e3].dgauss += 2*coefval*(edges[e1].coefg2*deriv2[0][2]+
            edges[e2].coefg2*deriv2[1][2]+edges[e3].coefg2*deriv2[3][2]);

         edges[e1].dgauss += coeff*coefval*(coefaG*dg[0][0]+coefbG*dg[1][0]+coefcG*dg[2][0]);
         edges[e2].dgauss += coeff*coefval*(coefaG*dg[0][1]+coefbG*dg[1][1]+coefcG*dg[2][1]);
         edges[e3].dgauss += coeff*coefval*(coefaG*dg[0][2]+coefbG*dg[1][2]+coefcG*dg[2][2]);
      }
   }

   // now add contribution of two-sphere
   double eps = 1.e-10;
   for (int iedge = 0; iedge < nedges; iedge++) {
      coefval = edges[iedge].gamma;
      coefvalS = edges[iedge].sigma;

      if (std::fabs(coefval) < eps) continue;

      ia = edges[iedge].vertices[0];
      ib = edges[iedge].vertices[1];

      if (vertices[ia].status==0 || vertices[ib].status==0) continue;

      coefaS = vertices[ia].coefs; coefaV = vertices[ia].coefv; 
      coefaM = vertices[ia].coefm; coefaG = vertices[ia].coefg;
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv; 
      coefbM = vertices[ib].coefm; coefbG = vertices[ib].coefg;

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;

      rab = edges[iedge].len; rab2 = rab*rab;

      twosphder(ra, ra2, rb, rb2, rab, rab2, surfa, surfb,
      vola, volb, r, phi, l, dsurfa2, dsurfb2, dvola2, dvolb2, 
      dr, dphi, dl, compder);

      ballwsurf[ia] -= coefval*surfa; 
      ballwsurf[ib] -= coefval*surfb; 
      ballwvol[ia]  -= coefval*vola; 
      ballwvol[ib]  -= coefval*volb; 

      val = coefe*pi*coefvalS*r*phi;
      ballwmean[ia] -= val;
      ballwmean[ib] -= val;

      val = coefe*pi*coefvalS*l;
      ballwgauss[ia] -= val;
      ballwgauss[ib] -= val;

      if (compder) {
         edges[iedge].dsurf  -= coefval* (coefaS*dsurfa2 + coefbS*dsurfb2);
         edges[iedge].dvol   -= coefval* (coefaV*dvola2 + coefbV*dvolb2);
         edges[iedge].dmean  -= coefval* (coefaM*dsurfa2/ra + coefbM*dsurfb2/rb);
         edges[iedge].dmean  -= coefe*coefvalS*pi*(r*dphi+phi*dr)*(coefaM+coefbM);
         edges[iedge].dgauss -= coefval* (coefaG*dsurfa2/ra2 + coefbG*dsurfb2/rb2);
         edges[iedge].dgauss -= coefe*coefvalS*pi*dl*(coefaG+coefbG);
      }
   }

   // now loop over vertices
   for (int i = 4; i < nvertices; i++) {
      coefval = vertices[i].gamma;
      if (vertices[i].info[0]==0) continue;
      if (vertices[i].info[7]==0) continue;
      if (coefval==0) continue;
      if (vertices[i].status==0) continue;

      ra = vertices[i].r; ra2 = ra*ra;
      surfa = 4*pi*ra*ra;
      vola  = surfa*ra/3;
      ballwsurf[i]  += coefval*surfa;
      ballwvol[i]   += coefval*vola;
      if (ra2 > 0) {
         ballwmean[i]  += ballwsurf[i]/ra;
         ballwgauss[i] += ballwsurf[i]/ra2;
      }
   }

   // compute total surface, volume (weighted, and unweighted)
   for (int i = 4; i < nvertices; i++) {
      if (vertices[i].info[0]==0) continue;
      if (vertices[i].status==0) continue;

      coefaS = vertices[i].coefs; coefaV = vertices[i].coefv;
      coefaM = vertices[i].coefm; coefaG = vertices[i].coefg;

      Surf           += ballwsurf[i];
      ballwsurf[i]    = ballwsurf[i]*coefaS;
      WSurf          += ballwsurf[i];

      Vol           += ballwvol[i];
      ballwvol[i]    = ballwvol[i]*coefaV;
      WVol          += ballwvol[i];

      Mean          += ballwmean[i];
      ballwmean[i]   = ballwmean[i]*coefaM;
      WMean         += ballwmean[i];

      Gauss         += ballwgauss[i];
      ballwgauss[i]  = ballwgauss[i]*coefaG;
      WGauss        += ballwgauss[i];
   }

   // shift as 4 first vertices are pseudo atoms
   int nballs = 0;
   for (int i = 0; i < nvertices; i++) if (vertices[i].status==1) nballs++;
   for (int i = 0; i < nballs; i++) {
      ballwsurf[i]   = ballwsurf[i+4];
      ballwvol[i]    = ballwvol[i+4];
      ballwmean[i]   = ballwmean[i+4];
      ballwgauss[i]  = ballwgauss[i+4];
   }

   if (!compder) return;

   // convert derivatives wrt to distance to derivatives wrt to coordinates
   for (int i = 0; i < nvertices; i++) {
      dsurfx[i] = 0.;
      dsurfy[i] = 0.;
      dsurfz[i] = 0.;
      dvolx[i] = 0.;
      dvoly[i] = 0.;
      dvolz[i] = 0.;
      dmeanx[i] = 0.;
      dmeany[i] = 0.;
      dmeanz[i] = 0.;
      dgaussx[i] = 0.;
      dgaussy[i] = 0.;
      dgaussz[i] = 0.;
   }

   for (int iedge = 0; iedge < nedges; iedge++) {
      ia = edges[iedge].vertices[0];
      ib = edges[iedge].vertices[1];

      for (int i = 0; i < 3; i++) {
         u[i] = vertices[ia].coord[i] - vertices[ib].coord[i];
      }

      rab  = edges[iedge].len;
      val1S  = edges[iedge].dsurf/rab;
      val1V  = edges[iedge].dvol/rab;
      val1M1 = edges[iedge].dmean/rab;
      val1G1 = edges[iedge].dgauss/rab;

      dsurfx[ia]  += u[0]*val1S;
      dsurfy[ia]  += u[1]*val1S;
      dsurfz[ia]  += u[2]*val1S;
      dsurfx[ib]  -= u[0]*val1S;
      dsurfy[ib]  -= u[1]*val1S;
      dsurfz[ib]  -= u[2]*val1S;
      dvolx[ia]   += u[0]*val1V;
      dvoly[ia]   += u[1]*val1V;
      dvolz[ia]   += u[2]*val1V;
      dvolx[ib]   -= u[0]*val1V;
      dvoly[ib]   -= u[1]*val1V;
      dvolz[ib]   -= u[2]*val1V;
      dmeanx[ia]  += u[0]*val1M1;
      dmeany[ia]  += u[1]*val1M1;
      dmeanz[ia]  += u[2]*val1M1;
      dmeanx[ib]  -= u[0]*val1M1;
      dmeany[ib]  -= u[1]*val1M1;
      dmeanz[ib]  -= u[2]*val1M1;
      dgaussx[ia] += u[0]*val1G1;
      dgaussy[ia] += u[1]*val1G1;
      dgaussz[ia] += u[2]*val1G1;
      dgaussx[ib] -= u[0]*val1G1;
      dgaussy[ib] -= u[1]*val1G1;
      dgaussz[ib] -= u[2]*val1G1;
   }

   // shift as 4 first vertices are pseudo atoms
   for (int i = 0; i < nballs; i++) {
      dsurfx[i]   = dsurfx[i+4];
      dsurfy[i]   = dsurfy[i+4];
      dsurfz[i]   = dsurfz[i+4];
      dvolx[i]    = dvolx[i+4];
      dvoly[i]    = dvoly[i+4];
      dvolz[i]    = dvolz[i+4];
      dmeanx[i]   = dmeanx[i+4];
      dmeany[i]   = dmeany[i+4];
      dmeanz[i]   = dmeanz[i+4];
      dgaussx[i]  = dgaussx[i+4];
      dgaussy[i]  = dgaussy[i+4];
      dgaussz[i]  = dgaussz[i+4];
   }
}
}
