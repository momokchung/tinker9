#include "ff/solv/alphamol.h"
#include "ff/solv/alffunc.h"
#include "math/const.h"

namespace tinker
{
void alphavol(double& WSurf, double& WVol, double& Surf, double& Vol, double* ballwsurf, double* ballwvol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, bool compder)
{
   int ia,ib,ic,id;
   int e1,e2,e3;
   int edge_list[6];
   double ra,ra2,rb,rb2,rc,rc2,rd,rd2;
   double rab,rac,rad,rbc,rbd,rcd;
   double rab2,rac2,rad2,rbc2,rbd2,rcd2;
   double val,val1S,val2S,val3S,val4S;
   double d1,d2,d3,d4;
   double val1V,val2V,val3V,val4V;
   double coefval;
   double surfa,surfb,surfc;
   double vola,volb,volc,vold;
   double r,phi,dr,dphi;
   double coefaS,coefbS,coefcS,coefdS;
   double coefaV,coefbV,coefcV,coefdV;
   double dsurfa2,dsurfb2;
   double dvola2,dvolb2;
   double u[3];
   double angle[6],cosine[6],sine[6];
   double deriv[6][6];
   double dsurfa3[3],dsurfb3[3],dsurfc3[3];
   double dvola3[3],dvolb3[3],dvolc3[3];
   double dvola[6],dvolb[6],dvolc[6],dvold[6];
   constexpr double twopi = 2 * pi;

   int nedges = edges.size();
   int nvertices = vertices.size();
   int nfaces = faces.size();
   int ntetra = tetra.size();

   // initialize results arrays
   WSurf = 0;
   Surf  = 0;
   WVol  = 0;
   Vol   = 0;
   for (int i = 0; i < nvertices; i++) {
      ballwsurf[i] = 0.;
      ballwvol[i] = 0.;
   }

   // initialize edge and vertex info
   for (int i = 0; i < nedges; i++) {
      edges[i].gamma = 1.;

      ia = edges[i].vertices[0];
      ib = edges[i].vertices[1];

      if (vertices[ia].status==0 || vertices[ib].status==0) continue;

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;

      coefaS = vertices[ia].coefs; coefbS = vertices[ib].coefs;
      coefaV = vertices[ia].coefv; coefbV = vertices[ib].coefv;

      rab2 = dist2(ia, ib);
      rab = std::sqrt(rab2);

      twosph(ra, ra2, rb, rb2, rab, rab2, surfa, surfb, vola, volb, r, phi);

      edges[i].len = rab;
      edges[i].surf   = (coefaS*surfa + coefbS*surfb)/twopi;
      edges[i].vol    = (coefaV*vola + coefbV*volb)/twopi;
      edges[i].dsurf  = 0;
      edges[i].dvol   = 0;

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
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv; 
      coefcS = vertices[ic].coefs; coefcV = vertices[ic].coefv; 
      coefdS = vertices[id].coefs; coefdV = vertices[id].coefv; 

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
         }
      }

      if (compder) {
         // Derivative: take into account the derivatives of the edge
         // weight in weightedinclusion-exclusion formula
         for (int iedge = 0; iedge < 6; iedge++) {
            int i1 = edge_list[iedge];
            val1S = edges[i1].surf;
            val1V = edges[i1].vol;
            for (int ie = 0; ie < 6; ie++) {
               int je = edge_list[ie];
               edges[je].dsurf  += val1S*deriv[iedge][ie];
               edges[je].dvol   += val1V*deriv[iedge][ie];
            }
         }

         // Derivative: take into account the derivatives of the vertex
         // weight in weighted inclusion-exclusion formula

         val1S = ra2*coefaS; val2S = rb2*coefbS;
         val3S = rc2*coefcS; val4S = rd2*coefdS;

         val1V = ra2*ra*coefaV/3; val2V = rb2*rb*coefbV/3;
         val3V = rc2*rc*coefcV/3; val4V = rd2*rd*coefdV/3;

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
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv;
      coefcS = vertices[ic].coefs; coefcV = vertices[ic].coefv;

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
      angle, surfa, surfb, surfc, vola, volb, volc, 
      dsurfa3, dsurfb3, dsurfc3, dvola3, dvolb3, dvolc3, compder);

      ballwsurf[ia] += coefval*surfa;
      ballwsurf[ib] += coefval*surfb;
      ballwsurf[ic] += coefval*surfc;

      ballwvol[ia] += coefval*vola;
      ballwvol[ib] += coefval*volb;
      ballwvol[ic] += coefval*volc;

      if (compder) {
         edges[e1].dsurf += coefval*(coefaS*dsurfa3[0]+coefbS*dsurfb3[0]+coefcS*dsurfc3[0]);
         edges[e2].dsurf += coefval*(coefaS*dsurfa3[1]+coefbS*dsurfb3[1]+coefcS*dsurfc3[1]);
         edges[e3].dsurf += coefval*(coefaS*dsurfa3[2]+coefbS*dsurfb3[2]+coefcS*dsurfc3[2]);

         edges[e1].dvol += coefval*(coefaV*dvola3[0]+coefbV*dvolb3[0]+coefcV*dvolc3[0]);
         edges[e2].dvol += coefval*(coefaV*dvola3[1]+coefbV*dvolb3[1]+coefcV*dvolc3[1]);
         edges[e3].dvol += coefval*(coefaV*dvola3[2]+coefbV*dvolb3[2]+coefcV*dvolc3[2]);
      }
   }

   // now add contribution of two-sphere
   double eps = 1.e-10;
   for (int iedge = 0; iedge < nedges; iedge++) {
      coefval = edges[iedge].gamma;

      if (std::fabs(coefval) < eps) continue;

      ia = edges[iedge].vertices[0];
      ib = edges[iedge].vertices[1];

      if (vertices[ia].status==0 || vertices[ib].status==0) continue;

      coefaS = vertices[ia].coefs; coefaV = vertices[ia].coefv;
      coefbS = vertices[ib].coefs; coefbV = vertices[ib].coefv;

      ra = vertices[ia].r; ra2 = ra*ra;
      rb = vertices[ib].r; rb2 = rb*rb;

      rab = edges[iedge].len; rab2 = rab*rab;

      twosphder(ra, ra2, rb, rb2, rab, rab2, surfa, surfb,
      vola, volb, r, phi, dsurfa2, dsurfb2, dvola2, dvolb2, 
      dr, dphi, compder);

      ballwsurf[ia] -= coefval*surfa; 
      ballwsurf[ib] -= coefval*surfb; 
      ballwvol[ia]  -= coefval*vola; 
      ballwvol[ib]  -= coefval*volb; 

      if (compder) {
         edges[iedge].dsurf  -= coefval* (coefaS*dsurfa2 + coefbS*dsurfb2);
         edges[iedge].dvol   -= coefval* (coefaV*dvola2 + coefbV*dvolb2);
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
   }

   // compute total surface, volume (weighted, and unweighted)
   for (int i = 4; i < nvertices; i++) {
      if (vertices[i].info[0]==0) continue;
      if (vertices[i].status==0) continue;

      coefaS = vertices[i].coefs; coefaV = vertices[i].coefv;

      Surf           += ballwsurf[i];
      ballwsurf[i]    = ballwsurf[i]*coefaS;
      WSurf          += ballwsurf[i];

      Vol           += ballwvol[i];
      ballwvol[i]    = ballwvol[i]*coefaV;
      WVol          += ballwvol[i];
   }

   // shift as 4 first vertices are pseudo atoms
   int nballs = 0;
   for (int i = 0; i < nvertices; i++) if (vertices[i].status==1) nballs++;
   for (int i = 0; i < nballs; i++) {
      ballwsurf[i]   = ballwsurf[i+4];
      ballwvol[i]    = ballwvol[i+4];
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
   }

   // shift as 4 first vertices are pseudo atoms
   for (int i = 0; i < nballs; i++) {
      dsurfx[i]   = dsurfx[i+4];
      dsurfy[i]   = dsurfy[i+4];
      dsurfz[i]   = dsurfz[i+4];
      dvolx[i]    = dvolx[i+4];
      dvoly[i]    = dvoly[i+4];
      dvolz[i]    = dvolz[i+4];
   }
}
}
