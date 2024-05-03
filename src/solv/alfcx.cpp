#include "ff/solv/alffunc.h"
#include "ff/solv/alphamol.h"

namespace tinker
{
inline void getcoord2(std::vector<Vertex>& vertices, int ia, int ja, double* a, double* b, double* cg, double& ra, double& rb);
inline void getcoord4(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la,
   double* a, double* b, double* c, double* d, double& ra, double& rb, double& rc, double& rd);
inline void getcoord5(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la,
   int ma, double* a, double* b, double* c, double* d, double* e,
   double& ra, double& rb, double& rc, double& rd, double& re);
void vertattach(double* a, double* b, double ra, double rb, int& testa, int& testb);

// "alfcx" builds the alpha complex based on
// the weighted Delaunay triangulation
void alfcx(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, double alpha)
{
   double ra,rb,rc,rd,re;
   double a[4],b[4],c[4],d[4],e[4],cg[3];

   int ntetra = tetra.size();
   std::bitset<6> *tetra_mask = new std::bitset<6>[ntetra];
   std::bitset<6> zero(std::string("000000"));
      
   for (int i = 0; i < ntetra; i++) tetra_mask[i] = zero;

   for (int i = 0; i < ntetra; i++) {
      for (int j = 0; j < 5; j++) tetra[i].info[j+2] = 0;
      for (int j = 0; j < 6; j++) tetra[i].info_edge[j] = -1;
   }

   int ntet_del = 0;
   int ntet_alp = 0;

   int i, j, k, l;
   for (int idx = 0; idx < ntetra; idx++) {
      // "dead" tetrahedron are ignored
      if (tetra[idx].info[1]==0) continue;

      ntet_del++;

      i = tetra[idx].vertices[0]; j = tetra[idx].vertices[1];
      k = tetra[idx].vertices[2]; l = tetra[idx].vertices[3];

      getcoord4(vertices, i, j, k, l, a, b, c, d, ra, rb, rc, rd);

      int iflag;
      alftetra(a, b, c, d, ra, rb, rc, rd, iflag, alpha);

      if (iflag==1) {
         tetra[idx].info[6] = 1;
         ntet_alp++;
      }
   }

   // loop over all triangles: each triangle is defined implicitly
   // as the interface between two tetrahedra i and j with i < j
   int ntrig = 0;
   for (int idx = 0; idx < ntetra; idx++) {
      // "dead" tetrahedron are ignored
      if (tetra[idx].info[1]==0) continue;

      for (int itrig = 0; itrig < 4; itrig++) {

         int jtetra = tetra[idx].neighbors[itrig];
         int jtrig = tetra[idx].nindex[itrig];

         if (jtetra==-1 || jtetra > idx) {
         // check the triangle defined by itetra and jtetra
         // If one of those tetrahedra belongs to the alpha complex,
         // the triangle belongs to the alpha complex
            if (tetra[idx].info[6]==1) {
               tetra[idx].info[2+itrig] = 1;
               ntrig++;
               if (jtetra>=0) {
                  tetra[jtetra].info[2+jtrig] = 1;
               }
               continue;
            }

            if (jtetra>=0) {
               if (tetra[jtetra].info[6]==1) {
                  tetra[idx].info[2+itrig] = 1;
                  tetra[jtetra].info[2+jtrig] = 1;
                  ntrig++;
                  continue;
               }
            }

            // If we are here, it means that the two
            // attached tetrahedra do not belong to the
            // alpha complex: need to check the triangle
            // itself

            // Define the 3 vertices of the triangle, as well as the 2
            // remaining vertices of the two tetrahedra attached to the
            // triangle
            i = tetra[idx].vertices[other3[itrig][0]];
            j = tetra[idx].vertices[other3[itrig][1]];
            k = tetra[idx].vertices[other3[itrig][2]];
            l = tetra[idx].vertices[itrig];

            int m;
            if (jtetra>=0) {
               m = tetra[jtetra].vertices[jtrig];
               getcoord5(vertices, i, j, k, l, m, a, b, c, d, e, ra, rb, rc, rd, re);
            }
            else {
               m = -1;
               getcoord4(vertices, i, j, k, l, a, b, c, d, ra, rb, rc, rd);
            }
            int irad, iattach;
            alftrig(a, b, c, d, e, ra, rb, rc, rd, re, m, irad, iattach, alpha);

            if (iattach==0 && irad == 1) {
               tetra[idx].info[2+itrig] = 1;
               ntrig++;
               if (jtetra >= 0) {
                  tetra[jtetra].info[2+jtrig] = 1;
               }
            }
         }
      }
   }

   // Now loop over all edges: each edge is defined implicitly
   // by the tetrahedra to which it belongs

   int nedge = 0;
   bool test_edge;
   int trig1, trig2, i1, i2, ia, ib, i_out;
   int testa, testb;
   int jtetra, ktetra, npass;
   int triga, trigb, ipair, trig_in, trig_out;
   bool done;

   std::vector<int> listcheck;

   for (int idx = 0; idx < ntetra; idx++) {

      if (tetra[idx].info[1]==0) continue;

      for (int iedge=0; iedge < 6; iedge++) {
         if (tetra_mask[idx][iedge]==1) continue;
         test_edge = false;

         // For each edge, check triangles attached to the edge
         // if at least one of these triangles is in alpha complex,
         // then the edge is in the alpha complex
         // We then put the two vertices directly in the alpha complex
         // Otherwise, build list of triangles to check

         // idx is one tetrahedron (a,b,c,d) containing the edge

         // iedge is the edge number in the tetrahedron idx, with:
         // iedge = 1 (c,d)
         // iedge = 2 (b,d)
         // iedge = 3 (b,c)
         // iedge = 4 (a,d)
         // iedge = 5 (a,c)
         // iedge = 6 (a,b)
        
         // Define indices of the edge

         i = tetra[idx].vertices[pair[iedge][0]];
         j = tetra[idx].vertices[pair[iedge][1]];

         // trig1 and trig2 are the two faces of idx that share iedge
         // i1 and i2 are the positions of the third vertices of
         // trig1 and trig2

         trig1 = face_info[iedge][0];
         i1 = face_pos[iedge][0];
         trig2 = face_info[iedge][1];
         i2 = face_pos[iedge][1];

         ia = tetra[idx].vertices[i1];
         ib = tetra[idx].vertices[i2];

         listcheck.clear();
         if (tetra[idx].info[2+trig1]==1) {
            test_edge = true;
         }
         else {
            listcheck.push_back(ia);
         }
         if (tetra[idx].info[2+trig2]==1) {
            test_edge = true;
         }
         else {
            listcheck.push_back(ib);
         }

         // now we look at the star of the edge:

         ktetra = idx;
         npass = 0;
         trig_out = trig1;
         jtetra = tetra[ktetra].neighbors[trig1];
         done = false;

         while (!done) {
            if (jtetra==-1) {
               if (npass==1) {
                  done = true;
               }
               else {
                  npass++;
                  ktetra = idx;
                  trig_out = trig2;
                  jtetra = tetra[ktetra].neighbors[trig_out];
               }
            }
            else {
               if (jtetra==idx) {
                  done = true;
               }
               else {
                  ipair = findedge(tetra[jtetra], i, j);
                  tetra_mask[jtetra][ipair] = 1;
                  trig_in = tetra[ktetra].nindex[trig_out];
                  triga = face_info[ipair][0];
                  i1 = face_pos[ipair][0];
                  trigb = face_info[ipair][1];
                  i2 = face_pos[ipair][1];
                  trig_out = triga;
                  i_out = i1;
                  if (trig_in == triga) {
                     trig_out = trigb;
                     i_out = i2;
                  }
                  if (tetra[jtetra].info[2+trig_out]==1) test_edge=true;
                  ktetra = jtetra;
                  jtetra = tetra[ktetra].neighbors[trig_out];
                  listcheck.push_back(tetra[ktetra].vertices[i_out]);
               }
            }
         }

         if (test_edge) {
            tetra[idx].info_edge[iedge] = 1;
            nedge++;
            vertices[i].info[7] = 1;
            vertices[j].info[7] = 1;
            continue;
         }

         // If we got here, it means that none of the triangles
         // in the star of the edge belongs to the alpha complex:
         // this is a singular edge.
         // We check if the edge is attached, and if alpha is
         // smaller than the radius of the sphere orthogonal
         // to the two balls corresponding to the edge

         getcoord2(vertices, i, j, a, b, cg, ra, rb);
         int irad, iattach;
         alfedge(vertices, a, b, ra, rb, cg, listcheck, irad, iattach, alpha);

         if (iattach==0 && irad == 1) {
            tetra[idx].info_edge[iedge] = 1;
            nedge++;
            vertices[i].info[7] = 1;
            vertices[j].info[7] = 1;
            continue;
         }

         // Edge is not in alpha complex: now check if the two vertices
         // could be attached to each other: 

         vertattach(a, b, ra, rb, testa, testb);

         if (testa==1) vertices[i].info[6] = 1;
         if (testb==1) vertices[j].info[6] = 1;
      }
   }

   // Now loop over vertices

   int nvert = 0;
   for (int i = 0; i < vertices.size(); i++) {
      if (vertices[i].info[0]==0) continue;
      if (vertices[i].info[6]==1) continue;
      nvert++;

      vertices[i].info[7] = 1;
   }

   delete [] tetra_mask;
}

// getcoord2 extracts two atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void getcoord2(std::vector<Vertex>& vertices, int ia, int ja, double* a, double* b, double* cg, double& ra, double& rb)
{
   double x;

   // get coordinates, build center of mass and center the two points
   for (int i = 0; i < 3; i++) {
      a[i] = vertices[ia].coord[i];
      b[i] = vertices[ja].coord[i];
      x = 0.5*(a[i] + b[i]);
      a[i] -= x;
      b[i] -= x;
      cg[i] = x;
   }

   ra = vertices[ia].r;
   rb = vertices[ja].r;

   a[3] = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] - ra*ra;
   b[3] = b[0]*b[0] + b[1]*b[1] + b[2]*b[2] - rb*rb;
}

// getcoord4 extracts four atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void getcoord4(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la, double* a, double* b,
   double* c, double* d, double& ra, double& rb, double& rc, double& rd)
{
   double x;

   // get coordinates, build center of mass and center the two points
   for (int i = 0; i < 3; i++) {
      a[i] = vertices[ia].coord[i];
      b[i] = vertices[ja].coord[i];
      c[i] = vertices[ka].coord[i];
      d[i] = vertices[la].coord[i];
      x = 0.25*(a[i] + b[i] + c[i] + d[i]);
      a[i] -= x;
      b[i] -= x;
      c[i] -= x;
      d[i] -= x;
   }

   ra = vertices[ia].r;
   rb = vertices[ja].r;
   rc = vertices[ka].r;
   rd = vertices[la].r;

   a[3] = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] - ra*ra;
   b[3] = b[0]*b[0] + b[1]*b[1] + b[2]*b[2] - rb*rb;
   c[3] = c[0]*c[0] + c[1]*c[1] + c[2]*c[2] - rc*rc;
   d[3] = d[0]*d[0] + d[1]*d[1] + d[2]*d[2] - rd*rd;
}

// getcoord5 extracts five atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void getcoord5(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la,
   int ma, double* a, double* b, double* c, double* d, double* e,
   double& ra, double& rb, double& rc, double& rd, double& re)
{
   double x;

   // get coordinates, build center of mass and center the two points
   for (int i = 0; i < 3; i++) {
      a[i] = vertices[ia].coord[i];
      b[i] = vertices[ja].coord[i];
      c[i] = vertices[ka].coord[i];
      d[i] = vertices[la].coord[i];
      e[i] = vertices[ma].coord[i];
      x = 0.2*(a[i] + b[i] + c[i] + d[i] + e[i]);
      a[i] -= x;
      b[i] -= x;
      c[i] -= x;
      d[i] -= x;
      e[i] -= x;
   }

   ra = vertices[ia].r;
   rb = vertices[ja].r;
   rc = vertices[ka].r;
   rd = vertices[la].r;
   re = vertices[ma].r;

   a[3] = a[0]*a[0] + a[1]*a[1] + a[2]*a[2] - ra*ra;
   b[3] = b[0]*b[0] + b[1]*b[1] + b[2]*b[2] - rb*rb;
   c[3] = c[0]*c[0] + c[1]*c[1] + c[2]*c[2] - rc*rc;
   d[3] = d[0]*d[0] + d[1]*d[1] + d[2]*d[2] - rd*rd;
   e[3] = e[0]*e[0] + e[1]*e[1] + e[2]*e[2] - re*re;
}

// vertattach tests if a vertex is attached to another vertex
void vertattach(double* a, double* b, double ra, double rb, int& testa, int& testb)
{
   double Dab[3];

   testa = 0;
   testb = 0;

   for(int i = 0; i < 3; i++) {
      Dab[i] = a[i] - b[i];
   }

   double ra2 = ra*ra;
   double rb2 = rb*rb;

   double dist2 = Dab[0] *Dab[0] + Dab[1]*Dab[1] + Dab[2]*Dab[2];

   double test1 = dist2 + ra2 - rb2;
   double test2 = dist2 - ra2 + rb2;

   if(test1 < 0) testa = 1;
   if(test2 < 0) testb = 1;

   return;
}
}
