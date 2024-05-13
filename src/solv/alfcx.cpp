#include "ff/solv/alfcx.h"
#include "ff/solv/alphamol.h"

namespace tinker
{
void Alfcx::alfcx(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
   std::vector<Edge>& edges, std::vector<Face>& faces, double alpha)
{
   double ra,rb,rc,rd,re;
   double a[4],b[4],c[4],d[4],e[4],cg[3];
   eps = alfcxeps;

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

      get_coord4(vertices, i, j, k, l, a, b, c, d, ra, rb, rc, rd);

      int iflag;
      alf_tetra(a, b, c, d, ra, rb, rc, rd, iflag, alpha);

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
               get_coord5(vertices, i, j, k, l, m, a, b, c, d, e, ra, rb, rc, rd, re);
            }
            else {
               m = -1;
               get_coord4(vertices, i, j, k, l, a, b, c, d, ra, rb, rc, rd);
            }
            int irad, iattach;
            alf_trig(a, b, c, d, e, ra, rb, rc, rd, re, m, irad, iattach, alpha);

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

      for (int iedge = 0; iedge < 6; iedge++) {
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

         get_coord2(vertices, i, j, a, b, cg, ra, rb);
         int irad, iattach;
         alf_edge(vertices, a, b, ra, rb, cg, listcheck, irad, iattach, alpha);

         if (iattach==0 && irad == 1) {
            tetra[idx].info_edge[iedge] = 1;
            nedge++;
            vertices[i].info[7] = 1;
            vertices[j].info[7] = 1;
            continue;
         }

         // Edge is not in alpha complex: now check if the two vertices
         // could be attached to each other:

         vertex_attach(a, b, ra, rb, testa, testb);

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

   alfcxedges(tetra, edges);

   alfcxfaces(tetra, faces);

   if (alfdebug and alfmeth==AlfMethod::AlphaMol) {
      printf("\n Number of tetrahedra in Delaunay complex : %d", ntet_del);
      printf("\n Number of tetrahedra in Alpha complex    : %d", ntet_alp);
      printf("\n Number of triangles in Alpha complex     : %d", ntrig);
      printf("\n Number of edges in Alpha complex         : %d", nedge);
      printf("\n Number of vertices in Alpha complex      : %d\n", nvert);
   }

   delete [] tetra_mask;
}

// "alfcxedges" generates the list of edges in
// the Alpha complex

void Alfcx::alfcxedges(std::vector<Tetrahedron>& tetra, std::vector<Edge>& edges)
{
   // define mask on edges of tetrahedra, to check if already seen
   int ntetra = tetra.size();
   std::bitset<6> *tetra_mask = new std::bitset<6>[ntetra];
   std::bitset<6> zero(std::string("000000"));
   for (int i = 0; i < ntetra; i++) {
      tetra_mask[i] = zero;
   }

   edges.clear();
   edges.reserve(ntetra);

   // define mask on edges of tetrahedra, to check if already seen
   // loop over all tetrahedron: if it belongs to the Delaunay triangulation,
   // check its edges; include in edge list if not seen before

   int i, j;
   int trig1, trig2;
   int jtetra, ktetra;
   int npass;
   bool done;
   int trig_in, trig_out;
   int triga, trigb;
   int ipair;
   int nedge;

   std::vector<std::pair<int,int>> tetra_list;

   for (int idx = 0; idx < ntetra; idx++) {

      if (tetra[idx].info[1]==0) continue;

      // check all six edges
      for (int iedge = 0; iedge < 6; iedge++) {

         // If this edge has already been considered
         // (from another tetrahedron), discard
         if (tetra_mask[idx][iedge] == 1) continue;

         // if this edge is not in alpha complex, discard
         if (tetra[idx].info_edge[iedge]==-1) continue;

         // iedge is the edge number in the tetrahedron idx, with:
         // iedge = 1  (c,d)
         // iedge = 2  (b,d)
         // iedge = 3  (b,c)
         // iedge = 4  (a,d)
         // iedge = 5  (a,c)
         // iedge = 6  (a,b)

         // define indices of the two vertices of the edge

         i = tetra[idx].vertices[pair[iedge][0]];
         j = tetra[idx].vertices[pair[iedge][1]];

         nedge = edges.size();
         Edge e = Edge(i, j);
         edges.push_back(e);

         tetra_list.clear();
         tetra_list.push_back(std::make_pair(idx, iedge));

         // trig1 and trig2 are the two faces of idx that share iedge

         trig1 = face_info[iedge][0];
         trig2 = face_info[iedge][1];

         // now we look at the star of the edge:

         ktetra = idx;
         npass = 0;
         trig_out = trig1;
         jtetra = tetra[ktetra].neighbors[trig1];
         done = false;

         while (!done) {
            // Leave this side of the star if we hit the convex hull
            // in this case, the edge is not buried

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
                  tetra_list.push_back(std::make_pair(jtetra, ipair));
                  tetra_mask[jtetra][ipair] = 1;
                  trig_in = tetra[ktetra].nindex[trig_out];
                  triga = face_info[ipair][0];
                  trigb = face_info[ipair][1];
                  trig_out = triga;
                  if (trig_in == triga) trig_out = trigb;
                  ktetra = jtetra;
                  jtetra = tetra[ktetra].neighbors[trig_out];
               }
            }
         }

         int np = tetra_list.size();
         for (int i = 0; i < np; i++) {
            jtetra = tetra_list[i].first;
            ipair = tetra_list[i].second;
            tetra[jtetra].info_edge[ipair] = nedge;
         }
      }
   }

   delete [] tetra_mask;
}

// "alfcxfaces" procedure generates the list of boundary
// faces in the alpha complex

void Alfcx::alfcxfaces(std::vector<Tetrahedron>& tetra, std::vector<Face>& faces)
{
   // Each triangle is defined implicitly as the interface
   // between two tetrahedra i and j

   int i, j, k;
   double coef;

   int ntetra = tetra.size();
   faces.clear();
   faces.reserve(ntetra);

   int e_1, e_2, e_3;

   for(int idx = 0; idx < ntetra; idx++) {
      // "dead" tetrahedron are ignored
      if (tetra[idx].info[1]==0) continue;

      for(int itrig = 0; itrig < 4; itrig++) {

         if (tetra[idx].info[2+itrig]==0) continue;

         int jtetra = tetra[idx].neighbors[itrig];

         i = tetra[idx].vertices[other3[itrig][0]];
         j = tetra[idx].vertices[other3[itrig][1]];
         k = tetra[idx].vertices[other3[itrig][2]];

         e_1 = tetra[idx].info_edge[face_edge[itrig][0]];
         e_2 = tetra[idx].info_edge[face_edge[itrig][1]];
         e_3 = tetra[idx].info_edge[face_edge[itrig][2]];

         if (jtetra==-1) {
            coef = 1.0;
            if (tetra[idx].info[6]==1) coef = 0.5;
            Face f = Face(i, j, k, e_1, e_2, e_3, coef);
            faces.push_back(f);
         }
         else if (jtetra > idx) {
            coef = 1.0;
            if (tetra[idx].info[6]==1 && tetra[jtetra].info[6]==1) {
               coef = 0.0;
            }
            else if (tetra[idx].info[6]==1 || tetra[jtetra].info[6]==1) {
               coef = 0.5;
            }
            Face f = Face(i, j, k, e_1, e_2, e_3, coef);
            faces.push_back(f);
         }
      }
   }
}

// "findedge" finds the index of the edge
// given two vertices of a tetrahedron

inline int Alfcx::findedge(Tetrahedron t, int i1, int j1)
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

// "alf_tetra" computes the radius R of the sphere orthogonal
// to the four spheres that define a tetrahedron [A,B,C,D]

void Alfcx::alf_tetra(double* a, double* b, double* c, double* d,
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

// "alf_trig" checks if a triangle ABC belons to the alpha complex

void Alfcx::alf_trig(double* a, double* b, double* c, double* d, double* e,
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
   triangle_attach(a, b, c, d, ra, rb, rc, rd, S, T, Dabc, attach, memory);

   // If attached, we can stop there, the triangle will
   // not be part of the alpha complex
   if (attach == 1) {
      iattach = 1;
      return;
   }

   // if e exists, check if a,b,c attached to e:
   if (ie >= 0) {
      triangle_attach(a, b, c, e, ra, rb, rc, re, S, T, Dabc, attach, memory);

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
   triangle_radius(a, b, c, ra, rb, rc, S, T, Dabc, testr, alpha, memory);

   if (testr == 1) irad = 1;
}

// "alf_edge" checks if the edge belongs to the alpha complex

void Alfcx::alf_edge(std::vector<Vertex>& vertices, double* a, double* b, double ra, double rb,
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

      edge_attach(a, b, c, ra, rb, rc, Dab, Sab, Tab, attach);

      if (attach==1) return;
   }

   iattach = 0;

   // edge is not attached; check radius

   int rad;
   edge_radius(a, b, ra, rb, Dab, Sab, Tab, rad, alpha);

   if (rad==1) irad = 1;
}

// edge_radius computes the radius of the smallest
// circumsphere to an edge, and compares it to alpha.
inline void Alfcx::edge_radius(double* a, double* b, double ra, double rb,
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
inline void Alfcx::edge_attach(double* a, double* b, double* c, double ra, double rb,
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

// triangle_attach checks if triangles are attached
inline void Alfcx::triangle_attach(double* a, double* b, double* c, double* d,
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

// triangle_radius computes the radius of the smallest circumsphere to a triangle
inline void Alfcx::triangle_radius(double* a, double* b, double* c, double ra, double rb,
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

// vertex_attach tests if a vertex is attached to another vertex
inline void Alfcx::vertex_attach(double* a, double* b, double ra, double rb, int& testa, int& testb)
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

// get_coord2 extracts two atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void Alfcx::get_coord2(std::vector<Vertex>& vertices, int ia, int ja, double* a, double* b, double* cg, double& ra, double& rb)
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

// get_coord4 extracts four atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void Alfcx::get_coord4(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la, double* a, double* b,
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

// get_coord5 extracts five atoms from the global array
// containing all atoms of the protein, centers them on (0,0,0),
// recomputes their weights and stores them in local arrays
inline void Alfcx::get_coord5(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la,
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
}
