#include "ff/solv/alffunc.h"
#include "ff/solv/alphamol.h"

namespace tinker
{
// "alfcxedges" generates the list of edges in
// the Alpha complex
void alfcxedges(std::vector<Tetrahedron>& tetra, std::vector<Edge>& edges)
{
   int face_info[6][2] = {
      { 0, 1},
      { 0, 2},
      { 0, 3},
      { 1, 2},
      { 1, 3},
      { 2, 3},
   };

   int pair[6][2] = {
      { 2, 3},
      { 1, 3},
      { 1, 2},
      { 0, 3},
      { 0, 2},
      { 0, 1},
   };

   // define mask on edges of tetrahedra, to check if already seen
   int ntetra = tetra.size();
   std::bitset<6> *tetra_mask = new std::bitset<6>[ntetra];
   edges.clear();
   edges.reserve(ntetra);

   std::bitset<6> zero(std::string("000000"));
   for (int i = 0; i < ntetra; i++) {
      tetra_mask[i] = zero;
   }

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
void alfcxfaces(std::vector<Tetrahedron>& tetra, std::vector<Face>& faces)
{
   int face_edge[4][3] = {
      { 2, 1, 0},
      { 4, 3, 0},
      { 5, 3, 1},
      { 5, 4, 2},
   };

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
}
