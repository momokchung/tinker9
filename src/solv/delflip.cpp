#include "tool/error.h"
#include "ff/solv/alphamol.h"

namespace tinker
{
void flip_1_4(std::vector<Tetrahedron>& tetra, int ipoint, int itetra, int& tetra_last,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill)
{
   int k, newtetra;
   int jtetra;
   int fact,idx;

   std::bitset<8> ikeep;
   char ival[4];

   int vertex[4], neighbour[4];
   int position[4];
   int idx_list[4][3]={{0,0,0},{0,1,1},{1,1,2},{2,2,2}};

   // store information about "old" tetrahedron
   ikeep = tetra[itetra].info;

   for (int i = 0; i < 4; i++) {
      vertex[i] = tetra[itetra].vertices[i];
      neighbour[i] = tetra[itetra].neighbors[i];
      ival[i] = tetra[itetra].nindex[i];
   }

   fact = -1;
   if (tetra[itetra].info[0] == 1) fact = 1;

   // The four new tetrahedra are going to be stored in
   // any free space in the tetrahedron list and
   // at the end of the list of known tetrahedra

   k = 0;
   while (!free.empty() && k < 4) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 4; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[3];

   // itetra is set to 0, and added to the "kill" list
   tetra[itetra].info[1] = 0;
   kill.push_back(itetra);

   // The tetrahedron is defined as (ijkl); four new tetrahedra are
   // created: jklp, iklp, ijlp, and ijkp, where p is the new
   // point to be included

   // For each new tetrahedron, define all four neighbours:
   // For each neighbour, I store the index of the vertex opposite to 
   // the common face in array tetra_nindex

   // tetrahedron jklp : neighbours are iklp, ijlp, ijkp and neighbour of (ijkl) on face jkl
   // tetrahedron iklp : neighbours are jklp, ijlp, ijkp and neighbour of (ijkl) on face ikl
   // tetrahedron ijlp : neighbours are jklp, iklp, ijkp and neighbour of (ijkl) on face ijl
   // tetrahedron ijkp : neighbours are jklp, iklp, ijlp and neighbour of (ijkl) on face ijk

   for (int i = 0; i < 4; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 4; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = vertex[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k] = idx_list[i][k];
         k++;
      }

      jtetra = neighbour[i];
      idx = ival[i];
      tetra[newtetra].vertices[3] = ipoint;
      tetra[newtetra].neighbors[3] = jtetra;
      tetra[newtetra].nindex[3] = ival[i];
      tetra[newtetra].info[2+i] = ikeep[2+i];

      if (jtetra !=-1 && idx != -1) {
         tetra[jtetra].neighbors[idx] = newtetra;
         tetra[jtetra].nindex[idx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      fact = -fact;
      tetra[newtetra].info[0] = 0;
      if (fact==1) tetra[newtetra].info[0] = 1;

   }

   // Now add all fours faces of itetra in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the
   // intersection of two tetrahedra

   // link_facet: jkl tetrahedra: jklp and neighbour of (ijkl) on jkl
   // link_facet: ikl tetrahedra: iklp and neighbour of (ijkl) on ikl
   // link_facet: ijl tetrahedra: ijlp and neighbour of (ijkl) on ijl
   // link_facet: ijk tetrahedra: ijkp and neighbour of (ijkl) on ijk

   for (int i = 0; i < 4; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      idx = tetra[newtetra].nindex[3];
      link_index.push(std::make_pair(3,idx));
   }
}

// flip_4_1 implements a 4->1 flip in 3D for regular triangulation
void flip_4_1(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra,
   int ltetra, int* ivertices, int idp, int jdp, int kdp, int ldp, bool test_acpo, int& ierr, int& tetra_last,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill)
{
   int p,o,a,b,c;
   int ishare,jshare,kshare,lshare;
   int idx,jdx,kdx,ldx;
   int test1,newtetra;
   std::bitset<8> ikeep, jkeep, kkeep, lkeep;

   ierr = 0;
   test1 = 1;
   if (test_acpo) test1 = -1;

   // if itetra, jtetra, ktetra, ltetra are inactive, cannot flip
   if (tetra[itetra].info[1]==0 || tetra[jtetra].info[1]==0 ||
      tetra[ktetra].info[1]==0 || tetra[ltetra].info[1]==0) {
      ierr = 1;
      return;
   }

   // store "old" info
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;
   kkeep = tetra[ktetra].info;
   lkeep = tetra[ltetra].info;

   // Define
   //     - ishare: index of tetrahedron sharing the face opposite to b in itetra
   //     - idx:    index of the vertex of ishare opposite to the face of ishare shared with itetra
   //     - jshare: index of tetrahedron sharing the face opposite to b in jtetra
   //     - jdx:    index of the vertex of jshare opposite to the face of jshare shared with jtetra
   //     - kshare: index of tetrahedron sharing the face opposite to b in ktetra
   //     - kdx:    index of the vertex of kshare opposite to the face of kshare shared with ktetra
   //     - lshare: index of tetrahedron sharing the face opposite to b in ltetra
   //     - ldx:    index of the vertex of lshare opposite to the face of lshare shared with ltetra

   ishare = tetra[itetra].neighbors[idp];
   jshare = tetra[jtetra].neighbors[jdp];
   kshare = tetra[ktetra].neighbors[kdp];
   lshare = tetra[ltetra].neighbors[ldp];

   idx = tetra[itetra].nindex[idp];
   jdx = tetra[jtetra].nindex[jdp];
   kdx = tetra[ktetra].nindex[kdp];
   ldx = tetra[ltetra].nindex[ldp];

   if (free.size() > 0) {
      newtetra = free.top();
      free.pop();
   }
   else {
      newtetra = tetra.size();
      Tetrahedron t;
      t.init();
      tetra.push_back(t);
   }

   tetra_last = newtetra;

   // itetra, jtetra, ktetra and ltetra become "available";
   // they are added to the "kill" zone

   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;
   tetra[ktetra].info[1] = 0;
   tetra[ltetra].info[1] = 0;
   kill.push_back(itetra);
   kill.push_back(jtetra);
   kill.push_back(ktetra);
   kill.push_back(ltetra);

   // I need :
   //     - the vertex b that is shared by all 4 tetrahedra
   //     - the vertices a, c, p and o
   //     - for each tetrahedron, find neighbour attached to the face
   //       oposite to b; this information is stored in *share,
   //       where * can be i, j, k or l

   a = ivertices[0];
   b = ivertices[1];
   c = ivertices[2];
   p = ivertices[3];
   o = ivertices[4];

   // For bookkeeping reason, p is set to be the last vertex of the new tetrahedron

   // Now I define the new tetrahedron: (acop)

   // tetrahedron acop : neighbor of (bcop) on face cpo, neighbor of (abop)
   //                    on face apo, neighbor of (abcp) on face acp
   //                    and neighbor of (abco) on face aco

   vertices[b].info[0] = 0;

   tetra[newtetra].vertices[0] = a;
   tetra[newtetra].neighbors[0] = lshare;
   tetra[newtetra].nindex[0] = ldx;
   tetra[newtetra].info[2] = lkeep[2+ldp];
   if (lshare != -1 && ldx != -1) {
      tetra[lshare].neighbors[ldx] = newtetra;
      tetra[lshare].nindex[ldx] = 0;
   }

   tetra[newtetra].vertices[1] = c;
   tetra[newtetra].neighbors[1] = kshare;
   tetra[newtetra].nindex[1] = kdx;
   tetra[newtetra].info[3] = kkeep[2+kdp];
   if (kshare != -1 && kdx != -1) {
      tetra[kshare].neighbors[kdx] = newtetra;
      tetra[kshare].nindex[kdx] = 1;
   }

   tetra[newtetra].vertices[2] = o;
   tetra[newtetra].neighbors[2] = ishare;
   tetra[newtetra].nindex[2] = idx;
   tetra[newtetra].info[4] = kkeep[2+idp];
   if (ishare != -1 && idx != -1) {
      tetra[ishare].neighbors[idx] = newtetra;
      tetra[ishare].nindex[idx] = 2;
   }

   tetra[newtetra].vertices[3] = p;
   tetra[newtetra].neighbors[3] = jshare;
   tetra[newtetra].nindex[3] = jdx;
   tetra[newtetra].info[5] = kkeep[2+jdp];
   if (jshare != -1 && jdx != -1) {
      tetra[jshare].neighbors[jdx] = newtetra;
      tetra[jshare].nindex[jdx] = 3;
   }

   tetra[newtetra].info[1] = 1;
   if (test1==1) {
      tetra[newtetra].info[0] = 1;
   }
   else {
      tetra[newtetra].info[0] = 0;
   }

   // Now add one link facet:
   // link_facet:	aco	tetrahedra:	acop and neighbour of (abco) on aco

   link_facet.push(std::make_pair(newtetra, jshare));
   link_index.push(std::make_pair(3, jdx));
}

// flip_2_3 implements a 2->3 flip in 3D for regular triangulation
void flip_2_3(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int* vertices,
   int* facei, int* facej,bool test_abpo, bool test_bcpo, bool test_capo, int& ierr, int& tetra_last,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill)
{
   int k,p,o;
   int it,jt,idx,jdx;
   int newtetra;
   std::bitset<8> ikeep,jkeep;
   int face[3];
   int itetra_touch[3],jtetra_touch[3];
   char jtetra_idx[3],itetra_idx[3];
   int idx_list[3][2]={{0,0},{0,1},{1,1}};
   int tests[3],position[3];

   ierr = 0;

   // f itetra or jtetra are inactive, cannot flip
   if (tetra[itetra].info[1] == 0 || tetra[jtetra].info[1] == 0) {
      ierr = 1;
      return;
   }

   // Define
   // - itetra_touch: the three tetrahedra that touches itetra on the
   //         faces opposite to the 3 vertices a,b,c
   // - itetra_idx: for the three tetrahedra defined by itetra_touch,
   //         index of the vertex opposite to the face common with itetra
   // - jtetra_touch: the three tetrahedra that touches jtetra on the
   //         faces opposite to the 3 vertices a,b,c
   // - jtetra_idx: for the three tetrahedra defined by jtetra_touch,
   //         index of the vertex opposite to the face common with jtetra

   for (int i = 0; i < 3; i++) {
      itetra_touch[i] = tetra[itetra].neighbors[facei[i]];
      jtetra_touch[i] = tetra[jtetra].neighbors[facej[i]];
      itetra_idx[i] = tetra[itetra].nindex[facei[i]];
      jtetra_idx[i] = tetra[jtetra].nindex[facej[i]];
   }

   // first three vertices define face that is removed
   face[0] = vertices[0];
   face[1] = vertices[1];
   face[2] = vertices[2];
   p = vertices[3];
   o = vertices[4];

   // The three new tetrahedra are stored in : 
   // - any free space in the tetrahedron list,
   // - at the end of the list of known tetrahedra if needed

   k = 0;
   while (!free.empty() && k < 3) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 3; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[2];

   // set itetra and jtetra to 0, and add them to kill list
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;

   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;

   kill.push_back(itetra);
   kill.push_back(jtetra);

   // Define the three new tetrahedra: (bcop), (acop) and (abop) as well as their neighbours

   // tetrahedron bcop : neighbours are acop, abop, neighbour of (abcp)
   //             on face bcp, and neighbour of (abco) on face bco
   // tetrahedron acop : neighbours are bcop, abop, neighbour of (abcp)
   //             on face acp, and neighbour of (abco) on face aco
   // tetrahedron abop : neighbours are bcop, acop, neighbour of (abcp)
   //             on face abp, and neighbour of (abco) on face abo

   tests[0] = 1;
   if (test_bcpo) tests[0] = -1;
   tests[1] = -1;
   if (test_capo) tests[1] = 1;
   tests[2] = 1;
   if (test_abpo) tests[2] = -1;

   for (int i = 0; i < 3; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 3; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = face[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k]=idx_list[i][k];
         k++;
      }

      tetra[newtetra].vertices[2] = o;
      it = itetra_touch[i];
      idx = itetra_idx[i];
      tetra[newtetra].neighbors[2] = it;
      tetra[newtetra].nindex[2] = idx;
      tetra[newtetra].info[5] = ikeep[2+facei[i]];
      if (it !=-1 && idx !=-1) {
         tetra[it].neighbors[idx]=newtetra;
         tetra[it].nindex[idx] = 2;
      }

      tetra[newtetra].vertices[3] = p;
      jt = jtetra_touch[i];
      jdx = jtetra_idx[i];
      tetra[newtetra].neighbors[3] = jt;
      tetra[newtetra].nindex[3] = jdx;
      tetra[newtetra].info[6] = jkeep[2+facej[i]];
      if (jt !=-1 && jdx !=-1) {
         tetra[jt].neighbors[jdx]=newtetra;
         tetra[jt].nindex[jdx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      if (tests[i]==1) {
         tetra[newtetra].info[0] = 1;
      }
      else {
         tetra[newtetra].info[0] = 0;
      }
   }

   // Now add all three faces of jtetra containing o in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the intersection of two tetrahedra

   // link_facet: bco tetrahedra: bcop and neighbour of (abco) on bco
   // link_facet: aco tetrahedra: acop and neighbour of (abco) on aco
   // link_facet: abo tetrahedra: abop and neighbour of (abco) on abo

   for (int i = 0; i < 3; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      idx = tetra[newtetra].nindex[3];
      link_index.push(std::make_pair(3,idx));
   }
}

// flip_3_2 implements a 3->2 flip in 3D for regular triangulation
void flip_3_2(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra, int* vertices,
   int* edgei, int* edgej, int* edgek, bool test_bcpo, bool test_acpo, int& ierr, int& tetra_last,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill)
{
   int k,p,o,c;
   int it,jt,kt,idx,jdx,kdx;
   int newtetra;
   std::bitset<8> ikeep,jkeep,kkeep;
   int edge[2],tests[2];
   int itetra_touch[2],jtetra_touch[2],ktetra_touch[2];
   int position[2];
   char itetra_idx[2],jtetra_idx[2],ktetra_idx[2];

   tests[0] = 1;
   if (test_bcpo) tests[0] = -1;
   tests[1] = 1;
   if (test_acpo) tests[1] = -1;
   ierr = 0;

   // if itetra, jtetra or ktetra are inactive, cannot flip
   if (tetra[itetra].info[1]==0 || tetra[jtetra].info[1]==0 || tetra[ktetra].info[1]==0)
   {
      ierr = 1;
      return;
   }

   // store old info
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;
   kkeep = tetra[ktetra].info;

   // Define
   //     - itetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in itetra, respectively
   //     - itetra_idx:   for the two tetrahedra defined by itetra_touch,
   //                     index position of the vertex opposite to the face common with itetra
   //     - jtetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in jtetra, respectively
   //     - jtetra_idx:   for the two tetrahedra defined by jtetra_touch,
   //                     index position of the vertex opposite to the face common with jtetra
   //     - ktetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in ktetra, respectively
   //     - ktetra_idx:   for the two tetrahedra defined by ktetra_touch,
   //                     index position of the vertex opposite to the face common with ktetra

   for (int i = 0; i < 2; i++) {
      itetra_touch[i] = tetra[itetra].neighbors[edgei[i]];
      jtetra_touch[i] = tetra[jtetra].neighbors[edgej[i]];
      ktetra_touch[i] = tetra[ktetra].neighbors[edgek[i]];
      itetra_idx[i]   = tetra[itetra].nindex[edgei[i]];
      jtetra_idx[i]   = tetra[jtetra].nindex[edgej[i]];
      ktetra_idx[i]   = tetra[ktetra].nindex[edgek[i]];
   }

   edge[0] = vertices[0];
   edge[1] = vertices[1];
   c       = vertices[2];
   p       = vertices[3];
   o       = vertices[4];


   // the two new tetrahedra are going to be stored "free" space, or at the end of the list
   k = 0;
   while (!free.empty() && k < 2) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 2; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[1];

   // itetra, jtetra and ktetra becomes "available"; they are added to the "kill" list
   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;
   tetra[ktetra].info[1] = 0;

   kill.push_back(itetra);
   kill.push_back(jtetra);
   kill.push_back(ktetra);

   // I need :
   //     - the two vertices that define their common edge (ab)
   //         these vertices are stored in the array edge
   //     - the vertices c, p and o that form the new triangle
   //     - for each vertex in the edge (ab), define the opposing
   //         faces in the three tetrahedra itetra, jtetra and ktetra, and
   //         the tetrahedron that share these faces with itetra, jtetra and
   //         ktetra, respectively. This information is stored
   //         in three arrays, itetra_touch, jtetra_touch and ktetra_touch

   // These information are given by the calling program

   // For bookkeeping reasons, I always set p to be the last vertex
   // of the new tetrahedra

   // Now I define the two new tetrahedra: (bcop) and (acop)
   // as well as their neighbours

   // tetrahedron bcop : neighbours are acop, neighbour of (abop)
   //             on face bpo, neighbour of (abcp) on face bcp
   //             and neighbour of (abco) on face (bco)
   // tetrahedron acop : neighbours are bcop, neighbour of (abop)
   //             on face apo, neighbour of (abcp) on face acp
   //             and neighbour of (abco) on face (aco)

   for (int i = 0; i < 2; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 2; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = edge[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k] = 0;
         k++;
      }

      tetra[newtetra].vertices[1] = c;
      kt = ktetra_touch[i];
      kdx = ktetra_idx[i];
      tetra[newtetra].neighbors[1] = kt;
      tetra[newtetra].nindex[1] = kdx;
      tetra[newtetra].info[3] = kkeep[2+edgek[i]];
      if (kdx != -1 && kt != -1) {
         tetra[kt].neighbors[kdx] = newtetra;
         tetra[kt].nindex[kdx] = 1;
      }

      tetra[newtetra].vertices[2] = o;
      it = itetra_touch[i];
      idx = itetra_idx[i];
      tetra[newtetra].neighbors[2] = it;
      tetra[newtetra].nindex[2] = idx;
      tetra[newtetra].info[4] = ikeep[2+edgek[i]];
      if (idx != -1 && it != -1) {
         tetra[it].neighbors[idx] = newtetra;
         tetra[it].nindex[idx] = 2;
      }

      tetra[newtetra].vertices[3] = p;
      jt = jtetra_touch[i];
      jdx = jtetra_idx[i];
      tetra[newtetra].neighbors[3] = jt;
      tetra[newtetra].nindex[3] = jdx;
      tetra[newtetra].info[5] = jkeep[2+edgej[i]];
      if (jdx != -1 && jt != -1) {
         tetra[jt].neighbors[jdx] = newtetra;
         tetra[jt].nindex[jdx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      if (tests[i] == 1) {
         tetra[newtetra].info[0] = 1;
      }
      else {
         tetra[newtetra].info[0] = 0;
      }
   }

   // Now add the two faces of ktetra containing (co) in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the
   // intersection of two tetrahedra

   // link_facet: bco tetrahedra: bcop and neighbour of (abco) on bco
   // link_facet: aco tetrahedra: acop and neighbour of (abco) on aco

   for (int i = 0; i < 2; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      link_index.push(std::make_pair(3, tetra[newtetra].nindex[3]));
   }
}

// find_tetra tests if four given points form an
// existing tetrahedron in the current Delaunay
inline void find_tetra(std::vector<Tetrahedron>& tetra, int itetra, int idx_c,
   int a, int b, int o, int& ifind, int& tetra_loc, int& idx_a, int& idx_b)
{
   // We are testing if tetrahedron (abpo) exists. If it exists, it is
   // a neighbour of abcp, on the face opposite to vertex c.
   // We test that tetrahedron and see if it contains o

   int ot, otx, otest;

   ot = tetra[itetra].neighbors[idx_c];
   otx = tetra[itetra].nindex[idx_c];
   otest = tetra[ot].vertices[otx];

   if (otest == o) {
      ifind = 1;
      tetra_loc = ot;

      // We found the tetrahedron, let us define the position
      // of a and b in this tetrahedron
      for (int i = 0; i < 4; i++) {
         if (tetra[tetra_loc].vertices[i] == a) {
            idx_a = i;
         }
         else if (tetra[tetra_loc].vertices[i] == b) {
            idx_b = i;
         }
      }
   }
   else {
      ifind = 0;
   }
}

// define_facet definess the triangle of intersection of two tetrahedra
inline void define_facet(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int idx_o, int* facei, int* facej)
{
   int ia, ib, ie, ig;
   int k;

   // I need to :
   //     - find the three vertices that define their common face
   //         these vertices are stored in the array triangle
   //     - find the vertices p and o

   // To define the common face of the two tetrahedra itetra and jtetra,
   // I look at the neighbours of itetra : one of them is jtetra!
   // This also provides p. The same procedure is repeated for jtetra,
   // to get o

   for (int i = 0; i < 3; i++) facei[i] = i;

   ia = tetra[itetra].vertices[0];
   for (int i = 0; i < 3; i++) {
      k = other[idx_o][i];
      ie = tetra[jtetra].vertices[k];
      if (ia == ie) {
         facej[0] = k;
         break;
      }
   }

   ib = tetra[itetra].vertices[1];
   ie = other2[idx_o][facej[0]][0];
   ig = other2[idx_o][facej[0]][1];
   if (ib == tetra[jtetra].vertices[ie]) {
      facej[1] = ie;
      facej[2] = ig;
   }
   else {
      facej[1] = ig;
      facej[2] = ie;
   }
}

// flip restores the regularity of Delaunay triangulation
void flip(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill)
{
   int ii,ij;
   int idxi,idxj,idxk,idxl;
   int p,o,a,b,c;
   int itetra,jtetra,idx_p,idx_o;
   int itest_abcp;
   int ierr,tetra_last;
   int ireflex,iflip,iorder;
   int ifind,tetra_ab,tetra_ac,tetra_bc;
   int idx_a,idx_b,idx_c;
   int ia,ib,ic;
   int facei[3],facej[3],edgei[2],edgej[2],edgek[2];
   int edge_val[3][2];
   int tetra_flip[3],list_flip[3];
   int vert_flip[5];
   std::pair<int,int> facet,index;
   bool convex,regular,test_abpo,test_bcpo,test_capo;
   bool test,test_abpc,test_bcpa,test_acpo,test_acpb;
   bool test_or[3][2];
   idx_a = 0;
   idx_b = 0;
   idx_c = 0;

   // go over all link facets
   while (!link_facet.empty()) {
      // First define the two tetrahedra that
      // contains the link facet as itetra and jtetra
      facet = link_facet.front();
      index = link_index.front();
      link_facet.pop();
      link_index.pop();

      itetra = facet.first;
      jtetra = facet.second;
      idx_p = index.first;
      idx_o = index.second;

      // if the link facet is on the convex hull, discard
      if (itetra==-1 || jtetra == -1) continue;

      // if these tetrahedra have already been discarded, discard this link facet
      if (tetra[itetra].info[1] == 0) {
         if (tetra[jtetra].info[1] == 0) {
            continue;
         }
         else {
            itetra = tetra[jtetra].neighbors[idx_o];
            idx_p = tetra[jtetra].nindex[idx_o];
         }
      }

      if (tetra[jtetra].info[1] == 0) {
         jtetra = tetra[itetra].neighbors[idx_p];
         idx_o = tetra[itetra].nindex[idx_p];
      }

      // define the vertices of the two tetrahedra
      a = tetra[itetra].vertices[0];
      b = tetra[itetra].vertices[1];
      c = tetra[itetra].vertices[2];
      p = tetra[itetra].vertices[3];

      o = tetra[jtetra].vertices[idx_o];

      itest_abcp = -1;
      if (tetra[itetra].info[0]==1) itest_abcp = 1;

      // check for local regularity (and convexity)
      regular_convex(vertices, a, b, c, p, o, itest_abcp, regular, convex, test_abpo, test_bcpo, test_capo);

      // if the link facet is locally regular, discard
      if (regular) continue;

      // define neighbors of the facet on itetra and jtetra
      define_facet(tetra, itetra, jtetra, idx_o, facei, facej);

      test_abpc = (itest_abcp != 1);

      // After discarding the trivial case, we now test if the tetrahedra
      // can be flipped. 

      // At this stage, I know that the link facet is not locally
      // regular. I still don t know if it is "flippable"

      // I first check if {itetra} U {jtetra} is convex. If it is, I
      // perform a 2-3 flip (this is the convexity test performed
      // at the same time as the regularity test)

      if (convex) {
         vert_flip[0] = a;
         vert_flip[1] = b;
         vert_flip[2] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         flip_2_3(tetra, itetra, jtetra, vert_flip, facei, facej, test_abpo, test_bcpo, test_capo, ierr, tetra_last, link_facet, link_index, free, kill);
         continue;
      }

      // The union of the two tetrahedra is not convex...
      // I now check the edges of the triangle in the link facet, and
      // check if they are "reflexes" (see definition in Edelsbrunner and
      // Shah, Algorithmica (1996), 15:223-241)

      ireflex = 0;
      iflip = 0;

      // First check edge (ab): 
      // - (ab) is reflex iff o and c lies on opposite sides of
      // the hyperplane defined by (abp). We therefore test the
      // orientation of (abpo) and (abpc): if they differ (ab)
      // is reflex
      // - if (ab) is reflex, we test if it is of degree 3.
      // (ab) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (abpo). The first two are itetra
      // and jtetra, so we only need to check if (abpo) exists.
      // since (abpo) contains p, (abp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      if (test_abpo != test_abpc) {
         ireflex++;
         find_tetra(tetra, itetra, 2, a, b, o, ifind, tetra_ab, idx_a, idx_b);

         if (ifind==1) {
            tetra_flip[iflip] = tetra_ab;
            list_flip[iflip] = 0;
            edge_val[iflip][0] = idx_a;
            edge_val[iflip][1] = idx_b;
            test_or[iflip][0] = test_bcpo;
            test_or[iflip][1] = !test_capo;
            iflip++;
         }
      }

      // Now check edge (ac): 
      // - (ac) is reflex iff o and b lies on opposite sides of
      // the hyperplane defined by (acp). We therefore test the
      // orientation of (acpo) and (acpb): if they differ (ac)
      // is reflex
      // - if (ac) is reflex, we test if it is of degree 3.
      // (ac) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (acpo). The first two are itetra
      // and jtetra, so we only need to check if (acpo) exists.
      // since (acpo) contains p, (acp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      test_acpo = !test_capo;
      test_acpb = !test_abpc;

      if (test_acpo != test_acpb) {

         ireflex++;
         find_tetra(tetra, itetra, 1, a, c, o, ifind, tetra_ac, idx_a, idx_c);

         if (ifind==1)  {
            tetra_flip[iflip] = tetra_ac;
            list_flip[iflip] = 1;
            edge_val[iflip][0] = idx_a;
            edge_val[iflip][1] = idx_c;
            test_or[iflip][0] = !test_bcpo;
            test_or[iflip][1] = test_abpo;
            iflip++;
         }
      }

      // Now check edge (bc): 
      // - (bc) is reflex iff o and a lies on opposite sides of
      // the hyperplane defined by (bcp). We therefore test the
      // orientation of (bcpo) and (bcpa): if they differ (bc)
      // is reflex
      // - if (bc) is reflex, we test if it is of degree 3.
      // (bc) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (bcpo). The first two are itetra
      // and jtetra, so we only need to check if (bcpo) exists.
      // since (bcpo) contains p, (bcp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      test_bcpa = test_abpc;
      if (test_bcpo != test_bcpa) {

         ireflex++;
         find_tetra(tetra, itetra, 0, b, c, o, ifind, tetra_bc, idx_b, idx_c);

         if (ifind==1)  {
            tetra_flip[iflip] = tetra_bc;
            list_flip[iflip] = 2;
            edge_val[iflip][0] = idx_b;
            edge_val[iflip][1] = idx_c;
            test_or[iflip][0] = test_capo;
            test_or[iflip][1] = !test_abpo;
            iflip++;
         }
      }

      if (ireflex != iflip) continue;

      if (iflip==1) {
         // only one edge is "flippable": we do a 3-2 flip
         iorder = list_flip[0];
         ia = table32[iorder][0];
         ib = table32[iorder][1];
         ic = table32[iorder][2];
         vert_flip[ia] = a;
         vert_flip[ib] = b;
         vert_flip[ic] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         ia = table32_2[iorder][0];
         ib = table32_2[iorder][1];
         edgei[0] = ia;
         edgei[1] = ib;
         edgej[0] = facej[ia];
         edgej[1] = facej[ib];
         edgek[0] = edge_val[0][0];
         edgek[1] = edge_val[0][1];
         flip_3_2(tetra, itetra, jtetra, tetra_flip[0], vert_flip, edgei, edgej, edgek, test_or[0][0], test_or[0][1], ierr, tetra_last, link_facet, link_index, free, kill);
      }
      else if (iflip==2) {
         // In this case, one point is redundant: the point common to
         // the two edges that can be flipped. We then perform a 4-1 flip
         iorder = list_flip[0] + list_flip[1] -1;
         vert_flip[table41[iorder][0]] = a;
         vert_flip[table41[iorder][1]] = b;
         vert_flip[table41[iorder][2]] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         ii = table41_2[iorder][0];
         ij = table41_2[iorder][1];
         idxi = iorder;
         idxj = facej[iorder];
         idxk = edge_val[0][ii];
         idxl = edge_val[1][ij];

         if (iorder==0) {
            test = test_bcpo;
         }
         else if (iorder==1) {
            test = !test_capo;
         }
         else {
            test = test_abpo;
         }
         flip_4_1(vertices, tetra, itetra, jtetra, tetra_flip[0], tetra_flip[1], vert_flip, idxi, idxj, idxk, idxl, test, ierr, tetra_last, link_facet, link_index, free, kill);
      }
      else {
         TINKER_THROW("DELFLIP  --  Three Edges Flippable");
      }
   }

   for (int i = 0; i < kill.size(); i++) {
      free.push(kill[i]);
   }
   kill.clear();
}
}
