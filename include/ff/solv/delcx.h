#pragma once
#include "ff/solv/alphamol.h"
#include "ff/solv/delsort.h"
#include "ff/solv/sosgmp.h"
#include <queue>
#include <stack>
#include <vector>

namespace tinker
{
class Delcx {
public:
   // Setup calculations
   void init(int natoms, AlfAtom* alfatoms, std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);

   // Compute 3D weighted Delaunay triangulation
   template <bool use_sos>
   void regular3D(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);

private:
   // locate tetrahedron in which point is inserted
   template <bool use_sos>
   void locate_jw(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int ipoint, int& tetra_loc, bool& iredundant);

   // go over full link_facet
   template <bool use_sos>
   void flip(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);

   // Check if a point is inside a tetrahedron
   template <bool use_sos>
   void inside_tetra(std::vector<Vertex>& vertices, int p, int a, int b, int c, int d, int iorient, bool& is_in, bool& redundant, int& ifail);

   // Check if a facet connects two tetrehedra that are convex
   template <bool use_sos>
   void regular_convex(std::vector<Vertex>& vertices, int a, int b, int c, int p, int o, int itest_abcp,
      bool& regular, bool& convex, bool& test_abpo, bool& test_bcpo, bool& test_capo);

   // sign associated with the missing inf point
   inline void missinf_sign(int i, int j, int k, int& l, int& sign);

   // flip 1-4: from 1 to 4 tetrahedra
   void flip_1_4(std::vector<Tetrahedron>& tetra, int ipoint, int itetra, int& tetra_last);

   // flip 4-1: from 4 to 1 tetrahedron
   void flip_4_1(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra,
      int ltetra, int* ivertices, int idp, int jdp, int kdp, int ldp, bool test_acpo, int& ierr, int& tetra_last);

   // flip 2-3: from 2 to 3 tetrahedra
   void flip_2_3(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int* vertices,
      int* facei, int* facej, bool test_abpo, bool test_bcpo, bool test_capo, int& ierr, int& tetra_last);

   // flip 3-2: from 3 to 2 tetrahedra
   void flip_3_2(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra, int* vertices,
      int* edgei, int* edgej, int* edgek, bool test_bcpo, bool test_acpo, int& ierr, int& tetra_last);

   // Define facet between two tetrahedron
   inline void define_facet(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int idx_o, int* facei, int* facej);

   // info about tetrahedron
   inline void find_tetra(std::vector<Tetrahedron>& tetra, int itetra, int idx_c,
      int a, int b, int o,int& ifind, int& tetra_loc, int& idx_a, int& idx_b);

   // reorder vertices of tetrahedra in increasing order
   inline void reorder_tetra(std::vector<Tetrahedron>& tetra);

   // remove "infinite" tetrahedron
   inline void remove_inf(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);

   // mark tetrahedron
   inline void mark_zero(std::vector<Tetrahedron>& tetra, int itetra, int ivertex);

   // remove flat tetrahedra
   template <bool use_sos>
   void peel(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int& flag);

   // Computes the volume of a tetrahedron
   inline double tetra_vol(double* a, double* b, double* c, double* d);

   // find edge in tetra defined by 2 vertices
   inline int findedge(Tetrahedron t, int i1, int j1);

   // Add bogus points as needed so that we have at least 4 points
   void addBogus(int npoints, AlfAtom* alfatoms, double* bcoord, double* brad);

   SOS sos;

   DelcxSort mysort;

protected:
   std::queue<std::pair<int,int>> link_facet;
   std::queue<std::pair<int,int>> link_index;
   std::stack<int> free;
   std::vector<int> kill;

   double eps = 1e-2;

   int inf4_1[4] = {1, 1, 0, 0};
   int sign4_1[4] = {-1, 1, 1, -1};
   int inf4_2[4][4] = {
      { -1, 1, 2, 2},
      { 1, -1, 2, 2},
      { 2, 2, -1, 0},
      { 2, 2, 0, -1}
   };
   int sign4_2[4][4] = {
      { 0, 1, -1, 1},
      { -1, 0, 1, -1},
      { 1, -1, 0, 1},
      { -1, 1, -1, 0}
   };
   int sign4_3[4] = {-1, 1, -1, 1};
   int inf5_2[4][4] = {
      { -1, 1, 0, 0},
      { 1, -1, 0, 0},
      { 0, 0, -1, 0},
      { 0, 0, 0, -1}
   };
   int sign5_2[4][4] = {
      { 0, -1, -1, 1},
      { 1, 0, -1, 1},
      { 1, 1, 0, 1},
      { -1, -1, -1, 0}
   };
   int inf5_3[4] = {0, 0, 2, 2};
   int sign5_3[4] = {1, 1, -1, 1};
   int order1[4][3] = {
      { 2, 1, 3},
      { 0, 2, 3},
      { 1, 0, 3},
      { 0, 1, 2}
   };

   int ord_rc[3][3] = {
      {0, 1, 2},
      {2, 0, 1},
      {1, 2, 0},
   };

   int order2[6][2] = {
      { 2, 3},
      { 3, 1},
      { 1, 2},
      { 0, 3},
      { 2, 0},
      { 0, 1}
   };
   int order3[6][2] = {
      { 0, 1},
      { 0, 2},
      { 0, 3},
      { 1, 2},
      { 1, 3},
      { 2, 3}
   };
   int idxList[4][3] = {
      { 0, 0, 0},
      { 0, 1, 1},
      { 1, 1, 2},
      { 2, 2, 2}
   };
   int table32[3][3] = {
      { 0, 1, 2},
      { 0, 2, 1},
      { 2, 0, 1}
   };
   int table32_2[3][2] = {
      { 0, 1},
      { 0, 2},
      { 1, 2}
   };
   int table41[3][3] = {
      { 1, 0, 2},
      { 0, 1, 2},
      { 0, 2, 1}
   };
   int table41_2[3][2] = {
      { 0, 0},
      { 1, 0},
      { 1, 1}
   };
   int order[3][2] = {
      { 1, 2},
      { 2, 0},
      { 0, 1}
   };
   int other[4][3] = {
      { 1, 2, 3},
      { 0, 2, 3},
      { 0, 1, 3},
      { 0, 1, 2}
   };
   int other2[4][4][2] = {
      {
         { -1, -1},
         { 2, 3},
         { 1, 3},
         { 1, 2}
      },
      {
         { 2, 3},
         { -1, -1},
         { 0, 3},
         { 0, 2}
      },
      {
         { 1, 3},
         { 0, 3},
         { -1, -1},
         { 0, 1}
      },
      {
         { 1, 2},
         { 0, 2},
         { 0, 1},
         { -1, -1}
      },
   };
};
}
