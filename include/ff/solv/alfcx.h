#pragma once
#include "ff/solv/alphamol.h"
#include <vector>

namespace tinker {
class Alfcx {
public:
   void alfcx(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
      std::vector<Edge>& edges, std::vector<Face>& faces, double alpha);

private:
   void alfcxedges(std::vector<Tetrahedron>& tetra, std::vector<Edge>& edges);

   void alfcxfaces(std::vector<Tetrahedron>& tetra, std::vector<Face>& faces);

   inline int findedge(Tetrahedron t, int i1, int j1);

   void alf_tetra(double* a, double* b, double* c, double* d,
      double ra, double rb, double rc, double rd, int& iflag, double alpha);

   void alf_trig(double* a, double* b, double* c, double* d, double* e,
      double ra,double rb, double rc, double rd, double re, int ie,
      int& irad,int& iattach, double alpha);

   void alf_edge(std::vector<Vertex>& vertices, double* a, double* b, double ra, double rb,
      double* cg, std::vector<int>& listcheck, int& irad, int& iattach, double alpha);

   inline void edge_radius(double* a, double* b, double ra, double rb,
      double* Dab, double* Sab, double* Tab, int& testr, double alpha);

   inline void edge_attach(double* a, double* b, double* c, double ra, double rb,
      double rc, double* Dab, double* Sab, double* Tab, int& testa);

   inline void triangle_attach(double* a, double* b, double* c, double* d,
      double ra, double rb, double rc, double rd, double S[3][4], double T[2][3],
      double Dabc, int& testa, int& memory);

   inline void triangle_radius(double* a, double* b, double* c, double ra, double rb,
      double rc, double S[3][4], double T[2][3], double Dabc, int& testr, double alpha, int& memory);

   inline void vertex_attach(double* a, double* b, double ra, double rb, int& testa, int& testb);

   inline void get_coord2(std::vector<Vertex>& vertices, int ia, int ja, double* a, double* b, double* cg, double& ra, double& rb);

   inline void get_coord4(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la, double* a, double* b,
      double* c, double* d, double& ra, double& rb, double& rc, double& rd);

   inline void get_coord5(std::vector<Vertex>& vertices, int ia, int ja, int ka, int la,
      int ma, double* a, double* b, double* c, double* d, double* e,
      double& ra, double& rb, double& rc, double& rd, double& re);

   // ALFCX_GMP alf_gmp;

protected:
   int other3[4][3] = {
      {1, 2, 3},
      {0, 2, 3},
      {0, 1, 3},
      {0, 1, 2} };

   int face_edge[4][3] = {
      { 2, 1, 0},
      { 4, 3, 0},
      { 5, 3, 1},
      { 5, 4, 2},
   };

   int face_info[6][2] = {
      {0, 1},
      {0, 2},
      {0, 3},
      {1, 2},
      {1, 3},
      {2, 3}};

   int face_pos[6][2] = {
      {1, 0},
      {2, 0},
      {3, 0},
      {2, 1},
      {3, 1},
      {3, 2}};

   int pair[6][2] = {
      {2, 3},
      {1, 3},
      {1, 2},
      {0, 3},
      {0, 2},
      {0, 1}};

   double eps;
};
}
