#pragma once
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"
#include <algorithm>
#include <bitset>
#include <cmath>
#include <queue>
#include <stack>
#include <vector>

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
class AlfAtom {
public:
   int index;
   double r;
   double coord[3];
   double coefs,coefv;

   AlfAtom() {}

   AlfAtom(int idx, double x, double y, double z, double r, double coefs, double coefv);

   ~AlfAtom();

private:
   double truncate_real(double x, int ndigit);
};

class Vertex {
public:
   double r;
   double coord[3];
   double w;
   double coefs,coefv;
   double gamma;

   std::bitset<8> info;
   bool status;

   Vertex() {}

   Vertex(double x, double y, double z, double r, double coefs, double coefv);

   ~Vertex();

private:
   double truncate_real(double x, int ndigit);
};

class Tetrahedron {
public:
   int vertices[4];
   int neighbors[4];
   std::bitset<8> info;
   int info_edge[6];
   int nindex[4];

   Tetrahedron();
   ~Tetrahedron();

   void init();
};

class Edge {
public:
   int vertices[2];
   double gamma;
   double len,surf,vol;
   double dsurf,dvol;

   Edge() {}

   Edge(int i, int j);

   ~Edge();
};

class Face {
public:
   int vertices[3];
   int edges[3];
   double gamma;

   Face() {}

   Face(int i, int j, int k, int e1, int e2, int e3, double S);

   ~Face();
};

enum class AlfMethod
{
   AlphaMol,
   AlphaMol2,
};

enum class AlfSort
{
   None,
   Sort3D,
   BRIO,
   Split,
   KDTree,
};

constexpr double deleps = 1e-4;
constexpr double delepsvol = 1e-4;
constexpr double alfeps = 1e-5;
TINKER_EXTERN double wsurf;
TINKER_EXTERN double wvol;
TINKER_EXTERN double* radii;
TINKER_EXTERN double* coefS;
TINKER_EXTERN double* coefV;
TINKER_EXTERN double* surf;
TINKER_EXTERN double* vol;
TINKER_EXTERN double* dsurfx;
TINKER_EXTERN double* dsurfy;
TINKER_EXTERN double* dsurfz;
TINKER_EXTERN double* dvolx;
TINKER_EXTERN double* dvoly;
TINKER_EXTERN double* dvolz;
TINKER_EXTERN std::vector<AlfAtom> alfatoms;
TINKER_EXTERN AlfMethod alfmeth;
TINKER_EXTERN AlfSort alfsort;
TINKER_EXTERN int alfdigit;
TINKER_EXTERN int alfnthd;
TINKER_EXTERN bool alfh;
TINKER_EXTERN bool alfdebug;

constexpr int inf4_1[4] = {1, 1, 0, 0};
constexpr int sign4_1[4] = {-1, 1, 1, -1};
constexpr int inf4_2[4][4] = {
   { -1, 1, 2, 2},
   { 1, -1, 2, 2},
   { 2, 2, -1, 0},
   { 2, 2, 0, -1}
};
constexpr int sign4_2[4][4] = {
   { 0, 1, -1, 1},
   { -1, 0, 1, -1},
   { 1, -1, 0, 1},
   { -1, 1, -1, 0}
};
constexpr int sign4_3[4] = {-1, 1, -1, 1};
constexpr int inf5_2[4][4] = {
   { -1, 1, 0, 0},
   { 1, -1, 0, 0},
   { 0, 0, -1, 0},
   { 0, 0, 0, -1}
};
constexpr int sign5_2[4][4] = {
   { 0, -1, -1, 1},
   { 1, 0, -1, 1},
   { 1, 1, 0, 1},
   { -1, -1, -1, 0}
};
constexpr int inf5_3[4] = {0, 0, 2, 2};
constexpr int sign5_3[4] = {1, 1, -1, 1};
constexpr int order1[4][3] = {
   { 2, 1, 3},
   { 0, 2, 3},
   { 1, 0, 3},
   { 0, 1, 2}
};
constexpr int ord_rc[3][3] = {
   {0, 1, 2},
   {2, 0, 1},
   {1, 2, 0},
};
constexpr int order2[6][2] = {
   { 2, 3},
   { 3, 1},
   { 1, 2},
   { 0, 3},
   { 2, 0},
   { 0, 1}
};
constexpr int order3[6][2] = {
   { 0, 1},
   { 0, 2},
   { 0, 3},
   { 1, 2},
   { 1, 3},
   { 2, 3}
};
constexpr int idxList[4][3] = {
   { 0, 0, 0},
   { 0, 1, 1},
   { 1, 1, 2},
   { 2, 2, 2}
};
constexpr int table32[3][3] = {
   { 0, 1, 2},
   { 0, 2, 1},
   { 2, 0, 1}
};
constexpr int table32_2[3][2] = {
   { 0, 1},
   { 0, 2},
   { 1, 2}
};
constexpr int table41[3][3] = {
   { 1, 0, 2},
   { 0, 1, 2},
   { 0, 2, 1}
};
constexpr int table41_2[3][2] = {
   { 0, 0},
   { 1, 0},
   { 1, 1}
};
constexpr int order[3][2] = {
   { 1, 2},
   { 2, 0},
   { 0, 1}
};
constexpr int other[4][3] = {
   { 1, 2, 3},
   { 0, 2, 3},
   { 0, 1, 3},
   { 0, 1, 2}
};
constexpr int other2[4][4][2] = {
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
constexpr int other3[4][3] = {
   {1, 2, 3},
   {0, 2, 3},
   {0, 1, 3},
   {0, 1, 2}
};
constexpr int face_info[6][2] = {
   {0, 1},
   {0, 2},
   {0, 3},
   {1, 2},
   {1, 3},
   {2, 3}
};
constexpr int face_pos[6][2] = {
   {1, 0},
   {2, 0},
   {3, 0},
   {2, 1},
   {3, 1},
   {3, 2}
};
constexpr int pair[6][2] = {
   {2, 3},
   {1, 3},
   {1, 2},
   {0, 3},
   {0, 2},
   {0, 1}
};

constexpr int hilbert_order = 52;
constexpr int hilbert_limit = 8;
constexpr int brio_threshold = 64;
constexpr double brio_ratio = 0.125;
TINKER_EXTERN int transgc[8][3][8];
TINKER_EXTERN int tsb1mod3[8];
}

namespace tinker {
/// \ingroup solv
void alfmol(int vers);
void alphamol(int natoms, AlfAtom* alfatoms, double* surf, double* vol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, int vers);
void alphamol1(int vers);
void alphamol2(int vers);
void initdelcx(int natoms, AlfAtom* alfatoms, std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
   std::queue<std::pair<int,int>>& link_facet,std::queue<std::pair<int,int>>& link_index,std::stack<int>& free,std::vector<int>& kill);
void delaunay(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill);
void minor2(double a11, double a21, int& res, double eps=1e-10);
void minor3(double a11, double a12, double a21, double a22, double a31, double a32, int& res, double eps=1e-10);
void minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps=1e-10);
void minor5(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
   double r3, double* coord_d, double r4, double* coord_e, double r5, int& res, double eps=1e-10);
void flip_1_4(std::vector<Tetrahedron>& tetra, int ipoint, int itetra, int& tetra_last,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill);
void flip(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
   std::queue<std::pair<int,int>>& link_facet, std::queue<std::pair<int,int>>& link_index, std::stack<int>& free, std::vector<int>& kill);
void regular_convex(std::vector<Vertex>& vertices, int a, int b, int c, int p, int o, int itest_abcp,
   bool& regular, bool& convex, bool& test_abpo, bool& test_bcpo, bool& test_capo);
void alfcx(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, double alpha);
void alfedge(std::vector<Vertex>& vertices, double* a, double* b, double ra, double rb, 
   double* cg, std::vector<int>& listcheck, int& irad, int& iattach, double alpha);
void alfcxedges(std::vector<Tetrahedron>& tetra, std::vector<Edge>& edges);
void alfcxfaces(std::vector<Tetrahedron>& tetra, std::vector<Face>& faces);
template <bool compder>
void alphavol(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
   std::vector<Edge>& edges, std::vector<Face>& faces, double* ballwsurf, double* ballwvol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz);
void alfboxsize(AlfAtom* alfatoms, int size, double& xmin, double& ymin, double& zmin, double& xmax, double& ymax, double& zmax, double& rmax);
void alforder(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double rmax, int nthreads, std::vector<int>& Nval);
void alfboxsize(AlfAtom* alfatoms, int size, double& xmin, double& ymin, double& zmin, double& xmax, double& ymax, double& zmax, double& rmax);
void initHilbert(int ndim);
void sort3DHilbert(AlfAtom *alfatoms, int size, int e, int d, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth);
void brioHilbert(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth);
void splitGrid(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int ncube, std::vector<int>& Nval);
void kdTree(std::vector<AlfAtom>& alfatoms, int nsplit_tot, std::vector<int>& Nval);
}
