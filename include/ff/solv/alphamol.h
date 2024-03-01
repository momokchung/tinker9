#pragma once
#include "ff/energybuffer.h"
#include "ff/precision.h"
#include "tool/rcman.h"
#include <tinker/detail/atoms.hh>
#include <bitset>
#include <cmath>
#include <queue>
#include <stack>
#include <vector>

namespace tinker {
/// \ingroup solv
void alphamol(int vers);
void initdelcx();
void delaunay();
void minor2(double a11, double a21, int& res, double eps=1e-10);
void minor3(double a11, double a12, double a21, double a22, double a31, double a32, int& res, double eps=1e-10);
void minor4(double* coord_a, double* coord_b, double* coord_c, double* coord_d, int& res, double eps=1e-10);
void minor5(double* coord_a, double r1, double* coord_b, double r2, double* coord_c,
   double r3, double* coord_d, double r4, double* coord_e, double r5, int& res, double eps=1e-10);
void flip_1_4(int ipoint, int itetra, int& tetra_last);
void flip();
void regular_convex(int a, int b, int c, int p, int o, int itest_abcp,
   bool& regular, bool& convex, bool& test_abpo, bool& test_bcpo, bool& test_capo) ;
void alfcx(double alpha);
void alfedge(double* a, double* b, double ra, double rb, 
   double* cg, std::vector<int>& listcheck, int& irad, int& iattach, double alpha);
void alfcxedges();
void alfcxfaces();
void alphavol(double& WSurf, double& WVol, double& WMean, double& WGauss,
   double& Surf, double& Vol, double& Mean, double& Gauss,
   double* ballwsurf, double* ballwvol, double* ballwmean, double* ballwgauss,
   double* dsurf_coord, double* dvol_coord, double* dmean_coord, double* dgauss_coord, bool compder);

}

//====================================================================//
//                                                                    //
//                          Global Variables                          //
//                                                                    //
//====================================================================//

namespace tinker {
class Vertex {
   public:
      double r;
      double coord[3];
      double w;
      double coefs,coefv,coefm,coefg;
      double gamma;

      std::bitset<8> info;
      bool status;

      Vertex() {}

      Vertex(double x, double y, double z, double r, double coefs, double coefv, double coefm, double coefg);

      ~Vertex();
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
   double sigma;
   double len,surf,vol; 
   double coefm1,coefm2,coefg1,coefg2;
   double dsurf,dvol,dmean,dgauss;

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

TINKER_EXTERN double tsurf;
TINKER_EXTERN double tvol;
TINKER_EXTERN double tmean;
TINKER_EXTERN double tgauss;
TINKER_EXTERN double wsurf;
TINKER_EXTERN double wvol;
TINKER_EXTERN double wmean;
TINKER_EXTERN double wgauss;
TINKER_EXTERN double* radii;
TINKER_EXTERN double* coefS;
TINKER_EXTERN double* coefV;
TINKER_EXTERN double* coefM;
TINKER_EXTERN double* coefG;
TINKER_EXTERN double* surf;
TINKER_EXTERN double* vol;
TINKER_EXTERN double* mean;
TINKER_EXTERN double* gauss;
TINKER_EXTERN double* dsurf;
TINKER_EXTERN double* dvol;
TINKER_EXTERN double* dmean;
TINKER_EXTERN double* dgauss;
constexpr double deleps = 1e-3;
constexpr double delepsvol = 1e-13;
constexpr double alfeps = 1e-10;
TINKER_EXTERN std::vector<Vertex> vertices;
TINKER_EXTERN std::vector<Tetrahedron> tetra;
TINKER_EXTERN std::vector<Edge> edges;
TINKER_EXTERN std::vector<Face> faces;
TINKER_EXTERN std::queue<std::pair<int,int>> link_facet;
TINKER_EXTERN std::queue<std::pair<int,int>> link_index;
TINKER_EXTERN std::stack<int> free;
TINKER_EXTERN std::vector<int> kill;

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
}
