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

enum class SymTyp
{
   None,
   Single,
   Linear,
   Planar,
   Mirror,
   Center,
};

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
TINKER_EXTERN double delcxeps;
TINKER_EXTERN double alfcxeps;
TINKER_EXTERN bool alfh;
TINKER_EXTERN bool alfdebug;
TINKER_EXTERN bool alfsos;

constexpr int hilbert_order = 52;
constexpr int hilbert_limit = 8;
constexpr int brio_threshold = 64;
constexpr double brio_ratio = 0.125;
TINKER_EXTERN int transgc[8][3][8];
TINKER_EXTERN int tsb1mod3[8];
}

namespace tinker {
/// \ingroup solv
void alfmola(int vers);
void alfmolb();
void alphamol(int natoms, AlfAtom* alfatoms, double* surf, double* vol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, int vers);
void alphamol1a(int vers);
void alphamol1b();
void alphamol2a(int vers);
void alphamol2b();

// AlphaMol2 spatial decomposition
void alfboxsize(AlfAtom* alfatoms, int size, double& xmin, double& ymin, double& zmin, double& xmax, double& ymax, double& zmax, double& rmax);
void alforder(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double rmax, int nthreads, std::vector<int>& Nval);
void initHilbert(int ndim);
void sort3DHilbert(AlfAtom *alfatoms, int size, int e, int d, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth);
void brioHilbert(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth);
void splitGrid(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int ncube, std::vector<int>& Nval);
void kdTree(std::vector<AlfAtom>& alfatoms, int nsplit_tot, std::vector<int>& Nval);

// break linear and planar symmetry
void chksymm(int n, double* mass, double* xref, double* yref, double* zref, SymTyp& symtyp);
void wiggle(int n, double* x, double* y, double* z, double eps);
void inertia(int n, double* mass, double* x, double* y, double* z);
}
