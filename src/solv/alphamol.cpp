#include "ff/solv/alphamol.h"
#include <algorithm>

namespace tinker {
void initalfatm()
{
   // resize alfatoms
   alfatoms.clear();
   alfatoms.reserve(atoms::n);

   // copy atoms into alfatoms list
   double xi, yi, zi, ri, cs, cv;
   for(int i = 0; i < atoms::n; i++) {
      xi = atoms::x[i];
      yi = atoms::y[i];
      zi = atoms::z[i];
      ri = radii[i];
      cs = coefS[i];
      cv = coefV[i];
      AlfAtom atm(i, xi, yi, zi, ri, cs, cv);
      alfatoms.push_back(atm);
   }

   // If needed, randomly shuffle the atoms
   bool shuffle = false;
   if (shuffle) {
      std::random_shuffle(alfatoms.begin(), alfatoms.end());
   }
}

void alfmol(int vers)
{
   // initialize alfatoms
   initalfatm();

   // run AlphaMol
   alphamol(alfatoms.size(), &(alfatoms[0]), wsurf, wvol, surf, vol,
      dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz, vers);
}

void alphamol(int natoms, AlfAtom* alfatoms, double& wsurf, double& wvol, double* surf, double* vol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, int vers)
{
   bool debug = false;
   clock_t start_s,stop_s;
   double total = 0;

   std::vector<Vertex> vertices;
   std::vector<Tetrahedron> tetra;
   std::vector<Edge> edges;
   std::vector<Face> faces;
   std::queue<std::pair<int,int>> link_facet;
   std::queue<std::pair<int,int>> link_index;
   std::stack<int> free;
   std::vector<int> kill;

   // initialize Delaunay procedure
   if (debug) {
      start_s = clock();
   }
   initdelcx(natoms, alfatoms, vertices, tetra, link_facet, link_index, free, kill);
   if (debug) {
      stop_s = clock();
      printf("\n Initdelcx compute time    : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // compute Delaunay triangulation
   if (debug) {
      start_s = clock();
   }
   delaunay(vertices, tetra, link_facet, link_index, free, kill);
   if (debug) {
      stop_s = clock();
      printf("\n Delaunay compute time     : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // generate alpha complex (with alpha=0.0)
   if (debug) {
      start_s = clock();
   }
   double alpha = 0;
   alfcx(vertices, tetra, alpha);
   if (debug) {
      stop_s = clock();
      printf("\n AlphaCx compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug) {
      start_s = clock();
   }
   alfcxedges(tetra, edges);
   if (debug) {
      stop_s = clock();
      printf("\n AlphaCxEdges compute time : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug) {
      start_s = clock();
   }
   alfcxfaces(tetra, faces);
   if (debug) {
      stop_s = clock();
      printf("\n AlphaCxFaces compute time : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug) {
      start_s = clock();
   }
   auto do_g = vers & calc::grad;
   alphavol(vertices, tetra, edges, faces, wsurf, wvol, surf, vol, dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz, do_g);
   if (debug) {
      stop_s = clock();
      printf("\n Volumes compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      total += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug) {
      printf("\n AlphaMol compute time     : %10.6f ms\n", total*1000);
      printf("\n Van der Waals Surface Area and Volume :\n");
      int width = 20;
      int precision = 4;
      printf("\n Total Area :              %*.*f Square Angstroms", width, precision, wsurf);
      printf("\n Total Volume :            %*.*f Cubic Angstroms", width, precision, wvol);
   }
}
}