#include "ff/solv/alphamol.h"
#include <algorithm>

namespace tinker {
void addBogus(int npoints, double* x, double* y, double* z, double* radii, double* bcoord, double* brad);

void initdelcx()
{
   // initialize vertices and tetra
   vertices.clear();
   vertices.reserve(atoms::n+4);
   tetra.clear();
   tetra.reserve(10*atoms::n);

   while (!link_facet.empty()) {
      link_facet.pop();
   }

   while (!link_index.empty()) {
      link_index.pop();
   }

   while (!free.empty()) {
      free.pop();
   }

   kill.clear();

   // set four "infinite" points
   double zero = 0.;
   for (int i = 0; i < 4; i++) {
      Vertex vert(zero, zero, zero, zero, zero, zero);
      vert.info[0] = 1;
      vert.status = 0;
      vertices.push_back(vert);
   }

   // copy atoms into vertex list
   double xi, yi, zi, ri, cs, cv;
   for (int i = 0; i < atoms::n; i++) {
      xi = atoms::x[i];
      yi = atoms::y[i];
      zi = atoms::z[i];
      ri = radii[i];
      cs = coefS[i];
      cv = coefV[i];
      Vertex vert(xi, yi, zi, ri, cs, cv);
      vert.info[0] = 1;
      vert.status = 1;
      vertices.push_back(vert);
   }

   // if n < 4, add "bogus" points
   if (atoms::n < 4) {
      int new_points = 4-atoms::n;
      double *bcoord = new double[3*new_points];
      double *brad   = new double[new_points];
      addBogus(atoms::n, atoms::x, atoms::y, atoms::z, radii, bcoord, brad); 
      for (int i = 0; i < new_points; i++) {
         xi = bcoord[3*i];
         yi = bcoord[3*i+1];
         zi = bcoord[3*i+2];
         ri = brad[i];
         cs = 1.;
         cv = 1;
         Vertex vert(xi, yi, zi, ri, cs, cv);
         vert.info[0] = 1;
         vert.status = 0;
         vertices.push_back(vert);
      }
      delete [] bcoord;
      delete [] brad;
   }

   // create an "infinite" tetrahedron
   Tetrahedron t;
   t.init();

   t.vertices[0] = 0;
   t.vertices[1] = 1;
   t.vertices[2] = 2;
   t.vertices[3] = 3;

   t.neighbors[0] = -1;
   t.neighbors[1] = -1;
   t.neighbors[2] = -1;
   t.neighbors[3] = -1;

   t.nindex[0] = -1;
   t.nindex[1] = -1;
   t.nindex[2] = -1;
   t.nindex[3] = -1;

   t.info[0] = 0;
   t.info[1] = 1;

   tetra.push_back(t);
}

void addBogus(int npoints, double* x, double* y, double* z, double* radii, double* bcoord, double* brad)
{
   if (npoints > 3) return;

   int np = 4 - npoints;
   double cx,cy,cz;
   double c1x,c1y,c1z;
   double c2x,c2y,c2z;
   double c3x,c3y,c3z;
   double u1x,u1y,u1z;
   double v1x,v1y,v1z;
   double w1x,w1y,w1z;
   double c32x,c32y,c32z;
   double Rmax, d, d1, d2, d3;

   for (int i = 0; i < 3 * np; ++i) bcoord[i] = 0;

   if (npoints==1) {
      Rmax = radii[0];
      bcoord[0] = x[0] + 3*Rmax;
      bcoord[3*1+1] = y[0] + 3*Rmax;
      bcoord[3*2+2] = z[0] + 3*Rmax;
      for (int i = 0; i < np; i++) {
         brad[i] = Rmax/20;
      }
   }
   else if (npoints==2) {
      Rmax = std::max(radii[0], radii[1]);
      c1x = x[0];
      c1y = y[0];
      c1z = z[0];
      c2x = x[1];
      c2y = y[1];
      c2z = z[1];
      cx = 0.5*(c1x+c2x);
      cy = 0.5*(c1y+c2y);
      cz = 0.5*(c1z+c2z);
      u1x = c2x-c1x; 
      u1y = c2y-c1y; 
      u1z = c2z-c1z; 
      if ((u1z!=0) || (u1x!=-u1y)) {
         v1x = u1z;
         v1y = u1z;
         v1z = -u1x-u1z;
      }
      else {
         v1x = -u1y-u1z;
         v1y = u1x;
         v1z = u1x;
      }
      w1x = u1y*v1z - u1z*v1y;
      w1y = u1z*v1x - u1x*v1z;
      w1z = u1x*v1y - u1y*v1x;
      d = std::sqrt(u1x*u1x + u1y*u1y + u1z*u1z);
      bcoord[0] = cx + (2*d+3*Rmax)*v1x;
      bcoord[0+3] = cx + (2*d+3*Rmax)*w1x;
      bcoord[1] = cy + (2*d+3*Rmax)*v1y;
      bcoord[1+3] = cy + (2*d+3*Rmax)*w1y;
      bcoord[2] = cz + (2*d+3*Rmax)*v1z;
      bcoord[2+3] = cz + (2*d+3*Rmax)*w1z;

      brad[0] = Rmax/20; brad[1] = Rmax/20;
   }
   else {
      Rmax = std::max(std::max(radii[0], radii[1]), radii[2]);
      c1x = x[0];
      c1y = y[0];
      c1z = z[0];
      c2x = x[1];
      c2y = y[1];
      c2z = z[1];
      c3x = x[2];
      c3y = y[2];
      c3z = z[2];
      cx = (c1x+c2x+c3x)/3;
      cy = (c1y+c2y+c3y)/3;
      cz = (c1z+c2z+c3z)/3;
      u1x = c2x-c1x;
      u1y = c2y-c1y;
      u1z = c2z-c1z;
      v1x = c3x-c1x;
      v1y = c3y-c1y;
      v1z = c3z-c1z;
      w1x = u1y*v1z - u1z*v1y;
      w1y = u1z*v1x - u1x*v1z;
      w1z = u1x*v1y - u1y*v1x;
      d1 = std::sqrt(w1x*w1x + w1y*w1y + w1z*w1z);
      if (d1 == 0) {
         if (u1x != 0) {
            w1x = u1y;
            w1y = -u1x;
            w1z = 0;
         }
         else if(u1y != 0) {
            w1x = u1y;
            w1y = -u1x;
            w1z = 0;
         }
         else {
            w1x = u1z;
            w1y = -u1z;
            w1z = 0;
         }
      }
      d1 = std::sqrt(u1x*u1x + u1y*u1y + u1z*u1z);
      d2 = std::sqrt(v1x*v1x + v1y*v1y + v1z*v1z);
      c32x = c3x-c2x;
      c32y = c3y-c2y;
      c32z = c3z-c2z;
      d3 = std::sqrt(c32x*c32x + c32y*c32y + c32z*c32z);
      d = std::max({d1,d2,d3});
      bcoord[0] = cx + (2*d+3*Rmax)*w1x;
      bcoord[1] = cy + (2*d+3*Rmax)*w1y;
      bcoord[2] = cz + (2*d+3*Rmax)*w1z;
      brad[0] = Rmax/20;
   }
}
}