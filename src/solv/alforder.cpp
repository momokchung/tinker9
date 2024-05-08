#include "ff/solv/alphamol.h"

namespace tinker {
void alforder(double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, double rmax, int nthreads, std::vector<int>& Nval)
{
   if (alfsort == AlfSort::None) return;

   int depth = 0;

   if (alfsort == AlfSort::Sort3D) {
      sort3DHilbert(&alfatoms[0], alfatoms.size(), 0, 0, xmin, ymin, zmin, xmax, ymax, zmax, depth);
   }
   else if (alfsort == AlfSort::BRIO) {
      brioHilbert(&alfatoms[0], alfatoms.size(), xmin, ymin, zmin, xmax, ymax, zmax, depth);
   }
   else if (alfsort == AlfSort::Split) {
      splitGrid(&alfatoms[0], alfatoms.size(), xmin, ymin, zmin, xmax, ymax, zmax, nthreads, Nval);
   }
   else if (alfsort == AlfSort::KDTree) {
      int nsplit = (int) std::log2((double) nthreads);
      if (nsplit > 0) kdTree(alfatoms, nsplit, Nval);
   }
}

void initHilbert(int ndim)
{
   int gc[8],N,mask,travel_bit;
   int c,f,g,k,v;

   for (int i = 0; i < 8; i++) gc[i] = 0;

   N = (ndim == 2) ? 4 : 8;
   mask = (ndim == 2) ? 3 : 7;

   // Generate the Gray code sequence.
   for (int i = 0; i < N; i++) {
      gc[i] = i ^ (i >> 1);
   }

   for (int e = 0; e < N; e++) {
      for (int d = 0; d < ndim; d++) {
         // Calculate the end point (f).
         f = e ^ (1 << d); // Toggle the d-th bit of 'e'.
         // travel_bit = 2**p, the bit we want to travel. 
         travel_bit = e ^ f;
         for (int i = 0; i < N; i++) {
            // // Rotate gc[i] left by (p + 1) % n bits.
            k = gc[i] * (travel_bit * 2);
            g = ((k | (k / N)) & mask);
            // Calculate the permuted Gray code by xor with the start point (e).
            transgc[e][d][i] = (g ^ e);
         }
      } // d
   } // e

   // Count the consecutive '1' bits (trailing) on the right.
   tsb1mod3[0] = 0;
   for (int i = 1; i < N; i++) {
      v = ~i; // Count the 0s.
      v = (v ^ (v - 1)) >> 1; // Set v's trailing 0s to 1s and zero rest
      for (c = 0; v; c++) {
         v >>= 1;
      }
      tsb1mod3[i] = c % ndim;
   }
}

inline int splitHilbert(AlfAtom *alfatoms, int size, int gc0, int gc1,
    double xmin, double xmax, double ymin, double ymax, double zmin, double zmax)
{
   int axis,d;
   int i,j;
   double splt;
   AlfAtom swapvert;

   // Find the current splitting axis. 'axis' is a value 0, or 1, or 2, which 
   // correspoding to x-, or y- or z-axis.
   axis = (gc0 ^ gc1) >> 1;

   // Calulate the split position along the axis.
   if (axis == 0) {
      splt = 0.5 *(xmin + xmax);
   }
   else if (axis == 1) {
      splt = 0.5 *(ymin + ymax);
   }
   else { // == 2
      splt = 0.5 *(zmin + zmax);
   }

   // Find the direction (+1 or -1) of the axis. If 'd' is +1, the direction
   // of the axis is to the positive of the axis, otherwise, it is -1.
   d = ((gc0 & (1<<axis)) == 0) ? 1 : -1;

   // Partition the vertices into left- and right-arrays such that left points
   // have Hilbert indices lower than the right points.
   i = 0;
   j = size - 1;

   // Partition the vertices into left- and right-arrays.
   if (d > 0) {
      do {
         for (; i < size; i++) {
            if (alfatoms[i].coord[axis] >= splt) break;
         }
         for (; j >= 0; j--) {
            if (alfatoms[j].coord[axis] < splt) break;
         }
         // Is the partition finished?
         if (i == (j + 1)) break;
         // Swap i-th and j-th vertices.
         swapvert = alfatoms[i];
         alfatoms[i] = alfatoms[j];
         alfatoms[j] = swapvert;
         // Continue patitioning the array;
      } while (true);
   }
   else {
      do {
         for (; i < size; i++) {
            if (alfatoms[i].coord[axis] <= splt) break;
         }
         for (; j >= 0; j--) {
            if (alfatoms[j].coord[axis] > splt) break;
         }
         // Is the partition finished?
         if (i == (j + 1)) break;
         // Swap i-th and j-th vertices.
         swapvert = alfatoms[i];
         alfatoms[i] = alfatoms[j];
         alfatoms[j] = swapvert;
         // Continue patitioning the array;
      } while (true);
   }

   return i;
}

void sort3DHilbert(AlfAtom *alfatoms, int size, int e, int d, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth)
{
   int N = 3;
   int mask = 7;
   int p[9],e_w,d_w,k,ei,di;
   double x1,x2,y1,y2,z1,z2;

   p[0] = 0;
   p[8] = size;

   // Sort the points according to the 1st order Hilbert curve in 3d.
   p[4] = splitHilbert(&alfatoms[0], p[8], transgc[e][d][3], transgc[e][d][4], xmin, xmax, ymin, ymax, zmin, zmax);
   p[2] = splitHilbert(&alfatoms[0], p[4], transgc[e][d][1], transgc[e][d][2], xmin, xmax, ymin, ymax, zmin, zmax);
   p[1] = splitHilbert(&alfatoms[0], p[2], transgc[e][d][0], transgc[e][d][1], xmin, xmax, ymin, ymax, zmin, zmax);
   p[3] = splitHilbert(&(alfatoms[p[2]]), p[4]-p[2], transgc[e][d][2], transgc[e][d][3], 
      xmin, xmax, ymin, ymax, zmin, zmax) + p[2];
   p[6] = splitHilbert(&(alfatoms[p[4]]), p[8]-p[4], transgc[e][d][5], transgc[e][d][6], 
      xmin, xmax, ymin, ymax, zmin, zmax)+p[4];
   p[5] = splitHilbert(&(alfatoms[p[4]]), p[6]-p[4], transgc[e][d][4], transgc[e][d][5], 
      xmin, xmax, ymin, ymax, zmin, zmax)+p[4];
   p[7] = splitHilbert(&(alfatoms[p[6]]), p[8]-p[6], transgc[e][d][6], transgc[e][d][7], 
      xmin, xmax, ymin, ymax, zmin, zmax)+p[6];

   if (hilbert_order > 0) {
      // A maximum order is prescribed. 
      if ((depth + 1) == hilbert_order) {
         // The maximum prescribed order is reached.
         return;
      }
   }

   // Recursively sort the points in sub-boxes.
   for (int w = 0; w < 8; w++) {
      // w is the local Hilbert index (NOT Gray code).
      // Sort into the sub-box either there are more than 2 points in it, or
      // the prescribed order of the curve is not reached yet.
      //if ((p[w+1] - p[w] > b->hilbert_limit) || (b->hilbert_order > 0)) {
      if ((p[w+1] - p[w]) > hilbert_limit) {
         // Calculcate the start point (ei) of the curve in this sub-box.
         // update e = e ^ (e(w) left_rotate (d+1)).
         if (w == 0) {
            e_w = 0;
         }
         else {
            // calculate e(w) = gc(2 * floor((w - 1) / 2)).
            k = 2 * ((w - 1) / 2);
            e_w = k ^ (k >> 1); // = gc(k).
         }
         k = e_w;
         e_w = ((k << (d+1)) & mask) | ((k >> (N-d-1)) & mask);
         ei = e ^ e_w;
         // Calulcate the direction (di) of the curve in this sub-box.
         // update d = (d + d(w) + 1) % N
         if (w == 0) {
            d_w = 0;
         }
         else {
            d_w = ((w % 2) == 0) ? tsb1mod3[w - 1] : tsb1mod3[w];
         }
         di = (d + d_w + 1) % N;
         // Calculate the bounding box of the sub-box.
         if (transgc[e][d][w] & 1) { // x-axis
            x1 = 0.5 * (xmin + xmax);
            x2 = xmax;
         }
         else {
            x1 = xmin;
            x2 = 0.5 * (xmin + xmax);
         }
         if (transgc[e][d][w] & 2) { // y-axis
            y1 = 0.5 * (ymin + ymax);
            y2 = ymax;
         }
         else {
            y1 = ymin;
            y2 = 0.5 * (ymin + ymax);
         }
         if (transgc[e][d][w] & 4) { // z-axis
            z1 = 0.5 * (zmin + zmax);
            z2 = zmax;
         }
         else {
            z1 = zmin;
            z2 = 0.5 * (zmin + zmax);
         }
         sort3DHilbert(&(alfatoms[p[w]]), p[w+1] - p[w], ei, di, x1, y1, z1, x2, y2, z2, depth+1);
      } // if (p[w+1] - p[w] > 1)
   } // w
}

void brioHilbert(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int depth)
{
   int middle = 0;

   if(size >= brio_threshold) {
      depth++;
      middle = size*brio_ratio;
      brioHilbert(alfatoms, middle, xmin, ymin, zmin, xmax, ymax, zmax, depth);
   }

   sort3DHilbert(&(alfatoms[middle]), size - middle, 0, 0, xmin, ymin, zmin, xmax, ymax, zmax, 0);
}

void splitGrid(AlfAtom *alfatoms, int size, double xmin, double ymin, double zmin, double xmax, double ymax, double zmax, int ncube, std::vector<int>& Nval)
{
   int Nx,Ny,Nz;
   int idx,Ix,Iy,Iz;
   double hx,hy,hz;
   double Xlim[2],Ylim[2],Zlim[2];
   double Point[3];
   double offset = 0.1;

   Xlim[0] = xmin - offset;
   Xlim[1] = xmax + offset;
   Ylim[0] = ymin - offset;
   Ylim[1] = ymax + offset;
   Zlim[0] = zmin - offset;
   Zlim[1] = zmax + offset;

   Nx = 1; Ny = 1; Nz = 1;
   if (ncube == 1) {
      Nx = 1; Ny = 1; Nz = 1;
   }
   else if (ncube == 2) {
      Nx = 2; Ny = 1; Nz = 1;
   }
   else if (ncube == 4) {
      Nx = 2; Ny = 2; Nz = 1;
   }
   else if (ncube == 8) {
      Nx = 2; Ny = 2; Nz = 2;
   }
   else if (ncube == 16) {
      Nx = 4; Ny = 2; Nz = 2;
   }
   else if (ncube == 32) {
      Nx = 4; Ny = 4; Nz = 2;
   }
   else if (ncube == 64) {
      Nx = 4; Ny = 4; Nz = 4;
   }

   hx = (Xlim[1]-Xlim[0])/Nx;
   hy = (Ylim[1]-Ylim[0])/Ny;
   hz = (Zlim[1]-Zlim[0])/Nz;

   std::vector<std::vector<AlfAtom>> splitAtoms;
   splitAtoms.resize(ncube);

   for (int j = 0; j < size; j++) {
      for (int k = 0; k < 3; k++) Point[k] = alfatoms[j].coord[k];
      Ix = (Point[0]-Xlim[0])/hx;
      Iy = (Point[1]-Ylim[0])/hy;
      Iz = (Point[2]-Zlim[0])/hz;
      idx = Ix + Iy*Nx + Iz*Nx*Ny;
      splitAtoms[idx].push_back(alfatoms[j]);
   }

   int nat = 0;
   Nval[0] = 0;
   for (int i = 0; i < ncube; i++) {
      Nval[i+1] = Nval[i] + splitAtoms[i].size();
      // std::cout << "i = " << i << " size = " << splitAtoms[i].size() << std::endl;
      for (int j = 0; j < splitAtoms[i].size(); j++) {
         alfatoms[nat+j] = splitAtoms[i][j];
      }
      nat += splitAtoms[i].size();
   }
}

struct alfatoms_cmp
{
   alfatoms_cmp (int index) : index_(index) {}

   bool operator() (const AlfAtom& atm1, const AlfAtom& atm2) const
   {
      return atm1.coord[index_] < atm2.coord[index_];
   }

   int index_;
};

void splitKDTree(std::vector<AlfAtom>& alfatoms, int begin, int end, int index, int nsplit, int nsplit_tot, std::vector<int>& v)
{
   if (nsplit >= nsplit_tot) return;

   const int dimensions = 3;
   int n = begin + (end - begin)/2;
   v.push_back(n);
   auto i = alfatoms.begin();
   std::nth_element(i + begin, i + n, i + end, alfatoms_cmp(index));

   index = (index+1) % dimensions;
   nsplit++;
   splitKDTree(alfatoms, begin, n, index, nsplit, nsplit_tot, v);
   splitKDTree(alfatoms, n , end, index, nsplit, nsplit_tot, v);
}

void kdTree(std::vector<AlfAtom>& alfatoms, int nsplit_tot, std::vector<int>& Nval)
{
   int begin = 0;
   int end = alfatoms.size();

   int nsplit= 0;
   int index = 0;

   std::vector<int> v;
   v.push_back(begin);
   v.push_back(end);

   splitKDTree(alfatoms, begin, end, index, nsplit, nsplit_tot, v);

   std::sort(v.begin(), v.end());

   for (int i = 0; i < v.size(); i++) Nval[i] = v[i];
}

void alfboxsize(AlfAtom* alfatoms, int size, double& xmin, double& ymin, double& zmin, double& xmax, double& ymax, double& zmax, double& rmax)
{
   xmin = alfatoms[0].coord[0];
   xmax = alfatoms[0].coord[0];
   ymin = alfatoms[0].coord[1];
   ymax = alfatoms[0].coord[1];
   zmin = alfatoms[0].coord[2];
   zmax = alfatoms[0].coord[2];
   rmax = alfatoms[0].r;

   for (int i = 1; i < size; i++) {
      double x = alfatoms[i].coord[0];
      double y = alfatoms[i].coord[1];
      double z = alfatoms[i].coord[2];
      double r = alfatoms[i].r;

      if (x < xmin) xmin = x;
      if (x > xmax) xmax = x;
      if (y < ymin) ymin = y;
      if (y > ymax) ymax = y;
      if (z < zmin) zmin = z;
      if (z > zmax) zmax = z;
      if (r > rmax) rmax = r;
   }
}
}