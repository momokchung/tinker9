#include "ff/solv/alphamol.h"
#include <tinker/detail/atoms.hh>
#include <algorithm>
#include <pthread.h>
#include <sys/time.h>

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
   if (alfmeth == AlfMethod::AlphaMol) {
      alphamol(alfatoms.size(), &(alfatoms[0]), wsurf, wvol, surf, vol,
         dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz, vers);
   }
   else if (alfmeth == AlfMethod::AlphaMol2) alphamol2(vers);
}

void alphamol(int natoms, AlfAtom* alfatoms, double& wsurf, double& wvol, double* surf, double* vol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, int vers)
{
   bool debug = false;
   clock_t start_s,stop_s;
   double tot_t = 0;

   std::vector<Vertex> vertices;
   std::vector<Tetrahedron> tetra;
   std::vector<Edge> edges;
   std::vector<Face> faces;
   std::queue<std::pair<int,int>> link_facet;
   std::queue<std::pair<int,int>> link_index;
   std::stack<int> free;
   std::vector<int> kill;

   // initialize Delaunay procedure
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   initdelcx(natoms, alfatoms, vertices, tetra, link_facet, link_index, free, kill);
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n Initdelcx compute time    : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // compute Delaunay triangulation
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   delaunay(vertices, tetra, link_facet, link_index, free, kill);
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n Delaunay compute time     : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // generate alpha complex (with alpha=0.0)
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   double alpha = 0;
   alfcx(vertices, tetra, alpha);
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n AlphaCx compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   alfcxedges(tetra, edges);
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n AlphaCxEdges compute time : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   alfcxfaces(tetra, faces);
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n AlphaCxFaces compute time : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug and alfmeth==AlfMethod::AlphaMol) {
      start_s = clock();
   }
   auto do_g = vers & calc::grad;
   alphavol(vertices, tetra, edges, faces, surf, vol, dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz, do_g);
   if (alfmeth==AlfMethod::AlphaMol) {
      wsurf = 0;
      wvol = 0;
      int nvertices = vertices.size();
      int nballs = 0;
      for (int i = 0; i < nvertices; i++) if (vertices[i].status==1) nballs++;
      for (int i = 0; i < nballs; i++) {
         wsurf += surf[i];
         wvol += vol[i];
      }
   }
   if (debug and alfmeth==AlfMethod::AlphaMol) {
      stop_s = clock();
      printf("\n Volumes compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (debug and alfmeth==AlfMethod::AlphaMol) {
      printf("\n AlphaMol compute time     : %10.6f ms\n", tot_t*1000);
      printf("\n Van der Waals Surface Area and Volume :\n");
      int width = 20;
      int precision = 4;
      printf("\n Total Area :              %*.*f Square Angstroms", width, precision, wsurf);
      printf("\n Total Volume :            %*.*f Cubic Angstroms", width, precision, wvol);
   }
}

inline void gettime(double& t1, double& u1);
inline double gettimediff(double t1, double u1, double t2, double u2);
inline bool inBox(double Point[3], double Xlim[2], double Ylim[2], double Zlim[2]);
void multimol(double buffer, int vers, int nthreads, std::vector<int>& Nval);
void* singlemol(void* data);

#define NUM_THREADS 128 
pthread_t threads[NUM_THREADS];
typedef struct thread_data {
   int vers;
   int N1;
   int N2;
   int nlist2;
   double buffer;
   double wsurf;
   double wvol;
   double* surf;
   double* vol;
   double* dsurfx;
   double* dsurfy;
   double* dsurfz;
   double* dvolx;
   double* dvoly;
   double* dvolz;
} thread_data;
thread_data info[NUM_THREADS];
int threadids[NUM_THREADS];

void alphamol2(int vers)
{
   bool debug = false;
   double t1,t2,u1,u2,diff;
   double tot_t = 0;

   // if needed, reorder  atoms
   if (debug) gettime(t1, u1);
   double xmin,ymin,zmin;
   double xmax,ymax,zmax;
   double rmax;
   std::vector<int> Nval(alfnthd + 1, 0);
   alfboxsize(&alfatoms[0], alfatoms.size(), xmin, ymin, zmin, xmax, ymax, zmax, rmax);
   alforder(xmin, ymin, zmin, xmax, ymax, zmax, rmax, alfnthd, Nval);
   if (debug) {
      gettime(t2, u2);
      diff = gettimediff(t1, u1, t2, u2);
      printf("\n Alforder compute time : %10.6f ms\n", diff*1000);
      tot_t += diff;
   }

   // run AlphaMol algorithm
   if (debug) gettime(t1, u1);
   int natoms = alfatoms.size();
   double buffer = 2*rmax;
   multimol(buffer, vers, alfnthd, Nval);
   if (debug) {
      gettime(t2, u2);
      diff = gettimediff(t1, u1, t2, u2);
      printf("\n MultiMol compute time : %10.6f ms\n", diff*1000);
      tot_t += diff;
   }

   if (debug) {
      printf("\n AlphaMol2 compute time : %10.6f ms\n", tot_t*1000);
      printf("\n Van der Waals Surface Area and Volume :\n");
      int width = 20;
      int precision = 4;
      printf("\n Total Area :              %*.*f Square Angstroms", width, precision, wsurf);
      printf("\n Total Volume :            %*.*f Cubic Angstroms", width, precision, wvol);
   }
}

inline void gettime(double& t1, double& u1)
{
   timeval tim;
   gettimeofday(&tim,NULL);
   t1 = tim.tv_sec;
   u1 = tim.tv_usec;
}

inline double gettimediff(double t1, double u1, double t2, double u2)
{
   return (t2-t1) + (u2-u1)*1.e-6;;
}

inline bool inBox(double Point[3], double Xlim[2], double Ylim[2], double Zlim[2])
{
   if(Point[0] < Xlim[0] || Point[0] >= Xlim[1]) return false;
   if(Point[1] < Ylim[0] || Point[1] >= Ylim[1]) return false;
   if(Point[2] < Zlim[0] || Point[2] >= Zlim[1]) return false;
   return true;
}

void multimol(double buffer, int vers, int nthreads, std::vector<int>& Nval)
{
   int N1,N2;
   int natoms = alfatoms.size();
   int nval = natoms/nthreads;

   for (int i = 0; i < nthreads; i++) {
      if (Nval[nthreads] == 0) {
         N1 = i*nval;
         N2 = N1 + nval;
         if (i == nthreads-1) N2 = natoms;
      }
      else {
         N1 = Nval[i];
         N2 = Nval[i+1];
      }

      threadids[i] = i;

      info[i].vers   = vers;
      info[i].N1     = N1;
      info[i].N2     = N2;
      info[i].buffer = buffer;
      info[i].surf   = surf;
      info[i].vol    = vol;
      info[i].dsurfx = dsurfx;
      info[i].dsurfy = dsurfy;
      info[i].dsurfz = dsurfz;
      info[i].dvolx  = dvolx;
      info[i].dvoly  = dvoly;
      info[i].dvolz  = dvolz;

      pthread_create(&threads[i], NULL, singlemol, (void*) &threadids[i]);
   }

   // for(int i = 0; i < natoms; i++) {
   //     std::cout << " i: " << i << " atom: " << alfatoms[i].index << std::endl;
   // }

   wsurf = 0;
   wvol = 0;
   for (int i = 0; i < nthreads; i++) {
      pthread_join(threads[i], NULL);
      wsurf += info[i].wsurf;
      wvol += info[i].wvol;
   }
}

void* singlemol(void* data)
{
   double xmin,ymin,zmin;
   double xmax,ymax,zmax;
   double r;
   double Xbox_buf[2],Ybox_buf[2],Zbox_buf[2];

   int threadid = *((int *) data);
   int N1       = info[threadid].N1;
   int N2       = info[threadid].N2;
   int natm     = N2 - N1;
   int natoms   = alfatoms.size();
   double buffer  = info[threadid].buffer;

   alfboxsize(&alfatoms[N1], natm, xmin, ymin, zmin, xmax, ymax, zmax, r);

   Xbox_buf[0] = xmin - buffer; Xbox_buf[1] = xmax + buffer;
   Ybox_buf[0] = ymin - buffer; Ybox_buf[1] = ymax + buffer;
   Zbox_buf[0] = zmin - buffer; Zbox_buf[1] = zmax + buffer;

   int nlist2;
   int *list2 = new int[natoms];
   double Point[3];
   bool test;

   nlist2 = 0;
   for (int i = 0; i < N1; i++) {
      for (int j = 0; j < 3; j++) Point[j] = alfatoms[i].coord[j];
      test = inBox(Point, Xbox_buf, Ybox_buf, Zbox_buf);
      if (test) {
         list2[nlist2] = i;
         nlist2++;
      }
   }
   for (int i = N2; i < natoms; i++) {
      for (int j = 0; j < 3; j++) Point[j] = alfatoms[i].coord[j];
      test = inBox(Point, Xbox_buf, Ybox_buf, Zbox_buf);
      if (test) {
         list2[nlist2] = i;
         nlist2++;
      }
   }

   int ntot = natm + nlist2;

   AlfAtom *newatoms = new AlfAtom[ntot];

   for (int i = 0; i < natm; i++) {
      newatoms[i] = alfatoms[N1+i];
   }
   // TODO: might not need
   sort3DHilbert(newatoms, natm, 0, 0, xmax, ymax, zmax, xmax, ymax, zmax, 0);

   int nat = natm;
   for (int i = 0; i < nlist2; i++) {
      int k = list2[i];
      newatoms[nat] = alfatoms[k];
      nat++;
   }

   int nfudge = 8;
   double tmp1;
   double tmp2;
   double* surfthd = new double[ntot+nfudge];
   double* volthd = new double[ntot+nfudge];
   double* dsurfxthd = new double[ntot+nfudge];
   double* dsurfythd = new double[ntot+nfudge];
   double* dsurfzthd = new double[ntot+nfudge];
   double* dvolxthd = new double[ntot+nfudge];
   double* dvolythd = new double[ntot+nfudge];
   double* dvolzthd = new double[ntot+nfudge];
   int vers = info[threadid].vers;

   alphamol(ntot, newatoms, tmp1, tmp2, surfthd, volthd,
      dsurfxthd, dsurfythd, dsurfzthd, dvolxthd, dvolythd, dvolzthd, vers);

   // transfer information to thread
   info[threadid].wsurf = 0;
   info[threadid].wvol = 0;

   for(int i = 0; i < natm; i++) {
      info[threadid].wsurf += surfthd[i];
      info[threadid].wvol += volthd[i];
   }

   auto do_g = vers & calc::grad;
   for (int i = 0; i < natm; i++) {
      info[threadid].surf[newatoms[i].index] = surfthd[i];
      info[threadid].vol[newatoms[i].index] = volthd[i];
      if (do_g) {
         info[threadid].dsurfx[newatoms[i].index] = dsurfxthd[i];
         info[threadid].dsurfy[newatoms[i].index] = dsurfythd[i];
         info[threadid].dsurfz[newatoms[i].index] = dsurfzthd[i];
         info[threadid].dvolx[newatoms[i].index] = dvolxthd[i];
         info[threadid].dvoly[newatoms[i].index] = dvolythd[i];
         info[threadid].dvolz[newatoms[i].index] = dvolzthd[i];
      }
   }

   // info[threadid].nlist2 = nlist2;

   delete[] newatoms;
   delete[] surfthd;
   delete[] volthd;
   delete[] dsurfxthd;
   delete[] dsurfythd;
   delete[] dsurfzthd;
   delete[] dvolxthd;
   delete[] dvolythd;
   delete[] dvolzthd;
   delete[] list2;

   return 0;
}
}