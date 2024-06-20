#include "ff/atom.h"
#include "ff/solv/alfcx.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/alphavol.h"
#include "ff/solv/delcx.h"
#include "tool/darray.h"
#include <tinker/detail/atomid.hh>
#include <tinker/routines.h>
#include <algorithm>
#include <pthread.h>
#include <random>
#include <sys/time.h>

namespace tinker {
void initalfatm()
{
   // resize alfatoms
   alfatoms.clear();
   alfatoms.reserve(n);

   // initialize atomic
   for (int i = 0; i < n; i++) {
      surf[i] = 0;
      vol[i] = 0;
      dsurfx[i] = 0;
      dsurfy[i] = 0;
      dsurfz[i] = 0;
      dvolx[i] = 0;
      dvoly[i] = 0;
      dvolz[i] = 0;
   }

   // check for symmetry
   SymTyp symtyp;

   // allocate
   double* xref = new double[n];
   double* yref = new double[n];
   double* zref = new double[n];

   // copy coordinates
   if (sizeof(pos_prec) == sizeof(double)) {
      darray::copyout(g::q0, n, xref, xpos);
      darray::copyout(g::q0, n, yref, ypos);
      darray::copyout(g::q0, n, zref, zpos);
      waitFor(g::q0);
   } else {
      std::vector<pos_prec> arrx(n), arry(n), arrz(n);
      darray::copyout(g::q0, n, arrx.data(), xpos);
      darray::copyout(g::q0, n, arry.data(), ypos);
      darray::copyout(g::q0, n, arrz.data(), zpos);
      waitFor(g::q0);
      for (int i = 0; i < n; ++i) {
         xref[i] = arrx[i];
         yref[i] = arry[i];
         zref[i] = arrz[i];
      }
   }

   // check symmetry of system
   chksymm(n, atomid::mass, xref, yref, zref, symtyp);

   // wiggle if system is symmetric
   double eps = 1e-3;
   bool dowiggle_linear = symtyp == SymTyp::Linear and n > 2;
   bool dowiggle_planar = symtyp == SymTyp::Planar and n > 3;
   bool dowiggle = dowiggle_linear or dowiggle_planar;
   if (dowiggle) wiggle(n, xref, yref, zref, eps);

   // copy atoms into alfatoms list
   double xi, yi, zi, ri, cs, cv;
   for (int i = 0; i < n; i++) {
      ri = radii[i];
      if (ri == 0) continue;
      xi = xref[i];
      yi = yref[i];
      zi = zref[i];
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

   // deallocate
   delete[] xref;
   delete[] yref;
   delete[] zref;
}

// "chksymm" examines the current coordinates for linearity,
// planarity, an internal mirror plane or center of inversion
void chksymm(int n, double* mass, double* xref, double* yref, double* zref, SymTyp& symtyp)
{
   // copy coordinates
   double* x = new double[n];
   double* y = new double[n];
   double* z = new double[n];
   for (int i = 0; i < n; i++) {
      x[i] = xref[i];
      y[i] = yref[i];
      z[i] = zref[i];
   }

   // move the atomic coordinates into the inertial frame
   inertia(n, mass, x, y, z);

   double eps = 1e-4;
   symtyp = SymTyp::None;
   bool xnul = true;
   bool ynul = true;
   bool znul = true;
   for (int i = 0; i < n; i++) {
      if (std::abs(x[i]) > eps) xnul = false;
      if (std::abs(y[i]) > eps) ynul = false;
      if (std::abs(z[i]) > eps) znul = false;
   }
   if (n == 3) symtyp = SymTyp::Planar;
   if (xnul) symtyp = SymTyp::Planar;
   if (ynul) symtyp = SymTyp::Planar;
   if (znul) symtyp = SymTyp::Planar;
   if (n == 2) symtyp = SymTyp::Linear;
   if (xnul and ynul) symtyp = SymTyp::Linear;
   if (xnul and znul) symtyp = SymTyp::Linear;
   if (ynul and znul) symtyp = SymTyp::Linear;
   if (n == 1) symtyp = SymTyp::Single;

   // test mean coords for mirror plane and inversion center
   if (symtyp == SymTyp::None) {
      double xave = 0;
      double yave = 0;
      double zave = 0;
      for (int i = 0; i < n; i++) {
         xave += x[i];
         yave += y[i];
         zave += z[i];
      }
      xave = std::abs(xave) / n;
      yave = std::abs(yave) / n;
      zave = std::abs(zave) / n;
      int nave = 0;
      if (xave < eps) nave++;
      if (yave < eps) nave++;
      if (zave < eps) nave++;
      if (nave != 0) symtyp = SymTyp::Mirror;
      if (nave == 3) symtyp = SymTyp::Center;
   }

   // deallocate
   delete[] x;
   delete[] y;
   delete[] z;
}

void ranvec(double vec[3])
{
   std::random_device rd;
   std::mt19937 gen(rd());
   std::uniform_int_distribution<> dis(0, 1);
   std::uniform_real_distribution<double> distribution1(0.1, 1.0);
   std::uniform_real_distribution<double> distribution2(-1.0, -0.1);

   double r[3];
   for (int i = 0; i < 3; i++) {
      if (dis(gen) == 0) r[i] = distribution1(gen);
      else r[i] = distribution2(gen);
   }

   double mag = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);

   // Normalize the vector
   vec[0] = r[0] / mag;
   vec[1] = r[1] / mag;
   vec[2] = r[2] / mag;
}

void wiggle(int n, double* x, double* y, double* z, double eps)
{
   double vec[3];
   for (int i = 0; i < n; i++) {
      ranvec(vec);
      x[i] += eps*vec[0];
      y[i] += eps*vec[1];
      z[i] += eps*vec[2];
   }
}

void inertia(int n, double* mass, double* x, double* y, double* z)
{
   int index[3];
   double weigh,total,dot;
   double xcm,ycm,zcm;
   double xx,xy,xz,yy,yz,zz;
   double xterm,yterm,zterm;
   double phi,theta,psi;
   double moment[3];
   double tensor[9],vec[9];

   // compute the position of the center of mass
   total = 0;
   xcm = 0;
   ycm = 0;
   zcm = 0;
   for (int i = 0; i < n; i++) {
      weigh = mass[i];
      total += weigh;
      xcm += x[i]*weigh;
      ycm += y[i]*weigh;
      zcm += z[i]*weigh;
   }
   xcm /= total;
   ycm /= total;
   zcm /= total;

   // compute and then diagonalize the inertia tensor
   xx = 0;
   xy = 0;
   xz = 0;
   yy = 0;
   yz = 0;
   zz = 0;
   for (int i = 0; i < n; i++) {
      weigh = mass[i];
      xterm = x[i] - xcm;
      yterm = y[i] - ycm;
      zterm = z[i] - zcm;
      xx += xterm*xterm*weigh;
      xy += xterm*yterm*weigh;
      xz += xterm*zterm*weigh;
      yy += yterm*yterm*weigh;
      yz += yterm*zterm*weigh;
      zz += zterm*zterm*weigh;
   }
   tensor[0] = yy + zz;
   tensor[1] = -xy;
   tensor[2] = -xz;
   tensor[3] = -xy;
   tensor[4] = xx + zz;
   tensor[5] = -yz;
   tensor[6] = -xz;
   tensor[7] = -yz;
   tensor[8] = xx + yy;
   int dim = 3;
   tinker_f_jacobi(&dim,tensor,moment,vec);

   // select the direction for each principal moment axis
   for (int i = 0; i < 2; i++) {
      for (int j = 0; j < n; j++) {
         xterm = vec[3*i+0] * (x[j]-xcm);
         yterm = vec[3*i+1] * (y[j]-ycm);
         zterm = vec[3*i+2] * (z[j]-zcm);
         dot = xterm + yterm + zterm;
         if (dot < 0.0) {
            for (int k = 0; k < 3; k++) {
               vec[3*i+k] = -vec[3*i+k];
            }
         }
         if (dot != 0) break;
      }
   }

   // moment axes must give a right-handed coordinate system
   xterm = vec[0] * (vec[4]*vec[8]-vec[7]*vec[5]);
   yterm = vec[1] * (vec[6]*vec[5]-vec[3]*vec[8]);
   zterm = vec[2] * (vec[3]*vec[7]-vec[6]*vec[4]);
   dot = xterm + yterm + zterm;
   if (dot < 0) {
      for (int j = 0; j < 3; j++) {
         vec[6+j] = -vec[6+j];
      }
   }

   // translate to origin, then apply Euler rotation matrix
   for (int i = 0; i < n; i++) {
      xterm = x[i] - xcm;
      yterm = y[i] - ycm;
      zterm = z[i] - zcm;
      x[i] = vec[0]*xterm + vec[1]*yterm + vec[2]*zterm;
      y[i] = vec[3]*xterm + vec[4]*yterm + vec[5]*zterm;
      z[i] = vec[6]*xterm + vec[7]*yterm + vec[8]*zterm;
   }
}

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

void alfmola(int vers)
{
   // initialize alfatoms
   clock_t start_s,stop_s;
   if (alfdebug) {
      start_s = clock();
   }
   initalfatm();
   if (alfdebug) {
      stop_s = clock();
      printf("\n Initalfatm compute time    : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
   }

   // run AlphaMola
   if (alfmeth == AlfMethod::AlphaMol) alphamol1a(vers);
   else if (alfmeth == AlfMethod::AlphaMol2) alphamol2a(vers);
}

void alfmolb()
{
   // run AlphaMolb
   if (alfmeth == AlfMethod::AlphaMol) alphamol1b();
   else if (alfmeth == AlfMethod::AlphaMol2) alphamol2b();
}

void alphamol(int natoms, AlfAtom* alfatoms, double* surf, double* vol,
   double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz, int vers)
{
   clock_t start_s,stop_s;
   double tot_t = 0;
   bool alfprint = alfdebug and alfmeth==AlfMethod::AlphaMol;

   std::vector<Vertex> vertices;
   std::vector<Tetrahedron> tetra;
   std::vector<Edge> edges;
   std::vector<Face> faces;

   Delcx delcx;
   Alfcx alfcx;
   AlphaVol alphavol;

   // initialize Delaunay procedure
   if (alfprint) {
      start_s = clock();
   }
   delcx.init(natoms, alfatoms, vertices, tetra);
   if (alfprint) {
      stop_s = clock();
      printf("\n Init compute time         : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // compute Delaunay triangulation
   if (alfprint) {
      start_s = clock();
   }
   if (alfsos) delcx.regular3D<true>(vertices, tetra);
   else delcx.regular3D<false>(vertices, tetra);
   if (alfprint) {
      stop_s = clock();
      printf("\n Regular3D compute time    : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   // generate alpha complex (with alpha=0.0)
   if (alfprint) {
      start_s = clock();
   }
   double alpha = 0;
   alfcx.alfcx(vertices, tetra, edges, faces, alpha);
   if (alfprint) {
      stop_s = clock();
      printf("\n AlphaCx compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
   }

   if (alfprint) {
      start_s = clock();
   }
   auto do_g = vers & calc::grad;
   if (do_g) alphavol.alphavol<true>(vertices, tetra, edges, faces, surf, vol, dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz);
   else alphavol.alphavol<false>(vertices, tetra, edges, faces, surf, vol, dsurfx, dsurfy, dsurfz, dvolx, dvoly, dvolz);

   if (alfprint) {
      stop_s = clock();
      printf("\n Volumes compute time      : %10.6f ms\n", (stop_s-start_s)/double(CLOCKS_PER_SEC)*1000);
      tot_t += (stop_s-start_s)/double(CLOCKS_PER_SEC);
      printf("\n AlphaMol compute time     : %10.6f ms\n", tot_t*1000);
   }
}

void* alphamol1thd(void* data)
{
   int vers = info[0].vers;
   int natoms = alfatoms.size();
   int nfudge = 8;
   double* surfthd = new double[natoms+nfudge];
   double* volthd = new double[natoms+nfudge];
   double* dsurfxthd = new double[natoms+nfudge];
   double* dsurfythd = new double[natoms+nfudge];
   double* dsurfzthd = new double[natoms+nfudge];
   double* dvolxthd = new double[natoms+nfudge];
   double* dvolythd = new double[natoms+nfudge];
   double* dvolzthd = new double[natoms+nfudge];
   alphamol(natoms, &(alfatoms[0]), surfthd, volthd,
      dsurfxthd, dsurfythd, dsurfzthd, dvolxthd, dvolythd, dvolzthd, vers);

   wsurf = 0;
   wvol = 0;
   for(int i = 0; i < natoms; i++) {
      wsurf += surfthd[i];
      wvol += volthd[i];
   }

   for (int i = 0; i < natoms; i++) {
      surf[alfatoms[i].index] = surfthd[i];
      vol[alfatoms[i].index] = volthd[i];
   }

   auto do_g = vers & calc::grad;
   if (do_g) {
      for (int i = 0; i < natoms; i++) {
         dsurfx[alfatoms[i].index] = dsurfxthd[i];
         dsurfy[alfatoms[i].index] = dsurfythd[i];
         dsurfz[alfatoms[i].index] = dsurfzthd[i];
         dvolx[alfatoms[i].index] = dvolxthd[i];
         dvoly[alfatoms[i].index] = dvolythd[i];
         dvolz[alfatoms[i].index] = dvolzthd[i];
      }
   }

   if (alfdebug and alfmeth==AlfMethod::AlphaMol) {
      printf("\n Van der Waals Surface Area and Volume :\n");
      int width = 20;
      int precision = 4;
      printf("\n Total Area :              %*.*f Square Angstroms", width, precision, wsurf);
      printf("\n Total Volume :            %*.*f Cubic Angstroms", width, precision, wvol);
   }

   delete[] surfthd;
   delete[] volthd;
   delete[] dsurfxthd;
   delete[] dsurfythd;
   delete[] dsurfzthd;
   delete[] dvolxthd;
   delete[] dvolythd;
   delete[] dvolzthd;

   return 0;
}

void alphamol1a(int vers)
{
   info[0].vers = vers;
   pthread_create(&threads[0], NULL, alphamol1thd, NULL);
}

void alphamol1b()
{
   pthread_join(threads[0], NULL);
}

inline void gettime(double& t1, double& u1);
inline double gettimediff(double t1, double u1, double t2, double u2);
inline bool inBox(double Point[3], double Xlim[2], double Ylim[2], double Zlim[2]);
void multimola(double buffer, int vers, int nthreads, std::vector<int>& Nval);
void multimolb(int nthreads);
void* singlemol(void* data);

void alphamol2a(int vers)
{
   double t1,t2,u1,u2,diff;
   double tot_t = 0;

   // if needed, reorder  atoms
   if (alfdebug) gettime(t1, u1);
   double xmin,ymin,zmin;
   double xmax,ymax,zmax;
   double rmax;
   std::vector<int> Nval(alfnthd + 1, 0);
   alfboxsize(&alfatoms[0], alfatoms.size(), xmin, ymin, zmin, xmax, ymax, zmax, rmax);
   alforder(xmin, ymin, zmin, xmax, ymax, zmax, rmax, alfnthd, Nval);
   if (alfdebug) {
      gettime(t2, u2);
      diff = gettimediff(t1, u1, t2, u2);
      printf("\n Alforder compute time : %10.6f ms\n", diff*1000);
      tot_t += diff;
   }

   // run AlphaMol algorithm
   if (alfdebug) gettime(t1, u1);
   double buffer = 2*rmax;
   multimola(buffer, vers, alfnthd, Nval);
   if (alfdebug) {
      gettime(t2, u2);
      diff = gettimediff(t1, u1, t2, u2);
      printf("\n MultiMola compute time : %10.6f ms\n", diff*1000);
      tot_t += diff;
   }
}

void alphamol2b()
{
   double t1,t2,u1,u2,diff;
   double tot_t = 0;

   // run AlphaMol algorithm
   if (alfdebug) gettime(t1, u1);
   multimolb(alfnthd);
   if (alfdebug) {
      gettime(t2, u2);
      diff = gettimediff(t1, u1, t2, u2);
      printf("\n MultiMolb compute time : %10.6f ms\n", diff*1000);
      tot_t += diff;
   }

   if (alfdebug) {
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

void multimola(double buffer, int vers, int nthreads, std::vector<int>& Nval)
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
}

void multimolb(int nthreads)
{
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
   double* surfthd = new double[ntot+nfudge];
   double* volthd = new double[ntot+nfudge];
   double* dsurfxthd = new double[ntot+nfudge];
   double* dsurfythd = new double[ntot+nfudge];
   double* dsurfzthd = new double[ntot+nfudge];
   double* dvolxthd = new double[ntot+nfudge];
   double* dvolythd = new double[ntot+nfudge];
   double* dvolzthd = new double[ntot+nfudge];
   int vers = info[threadid].vers;

   alphamol(ntot, newatoms, surfthd, volthd,
      dsurfxthd, dsurfythd, dsurfzthd, dvolxthd, dvolythd, dvolzthd, vers);

   // transfer information to thread
   info[threadid].wsurf = 0;
   info[threadid].wvol = 0;

   for(int i = 0; i < natm; i++) {
      info[threadid].wsurf += surfthd[i];
      info[threadid].wvol += volthd[i];
   }

   for (int i = 0; i < natm; i++) {
      info[threadid].surf[newatoms[i].index] = surfthd[i];
      info[threadid].vol[newatoms[i].index] = volthd[i];
   }
   auto do_g = vers & calc::grad;
   if (do_g) {
      for (int i = 0; i < natm; i++) {
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