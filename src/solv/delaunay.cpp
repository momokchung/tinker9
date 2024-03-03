#include "ff/solv/alffunc.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/delsort.h"

namespace tinker
{
inline void locatejw(int ipoint, int& tetra_loc, bool& iredundant);
inline void reordertetra();
inline void removeinf();
inline void peel(int& flag);

void delaunay()
{
   int tetra_loc;
   int ipoint;
   int tetra_last = -1;
   bool iredundant;
   int npoint = vertices.size()-4;

   for (int i = 0; i < npoint; i++) {
      ipoint = i+4;
      if (vertices[ipoint].info[1] == 0) continue;

      // first locate the point in the list of known tetrahedra
      tetra_loc = tetra_last;
      locatejw(ipoint, tetra_loc, iredundant);

      // if the point is redundant, move to next point
      if (iredundant) {
         vertices[ipoint].info[0] = 0;
         continue;
      }

      // otherwise, add point to tetrahedron: 1-4 flip
      int dummy;
      flip_1_4(ipoint, tetra_loc, dummy);

      // now scan link_facet list, and flip till list is empty
      flip();

      // At this stage, I should have a regular triangulation
      // of the i+4 points (i-th real points+4 "infinite" points)
      // Now add another point
   }

   // reorder the tetrahedra, such that vertices are in increasing order
   reordertetra();

   // I have the regular triangulation: I need to remove the
   // simplices including infinite points, and define the convex hull
   removeinf();

   // peel off flat tetrahedra at the boundary of the DT
   int nt = 0;
   for (int i = 0; i < tetra.size(); i++) if (tetra[i].info[1]!=0) nt++;
   if (nt > 1) {
      int flag;
      do {
         peel(flag);
      } while (flag != 0); 
   }
}

inline void insidetetra(int p, int a, int b, int c, int d, int iorient, bool& is_in, bool& redundant, int& ifail);

inline void locatejw(int ipoint, int& tetra_loc, bool& iredundant)
{
   // define starting tetrahedron
   iredundant = false;
   int ntetra = tetra.size();

   if (ntetra == 1) {
      tetra_loc = 0;
      return;
   }

   int itetra=-1;
   if (tetra_loc < 0) {
      for (int i = ntetra-1; i >=0; i--) {
         if (tetra[i].info[1] == 1) {
               itetra = i;
               break;
         }
      }
   }
   else {
      itetra = tetra_loc;
   }

   int a, b, c, d, iorient;
   bool test_in, test_red;
   int idx;

   do {
      a = tetra[itetra].vertices[0];
      b = tetra[itetra].vertices[1];
      c = tetra[itetra].vertices[2];
      d = tetra[itetra].vertices[3];
      iorient = -1;
      if (tetra[itetra].info[0]==1) iorient = 1;

      insidetetra(ipoint, a, b, c, d, iorient, test_in, test_red, idx);

      if (!test_in) itetra = tetra[itetra].neighbors[idx];

   } while (!test_in);

   tetra_loc = itetra;

   if (test_red) iredundant = true;
}

inline void insidetetra(int p, int a, int b, int c, int d, int iorient, bool& is_in, bool& redundant, int& ifail)
{

   int i,j,k,l;
   int ia,ib,ic,id,ie,idx;
   int ic1,ic5,ic1_k,ic1_l;
   int sign,sign5,sign_k,sign_l;
   int iswap,jswap,ninf,val;
   int tvert[4],infPoint[4];
   double xa,ya,xb,yb,xc,yc;
   double ra,rb,rc,rd,re;
   double Sij_1,Sij_2,Sij_3,Skl_1,Skl_2,Skl_3;
   double det_pijk,det_pjil,det_pkjl,det_pikl,det_pijkl;
   double coord_a[3],coord_b[3],coord_c[3],coord_d[3],coord_e[3];
   double detij[3],i_p[4],j_p[4],k_p[4],l_p[4];
   bool test_pijk,test_pjil,test_pkjl,test_pikl;

   bool ifprint = true;

   // initialize some values
   ia = 0;
   ib = 0;
   ic = 0;
   id = 0;
   ie = 0;
   iswap = 0;
   jswap = 0;
   val = 0;

   // If (i,j,k,l) is the tetrahedron in positive orientation, we need to test:
   // (p,i,j,k)
   // (p,j,i,l)
   // (p,k,j,l)
   // (p,i,k,l)
   // If all four are positive, than p is inside the tetrahedron.
   // All four tests relies on the sign of the corresponding 4x4
   // determinant. Interestingly, these four determinants share
   // some common lines, which can be used to speed up the computation.

   // Let us consider or example:

   // det(p,i,j,k) = | p(1) p(2) p(3) 1|
   //                | i(1) i(2) i(3) 1|
   //                | j(1) j(2) j(3) 1|
   //                | k(1) k(2) k(3) 1|

   // p appears in each determinant. The corresponding line can therefore
   // be substraced from all 3 other lines. Using the example above,
   // we find:

   // det(i,j,k,l) = -|ip(1) ip(2) ip(3)|
   //                 |jp(1) jp(2) jp(3)|
   //                 |kp(1) kp(2) kp(3)|

   // where xp(m) = x(m) - p(m) for x = i,j,k and m = 1,2,3

   // Now we notice that the first two lines of det(p,i,j,k) and det(p,i,j,l) are the same.

   // Let us define: Sij_3 = |ip(1) ip(2)| Sij_2 = |ip(1) ip(3)| and Sij_1 = |ip(2) ip(3)|
   //                        |jp(1) jp(2)|         |jp(1) jp(3)|             |jp(2) jp(3)|

   // We find:
   // det(p,i,j,k) = -kp(1)*Sij_1 + kp(2)*Sij_2 - kp(3)*Sij_3
   // and:
   // det(p,j,i,l) =  lp(1)*Sij_1 - lp(2)*Sij_2 + lp(3)*Sij_3

   // Similarly, if we define:

   // Skl_3 = |kp(1) kp(2)| Skl_2 = |kp(1) kp(3)| Skl_1 = |kp(2) kp(3)|
   //         |lp(1) lp(2)|         |lp(1) lp(3)|         |lp(2) lp(3)|

   // We find:
   // det(p,k,j,l) =  jp(1)*Skl_1 - jp(2)*Skl_2 + jp(3)*Skl_3
   // and:
   // det(p,i,k,l) = -ip(1)*Skl_1 + ip(2)*Skl_2 - ip(3)*Skl_3

   // Furthermore:
   // det(p,i,j,k,l) = -ip(4)*det(p,k,j,l)-jp(4)*det(p,i,k,l)
   //                  -kp(4)*det(p,j,i,l)-lp(4)*det(p,i,j,k)

   // The equations above hold for the general case; special care is
   // required to take in account infinite points (see below)

   is_in = false;
   redundant = false;

   tvert[0] = a;
   tvert[1] = b;
   tvert[2] = c;
   tvert[3] = d;

   for (int i = 0; i < 4; i++) infPoint[i] = 0;
   if (a<4) infPoint[0] = 1;
   if (b<4) infPoint[1] = 1;
   if (c<4) infPoint[2] = 1;
   if (d<4) infPoint[3] = 1;

   ninf = infPoint[0] + infPoint[1] + infPoint[2] + infPoint[3];

   // no infinite points
   if (ninf == 0)
   {
      for (i = 0; i < 3; i++) {
         i_p[i] = vertices[a].coord[i] - vertices[p].coord[i];
         j_p[i] = vertices[b].coord[i] - vertices[p].coord[i];
         k_p[i] = vertices[c].coord[i] - vertices[p].coord[i];
         l_p[i] = vertices[d].coord[i] - vertices[p].coord[i];
      }

      // compute 2x2 determinants Sij and Skl
      Sij_1 = i_p[1] * j_p[2] - i_p[2] * j_p[1];
      Sij_2 = i_p[0] * j_p[2] - i_p[2] * j_p[0];
      Sij_3 = i_p[0] * j_p[1] - i_p[1] * j_p[0];

      Skl_1 = k_p[1] * l_p[2] - k_p[2] * l_p[1];
      Skl_2 = k_p[0] * l_p[2] - k_p[2] * l_p[0];
      Skl_3 = k_p[0] * l_p[1] - k_p[1] * l_p[0];

      // perform tests, check all other four determinants
      det_pijk = -k_p[0] * Sij_1 + k_p[1] * Sij_2 - k_p[2] * Sij_3;
      det_pijk = det_pijk * iorient;
      test_pijk = (std::abs(det_pijk) > deleps);
      if (test_pijk && det_pijk > 0) {
         ifail = 3;
         return;
      }

      det_pjil = l_p[0] * Sij_1 - l_p[1] * Sij_2 + l_p[2] * Sij_3;
      det_pjil = det_pjil * iorient;
      test_pjil = (std::abs(det_pjil) > deleps);
      if (test_pjil && det_pjil > 0) {
         ifail = 2;
         return;
      }

      det_pkjl = j_p[0] * Skl_1 - j_p[1] * Skl_2 + j_p[2] * Skl_3;
      det_pkjl = det_pkjl * iorient;
      test_pkjl = (std::abs(det_pkjl) > deleps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = 0;
         return;
      }

      det_pikl = -i_p[0] * Skl_1 + i_p[1] * Skl_2 - i_p[2] * Skl_3;
      det_pikl = det_pikl * iorient;
      test_pikl = (std::abs(det_pikl) > deleps);
      if (test_pikl && det_pikl > 0) {
         ifail = 1;
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough.
      // In this case, we have to rank the indices.

      if (!test_pijk) {
         valsort4(p, a, b, c, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         minor4(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
               ifail = 3;
               return;
         }
      }

      if (!test_pjil) {
         valsort4(p, b, a, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         minor4(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
               ifail = 2;
               return;
         }
      }

      if (!test_pkjl) {
         valsort4(p, c, b, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         minor4(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
               ifail = 0;
               return;
         }
      }

      if (!test_pikl) {
         valsort4(p, a, c, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         minor4(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
               ifail = 1;
               return;
         }
      }

      // if we have gone that far, p is inside the tetrahedron
      is_in = true;

      //Now we check if p is redundant
      i_p[3] = vertices[a].w - vertices[p].w;
      j_p[3] = vertices[b].w - vertices[p].w;
      k_p[3] = vertices[c].w - vertices[p].w;
      l_p[3] = vertices[d].w - vertices[p].w;

      det_pijkl = -i_p[3] * det_pkjl - j_p[3] * det_pikl - k_p[3] * det_pjil - l_p[3] * det_pijk;

      // no need to multiply by iorient, since all minors contais iorient
      if (std::abs(det_pijkl) < deleps) {
         valsort5(p, a, b, c, d, ia, ib, ic, id, ie, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
            coord_e[i] = vertices[ie].coord[i];
         }
         ra = vertices[ia].r;
         rb = vertices[ib].r;
         rc = vertices[ic].r;
         rd = vertices[id].r;
         re = vertices[ie].r;
         minor5(coord_a, ra, coord_b, rb, coord_c, rc, coord_d, rd, coord_e, re, val);
         det_pijkl = val * jswap * iorient;
      }
      redundant = (det_pijkl < 0) ? true : false;
   }
   else if (ninf == 1) {
      // We know that one of the 4 vertices a,b,c or d is infinite.
      // To find which one it is, we use a map between
      // (inf(a),inf(b),inf(c),inf(d)) and X, where inf(i)
      // is 1 if i is infinite, 0 otherwise, and X = 0,1,2,3
      // if a,b,c or d are infinite, respectively.
      // A good mapping function is:
      // X = 2 - inf(a) - inf(a) -inf(b) + inf(d)

      idx = (2 - infPoint[0] - infPoint[0] - infPoint[1] + infPoint[3]);
      l = tvert[idx];

      i = tvert[order1[idx][0]];
      j = tvert[order1[idx][1]];
      k = tvert[order1[idx][2]];

      ic1 = inf4_1[l];
      sign = sign4_1[l];

      // let us look at the four determinant we need to compute:
      // det_pijk : unchanged
      // det_pjil : 1 infinite point (l), becomes det3_pji where
      // det3_pij = |p(ic1) p(ic2) 1|
      //            |i(ic1) i(ic2) 1|
      //            |j(ic1) j(ic2) 1|
      // and ic1 and ic2 depends on which infinite ( ic2 is always 3)
      // point is considered
      // det_pkjl : 1 infinite point (l), becomes det3_pkj
      // det_pikl : 1 infinite point (l), becomes det3_pik

      //get Coordinates
      for (int _i = 0; _i < 3; _i++) {
         i_p[_i] = vertices[i].coord[_i] - vertices[p].coord[_i];
         j_p[_i] = vertices[j].coord[_i] - vertices[p].coord[_i];
         k_p[_i] = vertices[k].coord[_i] - vertices[p].coord[_i];
      }

      detij[0] = i_p[0] * j_p[2] - i_p[2] * j_p[0];
      detij[1] = i_p[1] * j_p[2] - i_p[2] * j_p[1];
      detij[2] = i_p[0] * j_p[1] - i_p[1] * j_p[0];

      is_in = false;

      det_pijk = -k_p[0] * detij[1] + k_p[1] * detij[0] - k_p[2] * detij[2];
      det_pijk = det_pijk * iorient;
      test_pijk = (std::abs(det_pijk) > deleps);
      if (test_pijk && det_pijk > 0) {
         ifail = idx;
         return;
      }

      det_pjil = -detij[ic1] * sign * iorient;
      test_pjil = (std::abs(det_pjil) > deleps);
      if (test_pjil && det_pjil > 0) {
         ifail = order1[idx][2];
         return;
      }

      det_pkjl = k_p[ic1] * j_p[2] - k_p[2] * j_p[ic1];
      det_pkjl = sign * det_pkjl * iorient;
      test_pkjl = (std::abs(det_pkjl) > deleps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = order1[idx][0];
         return;
      }

      det_pikl = i_p[ic1] * k_p[2] - i_p[2] * k_p[ic1];
      det_pikl = sign * det_pikl * iorient;
      test_pikl = (std::abs(det_pikl) > deleps);
      if (test_pikl && det_pikl > 0) {
         ifail = order1[idx][1];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         valsort4(p, i, j, k, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         minor4(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
               ifail = idx;
               return;
         }
      }

      if (!test_pjil) {
         valsort3(p, j, i, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order1[idx][2];
            return;
         }
      }

      if (!test_pkjl) {
         valsort3(p, k, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order1[idx][0];
            return;
         }
      }

      if (!test_pikl) {
         valsort3(p, i, k, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order1[idx][1];
            return;
         }
      }

      // if we have gone so far, p is inside the tetrahedron
      is_in = true;

      // Now we check if p is redundant since
      // det_pijkl = det_pijk > 1 p cannot be redundant!
      redundant = false;
   }
   else if (ninf == 2) {
      // We know that two of the 4 vertices a,b,c or d are infinite
      // To find which one it is, we use a map between
      // (inf(a),inf(b),inf(c),inf(d)) and X, where inf(i)
      // is 1 if i is infinite, 0 otherwise, and X = 1,2,3,4,5,6
      // if (a,b), (a,c), (a,d), (b,c), (b,d), or (c,d) are
      // infinite, respectively.
      // A good mapping function is:
      // X = 2 - inf(a) - inf(a) +inf(c) + inf(d) + inf(d)

      idx = (2 - infPoint[0] - infPoint[0] + infPoint[2] + 2 * infPoint[3]);

      // the two infinite points :
      k = tvert[order3[idx][0]];
      l = tvert[order3[idx][1]];

      // the two finite points
      i = tvert[order2[idx][0]];
      j = tvert[order2[idx][1]];

      ic1_k = inf4_1[k];
      ic1_l = inf4_1[l];
      sign_k = sign4_1[k];
      sign_l = sign4_1[l];
      ic1 = inf4_2[l][k];
      sign = sign4_2[l][k];

      // get coordinates
      for (int _i = 0; _i < 3; _i++) {
         i_p[_i] = vertices[i].coord[_i] - vertices[p].coord[_i];
         j_p[_i] = vertices[j].coord[_i] - vertices[p].coord[_i];
      }

      // Perform test; first set is_in .false.
      is_in = false;

      // det_pijk is now det3_pij with k as infinite point
      det_pijk = i_p[ic1_k] * j_p[2] - i_p[2] * j_p[ic1_k];
      det_pijk = det_pijk * sign_k * iorient;
      test_pijk = (std::abs(det_pijk) > deleps);
      if (test_pijk && det_pijk > 0) {
         ifail = order3[idx][1];
         return;
      }

      // det_pjil is now det3_pji with l as infinite point
      det_pjil = i_p[2] * j_p[ic1_l] - i_p[ic1_l] * j_p[2];
      det_pjil = det_pjil * sign_l * iorient;
      test_pjil = (std::abs(det_pjil) > deleps);
      if (test_pjil && det_pjil > 0) {
         ifail = order3[idx][0];
         return;
      }

      // det_pkjl is now -det2_pj (k,l infinite)
      det_pkjl = j_p[ic1] * sign * iorient;
      test_pkjl = (std::abs(det_pkjl) > deleps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = order2[idx][0];
         return;
      }

      // det_pikl is now det2_pi (k,l infinite)
      det_pikl = -i_p[ic1] * sign * iorient;
      test_pikl = (std::abs(det_pikl) > deleps);
      if (test_pikl && det_pikl > 0) {
         ifail = order2[idx][1];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         valsort3(p, i, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1_k];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1_k];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1_k];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         val = val * sign_k * jswap * iorient;
         if (val == 1) {
            ifail = order3[idx][1];
            return;
         }
      }

      if (!test_pjil) {
         valsort3(p, j, i, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1_l];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1_l];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1_l];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         val = val * sign_l * jswap * iorient;
         if (val == 1) {
            ifail = order3[idx][0];
            return;
         }
      }

      if (!test_pkjl) {
         valsort2(p, j, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         minor2(xa, xb, val);
         val = -val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order2[idx][0];
            return;
         }
      }

      if (!test_pikl) {
         valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         minor2(xa, xb, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order2[idx][1];
            return;
         }
      }

      // if we have gone so far, p is inside the tetrahedron
      is_in = true;

      // now we check if p is redundant det_pijkl becomes det3_pij
      ic5 = inf5_2[l][k];
      sign5 = sign5_2[l][k];
      det_pijkl = i_p[ic5] * j_p[2] - i_p[2] * j_p[ic5];
      if (std::abs(det_pijkl) < deleps) {
         valsort3(p, i, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic5];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic5];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic5];
         yc = vertices[ic].coord[temp];
         minor3(xa, ya, xb, yb, xc, yc, val);
         det_pijkl = val*jswap;
      }
      det_pijkl = det_pijkl * sign5 * iorient;
      redundant = (det_pijkl < 0) ? true : false;
   }
   else if (ninf == 3) {
      // We know that three of the 4 vertices a,b,c or d are infinite
      // To find which one is finite, we use a map between
      // (inf(a),inf(b),inf(c),inf(d)) and X, where inf(i)
      // is 1 if i is infinite, 0 otherwise, and X = 0, 1, 2, 3
      // if a,b,c or d are finite, respectively.
      // A good mapping function is:
      // X = inf(a) + inf(a) +inf(b) - inf(d)

      idx = 2 * infPoint[0] + infPoint[1] - infPoint[3];

      i = tvert[idx];
      j = tvert[order1[idx][0]];
      k = tvert[order1[idx][1]];
      l = tvert[order1[idx][2]];

      // index of the "missing" infinite point (i.e. the fourth infinite point)
      missinf_sign(j, k, l, ie, iswap);

      // get coordinates
      for (int _i = 0; _i < 3; _i++) {
         i_p[_i] = vertices[i].coord[_i] - vertices[p].coord[_i];
      }

      // perform test; first set is_in to .false.
      is_in = false;

      // det_pijk is now - det2_pi (missing j,k)
      det_pijk = i_p[inf4_2[k][j]] * iorient * sign4_2[k][j];
      test_pijk = (std::abs(det_pijk) > deleps);
      if (test_pijk && det_pijk > 0) {
         ifail = order1[idx][2];
         return;
      }

      // det_pjil is now det2_pi (missing j,l)
      det_pjil = -i_p[inf4_2[l][j]] * iorient * sign4_2[l][j];
      test_pjil = (std::abs(det_pjil) > deleps);
      if (test_pjil && det_pjil > 0) {
         ifail = order1[idx][1];
         return;
      }

      // det_pkjl is now det1_p
      det_pkjl = iorient * iswap * sign4_3[ie];
      if (det_pkjl > 0) {
         ifail = idx;
         return;
      }

      // det_ikl is now - det2_pi (missing k,l)
      det_pikl = i_p[inf4_2[l][k]] * iorient * sign4_2[l][k];
      test_pikl = (std::abs(det_pikl) > deleps);
      if (test_pikl && det_pikl > 0) {
         ifail = order1[idx][0];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[k][j]];
         xb = vertices[ib].coord[inf4_2[k][j]];
         minor2(xa, xb, val);
         val = -val * sign4_2[k][j] * iorient * jswap;
         if (val == 1) {
            ifail = order1[idx][2];
            return;
         }
      }

      if (!test_pjil) {
         valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[l][j]];
         xb = vertices[ib].coord[inf4_2[l][j]];
         minor2(xa, xb, val);
         val = val * sign4_2[l][j] * iorient * jswap;
         if (val == 1) {
            ifail = order1[idx][1];
            return;
         }
      }

      if (!test_pikl) {
         valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[l][k]];
         xb = vertices[ib].coord[inf4_2[l][k]];
         minor2(xa, xb, val);
         val = -val * sign4_2[l][k] * iorient * jswap;
         if (val == 1) {
            ifail = order1[idx][0];
            return;
         }
      }

      is_in = true;

      // now check for redundancy det_pijkl becomes -det2_pi
      ic1 = inf5_3[ie];
      sign5 = sign5_3[ie];
      det_pijkl = -i_p[ic1];
      if (std::abs(det_pijkl) < deleps) {
         valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         minor2(xa, xb, val);
         det_pijkl = val * jswap;
      }
      det_pijkl = -iorient * det_pijkl * sign5 * iswap;
      redundant = (det_pijkl < 0) ? true : false;
   }
   else {
      // In the case all four points ia, ib, ic, and id are infinite,
      // then is_in = true and redundant = false
      is_in = true;
      redundant = false;
   }
}

inline void reordertetra()
{
   int ntetra = tetra.size();
   int vert[4], idx[4], neighbor[4];
   int n = 4;
   int nswap;
   char nidx[4];
   bool nsurf[4];

   for (int i = 0; i < ntetra; i++)
   {
      if (tetra[i].info[1]==0) continue;

      for (int j = 0; j < 4; j++) {
         vert[j] = tetra[i].vertices[j];
      }
      sort4sign(vert, idx, nswap, n);

      for (int j = 0; j < 4; j++) {
         neighbor[j] = tetra[i].neighbors[idx[j]];
         nidx[j] = tetra[i].nindex[idx[j]];
         std::string s = tetra[i].info.to_string();
         nsurf[j] = (s[2+idx[j]]=='1');
         if (neighbor[j] != -1) {
            tetra[neighbor[j]].nindex[nidx[j]]=j;
         }
      }

      for (int j = 0; j < 4; j++) {
         tetra[i].vertices[j] = vert[j];
         tetra[i].neighbors[j] = neighbor[j];
         tetra[i].nindex[j] = nidx[j];
         tetra[i].info.set(2+j,nsurf[j]);
      }

      if (nswap==-1) {
         if (tetra[i].info[0] == 0) {
            tetra[i].info[0] = 1;
         }
         else {
            tetra[i].info[0] = 0;
         }
      }
   }
}

inline void markzero(int itetra, int ivertex);

inline void removeinf()
{
   int a, b, c, d;
   int ntetra = tetra.size();

   for (int i = 0; i < ntetra; i++) {
      if (tetra[i].info[1]==0) continue;

      a = tetra[i].vertices[0];
      b = tetra[i].vertices[1];
      c = tetra[i].vertices[2];
      d = tetra[i].vertices[3];

      if (a<4 || b<4 || c<4 || d<4) {
         tetra[i].info[2] = 1;
         tetra[i].info[1] = 0;
         if (a < 4) markzero(i, 0);
         if (b < 4) markzero(i, 1);
         if (c < 4) markzero(i, 2);
         if (d < 4) markzero(i, 3);
      }
   }

   for (int i = 0; i < 4; i++) {
      vertices[i].info[0] = 0;
   }
}

inline void markzero(int itetra, int ivertex)
{
   int jtetra, jvertex;

   jtetra = tetra[itetra].neighbors[ivertex];

   if(jtetra != -1) {
      jvertex = tetra[itetra].nindex[ivertex];
      tetra[jtetra].neighbors[jvertex] = -1;
   }
}

double tetravol(double* a, double* b, double* c, double* d);

inline void peel(int& flag)
{
   int ia, ib, ic, id;
   int k, l;
   int ntetra = tetra.size();
   double coorda[3], coordb[3], coordc[3], coordd[3];

   flag = 0;
   for (int i = 0; i < ntetra; i++) {

      if (tetra[i].info[1]==0) continue;

      bool itest = false;
      for (int j = 0; j < 4; j++) {
         if (tetra[i].neighbors[j]==-1) itest = true;
      }

      if (!itest) continue; 

      // This is a tetrahedron at the boundary: we test
      // if it is flat, i.e. if its volume is 0

      ia = tetra[i].vertices[0];
      ib = tetra[i].vertices[1];
      ic = tetra[i].vertices[2];
      id = tetra[i].vertices[3];

      for (int j = 0; j < 3; j++) {
         coorda[j] = vertices[ia].coord[j];
         coordb[j] = vertices[ib].coord[j];
         coordc[j] = vertices[ic].coord[j];
         coordd[j] = vertices[id].coord[j];
      }

      double vol = tetravol(coorda, coordb, coordc, coordd);

      if (std::abs(vol) < delepsvol) {
         flag++;
         tetra[i].info[2] = 1;
      }
   }

   // Now we remove those flat tetrahedra, and update the links
   // to their neighbours
   for (int i = 0; i < ntetra; i++) {
      if (tetra[i].info[2]==1) {
         if (tetra[i].info[1]==1) {
            tetra[i].info[1] = 0;
            for (int j = 0; j < 4; j++) {
               k = tetra[i].neighbors[j];
               if (k != -1) {
                  l = tetra[i].nindex[j];
                  tetra[k].neighbors[l] = -1;
               }
            }
         }
      }
   }
}

// computes the volume of a tetrahedron
inline double tetravol(double* a, double* b, double* c, double* d)
{
   double vol;
   double ad[3], bd[3], cd[3];
   double Sbcd[3];

   // The volume of the tetrahedron is proportional to:
   // vol = det | a(1)  a(2)  a(3)  1|
   //           | b(1)  b(2)  b(3)  1|
   //           | c(1)  c(2)  c(3)  1|
   //           | d(1)  d(2)  d(3)  1|
   // After substracting the last row from the first 3 rows, and
   // developping with respect to the last column, we obtain:
   // vol = det | ad(1)  ad(2)  ad(3) |
   //           | bd(1)  bd(2)  bd(3) |
   //           | cd(1)  cd(2)  cd(3) |
   // where ad(i) = a(i) - d(i), ...

   for(int i = 0; i < 3; i++) {
      ad[i] = a[i] - d[i];
      bd[i] = b[i] - d[i];
      cd[i] = c[i] - d[i];
   }

   Sbcd[2] = bd[0]*cd[1] - cd[0]*bd[1];
   Sbcd[1] = bd[0]*cd[2] - cd[0]*bd[2];
   Sbcd[0] = bd[1]*cd[2] - cd[1]*bd[2];

   vol = ad[0]*Sbcd[0] - ad[1]*Sbcd[1] + ad[2]*Sbcd[2];

   return vol;
}
}
