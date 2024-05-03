#include "ff/solv/alffunc.h"
#include "ff/solv/alphamol.h"
#include "ff/solv/delsort.h"
#include <cassert>
#include <cmath>

namespace tinker
{
// "regular_convex" checks for local regularity and convexity
void regular_convex(std::vector<Vertex>& vertices, int a, int b, int c, int p, int o, int itest_abcp,
   bool& regular, bool& convex, bool& test_abpo, bool& test_bcpo, bool& test_capo) 
{
   int i,j,k,l,m;
   int ia,ib,ic,id,ie;
   int ninf,infp,info,iswap,iswap2,idx,val;
   int icol1,sign1,icol2,sign2,icol4,sign4,icol5,sign5;
   int list[3],infPoint[3];
   double det_abpo,det_bcpo,det_capo,det_abcpo,det_abpc;
   double xa,ya,xb,yb,xc,yc;
   double ra,rb,rc,rd,re;
   double a_p[4],b_p[4],c_p[4],o_p[4];
   double i_p[3],j_p[3];
   double Mbo[3],Mca[3],Mjo[3],Mio[3];
   double coord1[3],coord2[3],coord3[3],coord4[3],coord5[3];
   bool testc[3];

   bool ifprint = true;

   ia = 0;
   ib = 0;
   ic = 0;
   id = 0;
   ie = 0;
   iswap = 0;
   iswap2 = 0;
   val = 0;
   icol4 = 0;

   regular = true;
   convex  = true;
   test_abpo = false;
   test_bcpo = false;
   test_capo = false;

   // To test if the union of the two tetrahedron is convex, check the position of o w.r.t
   // three faces (a,b,p), (b,c,p) and (c,a,p) of (a,b,c,p).evaluate the three determinants:
   //     det(a,b,p,o)
   //     det(b,c,p,o)
   //     det(c,a,p,o)
   // If the three determinants are positive, & det(a,b,c,p) is negative,the union is convex
   // If the three determinants are negative, & det(a,b,c,p) is positive,the union is convex
   // In all other cases, the union is non convex
   // The regularity is tested by computing det(a,b,c,p,o)

   // first count how many infinite points (except o)
   // only a and/or b and/or c can be infinite:
   infPoint[0] = 0;
   infPoint[1] = 0;
   infPoint[2] = 0;
   if (a < 4) infPoint[0] = 1;
   if (b < 4) infPoint[1] = 1;
   if (c < 4) infPoint[2] = 1;

   ninf = infPoint[0] + infPoint[1] + infPoint[2];

   list[0] = a;
   list[1] = b;
   list[2] = c;

   // general case:no inf points
   if (ninf == 0) {
      // First, simple case:
      // if o is infinite, then det(a,b,c,p,o) = -det(a,b,c,p)
      // and consequently (a,b,c,p,o) is regular:nothing to do!
      if (o < 4) {
         regular = true;
         return;
      }

      // The three determinants det(a,b,p,o), det(b,c,p,o), and det(c,a,p,o) are "real" 4x4 determinants.
      // Subtract the row corresponding to p from the other row, and develop with respect to p.
      // The determinants become:
      //     det(a,b,p,o)= - | ap(1) ap(2) ap(3) |
      //                     | bp(1) bp(2) bp(3) |
      //                     | op(1) op(2) op(3) |

      //     det(b,c,p,o)=  -| bp(1) bp(2) bp(3) |
      //                     | cp(1) cp(2) cp(3) |
      //                     | op(1) op(2) op(3) |

      //     det(c,a,p,o)=  -| cp(1) cp(2) cp(3) |
      //                     | ap(1) ap(2) ap(3) |
      //                     | op(1) op(2) op(3) |

      // Where ip(j) = i(j) - p(j) for all i in {a,b,c,o} and j in {1,2,3}
      // Compute two types of minors:
      //     Mbo_ij = bp(i)op(j) - bp(j)op(i) and Mca_ij = cp(i)ap(j) - cp(j)op(i)

      // Store Mbo_12 in Mbo(3), Mbo_13 in Mbo(2),...

      // get the coordinates
      for (m = 0; m < 3; m++) {
         a_p[m] = vertices[a].coord[m] - vertices[p].coord[m];
         b_p[m] = vertices[b].coord[m] - vertices[p].coord[m];
         c_p[m] = vertices[c].coord[m] - vertices[p].coord[m];
         o_p[m] = vertices[o].coord[m] - vertices[p].coord[m];
      }
      a_p[3] = vertices[a].w - vertices[p].w;
      b_p[3] = vertices[b].w - vertices[p].w;
      c_p[3] = vertices[c].w - vertices[p].w;
      o_p[3] = vertices[o].w - vertices[p].w;

      //c ompute 2x2 determinants Mbo and Mca
      Mbo[0] = b_p[1] * o_p[2] - b_p[2] * o_p[1];
      Mbo[1] = b_p[0] * o_p[2] - b_p[2] * o_p[0];
      Mbo[2] = b_p[0] * o_p[1] - b_p[1] * o_p[0];

      Mca[0] = c_p[1] * a_p[2] - c_p[2] * a_p[1];
      Mca[1] = c_p[0] * a_p[2] - c_p[2] * a_p[0];
      Mca[2] = c_p[0] * a_p[1] - c_p[1] * a_p[0];

      det_abpo = -a_p[0] * Mbo[0] + a_p[1] * Mbo[1] - a_p[2] * Mbo[2];
      det_bcpo =  c_p[0] * Mbo[0] - c_p[1] * Mbo[1] + c_p[2] * Mbo[2];
      det_capo = -o_p[0] * Mca[0] + o_p[1] * Mca[1] - o_p[2] * Mca[2];
      det_abpc = -b_p[0] * Mca[0] + b_p[1] * Mca[1] - b_p[2] * Mca[2];

      // To compute:
      // det(a,b,c,p,o) = | a(1) a(2) a(3) a(4) 1 |
      //                  | b(1) b(2) b(3) b(4) 1 |
      //                  | c(1) c(2) c(3) c(4) 1 |
      //                  | p(1) p(2) p(3) p(4) 1 |
      //                  | o(1) o(2) o(3) o(4) 1 |
      // First substract row p :
      // det(a,b,c,p,o) = -| ap(1) ap(2) ap(3) ap(4) |
      //                   | bp(1) bp(2) bp(3) bp(4) |
      //                   | cp(1) cp(2) cp(3) cp(4) |
      //                   | op(1) op(2) op(3) op(4) |

      // expand w.r.t last column
      det_abcpo = -a_p[3] * det_bcpo - b_p[3] * det_capo - c_p[3] * det_abpo + o_p[3] * det_abpc;

      // if (a,b,c,p,o) regular, no need to flip
      if (std::abs(det_abcpo) < deleps) {
         valsort5(a, b, c, p, o, ia, ib, ic, id, ie, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
            coord5[i] = vertices[ie].coord[i];
         }
         ra = vertices[ia].r; rb = vertices[ib].r;
         rc = vertices[ic].r; rd = vertices[id].r;
         re = vertices[ie].r;
         minor5(coord1, ra, coord2, rb, coord3, rc, coord4, rd, coord5, re, val);
         det_abcpo = val * iswap;
      }

      if ((det_abcpo * itest_abcp) < 0) {
         regular = true;
         return;
      }
      regular = false;

      // if not regular, we test for convexity
      if (std::abs(det_abpo) < deleps) {
         valsort4(a, b, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         minor4(coord1, coord2, coord3, coord4, val);
         det_abpo = val * iswap;
      }
      if (std::abs(det_bcpo) < deleps) {
         valsort4(b, c, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         minor4(coord1, coord2, coord3, coord4, val);
         det_bcpo = val * iswap;
      }
      if (std::abs(det_capo) < deleps) {
         valsort4(c, a, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         minor4(coord1, coord2, coord3, coord4, val);
         det_capo = val * iswap;
      }

      test_abpo = (det_abpo > 0);
      test_bcpo = (det_bcpo > 0);
      test_capo = (det_capo > 0);

      convex = false;
      if ((itest_abcp * det_abpo) > 0) return;
      if ((itest_abcp * det_bcpo) > 0) return;
      if ((itest_abcp * det_capo) > 0) return;
      convex = true;
   }
   else if (ninf == 1) { 
      // Define X as infinite point, and (i,j) the pair of finite points.
      // If X = a, (i,j) = (b,c)
      // If X = b, (i,j) = (c,a)
      // If X = c, (i,j) = (a,b)
      // If we define inf(a) = 1 if a infinite, 0 otherwise,
      // then idx_X  = 2 - inf(a) + inf(c)

      idx = 1 - infPoint[0] + infPoint[2];
      infp = list[idx];
      i = list[order[idx][0]];
      j = list[order[idx][1]];

      // get the coordinates
      for (m = 0; m < 3; m++) {
         i_p[m] = vertices[i].coord[m] - vertices[p].coord[m];
         j_p[m] = vertices[j].coord[m] - vertices[p].coord[m];
      }

      if (o > 3) {
         // first case: o is finite
         for (m = 0; m < 3; m++) {
            o_p[m] = vertices[o].coord[m] - vertices[p].coord[m];
         }

         icol1 = inf4_1[infp];
         sign1 = sign4_1[infp];

         // The three 4x4 determinants become:
         //     -det(i,p,o) [X missing], det(j,p,o) [X missing],det(i,j,p,o)
         // And the 5x5 determinant becomes:
         //     -det(i,j,p,o)

         Mjo[0] = j_p[0] * o_p[2] - j_p[2] * o_p[0];
         Mjo[1] = j_p[1] * o_p[2] - j_p[2] * o_p[1];
         Mjo[2] = j_p[0] * o_p[1] - j_p[1] * o_p[0];

         // The correspondence between a,b,c and i,j is not essential
         // We use corresponce for a infinite; in thetwo other cases
         // (b infinite or c infinite),computed determinants are the
         // same , but not in the same order

         det_abpo = i_p[icol1] * o_p[2] - i_p[2] * o_p[icol1];
         if (std::abs(det_abpo) < deleps) {
            int temp = 2;
            valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_abpo = -val * iswap;
         }
         det_abpo = det_abpo * sign1;
         det_capo = -Mjo[icol1];
         if (std::abs(det_capo) < deleps) {
            int temp = 2;
            valsort3(j, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_capo = val * iswap;
         }
         det_capo = det_capo * sign1;
         det_bcpo = -i_p[0] * Mjo[1] + i_p[1] * Mjo[0] - i_p[2] * Mjo[2];
         if (std::abs(det_bcpo) < deleps) {
            valsort4(i, j, p, o, ia, ib, ic, id, iswap);
            for (int i = 0; i < 3; i++) {
               coord1[i] = vertices[ia].coord[i];
               coord2[i] = vertices[ib].coord[i];
               coord3[i] = vertices[ic].coord[i];
               coord4[i] = vertices[id].coord[i];
            }
            minor4(coord1, coord2, coord3, coord4, val);
            det_bcpo = val * iswap;
      }
      det_abcpo = -det_bcpo;
      }
      else {
         // second case: o is infinite
         info = o;

         // The three 4x4 determinants become:
         //     -det(i,p) [o,X missing]
         //     det(j,p) [o,X missing]
         //     det(i,j,p) [o missing]
         // And the 5x5 determinant becomes:
         //     det(i,j,p) [o,X missing]

         icol1 = inf4_2[infp][info];
         sign1 = sign4_2[infp][info];

         icol2 = inf4_1[info];
         sign2 = sign4_1[info];

         icol5 = inf5_2[infp][info];
         sign5 = sign5_2[infp][info];

         det_abpo = -i_p[icol1] * sign1;
         if (std::abs(det_abpo) < deleps) {
            valsort2(i, p, ia, ib, iswap);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            minor2(xa, xb, val);
            det_abpo = -val * iswap * sign1;
         }
         det_capo = j_p[icol1] * sign1;
         if (std::abs(det_capo) < deleps) {
            valsort2(j, p, ia, ib, iswap);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            minor2(xa, xb, val);
            det_capo = val * iswap * sign1;
         }
         det_bcpo = i_p[icol2] * j_p[2] - i_p[2] * j_p[icol2];
         if (std::abs(det_bcpo) < deleps) {
            int temp = 2;
            valsort3(i, j, p, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol2];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol2];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol2];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_bcpo = val * iswap;
         }
         det_bcpo = det_bcpo * sign2;
         det_abcpo = i_p[icol5] * j_p[2] - i_p[2] * j_p[icol5];
         if (std::abs(det_abcpo) < deleps) {
            int temp = 2;
            valsort3(i, j, p, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol5];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol5];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol5];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_abcpo = val * iswap;
         }
         det_abcpo = det_abcpo * sign5;
      }

      // test if (a,b,c,p,o) regular, in which case there is no need to flip
      if ((det_abcpo * itest_abcp) < 0) {
         regular = true;
         return;
      }
      regular = false;

      // if not regular, we test for convexity

      testc[0] = (det_abpo > 0);
      testc[1] = (det_bcpo > 0);
      testc[2] = (det_capo > 0);
      test_abpo = testc[ord_rc[idx][0]];
      test_bcpo = testc[ord_rc[idx][1]];
      test_capo = testc[ord_rc[idx][2]];

      convex = false;
      if ((itest_abcp * det_abpo) > 0) return;
      if ((itest_abcp * det_bcpo) > 0) return;
      if ((itest_abcp * det_capo) > 0) return;
      convex = true;
   }
   else if (ninf == 2) {
      // define(k,l) as the two infinite points, and i be finite
      // If i = a, (k,l) = (b,c)
      // If i = b, (k,l) = (c,a)
      // If i = c, (k,l) = (a,b)
      // Again: idx = 2 + inf(a) - inf(c)

      idx = 1 + infPoint[0] - infPoint[2];
      i = list[idx];
      k = list[order[idx][0]];
      l = list[order[idx][1]];

      // get the coordinates
      for (m = 0; m < 3; m++) {
         i_p[m] = vertices[i].coord[m] - vertices[p].coord[m];
      }

      if (o > 3) {
         // first case: o is finite

         // The three 4x4 determinants become:
         //     det(i,p,o) [k missing]
         //     -det(i,p,o) [l missing]
         //     S*det(p,o) [k,l missing, with S =1 if k<l, -1 otherwise]
         // The 5x5 determinants become:
         //     S*det(i,p,o) [k,l missing, with S=1 if k<l, -1 otherwise]

         for (m = 0; m < 3; m++) {
            o_p[m] = vertices[o].coord[m] - vertices[p].coord[m];
         }
         icol1 = inf4_1[k];
         sign1 = sign4_1[k];
         icol2 = inf4_1[l];
         sign2 = sign4_1[l];
         icol4 = inf4_2[l][k];
         sign4 = sign4_2[l][k];
         icol5 = inf5_2[l][k];
         sign5 = sign5_2[l][k];

         Mio[0] = i_p[0] * o_p[2] - i_p[2] * o_p[0];
         Mio[1] = i_p[1] * o_p[2] - i_p[2] * o_p[1];
         Mio[2] = i_p[0] * o_p[1] - i_p[1] * o_p[0];

         // The correspondence between a,b,c and i,j,k is not essential
         // use the correspondence for a finite; in the two other cases
         // (b finite or c finite),have computed the same determinants,
         // but not in the same order

         det_abpo = -Mio[icol1] * sign1;
         if (std::abs(det_abpo) < deleps) {
            int temp = 2;
            valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_abpo = val * iswap * sign1;
         }
         det_capo = Mio[icol2] * sign2;
         if (std::abs(det_capo) < deleps) {
            int temp = 2;
            valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol2];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol2];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol2];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_capo = -val * iswap * sign2;
         }
         det_bcpo = -o_p[icol4] * sign4;
         if (std::abs(det_bcpo) < deleps) {
            valsort2(p, o, ia, ib, iswap);
            xa = vertices[ia].coord[icol4];
            xb = vertices[ib].coord[icol4];
            minor2(xa, xb, val);
            det_bcpo = val * sign4 * iswap;
         }
         det_abcpo = -Mio[icol5] * sign5;
         if (std::abs(det_abcpo) < deleps) {
            int temp = 2;
            valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol5];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol5];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol5];
            yc = vertices[ic].coord[temp];
            minor3(xa, ya, xb, yb, xc, yc, val);
            det_abcpo = val * iswap * sign5;
         }
      }
      else {
         // second case: o is infinite
         info = o;

         // The three 4x4 determinants become:
         //     det(i,p) [o,k missing]
         //     -det(i,p) [o,l missing]
         //     Const [o,k,l missing]
         // The 5x5 determinants become:
         //     Const*det(i,p) [o,k,l missing]

         icol1 = inf4_2[k][info];
         sign1 = sign4_2[k][info];
         icol2 = inf4_2[l][info];
         sign2 = sign4_2[l][info];

         missinf_sign(info, k, l, icol4, iswap);

         det_abpo = i_p[icol1] * sign1;
         if (std::abs(det_abpo) < deleps) {
            valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            minor2(xa, xb, val);
            det_abpo = val * iswap2 * sign1;
         }
         det_capo = -i_p[icol2] * sign2;
         if (std::abs(det_capo) < deleps) {
            valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[icol2];
            xb = vertices[ib].coord[icol2];
            minor2(xa, xb, val);
            det_capo = -val * iswap2 * sign2;
         }
         det_bcpo = sign4_3[icol4] * iswap;
         det_abcpo = sign5_3[icol4] * iswap * i_p[inf5_3[icol4]];
         if (std::abs(det_abcpo) < deleps) {
            valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[inf5_3[icol4]];
            xb = vertices[ib].coord[inf5_3[icol4]];
            minor2(xa, xb, val);
            det_abcpo = val * iswap2 * iswap * sign5_3[icol4];
         }
      }
      // if (a,b,c,p,o) regular, no need to flip
      if ((det_abcpo * itest_abcp) < 0) {
         regular = true;
         return;
      }
      regular = false;

      // if not regular, we test for convexity
      testc[0] = (det_abpo > 0);
      testc[1] = (det_bcpo > 0);
      testc[2] = (det_capo > 0);
      test_abpo = testc[ord_rc[idx][0]];
      test_bcpo = testc[ord_rc[idx][1]];
      test_capo = testc[ord_rc[idx][2]];

      convex = false;
      if ((itest_abcp * det_abpo) > 0) return;
      if ((itest_abcp * det_bcpo) > 0) return;
      if ((itest_abcp * det_capo) > 0) return;
      convex = true;
   }
   else if (ninf == 3) {
      assert(true);
      // this should not happen
   }
}
}
