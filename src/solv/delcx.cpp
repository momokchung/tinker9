#include "ff/solv/alphamol.h"
#include "ff/solv/delcx.h"
#include "ff/solv/delsort.h"
#include "tool/error.h"
#include <cassert>
#include <cmath>

namespace tinker
{
// "init" initializes and sets up Delaunay triangulation

void Delcx::init(int natoms, AlfAtom* alfatoms, std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra)
{
   sos.init_sos_gmp();

   // initialize vertices and tetra
   vertices.clear();
   vertices.reserve(natoms+4);
   tetra.clear();
   tetra.reserve(10*natoms);

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
   double xi, yi, zi, ri, cs, cv, cm, cg;
   for (int i = 0; i < natoms; i++) {
      xi = alfatoms[i].coord[0];
      yi = alfatoms[i].coord[1];
      zi = alfatoms[i].coord[2];
      ri = alfatoms[i].r;
      cs = alfatoms[i].coefs;
      cv = alfatoms[i].coefv;
      Vertex vert(xi, yi, zi, ri, cs, cv);
      vert.info[0] = 1;
      vert.status = 1;
      vertices.push_back(vert);
   }

   // if natoms < 4, add "bogus" points
   if (natoms < 4) {
      int new_points = 4-natoms;
      double *bcoord = new double[3*new_points];
      double *brad   = new double[new_points];
      addBogus(natoms, alfatoms, bcoord, brad);
      for (int i = 0; i < new_points; i++) {
         xi = bcoord[3*i];
         yi = bcoord[3*i+1];
         zi = bcoord[3*i+2];
         ri = brad[i];
         cs = 1.;
         cv = 1;
         cm = 1;
         cg = 1;
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

// "regular3D" computes the weighted 3D Delaunay triangulation
template <bool use_sos>
void Delcx::regular3D(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra)
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
      locate_jw<use_sos>(vertices, tetra, ipoint, tetra_loc, iredundant);

      // if the point is redundant, move to next point
      if (iredundant) {
         vertices[ipoint].info[0] = 0;
         continue;
      }

      // otherwise, add point to tetrahedron: 1-4 flip
      int dummy;
      flip_1_4(tetra, ipoint, tetra_loc, dummy);

      // now scan link_facet list, and flip till list is empty
      flip<use_sos>(vertices, tetra);

      // At this stage, I should have a regular triangulation
      // of the i+4 points (i-th double points+4 "infinite" points)
      // Now add another point
   }

   // reorder the tetrahedra, such that vertices are in increasing order
   reorder_tetra(tetra);

   // I have the regular triangulation: I need to remove the
   // simplices including infinite points, and define the convex hull
   remove_inf(vertices, tetra);

   // peel off flat tetrahedra at the boundary of the DT
   // int nt = 0;
   // for (int i = 0; i < tetra.size(); i++) if (tetra[i].info[1]!=0) nt++;
   // if (nt > 1) {
   //     int flag;
   //     do {
   //         peel<use_sos>(vertices, tetra, flag);
   //     } while (flag != 0);
   // }
   int flag;
   do {
      peel<use_sos>(vertices, tetra, flag);
   } while (flag != 0);

   sos.clear_sos_gmp();
}

// "locate_jw" uses the "jump-and-walk" technique: first N active
// tetrahedra are randomly chosen. The "distances" between
// these tetrahedra and the point to be added are computed,
// and the tetrahedron closest to the point is chosen as
// a starting point. The program then "walks" from that tetrahedron
// to the point, till we find a tetrahedron that contains
// the point.
// It also checks if the point is redundant in the current
// tetrahedron. If it is, the search terminates.

template <bool use_sos>
void Delcx::locate_jw(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int ipoint, int& tetra_loc, bool& iredundant)
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

      inside_tetra<use_sos>(vertices, ipoint, a, b, c, d, iorient, test_in, test_red, idx);

      if (!test_in) itetra = tetra[itetra].neighbors[idx];

   } while (!test_in);

   tetra_loc = itetra;

   if (test_red) iredundant = true;
}

// flip restores the regularity of Delaunay triangulation
template <bool use_sos>
void Delcx::flip(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra)
{
   int ii,ij;
   int idxi,idxj,idxk,idxl;
   int p,o,a,b,c;
   int itetra,jtetra,idx_p,idx_o;
   int itest_abcp;
   int ierr,tetra_last;
   int ireflex,iflip,iorder;
   int ifind,tetra_ab,tetra_ac,tetra_bc;
   int idx_a,idx_b,idx_c;
   int ia,ib,ic;
   int facei[3],facej[3],edgei[2],edgej[2],edgek[2];
   int edge_val[3][2];
   int tetra_flip[3],list_flip[3];
   int vert_flip[5];
   std::pair<int,int> facet,index;
   bool convex,regular,test_abpo,test_bcpo,test_capo;
   bool test,test_abpc,test_bcpa,test_acpo,test_acpb;
   bool test_or[3][2];
   idx_a = 0;
   idx_b = 0;
   idx_c = 0;

   // go over all link facets
   while (!link_facet.empty()) {
      // First define the two tetrahedra that
      // contains the link facet as itetra and jtetra
      facet = link_facet.front();
      index = link_index.front();
      link_facet.pop();
      link_index.pop();

      itetra = facet.first;
      jtetra = facet.second;
      idx_p = index.first;
      idx_o = index.second;

      // if the link facet is on the convex hull, discard
      if (itetra==-1 || jtetra == -1) continue;

      // if these tetrahedra have already been discarded, discard this link facet
      if (tetra[itetra].info[1] == 0) {
         if (tetra[jtetra].info[1] == 0) {
            continue;
         }
         else {
            itetra = tetra[jtetra].neighbors[idx_o];
            idx_p = tetra[jtetra].nindex[idx_o];
         }
      }

      if (tetra[jtetra].info[1] == 0) {
         jtetra = tetra[itetra].neighbors[idx_p];
         idx_o = tetra[itetra].nindex[idx_p];
      }

      // define the vertices of the two tetrahedra
      a = tetra[itetra].vertices[0];
      b = tetra[itetra].vertices[1];
      c = tetra[itetra].vertices[2];
      p = tetra[itetra].vertices[3];

      o = tetra[jtetra].vertices[idx_o];

      itest_abcp = -1;
      if (tetra[itetra].info[0]==1) itest_abcp = 1;

      // check for local regularity (and convexity)
      regular_convex<use_sos>(vertices, a, b, c, p, o, itest_abcp, regular, convex, test_abpo, test_bcpo, test_capo);

      // if the link facet is locally regular, discard
      if (regular) continue;

      // define neighbors of the facet on itetra and jtetra
      define_facet(tetra, itetra, jtetra, idx_o, facei, facej);

      test_abpc = (itest_abcp != 1);

      // After discarding the trivial case, we now test if the tetrahedra
      // can be flipped.

      // At this stage, I know that the link facet is not locally
      // regular. I still don t know if it is "flippable"

      // I first check if {itetra} U {jtetra} is convex. If it is, I
      // perform a 2-3 flip (this is the convexity test performed
      // at the same time as the regularity test)

      if (convex) {
         vert_flip[0] = a;
         vert_flip[1] = b;
         vert_flip[2] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         flip_2_3(tetra, itetra, jtetra, vert_flip, facei, facej, test_abpo, test_bcpo, test_capo, ierr, tetra_last);
         continue;
      }

      // The union of the two tetrahedra is not convex...
      // I now check the edges of the triangle in the link facet, and
      // check if they are "reflexes" (see definition in Edelsbrunner and
      // Shah, Algorithmica (1996), 15:223-241)

      ireflex = 0;
      iflip = 0;

      // First check edge (ab):
      // - (ab) is reflex iff o and c lies on opposite sides of
      // the hyperplane defined by (abp). We therefore test the
      // orientation of (abpo) and (abpc): if they differ (ab)
      // is reflex
      // - if (ab) is reflex, we test if it is of degree 3.
      // (ab) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (abpo). The first two are itetra
      // and jtetra, so we only need to check if (abpo) exists.
      // since (abpo) contains p, (abp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      if (test_abpo != test_abpc) {
         ireflex++;
         find_tetra(tetra, itetra, 2, a, b, o, ifind, tetra_ab, idx_a, idx_b);

         if (ifind==1) {
            tetra_flip[iflip] = tetra_ab;
            list_flip[iflip] = 0;
            edge_val[iflip][0] = idx_a;
            edge_val[iflip][1] = idx_b;
            test_or[iflip][0] = test_bcpo;
            test_or[iflip][1] = !test_capo;
            iflip++;
         }
      }

      // Now check edge (ac):
      // - (ac) is reflex iff o and b lies on opposite sides of
      // the hyperplane defined by (acp). We therefore test the
      // orientation of (acpo) and (acpb): if they differ (ac)
      // is reflex
      // - if (ac) is reflex, we test if it is of degree 3.
      // (ac) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (acpo). The first two are itetra
      // and jtetra, so we only need to check if (acpo) exists.
      // since (acpo) contains p, (acp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      test_acpo = !test_capo;
      test_acpb = !test_abpc;

      if (test_acpo != test_acpb) {

         ireflex++;
         find_tetra(tetra, itetra, 1, a, c, o, ifind, tetra_ac, idx_a, idx_c);

         if (ifind==1)  {
            tetra_flip[iflip] = tetra_ac;
            list_flip[iflip] = 1;
            edge_val[iflip][0] = idx_a;
            edge_val[iflip][1] = idx_c;
            test_or[iflip][0] = !test_bcpo;
            test_or[iflip][1] = test_abpo;
            iflip++;
         }
      }

      // Now check edge (bc):
      // - (bc) is reflex iff o and a lies on opposite sides of
      // the hyperplane defined by (bcp). We therefore test the
      // orientation of (bcpo) and (bcpa): if they differ (bc)
      // is reflex
      // - if (bc) is reflex, we test if it is of degree 3.
      // (bc) is of degree 3 if it is shared by 3 tetrahedra,
      // namely (abcp), (abco) and (bcpo). The first two are itetra
      // and jtetra, so we only need to check if (bcpo) exists.
      // since (bcpo) contains p, (bcp) should then be a link facet
      // of p, so we test all tetrahedra that define link facets

      test_bcpa = test_abpc;
      if (test_bcpo != test_bcpa) {

         ireflex++;
         find_tetra(tetra, itetra, 0, b, c, o, ifind, tetra_bc, idx_b, idx_c);

         if (ifind==1)  {
            tetra_flip[iflip] = tetra_bc;
            list_flip[iflip] = 2;
            edge_val[iflip][0] = idx_b;
            edge_val[iflip][1] = idx_c;
            test_or[iflip][0] = test_capo;
            test_or[iflip][1] = !test_abpo;
            iflip++;
         }
      }

      if (ireflex != iflip) continue;

      if (iflip==1) {
         // only one edge is "flippable": we do a 3-2 flip
         iorder = list_flip[0];
         ia = table32[iorder][0];
         ib = table32[iorder][1];
         ic = table32[iorder][2];
         vert_flip[ia] = a;
         vert_flip[ib] = b;
         vert_flip[ic] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         ia = table32_2[iorder][0];
         ib = table32_2[iorder][1];
         edgei[0] = ia;
         edgei[1] = ib;
         edgej[0] = facej[ia];
         edgej[1] = facej[ib];
         edgek[0] = edge_val[0][0];
         edgek[1] = edge_val[0][1];
         flip_3_2(tetra, itetra, jtetra, tetra_flip[0], vert_flip, edgei, edgej, edgek, test_or[0][0], test_or[0][1], ierr, tetra_last);
      }
      else if (iflip==2) {
         // In this case, one point is redundant: the point common to
         // the two edges that can be flipped. We then perform a 4-1 flip
         iorder = list_flip[0] + list_flip[1] -1;
         vert_flip[table41[iorder][0]] = a;
         vert_flip[table41[iorder][1]] = b;
         vert_flip[table41[iorder][2]] = c;
         vert_flip[3] = p;
         vert_flip[4] = o;
         ii = table41_2[iorder][0];
         ij = table41_2[iorder][1];
         idxi = iorder;
         idxj = facej[iorder];
         idxk = edge_val[0][ii];
         idxl = edge_val[1][ij];

         if (iorder==0) {
            test = test_bcpo;
         }
         else if (iorder==1) {
            test = !test_capo;
         }
         else {
            test = test_abpo;
         }
         flip_4_1(vertices, tetra, itetra, jtetra, tetra_flip[0], tetra_flip[1], vert_flip, idxi, idxj, idxk, idxl, test, ierr, tetra_last);
      }
      else {
         TINKER_THROW("DELCX::FLIP  --  Three Edges Flippable");
      }
   }

   for (int i = 0; i < kill.size(); i++) {
      free.push(kill[i]);
   }
   kill.clear();
}

// "inside_tetra" tests if a point p is inside a tetrahedron
// defined by four points (a,b,c,d) with orientation "iorient".
// If p is inside the tetrahedron, check if redundant

template <bool use_sos>
void Delcx::inside_tetra(std::vector<Vertex>& vertices, int p, int a, int b, int c, int d, int iorient, bool& is_in, bool& redundant, int& ifail)
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
      test_pijk = (std::abs(det_pijk) > eps);
      if (test_pijk && det_pijk > 0) {
         ifail = 3;
         return;
      }

      det_pjil = l_p[0] * Sij_1 - l_p[1] * Sij_2 + l_p[2] * Sij_3;
      det_pjil = det_pjil * iorient;
      test_pjil = (std::abs(det_pjil) > eps);
      if (test_pjil && det_pjil > 0) {
         ifail = 2;
         return;
      }

      det_pkjl = j_p[0] * Skl_1 - j_p[1] * Skl_2 + j_p[2] * Skl_3;
      det_pkjl = det_pkjl * iorient;
      test_pkjl = (std::abs(det_pkjl) > eps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = 0;
         return;
      }

      det_pikl = -i_p[0] * Skl_1 + i_p[1] * Skl_2 - i_p[2] * Skl_3;
      det_pikl = det_pikl * iorient;
      test_pikl = (std::abs(det_pikl) > eps);
      if (test_pikl && det_pikl > 0) {
         ifail = 1;
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough.
      // In this case, we have to rank the indices.

      if (!test_pijk) {
         mysort.valsort4(p, a, b, c, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
            ifail = 3;
            return;
         }
      }

      if (!test_pjil) {
         mysort.valsort4(p, b, a, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
            ifail = 2;
            return;
         }
      }

      if (!test_pkjl) {
         mysort.valsort4(p, c, b, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
            ifail = 0;
            return;
         }
      }

      if (!test_pikl) {
         mysort.valsort4(p, a, c, d, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord_a, coord_b, coord_c, coord_d, val);
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
      if (std::abs(det_pijkl) < eps) {
         mysort.valsort5(p, a, b, c, d, ia, ib, ic, id, ie, jswap);
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
         sos.sos_minor5<use_sos>(coord_a, ra, coord_b, rb, coord_c, rc, coord_d, rd, coord_e, re, val);
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
      test_pijk = (std::abs(det_pijk) > eps);
      if (test_pijk && det_pijk > 0) {
         ifail = idx;
         return;
      }

      det_pjil = -detij[ic1] * sign * iorient;
      test_pjil = (std::abs(det_pjil) > eps);
      if (test_pjil && det_pjil > 0) {
         ifail = order1[idx][2];
         return;
      }

      det_pkjl = k_p[ic1] * j_p[2] - k_p[2] * j_p[ic1];
      det_pkjl = sign * det_pkjl * iorient;
      test_pkjl = (std::abs(det_pkjl) > eps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = order1[idx][0];
         return;
      }

      det_pikl = i_p[ic1] * k_p[2] - i_p[2] * k_p[ic1];
      det_pikl = sign * det_pikl * iorient;
      test_pikl = (std::abs(det_pikl) > eps);
      if (test_pikl && det_pikl > 0) {
         ifail = order1[idx][1];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         mysort.valsort4(p, i, j, k, ia, ib, ic, id, jswap);
         for (int i = 0; i < 3; i++) {
            coord_a[i] = vertices[ia].coord[i];
            coord_b[i] = vertices[ib].coord[i];
            coord_c[i] = vertices[ic].coord[i];
            coord_d[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord_a, coord_b, coord_c, coord_d, val);
         val = val * jswap * iorient;
         if (val == 1) {
            ifail = idx;
            return;
         }
      }

      if (!test_pjil) {
         mysort.valsort3(p, j, i, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order1[idx][2];
            return;
         }
      }

      if (!test_pkjl) {
         mysort.valsort3(p, k, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
         val = val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order1[idx][0];
            return;
         }
      }

      if (!test_pikl) {
         mysort.valsort3(p, i, k, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
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
      test_pijk = (std::abs(det_pijk) > eps);
      if (test_pijk && det_pijk > 0) {
         ifail = order3[idx][1];
         return;
      }

      // det_pjil is now det3_pji with l as infinite point
      det_pjil = i_p[2] * j_p[ic1_l] - i_p[ic1_l] * j_p[2];
      det_pjil = det_pjil * sign_l * iorient;
      test_pjil = (std::abs(det_pjil) > eps);
      if (test_pjil && det_pjil > 0) {
         ifail = order3[idx][0];
         return;
      }

      // det_pkjl is now -det2_pj (k,l infinite)
      det_pkjl = j_p[ic1] * sign * iorient;
      test_pkjl = (std::abs(det_pkjl) > eps);
      if (test_pkjl && det_pkjl > 0) {
         ifail = order2[idx][0];
         return;
      }

      // det_pikl is now det2_pi (k,l infinite)
      det_pikl = -i_p[ic1] * sign * iorient;
      test_pikl = (std::abs(det_pikl) > eps);
      if (test_pikl && det_pikl > 0) {
         ifail = order2[idx][1];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         mysort.valsort3(p, i, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1_k];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1_k];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1_k];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
         val = val * sign_k * jswap * iorient;
         if (val == 1) {
            ifail = order3[idx][1];
            return;
         }
      }

      if (!test_pjil) {
         mysort.valsort3(p, j, i, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic1_l];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic1_l];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic1_l];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
         val = val * sign_l * jswap * iorient;
         if (val == 1) {
            ifail = order3[idx][0];
            return;
         }
      }

      if (!test_pkjl) {
         mysort.valsort2(p, j, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         sos.sos_minor2<use_sos>(xa, xb, val);
         val = -val * sign * jswap * iorient;
         if (val == 1) {
            ifail = order2[idx][0];
            return;
         }
      }

      if (!test_pikl) {
         mysort.valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         sos.sos_minor2<use_sos>(xa, xb, val);
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
      if (std::abs(det_pijkl) < eps) {
         mysort.valsort3(p, i, j, ia, ib, ic, jswap);
         int temp = 2;
         xa = vertices[ia].coord[ic5];
         ya = vertices[ia].coord[temp];
         xb = vertices[ib].coord[ic5];
         yb = vertices[ib].coord[temp];
         xc = vertices[ic].coord[ic5];
         yc = vertices[ic].coord[temp];
         sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
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
      test_pijk = (std::abs(det_pijk) > eps);
      if (test_pijk && det_pijk > 0) {
         ifail = order1[idx][2];
         return;
      }

      // det_pjil is now det2_pi (missing j,l)
      det_pjil = -i_p[inf4_2[l][j]] * iorient * sign4_2[l][j];
      test_pjil = (std::abs(det_pjil) > eps);
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
      test_pikl = (std::abs(det_pikl) > eps);
      if (test_pikl && det_pikl > 0) {
         ifail = order1[idx][0];
         return;
      }

      // At this stage, either all four determinants are positive,
      // or one of the determinant is not precise enough

      if (!test_pijk) {
         mysort.valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[k][j]];
         xb = vertices[ib].coord[inf4_2[k][j]];
         sos.sos_minor2<use_sos>(xa, xb, val);
         val = -val * sign4_2[k][j] * iorient * jswap;
         if (val == 1) {
            ifail = order1[idx][2];
            return;
         }
      }

      if (!test_pjil) {
         mysort.valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[l][j]];
         xb = vertices[ib].coord[inf4_2[l][j]];
         sos.sos_minor2<use_sos>(xa, xb, val);
         val = val * sign4_2[l][j] * iorient * jswap;
         if (val == 1) {
            ifail = order1[idx][1];
            return;
         }
      }

      if (!test_pikl) {
         mysort.valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[inf4_2[l][k]];
         xb = vertices[ib].coord[inf4_2[l][k]];
         sos.sos_minor2<use_sos>(xa, xb, val);
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
      if (std::abs(det_pijkl) < eps) {
         mysort.valsort2(p, i, ia, ib, jswap);
         xa = vertices[ia].coord[ic1];
         xb = vertices[ib].coord[ic1];
         sos.sos_minor2<use_sos>(xa, xb, val);
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

// "regular_convex" checks for local regularity and convexity

template <bool use_sos>
void Delcx::regular_convex(std::vector<Vertex>& vertices, int a, int b, int c, int p, int o, int itest_abcp,
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

      // The three determinants det(a,b,p,o), det(b,c,p,o), and det(c,a,p,o) are "double" 4x4 determinants.
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
      if (std::abs(det_abcpo) < eps) {
         mysort.valsort5(a, b, c, p, o, ia, ib, ic, id, ie, iswap);
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
         sos.sos_minor5<use_sos>(coord1, ra, coord2, rb, coord3, rc, coord4, rd, coord5, re, val);
         det_abcpo = val * iswap;
      }

      if ((det_abcpo * itest_abcp) < 0) {
         regular = true;
         return;
      }
      regular = false;

      // if not regular, we test for convexity
      if (std::abs(det_abpo) < eps) {
         mysort.valsort4(a, b, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord1, coord2, coord3, coord4, val);
         det_abpo = val * iswap;
      }
      if (std::abs(det_bcpo) < eps) {
         mysort.valsort4(b, c, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord1, coord2, coord3, coord4, val);
         det_bcpo = val * iswap;
      }
      if (std::abs(det_capo) < eps) {
         mysort.valsort4(c, a, p, o, ia, ib, ic, id, iswap);
         for (int i = 0; i < 3; i++) {
            coord1[i] = vertices[ia].coord[i];
            coord2[i] = vertices[ib].coord[i];
            coord3[i] = vertices[ic].coord[i];
            coord4[i] = vertices[id].coord[i];
         }
         sos.sos_minor4<use_sos>(coord1, coord2, coord3, coord4, val);
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
         if (std::abs(det_abpo) < eps) {
            int temp = 2;
            mysort.valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
            det_abpo = -val * iswap;
         }
         det_abpo = det_abpo * sign1;
         det_capo = -Mjo[icol1];
         if (std::abs(det_capo) < eps) {
            int temp = 2;
            mysort.valsort3(j, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
            det_capo = val * iswap;
         }
         det_capo = det_capo * sign1;
         det_bcpo = -i_p[0] * Mjo[1] + i_p[1] * Mjo[0] - i_p[2] * Mjo[2];
         if (std::abs(det_bcpo) < eps) {
            mysort.valsort4(i, j, p, o, ia, ib, ic, id, iswap);
            for (int i = 0; i < 3; i++) {
               coord1[i] = vertices[ia].coord[i];
               coord2[i] = vertices[ib].coord[i];
               coord3[i] = vertices[ic].coord[i];
               coord4[i] = vertices[id].coord[i];
            }
            sos.sos_minor4<use_sos>(coord1, coord2, coord3, coord4, val);
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
         if (std::abs(det_abpo) < eps) {
            mysort.valsort2(i, p, ia, ib, iswap);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            sos.sos_minor2<use_sos>(xa, xb, val);
            det_abpo = -val * iswap * sign1;
         }
         det_capo = j_p[icol1] * sign1;
         if (std::abs(det_capo) < eps) {
            mysort.valsort2(j, p, ia, ib, iswap);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            sos.sos_minor2<use_sos>(xa, xb, val);
            det_capo = val * iswap * sign1;
         }
         det_bcpo = i_p[icol2] * j_p[2] - i_p[2] * j_p[icol2];
         if (std::abs(det_bcpo) < eps) {
            int temp = 2;
            mysort.valsort3(i, j, p, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol2];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol2];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol2];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
            det_bcpo = val * iswap;
         }
         det_bcpo = det_bcpo * sign2;
         det_abcpo = i_p[icol5] * j_p[2] - i_p[2] * j_p[icol5];
         if (std::abs(det_abcpo) < eps) {
            int temp = 2;
            mysort.valsort3(i, j, p, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol5];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol5];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol5];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
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
         if (std::abs(det_abpo) < eps) {
            int temp = 2;
            mysort.valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol1];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol1];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol1];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
            det_abpo = val * iswap * sign1;
         }
         det_capo = Mio[icol2] * sign2;
         if (std::abs(det_capo) < eps) {
            int temp = 2;
            mysort.valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol2];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol2];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol2];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
            det_capo = -val * iswap * sign2;
         }
         det_bcpo = -o_p[icol4] * sign4;
         if (std::abs(det_bcpo) < eps) {
            mysort.valsort2(p, o, ia, ib, iswap);
            xa = vertices[ia].coord[icol4];
            xb = vertices[ib].coord[icol4];
            sos.sos_minor2<use_sos>(xa, xb, val);
            det_bcpo = val * sign4 * iswap;
         }
         det_abcpo = -Mio[icol5] * sign5;
         if (std::abs(det_abcpo) < eps) {
            int temp = 2;
            mysort.valsort3(i, p, o, ia, ib, ic, iswap);
            xa = vertices[ia].coord[icol5];
            ya = vertices[ia].coord[temp];
            xb = vertices[ib].coord[icol5];
            yb = vertices[ib].coord[temp];
            xc = vertices[ic].coord[icol5];
            yc = vertices[ic].coord[temp];
            sos.sos_minor3<use_sos>(xa, ya, xb, yb, xc, yc, val);
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
         if (std::abs(det_abpo) < eps) {
            mysort.valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[icol1];
            xb = vertices[ib].coord[icol1];
            sos.sos_minor2<use_sos>(xa, xb, val);
            det_abpo = val * iswap2 * sign1;
         }
         det_capo = -i_p[icol2] * sign2;
         if (std::abs(det_capo) < eps) {
            mysort.valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[icol2];
            xb = vertices[ib].coord[icol2];
            sos.sos_minor2<use_sos>(xa, xb, val);
            det_capo = -val * iswap2 * sign2;
         }
         det_bcpo = sign4_3[icol4] * iswap;
         det_abcpo = sign5_3[icol4] * iswap * i_p[inf5_3[icol4]];
         if (std::abs(det_abcpo) < eps) {
            mysort.valsort2(i, p, ia, ib, iswap2);
            xa = vertices[ia].coord[inf5_3[icol4]];
            xb = vertices[ib].coord[inf5_3[icol4]];
            sos.sos_minor2<use_sos>(xa, xb, val);
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

// sign associated with the missing infinite point
inline void Delcx::missinf_sign(int i, int j, int k, int& l, int& sign)
{
   int a, b, c, d;
   l = 6 - i - j - k;
   a = i;
   b = j;
   c = k;
   sign = 1;
   if (a > b) {
      d = a;
      a = b;
      b = d;
      sign = -sign;
   }
   if (a > c) {
      d = a;
      a = c;
      c = d;
      sign = -sign;
   }
   if (b > c) {
      sign = -sign;
   }
}

// flip_1_4 implements a 4->1 flip in 3D for regular triangulation
void Delcx::flip_1_4(std::vector<Tetrahedron>& tetra, int ipoint, int itetra, int& tetra_last)
{
   int k, newtetra;
   int jtetra;
   int fact,idx;

   std::bitset<8> ikeep;
   char ival[4];

   int vertex[4], neighbour[4];
   int position[4];
   int idx_list[4][3]={{0,0,0},{0,1,1},{1,1,2},{2,2,2}};

   // store information about "old" tetrahedron
   ikeep = tetra[itetra].info;

   for (int i = 0; i < 4; i++) {
      vertex[i] = tetra[itetra].vertices[i];
      neighbour[i] = tetra[itetra].neighbors[i];
      ival[i] = tetra[itetra].nindex[i];
   }

   fact = -1;
   if (tetra[itetra].info[0] == 1) fact = 1;

   // The four new tetrahedra are going to be stored in
   // any free space in the tetrahedron list and
   // at the end of the list of known tetrahedra

   k = 0;
   while (!free.empty() && k < 4) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 4; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[3];

   // itetra is set to 0, and added to the "kill" list
   tetra[itetra].info[1] = 0;
   kill.push_back(itetra);

   // The tetrahedron is defined as (ijkl); four new tetrahedra are
   // created: jklp, iklp, ijlp, and ijkp, where p is the new
   // point to be included

   // For each new tetrahedron, define all four neighbours:
   // For each neighbour, I store the index of the vertex opposite to
   // the common face in array tetra_nindex

   // tetrahedron jklp : neighbours are iklp, ijlp, ijkp and neighbour of (ijkl) on face jkl
   // tetrahedron iklp : neighbours are jklp, ijlp, ijkp and neighbour of (ijkl) on face ikl
   // tetrahedron ijlp : neighbours are jklp, iklp, ijkp and neighbour of (ijkl) on face ijl
   // tetrahedron ijkp : neighbours are jklp, iklp, ijlp and neighbour of (ijkl) on face ijk

   for (int i = 0; i < 4; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 4; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = vertex[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k] = idx_list[i][k];
         k++;
      }

      jtetra = neighbour[i];
      idx = ival[i];
      tetra[newtetra].vertices[3] = ipoint;
      tetra[newtetra].neighbors[3] = jtetra;
      tetra[newtetra].nindex[3] = ival[i];
      tetra[newtetra].info[2+i] = ikeep[2+i];

      if (jtetra !=-1 && idx != -1) {
         tetra[jtetra].neighbors[idx] = newtetra;
         tetra[jtetra].nindex[idx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      fact = -fact;
      tetra[newtetra].info[0] = 0;
      if (fact==1) tetra[newtetra].info[0] = 1;

   }

   // Now add all fours faces of itetra in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the
   // intersection of two tetrahedra

   // link_facet: jkl tetrahedra: jklp and neighbour of (ijkl) on jkl
   // link_facet: ikl tetrahedra: iklp and neighbour of (ijkl) on ikl
   // link_facet: ijl tetrahedra: ijlp and neighbour of (ijkl) on ijl
   // link_facet: ijk tetrahedra: ijkp and neighbour of (ijkl) on ijk

   for (int i = 0; i < 4; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      idx = tetra[newtetra].nindex[3];
      link_index.push(std::make_pair(3,idx));
   }
}

// flip_4_1 implements a 4->1 flip in 3D for regular triangulation
void Delcx::flip_4_1(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra,
   int ltetra, int* ivertices, int idp, int jdp, int kdp, int ldp, bool test_acpo, int& ierr, int& tetra_last)
{
   int p,o,a,b,c;
   int ishare,jshare,kshare,lshare;
   int idx,jdx,kdx,ldx;
   int test1,newtetra;
   std::bitset<8> ikeep, jkeep, kkeep, lkeep;

   ierr = 0;
   test1 = 1;
   if (test_acpo) test1 = -1;

   // if itetra, jtetra, ktetra, ltetra are inactive, cannot flip
   if (tetra[itetra].info[1]==0 || tetra[jtetra].info[1]==0 ||
      tetra[ktetra].info[1]==0 || tetra[ltetra].info[1]==0) {
      ierr = 1;
      return;
   }

   // store "old" info
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;
   kkeep = tetra[ktetra].info;
   lkeep = tetra[ltetra].info;

   // Define
   //     - ishare: index of tetrahedron sharing the face opposite to b in itetra
   //     - idx:    index of the vertex of ishare opposite to the face of ishare shared with itetra
   //     - jshare: index of tetrahedron sharing the face opposite to b in jtetra
   //     - jdx:    index of the vertex of jshare opposite to the face of jshare shared with jtetra
   //     - kshare: index of tetrahedron sharing the face opposite to b in ktetra
   //     - kdx:    index of the vertex of kshare opposite to the face of kshare shared with ktetra
   //     - lshare: index of tetrahedron sharing the face opposite to b in ltetra
   //     - ldx:    index of the vertex of lshare opposite to the face of lshare shared with ltetra

   ishare = tetra[itetra].neighbors[idp];
   jshare = tetra[jtetra].neighbors[jdp];
   kshare = tetra[ktetra].neighbors[kdp];
   lshare = tetra[ltetra].neighbors[ldp];

   idx = tetra[itetra].nindex[idp];
   jdx = tetra[jtetra].nindex[jdp];
   kdx = tetra[ktetra].nindex[kdp];
   ldx = tetra[ltetra].nindex[ldp];

   if (free.size() > 0) {
      newtetra = free.top();
      free.pop();
   }
   else {
      newtetra = tetra.size();
      Tetrahedron t;
      t.init();
      tetra.push_back(t);
   }

   tetra_last = newtetra;

   // itetra, jtetra, ktetra and ltetra become "available";
   // they are added to the "kill" zone

   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;
   tetra[ktetra].info[1] = 0;
   tetra[ltetra].info[1] = 0;
   kill.push_back(itetra);
   kill.push_back(jtetra);
   kill.push_back(ktetra);
   kill.push_back(ltetra);

   // I need :
   //     - the vertex b that is shared by all 4 tetrahedra
   //     - the vertices a, c, p and o
   //     - for each tetrahedron, find neighbour attached to the face
   //       oposite to b; this information is stored in *share,
   //       where * can be i, j, k or l

   a = ivertices[0];
   b = ivertices[1];
   c = ivertices[2];
   p = ivertices[3];
   o = ivertices[4];

   // For bookkeeping reason, p is set to be the last vertex of the new tetrahedron

   // Now I define the new tetrahedron: (acop)

   // tetrahedron acop : neighbor of (bcop) on face cpo, neighbor of (abop)
   //                    on face apo, neighbor of (abcp) on face acp
   //                    and neighbor of (abco) on face aco

   vertices[b].info[0] = 0;

   tetra[newtetra].vertices[0] = a;
   tetra[newtetra].neighbors[0] = lshare;
   tetra[newtetra].nindex[0] = ldx;
   tetra[newtetra].info[2] = lkeep[2+ldp];
   if (lshare != -1 && ldx != -1) {
      tetra[lshare].neighbors[ldx] = newtetra;
      tetra[lshare].nindex[ldx] = 0;
   }

   tetra[newtetra].vertices[1] = c;
   tetra[newtetra].neighbors[1] = kshare;
   tetra[newtetra].nindex[1] = kdx;
   tetra[newtetra].info[3] = kkeep[2+kdp];
   if (kshare != -1 && kdx != -1) {
      tetra[kshare].neighbors[kdx] = newtetra;
      tetra[kshare].nindex[kdx] = 1;
   }

   tetra[newtetra].vertices[2] = o;
   tetra[newtetra].neighbors[2] = ishare;
   tetra[newtetra].nindex[2] = idx;
   tetra[newtetra].info[4] = kkeep[2+idp];
   if (ishare != -1 && idx != -1) {
      tetra[ishare].neighbors[idx] = newtetra;
      tetra[ishare].nindex[idx] = 2;
   }

   tetra[newtetra].vertices[3] = p;
   tetra[newtetra].neighbors[3] = jshare;
   tetra[newtetra].nindex[3] = jdx;
   tetra[newtetra].info[5] = kkeep[2+jdp];
   if (jshare != -1 && jdx != -1) {
      tetra[jshare].neighbors[jdx] = newtetra;
      tetra[jshare].nindex[jdx] = 3;
   }

   tetra[newtetra].info[1] = 1;
   if (test1==1) {
      tetra[newtetra].info[0] = 1;
   }
   else {
      tetra[newtetra].info[0] = 0;
   }

   // Now add one link facet:
   // link_facet: aco tetrahedra: acop and neighbour of (abco) on aco

   link_facet.push(std::make_pair(newtetra, jshare));
   link_index.push(std::make_pair(3, jdx));
}

// flip_2_3 implements a 2->3 flip in 3D for regular triangulation
void Delcx::flip_2_3(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int* vertices,
   int* facei, int* facej,bool test_abpo, bool test_bcpo, bool test_capo, int& ierr, int& tetra_last)
{
   int k,p,o;
   int it,jt,idx,jdx;
   int newtetra;
   std::bitset<8> ikeep,jkeep;
   int face[3];
   int itetra_touch[3],jtetra_touch[3];
   char jtetra_idx[3],itetra_idx[3];
   int idx_list[3][2]={{0,0},{0,1},{1,1}};
   int tests[3],position[3];

   ierr = 0;

   // f itetra or jtetra are inactive, cannot flip
   if (tetra[itetra].info[1] == 0 || tetra[jtetra].info[1] == 0) {
      ierr = 1;
      return;
   }

   // Define
   // - itetra_touch: the three tetrahedra that touches itetra on the
   //         faces opposite to the 3 vertices a,b,c
   // - itetra_idx: for the three tetrahedra defined by itetra_touch,
   //         index of the vertex opposite to the face common with itetra
   // - jtetra_touch: the three tetrahedra that touches jtetra on the
   //         faces opposite to the 3 vertices a,b,c
   // - jtetra_idx: for the three tetrahedra defined by jtetra_touch,
   //         index of the vertex opposite to the face common with jtetra

   for (int i = 0; i < 3; i++) {
      itetra_touch[i] = tetra[itetra].neighbors[facei[i]];
      jtetra_touch[i] = tetra[jtetra].neighbors[facej[i]];
      itetra_idx[i] = tetra[itetra].nindex[facei[i]];
      jtetra_idx[i] = tetra[jtetra].nindex[facej[i]];
   }

   // first three vertices define face that is removed
   face[0] = vertices[0];
   face[1] = vertices[1];
   face[2] = vertices[2];
   p = vertices[3];
   o = vertices[4];

   // The three new tetrahedra are stored in :
   // - any free space in the tetrahedron list,
   // - at the end of the list of known tetrahedra if needed

   k = 0;
   while (!free.empty() && k < 3) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 3; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[2];

   // set itetra and jtetra to 0, and add them to kill list
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;

   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;

   kill.push_back(itetra);
   kill.push_back(jtetra);

   // Define the three new tetrahedra: (bcop), (acop) and (abop) as well as their neighbours

   // tetrahedron bcop : neighbours are acop, abop, neighbour of (abcp)
   //             on face bcp, and neighbour of (abco) on face bco
   // tetrahedron acop : neighbours are bcop, abop, neighbour of (abcp)
   //             on face acp, and neighbour of (abco) on face aco
   // tetrahedron abop : neighbours are bcop, acop, neighbour of (abcp)
   //             on face abp, and neighbour of (abco) on face abo

   tests[0] = 1;
   if (test_bcpo) tests[0] = -1;
   tests[1] = -1;
   if (test_capo) tests[1] = 1;
   tests[2] = 1;
   if (test_abpo) tests[2] = -1;

   for (int i = 0; i < 3; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 3; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = face[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k]=idx_list[i][k];
         k++;
      }

      tetra[newtetra].vertices[2] = o;
      it = itetra_touch[i];
      idx = itetra_idx[i];
      tetra[newtetra].neighbors[2] = it;
      tetra[newtetra].nindex[2] = idx;
      tetra[newtetra].info[5] = ikeep[2+facei[i]];
      if (it !=-1 && idx !=-1) {
         tetra[it].neighbors[idx]=newtetra;
         tetra[it].nindex[idx] = 2;
      }

      tetra[newtetra].vertices[3] = p;
      jt = jtetra_touch[i];
      jdx = jtetra_idx[i];
      tetra[newtetra].neighbors[3] = jt;
      tetra[newtetra].nindex[3] = jdx;
      tetra[newtetra].info[6] = jkeep[2+facej[i]];
      if (jt !=-1 && jdx !=-1) {
         tetra[jt].neighbors[jdx]=newtetra;
         tetra[jt].nindex[jdx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      if (tests[i]==1) {
         tetra[newtetra].info[0] = 1;
      }
      else {
         tetra[newtetra].info[0] = 0;
      }
   }

   // Now add all three faces of jtetra containing o in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the intersection of two tetrahedra

   // link_facet: bco tetrahedra: bcop and neighbour of (abco) on bco
   // link_facet: aco tetrahedra: acop and neighbour of (abco) on aco
   // link_facet: abo tetrahedra: abop and neighbour of (abco) on abo

   for (int i = 0; i < 3; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      idx = tetra[newtetra].nindex[3];
      link_index.push(std::make_pair(3,idx));
   }
}

// flip_3_2 implements a 3->2 flip in 3D for regular triangulation
void Delcx::flip_3_2(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int ktetra, int* vertices,
   int* edgei, int* edgej, int* edgek, bool test_bcpo, bool test_acpo, int& ierr, int& tetra_last)
{
   int k,p,o,c;
   int it,jt,kt,idx,jdx,kdx;
   int newtetra;
   std::bitset<8> ikeep,jkeep,kkeep;
   int edge[2],tests[2];
   int itetra_touch[2],jtetra_touch[2],ktetra_touch[2];
   int position[2];
   char itetra_idx[2],jtetra_idx[2],ktetra_idx[2];

   tests[0] = 1;
   if (test_bcpo) tests[0] = -1;
   tests[1] = 1;
   if (test_acpo) tests[1] = -1;
   ierr = 0;

   // if itetra, jtetra or ktetra are inactive, cannot flip
   if (tetra[itetra].info[1]==0 || tetra[jtetra].info[1]==0 || tetra[ktetra].info[1]==0)
   {
      ierr = 1;
      return;
   }

   // store old info
   ikeep = tetra[itetra].info;
   jkeep = tetra[jtetra].info;
   kkeep = tetra[ktetra].info;

   // Define
   //     - itetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in itetra, respectively
   //     - itetra_idx:   for the two tetrahedra defined by itetra_touch,
   //                     index position of the vertex opposite to the face common with itetra
   //     - jtetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in jtetra, respectively
   //     - jtetra_idx:   for the two tetrahedra defined by jtetra_touch,
   //                     index position of the vertex opposite to the face common with jtetra
   //     - ktetra_touch: indices of the two tetrahedra that share the
   //                     faces opposite to a and b in ktetra, respectively
   //     - ktetra_idx:   for the two tetrahedra defined by ktetra_touch,
   //                     index position of the vertex opposite to the face common with ktetra

   for (int i = 0; i < 2; i++) {
      itetra_touch[i] = tetra[itetra].neighbors[edgei[i]];
      jtetra_touch[i] = tetra[jtetra].neighbors[edgej[i]];
      ktetra_touch[i] = tetra[ktetra].neighbors[edgek[i]];
      itetra_idx[i]   = tetra[itetra].nindex[edgei[i]];
      jtetra_idx[i]   = tetra[jtetra].nindex[edgej[i]];
      ktetra_idx[i]   = tetra[ktetra].nindex[edgek[i]];
   }

   edge[0] = vertices[0];
   edge[1] = vertices[1];
   c       = vertices[2];
   p       = vertices[3];
   o       = vertices[4];


   // the two new tetrahedra are going to be stored "free" space, or at the end of the list
   k = 0;
   while (!free.empty() && k < 2) {
      position[k] = free.top();
      free.pop();
      k++;
   }

   for (int l = k; l < 2; l++) {
      Tetrahedron t;
      t.init();
      position[l] = tetra.size();
      tetra.push_back(t);
   }
   tetra_last = position[1];

   // itetra, jtetra and ktetra becomes "available"; they are added to the "kill" list
   tetra[itetra].info[1] = 0;
   tetra[jtetra].info[1] = 0;
   tetra[ktetra].info[1] = 0;

   kill.push_back(itetra);
   kill.push_back(jtetra);
   kill.push_back(ktetra);

   // I need :
   //     - the two vertices that define their common edge (ab)
   //         these vertices are stored in the array edge
   //     - the vertices c, p and o that form the new triangle
   //     - for each vertex in the edge (ab), define the opposing
   //         faces in the three tetrahedra itetra, jtetra and ktetra, and
   //         the tetrahedron that share these faces with itetra, jtetra and
   //         ktetra, respectively. This information is stored
   //         in three arrays, itetra_touch, jtetra_touch and ktetra_touch

   // These information are given by the calling program

   // For bookkeeping reasons, I always set p to be the last vertex
   // of the new tetrahedra

   // Now I define the two new tetrahedra: (bcop) and (acop)
   // as well as their neighbours

   // tetrahedron bcop : neighbours are acop, neighbour of (abop)
   //             on face bpo, neighbour of (abcp) on face bcp
   //             and neighbour of (abco) on face (bco)
   // tetrahedron acop : neighbours are bcop, neighbour of (abop)
   //             on face apo, neighbour of (abcp) on face acp
   //             and neighbour of (abco) on face (aco)

   for (int i = 0; i < 2; i++) {
      newtetra = position[i];
      k = 0;
      for (int j = 0; j < 2; j++) {
         if (j==i) continue;
         tetra[newtetra].vertices[k] = edge[j];
         tetra[newtetra].neighbors[k] = position[j];
         tetra[newtetra].nindex[k] = 0;
         k++;
      }

      tetra[newtetra].vertices[1] = c;
      kt = ktetra_touch[i];
      kdx = ktetra_idx[i];
      tetra[newtetra].neighbors[1] = kt;
      tetra[newtetra].nindex[1] = kdx;
      tetra[newtetra].info[3] = kkeep[2+edgek[i]];
      if (kdx != -1 && kt != -1) {
         tetra[kt].neighbors[kdx] = newtetra;
         tetra[kt].nindex[kdx] = 1;
      }

      tetra[newtetra].vertices[2] = o;
      it = itetra_touch[i];
      idx = itetra_idx[i];
      tetra[newtetra].neighbors[2] = it;
      tetra[newtetra].nindex[2] = idx;
      tetra[newtetra].info[4] = ikeep[2+edgek[i]];
      if (idx != -1 && it != -1) {
         tetra[it].neighbors[idx] = newtetra;
         tetra[it].nindex[idx] = 2;
      }

      tetra[newtetra].vertices[3] = p;
      jt = jtetra_touch[i];
      jdx = jtetra_idx[i];
      tetra[newtetra].neighbors[3] = jt;
      tetra[newtetra].nindex[3] = jdx;
      tetra[newtetra].info[5] = jkeep[2+edgej[i]];
      if (jdx != -1 && jt != -1) {
         tetra[jt].neighbors[jdx] = newtetra;
         tetra[jt].nindex[jdx] = 3;
      }

      tetra[newtetra].info[1] = 1;

      if (tests[i] == 1) {
         tetra[newtetra].info[0] = 1;
      }
      else {
         tetra[newtetra].info[0] = 0;
      }
   }

   // Now add the two faces of ktetra containing (co) in the link_facet queue.
   // Each link_facet (a triangle) is implicitly defined as the
   // intersection of two tetrahedra

   // link_facet: bco tetrahedra: bcop and neighbour of (abco) on bco
   // link_facet: aco tetrahedra: acop and neighbour of (abco) on aco

   for (int i = 0; i < 2; i++) {
      newtetra = position[i];
      link_facet.push(std::make_pair(newtetra, tetra[newtetra].neighbors[3]));
      link_index.push(std::make_pair(3, tetra[newtetra].nindex[3]));
   }
}

// define_facet definess the triangle of intersection of two tetrahedra
inline void Delcx::define_facet(std::vector<Tetrahedron>& tetra, int itetra, int jtetra, int idx_o, int* facei, int* facej)
{
   int ia, ib, ie, ig;
   int k;

   // I need to :
   //     - find the three vertices that define their common face
   //         these vertices are stored in the array triangle
   //     - find the vertices p and o

   // To define the common face of the two tetrahedra itetra and jtetra,
   // I look at the neighbours of itetra : one of them is jtetra!
   // This also provides p. The same procedure is repeated for jtetra,
   // to get o

   for (int i = 0; i < 3; i++) facei[i] = i;

   ia = tetra[itetra].vertices[0];
   for (int i = 0; i < 3; i++) {
      k = other[idx_o][i];
      ie = tetra[jtetra].vertices[k];
      if (ia == ie) {
         facej[0] = k;
         break;
      }
   }

   ib = tetra[itetra].vertices[1];
   ie = other2[idx_o][facej[0]][0];
   ig = other2[idx_o][facej[0]][1];
   if (ib == tetra[jtetra].vertices[ie]) {
      facej[1] = ie;
      facej[2] = ig;
   }
   else {
      facej[1] = ig;
      facej[2] = ie;
   }
}

// find_tetra tests if four given points form an
// existing tetrahedron in the current Delaunay
inline void Delcx::find_tetra(std::vector<Tetrahedron>& tetra, int itetra, int idx_c,
   int a, int b, int o,int& ifind, int& tetra_loc, int& idx_a, int& idx_b)
{
   // We are testing if tetrahedron (abpo) exists. If it exists, it is
   // a neighbour of abcp, on the face opposite to vertex c.
   // We test that tetrahedron and see if it contains o

   int ot, otx, otest;

   ot = tetra[itetra].neighbors[idx_c];
   otx = tetra[itetra].nindex[idx_c];
   otest = tetra[ot].vertices[otx];

   if (otest == o) {
      ifind = 1;
      tetra_loc = ot;

      // We found the tetrahedron, let us define the position
      // of a and b in this tetrahedron
      for (int i = 0; i < 4; i++) {
         if (tetra[tetra_loc].vertices[i] == a) {
            idx_a = i;
         }
         else if (tetra[tetra_loc].vertices[i] == b) {
            idx_b = i;
         }
      }
   }
   else {
      ifind = 0;
   }
}

// "reorde_tetra" reorders the vertices of a list of tetrahedron,
// such that now the indices are in increasing order

inline void Delcx::reorder_tetra(std::vector<Tetrahedron>& tetra)
{
   int ntetra = tetra.size();
   int vert[4],idx[4],neighbor[4];
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
      mysort.sort4_sign(vert, idx, nswap, n);

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

// "remove_inf" sets to 0 the status of tetrahedron
// that contains infinite points

inline void Delcx::remove_inf(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra)
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
         if (a < 4) mark_zero(tetra, i, 0);
         if (b < 4) mark_zero(tetra, i, 1);
         if (c < 4) mark_zero(tetra, i, 2);
         if (d < 4) mark_zero(tetra, i, 3);
      }
   }

   for (int i = 0; i < 4; i++) {
      vertices[i].info[0] = 0;
   }
}

inline void Delcx::mark_zero(std::vector<Tetrahedron>& tetra, int itetra, int ivertex)
{
   int jtetra, jvertex;

   jtetra = tetra[itetra].neighbors[ivertex];

   if(jtetra != -1) {
      jvertex = tetra[itetra].nindex[ivertex];
      tetra[jtetra].neighbors[jvertex] = -1;
   }
}

// "peel" removes the flat tetrahedra at the boundary of the DT

template <bool use_sos>
void Delcx::peel(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra, int& flag)
{
   int ia, ib, ic, id;
   int k, l;
   int res;
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

      double vol = tetra_vol(coorda, coordb, coordc, coordd);

      if (std::abs(vol) < eps) {
         sos.minor4<use_sos>(coorda, coordb, coordc, coordd, res);
         if (res == 0) {
            flag++;
            tetra[i].info[2] = 1;
         }
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
inline double Delcx::tetra_vol(double* a, double* b, double* c, double* d)
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

// "findedge" finds the index of the edge
// given two vertices of a tetrahedron

inline int Delcx::findedge(Tetrahedron t, int i1, int j1)
{
   int ipair;

   if (i1==t.vertices[0]) {
      if (j1==t.vertices[1]) ipair = 5;
      else if (j1==t.vertices[2]) ipair = 4;
      else ipair = 3;
   }
   else if (i1==t.vertices[1]) {
      if (j1==t.vertices[2]) ipair = 2;
      else ipair = 1;
   }
   else {
      ipair = 0;
   }
   return ipair;
}

// "addBogus" adds bogus points so that nvertices >= 4

void Delcx::addBogus(int npoints, AlfAtom* alfatoms, double* bcoord, double* brad)
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
      Rmax = alfatoms[0].r;
      bcoord[0] = alfatoms[0].coord[0] + 3*Rmax;
      bcoord[3*1+1] = alfatoms[0].coord[1] + 3*Rmax;
      bcoord[3*2+2] = alfatoms[0].coord[2] + 3*Rmax;
      for (int i = 0; i < np; i++) {
         brad[i] = Rmax/20;
      }
   }
   else if (npoints==2) {
      Rmax = std::fmax(alfatoms[0].r, alfatoms[1].r);
      c1x = alfatoms[0].coord[0];
      c1y = alfatoms[0].coord[1];
      c1z = alfatoms[0].coord[2];
      c2x = alfatoms[1].coord[0];
      c2y = alfatoms[1].coord[1];
      c2z = alfatoms[1].coord[2];
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
      Rmax = std::fmax(std::fmax(alfatoms[0].r, alfatoms[1].r), alfatoms[2].r);
      c1x = alfatoms[0].coord[0];
      c1y = alfatoms[0].coord[1];
      c1z = alfatoms[0].coord[2];
      c2x = alfatoms[1].coord[0];
      c2y = alfatoms[1].coord[1];
      c2z = alfatoms[1].coord[2];
      c3x = alfatoms[2].coord[0];
      c3y = alfatoms[2].coord[1];
      c3z = alfatoms[2].coord[2];
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
      d = std::fmax(d1,std::fmax(d2,d3));
      bcoord[0] = cx + (2*d+3*Rmax)*w1x;
      bcoord[1] = cy + (2*d+3*Rmax)*w1y;
      bcoord[2] = cz + (2*d+3*Rmax)*w1z;
      brad[0] = Rmax/20;
   }
}

// explicit instatiation
template void Delcx::regular3D<true>(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);
template void Delcx::regular3D<false>(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra);
}
