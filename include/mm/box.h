#pragma once
#include "tool/rcman.h"
#include "tool/realn.h"

namespace tinker {
/// \ingroup box
enum class BoxShape
{
   UNBOUND,
   ORTHO,
   MONO,
   TRI,
   OCT
};
constexpr BoxShape UNBOUND_BOX = BoxShape::UNBOUND;
constexpr BoxShape ORTHO_BOX = BoxShape::ORTHO;
constexpr BoxShape MONO_BOX = BoxShape::MONO;
constexpr BoxShape TRI_BOX = BoxShape::TRI;
constexpr BoxShape OCT_BOX = BoxShape::OCT;

/**
 * \ingroup box
 * \brief Periodic boundary conditions (PBC).
 *
 * PBC in Tinker is defined by lengths of three axes and three angles:
 * `a-axis`, `b-axis`, `c-axis`, `alpha`, `beta`, and `gamma`.
 *
 * In cartesian coordinates:
 *    - `a-axis` is always aligned with the x axis, so the vector of `a-axis`
 *       is `(ax,ay,az) = (a,0,0)`;
 *    - `b-axis` does not have z component, so the vector of `b-axis` is
 *       `(bx,by,bz) = (bx,by,0)`;
 *    - `c-axis` is `(cx,cy,cz)`.
 *
 * Tinker has another 3x3 matrix `lvec`:
 *    - `lvec(:,1)` (or `lvec[0][]`): `(ax,bx,cx)`;
 *    - `lvec(:,2)` (or `lvec[1][]`): `(ay,by,cy)`;
 *    - `lvec(:,3)` (or `lvec[2][]`): `(az,bz,cz)`.
 *
 * Triclinic:
 *    - `lvec(:,1)` (or `lvec[0][]`): `(ax,bx,cx)`;
 *    - `lvec(:,2)` (or `lvec[1][]`): `( 0,by,cy)`;
 *    - `lvec(:,3)` (or `lvec[2][]`): `( 0, 0,cz)`.
 *
 * Monoclinic:
 *    - `lvec(:,1)` (or `lvec[0][]`): `(a,0,cx)`;
 *    - `lvec(:,2)` (or `lvec[1][]`): `(0,b, 0)`;
 *    - `lvec(:,3)` (or `lvec[2][]`): `(0,0,cz)`;
 *    - `alpha` and `gamma` are 90 degrees.
 *
 * Orthogonal:
 *    - `lvec(:,1)` (or `lvec[0][]`): `(a,0,0)`;
 *    - `lvec(:,2)` (or `lvec[1][]`): `(0,b,0)`;
 *    - `lvec(:,3)` (or `lvec[2][]`): `(0,0,c)`;
 *    - `alpha`, `beta`, and `gamma` are 90 degrees.
 *
 * Reciprocal lattice (`recip`), is the inverse of `lvec`:
 *    - Fortran: `recip(:,1)`, `recip(:,2)`, `recip(:,3)`;
 *    - C++: `recip[0][]`, `recip[1][]`, `recip[2][]`;
 *    - Fractional coordinates `fi (i=1,2,3) = recip(:,i) (xr,yr,zr)`;
 *    - Cartesian coordinates `wr (w=1,2,3 or x,y,z) = inv_recip(:,w) (f1,f2,f3)
         = lvec(:,w) (f1,f2,f3)`.
 */
struct Box
{
   BoxShape box_shape;
   real3 lvec1, lvec2, lvec3;
   real3 recipa, recipb, recipc;
};

void boxData(RcOp);
void boxExtent(double new_extent);
void boxSetDefault(const Box&);
void boxGetDefault(Box&);
void boxSetDefaultRecip();
void boxSetTinkerModule(const Box&);
void boxGetAxesAngles(const Box&, double& a, double& b, double& c, //
   double& alpha, double& beta, double& gamma);

/// \ingroup box
/// \brief Similar to Tinker `lattice` subroutine.
void boxLattice(Box& p, BoxShape sh, double a, double b, double c, //
   double alpha_deg, double beta_deg, double gamma_deg);

void boxCopyin();

/// \ingroup box
/// \brief Get the volume of the PBC box.
/// \note This function may calculate the volume on-the-fly instead of using
/// an internal variable to save the volume.
/// \note The volume is undefined for the unbound box.
real boxVolume();
}

#include "mod/box.h"