#pragma once
#include "ff/solv/alphamol.h"

namespace tinker
{
class AlphaVol {
public:
    template <bool compder>
    void alphavol(std::vector<Vertex>& vertices, std::vector<Tetrahedron>& tetra,
        std::vector<Edge>& edges, std::vector<Face>& faces, double* ballwsurf, double* ballwvol,
        double* dsurfx, double* dsurfy, double* dsurfz, double* dvolx, double* dvoly, double* dvolz);

private:
    double eps = 1e-14;

    inline double dist2(std::vector<Vertex>& vertices, int n1, int n2);

    inline void twosph(double ra, double ra2, double rb, double rb2,
    double rab, double rab2, double& surfa, double& surfb,
    double& vola, double& volb, double& r, double& phi);

    template <bool compder>
    inline void twosphder(double ra, double ra2, double rb, double rb2, double rab, double rab2,
    double& surfa, double& surfb, double& vola, double& volb, double& r, double& phi,
    double& dsurfa, double& dsurfb, double& dvola, double& dvolb, double& dr, double& dphi);

    template <bool compder>
    inline void threesphder(double ra, double rb,double rc, double ra2,
    double rb2, double rc2, double rab, double rac, double rbc,
    double rab2, double rac2, double rbc2, double *angle,
    double& surfa, double& surfb, double& surfc, double& vola, double& volb, double& volc,
    double* dsurfa, double* dsurfb, double* dsurfc, double* dvola, double* dvolb, double* dvolc);
        
    inline double plane_dist(double ra2, double rb2, double rab2);

    template <bool compder>
    inline void tetdihedder(double r12sq, double r13sq, double r14sq,
    double r23sq, double r24sq, double r34sq, double* angle,
    double* cosine, double* sine, double deriv[6][6]);

    template <bool compder>
    inline void tetdihedder3(double r12sq, double r13sq, double r14sq,
    double r23sq, double r24sq, double r34sq, double* angle,
    double* cosine, double* sine, double deriv[6][3]);

    template <bool compder>
    inline void tet3dihedcos(double r12sq, double r13sq, double r14sq,
    double r23sq, double r24sq,double r34sq, double* cosine, double deriv[3][3]);

    template <bool compder>
    inline void tetvorder(double ra2,double rb2,double rc2,double rd2,
    double rab, double rac, double rad, double rbc, double rbd,
    double rcd, double rab2, double rac2, double rad2,double rbc2,
    double rbd2, double rcd2, double* cos_ang, double* sin_ang,
    double deriv[6][6], double& vola, double& volb, double& volc, 
    double& vold, double* dvola, double* dvolb, double* dvolc, double* dvold);
};
}
