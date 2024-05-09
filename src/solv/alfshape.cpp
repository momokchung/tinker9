#include "ff/solv/alphamol.h"

namespace tinker {
AlfAtom::AlfAtom(int idx, double x, double y, double z, double r, double coefs, double coefv)
{
    this->index = idx;
    this->coord[0] = truncate_real(x,alfdigit);
    this->coord[1] = truncate_real(y,alfdigit);
    this->coord[2] = truncate_real(z,alfdigit);
    this->r = truncate_real(r,alfdigit);
    this->coefs = coefs;
    this->coefv = coefv;
}

AlfAtom::~AlfAtom() {}

double AlfAtom::truncate_real(double x, int ndigit)
{
    double x_out,y;
    double fact;

    int mantissa;
    int digit;

    mantissa = (int) x;
    y = x - mantissa;

    x_out = mantissa;
    fact = 1;
    for (int i = 0; i < ndigit; i++) {
        fact *= 10;
        digit = (int) std::round(y*10);
        y = 10*(y-digit/10.0);
        x_out += digit/fact;
    }

    return x_out;
}

Vertex::Vertex(double x, double y, double z, double r, double coefs, double coefv)
{
   double x1 = truncate_real(x, alfdigit);
   double y1 = truncate_real(y, alfdigit);
   double z1 = truncate_real(z, alfdigit);
   double r1 = truncate_real(r, alfdigit);
   this->coord[0] = x1;
   this->coord[1] = y1;
   this->coord[2] = z1;
   this->r = r1;
   this->coefs = coefs;
   this->coefv = coefv;

   std::bitset<8> b(std::string("00000000"));
   this->info = b;
   this->info[1] = 1;

   double t1 = (double) pow(10, alfdigit/2);
   double t2 = (double) pow(10, alfdigit);
   long long ival1= std::round(r1*t1);
   long long ival2 = -ival1*ival1;
   ival1 = std::round(x1*t1);
   ival2 += ival1*ival1;
   ival1 = std::round(y1*t1);
   ival2 += ival1*ival1;
   ival1 = std::round(z1*t1);
   ival2 += ival1*ival1;
   this->w = (double) ival2/t2;

   this->gamma = 0;
}

Vertex::~Vertex() {}

double Vertex::truncate_real(double x, int ndigit)
{
   double x_out,y;
   double fact;

   int mantissa;
   int digit;

   mantissa = (int) x;
   y = x - mantissa;

   x_out = mantissa;
   fact = 1;
   for (int i = 0; i < ndigit; i++) {
       fact *= 10;
       digit = (int) std::round(y*10);
       y = 10*(y-digit/10.0);
       x_out += digit/fact;
   }

   return x_out;
}

Tetrahedron::Tetrahedron() {
    for (int i = 0; i < 4; i++) {
        this->vertices[i] = -1;
        this->neighbors[i] = -1;
        this->nindex[i] = -1;
    }
    std::bitset<8> b(std::string("00000000"));
    this->info = b;
}
void Tetrahedron::init() {
    for (int i = 0; i < 4; i++) {
        this->vertices[i] = -1;
        this->neighbors[i] = -1;
        this->nindex[i] = -1;
    }
    std::bitset<8> b(std::string("00000000"));
    this->info = b;
}
Tetrahedron::~Tetrahedron() {}

Edge::Edge(int i, int j) {
   vertices[0] = i;
   vertices[1] = j;
   gamma = 0.0;
}
Edge::~Edge() {}

Face::Face(int i, int j, int k, int e1, int e2, int e3, double S) {
   vertices[0] = i;
   vertices[1] = j;
   vertices[2] = k;
   edges[0] = e1;
   edges[1] = e2;
   edges[2] = e3;
   gamma = S;
}
Face::~Face() {}
}
