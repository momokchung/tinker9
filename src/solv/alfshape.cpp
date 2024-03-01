#include "ff/solv/alphamol.h"

namespace tinker {
Vertex::Vertex(double x, double y, double z, double r, double coefs, double coefv, double coefm, double coefg)
{
   this->coord[0] = x;
   this->coord[1] = y;
   this->coord[2] = z;
   this->r = r;
   this->coefs = coefs;
   this->coefv = coefv;
   this->coefm = coefm;
   this->coefg = coefg;

   std::bitset<8> b(std::string("00000000"));
   this->info = b;
   this->info[1] = 1;

   this->w = x*x + y*y + z*z - r*r;

   this->gamma = 0;
}
Vertex::~Vertex() {}

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
   sigma = 0.0;
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
