#pragma once
#include "ff/precision.h"
#include "tool/genunit.h"

namespace tinker {
enum class NblGK
{
   UNDEFINED,
   DOUBLE_LOOP,
   NEIGHBOR_LIST,
};

void nblistgkData(RcOp); ///< Sets up data on device.
void nblistgkRefresh();  ///< Updates the neighbor lists.

class DLoop;

typedef GenericUnit<DLoop, GenericUnitVersion::ENABLE_ON_DEVICE> DLoopUnit;

class DLoop
{
public:
   struct ScaleInfo
   {
      int (*js)[2];       ///< Atom pairs of atoms with exclusion rules (of length #ns).
      unsigned int* bit0; ///< Bits array, stored as 32-bit unsigned integers.
      int ns;             ///< Number of pairs of atoms with exclusion rules.

      void init();
      void set(int nns, int (*jjs)[2]);
   };

   static constexpr int BLOCK = 32;  ///< Number of atom per block.
                                     ///< Equal to #WARP_SIZE and \c sizeof(int).

   ScaleInfo si1; ///< #ScaleInfo object 1.
   ScaleInfo si2; ///< #ScaleInfo object 2.
   ScaleInfo si3; ///< #ScaleInfo object 3.
   ScaleInfo si4; ///< #ScaleInfo object 4.

   int n;         ///< Number of atoms.

   int* iakp;     ///< List of all block pairs. Length #nakp.
                  ///< The pair `(x,y)` was encoded via triangular number and stored as `tri(x)+y`.
   int nakp;      ///< Length of #iakp. (#nak+1)*#nak/2.

   int* iakpl;    ///< List of block pairs subject to exclusion rules. Length #nakpl.
                  ///< The pair `(x,y)` was encoded via triangular number and stored as `tri(x)+y`.
   int nakpl;     ///< Length of #iakpl.

   int* iakpa;    ///< List of block pairs not subject to exclusion rules. Length #nakpa.
                  ///< The pair `(x,y)` was encoded via triangular number and stored as `tri(x)+y`.
   int nakpa;     ///< Length of #iakpa.

   int* worker;   ///< Work array of length `max(2*#n,128)`.

   ~DLoop();

   static void dataAlloc(DLoopUnit& u, int n, int nstype, //
   int ns1 = 0, int (*js1)[2] = nullptr,                  //
   int ns2 = 0, int (*js2)[2] = nullptr,                  //
   int ns3 = 0, int (*js3)[2] = nullptr,                  //
   int ns4 = 0, int (*js4)[2] = nullptr);

   static void dataInit(DLoopUnit u);

private:
   int* iakpl_rev; ///< Reverse lookup array. Length #nakp. `iakpl_rev[iakpl[i]] = i`.
   int* akpf;      ///< Compressed bit flags of block-block pair. Length #nakpk.
   int nak;        ///< Number of blocks.
   int nakpk;      ///< Length of #akpf.
   int cap_nakpl;  ///< Capacity of iakpl. Initial value (32+8*nak).
   int cap_nakpa;  ///< Capacity of iakpa. Initial value (32+8*nak).
   int nstype;     ///< Number of #ScaleInfo objects in use.

   friend void dloopDataInit_cu(DLoopUnit);
};

TINKER_EXTERN DLoopUnit vdloop_unit;
TINKER_EXTERN DLoopUnit mdloop_unit;
TINKER_EXTERN DLoopUnit udloop_unit;
}
