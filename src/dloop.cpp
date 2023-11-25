#include "ff/dloop.h"
#include "ff/atom.h"
#include "ff/potent.h"
#include "ff/spatial.h"
#include <tinker/detail/limits.hh>

namespace tinker {
N2::~N2()
{
   darray::deallocate(iakp);
}
}

namespace tinker {
void dloopData(RcOp op)
{
   if (!use(Potent::SOLV) or limits::use_mlist) return;

   if (op & RcOp::DEALLOC) {
      N2Unit::clear();
      mn2_unit.close();
   }

   if (op & RcOp::ALLOC) {
      assert(N2Unit::size() == 0);
      n2Alloc(mn2_unit);
   }

   if (op & RcOp::INIT) {
      n2DataInit(mn2_unit);
   }
}
}

namespace tinker {
TINKER_FVOID2(acc0, cu1, n2DataInit, N2Unit&);
void n2DataInit(N2Unit& n2u)
{
   TINKER_FCALL2(acc0, cu1, n2DataInit, n2u);
}

void n2Alloc(N2Unit& n2u)
{
   n2u = N2Unit::open();
   auto& st = *n2u;
   st.nak = (n + Spatial::BLOCK - 1) / Spatial::BLOCK;
   st.nakp = (st.nak + 1) * st.nak / 2;
   darray::allocate(st.nakp, &st.iakp);
}
}
