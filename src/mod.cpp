#define TINKER_EXTERN_DEFINITION_FILE 1

#include "ff/modamoeba.h"
#include "ff/atom.h"
#include "ff/box.h"
#include "ff/echarge.h"
#include "ff/echglj.h"
#include "ff/elec.h"
#include "ff/energy.h"
#include "ff/energybuffer.h"
#include "ff/evalence.h"
#include "ff/evdw.h"
#include "ff/solv/solute.h"
#include "ff/hippo/edisp.h"
#include "ff/hippo/erepel.h"
#include "ff/modhippo.h"
#include "ff/molecule.h"
#include "ff/nblist.h"
#include "ff/pme.h"
#include "ff/spatial.h"

#include "md/lflpiston.h"
#include "md/misc.h"
#include "md/osrw.h"
#include "md/pq.h"
#include "md/rattle.h"

#include "tool/accasync.h"
#include "tool/cudalib.h"
#include "tool/gpucard.h"
#include "tool/platform.h"
#include "tool/rcman.h"
