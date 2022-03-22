#pragma once
#include "glob/cflux.h"
#include "glob/ctrpot.h"
#include "glob/mplpot.h"
#include "glob/mpole.h"
#include "tool/rcman.h"

namespace tinker {
void cflux_data(RcOp);
void zero_pot();

void alterchg();
void alterchg_acc();

void dcflux(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, virial_buffer vir);
void dcflux_acc(int vers, grad_prec* gx, grad_prec* gy, grad_prec* gz, virial_buffer vir);
}