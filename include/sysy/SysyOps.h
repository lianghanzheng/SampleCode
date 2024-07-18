#ifndef SYSY_SYSYOPS_H_
#define SYSY_SYSYOPS_H_

#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"

#include "sysy/SysyDialect.h"

#define GET_OP_CLASSES
#include "sysy/SysyOps.h.inc"

#endif // SYSY_SYSYOPS_H_