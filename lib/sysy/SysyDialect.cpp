#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "sysy/SysyDialect.h"
#include "sysy/SysyOps.h"

#include "sysy/SysyOpsDialect.cpp.inc"
#define GET_OP_CLASSES
#include "sysy/SysyOps.cpp.inc"

namespace mlir {
namespace sysy {

void SysyDialect::initialize() {
  addOperations<
#     define GET_OP_LIST
#     include "sysy/SysyOps.cpp.inc"
  >();
}

} // namespace sysy
} // namespace mlir