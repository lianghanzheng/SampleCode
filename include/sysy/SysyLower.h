#ifndef SYSY_SYSYLOWER_H_
#define SYSY_SYSYLOWER_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include <memory>

namespace mlir {

class Pass;

namespace sysy {

#define GEN_PASS_DECL
#include "sysy/SysyLower.h.inc"

#define GEN_PASS_REGISTRATION
#include "sysy/SysyLower.h.inc"

std::unique_ptr<Pass> createTensorLoweringPass();

} // namesapce sysy
} // namespace mlir

#endif // SYSY_SYSYLOWER_H_