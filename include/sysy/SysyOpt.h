#ifndef SYSY_SYSYOPT_H_
#define SYSY_SYSYOPT_H_

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "mlir/Pass/Pass.h"
#include <vector>

namespace mlir {
namespace sysy {

using llvm::SmallVector;
using llvm::StringRef;
using mlir::affine::AffineForOp;

class AffineTilePass
    : public PassWrapper<AffineTilePass,
                         OperationPass<mlir::func::FuncOp>> {
private:
  void runOnOperation() override;
  StringRef getArgument() const final { return "affine-tiling"; }
  StringRef getDescription() const final {
    return "Tile the affine loops lowered from `sysy` dialect";
  }
}; // class AffineTilePass


} // namespace sysy
} // namespace mlir

#endif // SYSY_SYSYOPT_H_