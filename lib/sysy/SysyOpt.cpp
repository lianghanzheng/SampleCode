#include "sysy/SysyOpt.h"

namespace mlir {
namespace sysy {

using namespace mlir::affine;  
using llvm::SmallVector;
using llvm::ArrayRef;

void AffineTilePass::runOnOperation() {
  getOperation().walk([&](AffineForOp op) {
    SmallVector<unsigned, 6> tileSizes {4, 8, 16};
    SmallVector<AffineForOp> loopNest;
    getPerfectlyNestedLoops(loopNest, op);
    llvm::errs() << loopNest.size() << "\n";
    
    if (failed(tilePerfectlyNested(loopNest, {4, 8, 16}))) {
      signalPassFailure();
    }
  });
}
  
} // namespace sysy
} // namespace mlir