#ifndef SYSY_SYSYLOWER_H_
#define SYSY_SYSYLOWER_H_

include "mlir/Pass/PassBase.td"

def SysyLower : Pass<"sysy-lower"> {
  let summary = "Convert 'sysy' dialect to MLIR upstream dialects.";
  let description = [{
      This pass lowers the `sysy` dialect to affine, memref, arith, etc.
  }];
  let dependentDialects = [
      "mlir::sysy::SysyDialect",
      "mlir::arith::ArithDialect",
      "mlir::memref::MemRefDialect",
      "mlir::affine::AffineDialect"
  ];
}

#endif // SYSY_SYSYLOWER_H_