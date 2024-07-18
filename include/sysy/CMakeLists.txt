add_mlir_dialect(SysyOps sysy)
add_mlir_doc(SysyOps -gen-doc -dialect sysy)

set(LLVM_TARGET_DEFINITIONS SysyLower.td)
mlir_tablegen(SysyLower.h.inc -gen-pass-decls -name sysy)
add_public_tablegen_target(MLIRSysyLowerIncGen)