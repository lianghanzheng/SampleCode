add_mlir_dialect_library(MLIRSysy
  SysyDialect.cpp
  SysyLower.cpp
  SysyOpt.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sysy

  DEPENDS
  MLIRSysyOpsIncGen
  MLIRSysyLowerIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
  MLIRInferTypeOpInterface
  MLIRPass
  MLIRArithDialect
  MLIRMemRefDialect
  MLIRAffineDialect
  MLIRFuncDialect
  MLIRTransforms
  MLIRFuncTransforms
  LLVMSupport
  MLIRAffineUtils
  MLIRAffineAnalysis
)