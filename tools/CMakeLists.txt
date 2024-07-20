get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

set(LIBS 
  ${dialect_libs}
  ${conversion_libs}
  MLIRCAPIRegisterEverything
  MLIRPass
  MLIRDialect
  MLIRTransforms
  MLIRAnalysis
  MLIRIR
  MLIROptLib
  MLIRSysy
  MLIRFuncToLLVM
  MLIROpenMPToLLVMIRTranslation
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRMemRefToLLVM
  MLIROpenMPToLLVM
)

add_llvm_executable(sysy-opt
  driver.cpp
)

llvm_update_compile_flags(sysy-opt)

target_link_libraries(sysy-opt PRIVATE ${LIBS})