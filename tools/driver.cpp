#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"

#include "sysy/SysyDialect.h"
#include "sysy/SysyLower.h"
#include "sysy/SysyOpt.h"

void sysyToLLVMPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sysy::createSysyLower());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  // cleanup
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void standardToLLVMPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createConvertSCFToCFPass());
  pm.addPass(mlir::createConvertControlFlowToLLVMPass());
  pm.addPass(mlir::createArithToLLVMConversionPass());
  pm.addPass(mlir::createConvertFuncToLLVMPass());
  pm.addPass(mlir::memref::createExpandStridedMetadataPass());
  pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  // cleanup
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createSCCPPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createSymbolDCEPass());
}

void sysyToParallelLLVMPipelineBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sysy::createSysyLower());
}

void sysyTensorLoweringBuilder(mlir::OpPassManager &pm) {
  pm.addPass(mlir::sysy::createTensorLoweringPass());
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  mlir::registerAllPasses();
  //mlir::PassRegistration<mlir::sysy::AffineTilePass>();

  registry.insert<::mlir::sysy::SysyDialect>();
  mlir::sysy::registerSysyLowerPass();
  mlir::PassPipelineRegistration<>(
      "sysy-to-llvm", "Pass collection loweres sysy dialect to LLVM",
      sysyToLLVMPipelineBuilder);
  mlir::PassPipelineRegistration<>(
      "standard-to-llvm", "Lower the underlying dialect of sysy to LLVM",
      standardToLLVMPipelineBuilder);
  mlir::PassPipelineRegistration<>(
      "sysy-to-llvm-parallel", 
      "Pass collection loweres Sysy dialect to parallel LLVM through openmp",
      sysyToParallelLLVMPipelineBuilder);
  mlir::PassPipelineRegistration<>(
      "sysy-tensor-to-memref",
      "Partial lowering form tensor to memref",
      sysyTensorLoweringBuilder);


  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Sysy Dialect Driver", registry)
  );
}