# 编写一个类 `mlir-opt` 工具

## 一个空壳

只需要编写这样一段代码，我们就能够得到 `mlir-opt` 的壳子，拥有其所有的内置dialect和pass的信息。

```cpp
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/IR/DialectRegistry.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registery;
  mlir::registerAllDialects(registery);
  mlir::registerAllPasses();
  
  return mlir::asMainReturnCode(
      return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "Sysy Dialect Driver", registry)
  ));
}
```

## 使用ODS添加自定义的dialect

### ODS Framework

ODS框架基于tabelgen实现，提供了一种定义dialect、op、type等的便利的定义方法，不用写一些很长的模板文件。

在使用ODS时，我个人的习惯是进入 `include` 目录 -> 编写 `.td` 文件和 `.h` 文件 -> 更新cmake文件 -> 进入 `lib` 目录 -> 编写 `.cpp` 文件 -> 更新cmake文件 -> 编译。

### 定义 `sysy` dialect

所有的dialect都是 `Dialect` 的实例：

```tablegen
def Sysy_Dialect : Dialect {
  let name = "sysy";
  let summary = "Extended features of SysY2022";
  let description = [{
      A high-level dialect describes the numeric computation
      opration defined in the extended SysY2022 specification.}];
  let cppNamespace = "::mlir::sysy";
}
```

Tablegen文件通过工具 `mlir-tblgen` 能够生成的 `.h.inc`/`.cpp.inc` 文件。

```sh
mlir-tblgen -gen-dialect-decls SysyDialect.td -I/opt/llvm/include/
mlir_tblgen -gen-dialect-defs SysyDialect.td -I/opt/llvm/include/
```

中两个生成的文件通常被命名为 `SysyDialect.h.inc` 和 `SysyDialect.cpp.inc`。

在CMakeFile中，需要使用LLVM项目提供的命令来创建目标：

```cmake
set(LLVM_TARGET_DEFINITIONS SysyDialect.td)
mlir_tablegen(SysyDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SysyDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(MLIRSysyDialectGen)
```

当然，这些生成的文件需要引用在对应的 `.h`/`.cpp` 文件中才能发挥作用

```cpp
// SysyDialect.h
#include "mlir/IR/DialectImplementation.h"

#include "sysy/SysyDialect.h.inc"
```

```cpp
// SysyDialect.cpp
#include "sysy/SysyDialect.h"

#include "sysy/SysyDialect.cpp.inc"

namespace mlir {
namespace sysy {

void SysyDialect::initialize() {

}

} // namespace sysy
} // namespace mlir
```

### 在cmake中添加MLIR库的定义

在MLIR项目中，一个dialect库通常以如下形式定义：

```cmake
add_mlir_dialect_library(MLIRSysy
  SysyDialect.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/sysy

  DEPENDS
  MLIRSysyDialectGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRSupport
)
```

在链接驱动时，就可以把 `MLIRSysy` 作为我们的dialect的库加进去

### 注册进上下文

拥有了这些定义后就可以向 `sysy-opt` 这个空壳子里加上我们的 `sysy` dialect了：

```cpp
registry.insert<::mlir::sysy::SysyDialect>();
```

# 添加 `matmul` 操作

```tablegen
def MemType : MemRefOf<[AnyInteger, F32]>;

def Sysy_MatmulOp : Op<Sysy_Dialect, "matmul"> {
  let summary = "sysy 2D tensor multiplication";
  let description = [{
      Matmul operation will be converted into affine loops then be optimized.
  }];
  
  let arguments = (ins MemType:$dst, MemType:$lhs, MemType:$rhs);
  let results = (outs );
  let assemblyFormat = "$dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)";
}
```

和定义dialect一样，所有的操作也是对 `Op` 的实例化。
除了描述性的信息之外，我们为了让 `sysy.matmul` 能够支持浮点类型和整型，额外定义了 `MemType` 类型。
可用的基本类型在 `mlir/IR/CommonTypeConstraints.td` 中能够找到，以及很多形似 `MemType` 的定义。

通过填写 `assemblyFormat` 字段能够自定义操作的格式。
ODS系统中预留了一些关键字，如 `attr-dict`、 `type` 等。

## 编写 `.h` 和更新cmake

和 `SysyDialect.h` 一样，`SysyOps.h` 中也要引入tablegen生成的 `.h.inc` 文件。
不过为了使 `.h.inc` 的文件内容有效，需要额外加上宏：

```cpp
#define GET_OP_CLASSES
#include "sysy/SysyOps.h.inc"
```

接下来更新 `CMakeLists.txt`。
除了dialect以外，MLIR允许用户自定义op，type，trait...，再按照原来的方式编写cmake会变得很长。
于是MLIR提供了一个简化的命令：

```cmake
add_mlir_dialect(SysyOps sysy)
add_mlir_doc(SysyOps -gen-doc -dialect sysy)
```

它等效于：

```cmake
set(LLVM_TARGET_DEFINITIONS SysyOps.td)
mlir_tablegen(SysyOps.h.inc -gen-op-decls)
mlir_tablegen(SysyOps.cpp.inc -gen-op-defs)
mlir_tablegen(SysyOpsDialect.h.inc -gen-dialect-decls)
mlir_tablegen(SysyOpsDialect.cpp.inc -gen-dialect-defs)
mlir_tablegen(SysyOpsTypes.h.inc -gen-typedef-decls -typedefs-dialect=poly)
mlir_tablegen(SysyOpsTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=poly)
add_public_tablegen_target(MLIRSysyOpsIncGen)
```

## 更新 `.cpp`

方便起见，我们没有创建 `SysyOps.cpp`，而是在 `SysyDialect.cpp` 中更新。
主要注意两个地方的宏定义是不一样的。

```cpp
#define GET_OP_CLASSES
#include "sysy/SysyOps.cpp.inc"

void SysyDialect::initialize() {
  addOperations<
#     define GET_OP_LIST
#     include "sysy/SysyOps.cpp.inc"
  >();
}
```

# 进行降级

## 编写 `.td` 文件

仍然是一些描述性信息：

```tablegen
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
```

## 编写 `.h` 和更新cmake

同样需要注意的事情是要引入不同的宏：

```cpp
#define GEN_PASS_DECL
#include "sysy/SysyLower.h.inc"

#define GEN_PASS_REGISTRATION
#include "sysy/SysyLower.h.inc"
```

由于 `add_mlir_dialect` 和 `add_mlir_doc` 不会自动生成pass相关的文件，所以需要自己手动添加：

```cmake
set(LLVM_TARGET_DEFINITIONS SysyLower.td)
mlir_tablegen(SysyLower.h.inc -gen-pass-decls -name sysy)
add_public_tablegen_target(MLIRSysyLowerIncGen)
```

## 更新 `.cpp`

这会需要单独编写 `SysyLower.cpp`。
MLIR的降级包括两个部分，一个是Op的转换，一个是类型的转换。
两部分代码独立编写

### TypeConverter

```cpp
class SysyLowerTypeConverter : public TypeConverter {/*...*/};
```

这个类定义了类型如何转换。
不过由于在sysy dialect中没有引入自定义的类型，所以只需要直接返回原本的类型即可。

### Opconverter

```cpp
struct ConvertMatmul : public OpConversionPattern<MatmulOp> {
  // ...
    LogicalResult matchAndRewrite(/*...*/) const override {/*...*/}
  // ...
};
```

这个结构体定义如何转换Op。

### 组合起来

```cpp
struct SysyLower : impl::SysyLowerBase<SysyLower> {/*...*/};
```

这里继承的结构体是由tablegen文件生成的，作用类似于将前两个数据对象粘起来的胶水。
记录了降级过程中用到了哪些MLIR自定义的dialect。

### 注册

在 `driver.cpp` 中要把降级的pass注册了才能使用。

```cpp
mlir::sysy::registerSysyLowerPass();
```

# 使用pass manager组合选项

在降级过程中会调用大量pass进行降级，通过使用pass manager可以将它们合并为一个选项。

```cpp
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

int main() {
  // ...
  mlir::PassPipelineRegistration<>(
      "standard-to-llvm", "Lower the underlying dialect of sysy to LLVM",
      standardToLLVMPipelineBuilder);
  // ...
}
```