#ifndef SYSY_SYSYOPS_TD_
#define SYSY_SYSYOPS_TD_

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "SysyDialect.td"

def AttrType : AnyAttrOf<[FloatElementsAttr<32>, AnyIntElementsAttr]>;
def MemType : MemRefOf<[AnyInteger, F32]>;
def TensorType : RankedTensorOf<[AnyInteger, F32]>;
def CompositeType : AnyTypeOf<[MemType, TensorType]>;

def Sysy_ConstantOp : Op<Sysy_Dialect, "constant"> {
  let summary = "constant";
  let description = [{
      A constant literal in Sysy Dialect.
  }];

  let arguments = (ins AttrType:$value);
  let results = (outs TensorType);
}

def Sysy_MatmulOp : Op<Sysy_Dialect, "matmul"> {
  let summary = "sysy 2D tensor multiplication";
  let description = [{
      Matmul operation will be converted into affine loops then be optimized.
  }];
  
  let arguments = (ins CompositeType:$dst, CompositeType:$lhs, CompositeType:$rhs);
  let results = (outs );
  let assemblyFormat = "$dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)";
}

def Sysy_AddOp : Op<Sysy_Dialect, "add"> {
  let summary = "sysy tensor addition";
  let description = [{
      Addtion for sysy tensor in various dimensions.
  }];

  let arguments = (ins MemType:$dst, MemType:$lhs, MemType:$rhs);
  let assemblyFormat = "$dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)";
}

def Sysy_SubOp : Op<Sysy_Dialect, "sub"> {
  let summary = "sysy tensor substraction";
  let description = [{
      Substraction for sysy tensor in various dimensions.
  }];

  let arguments = (ins MemType:$dst, MemType:$lhs, MemType:$rhs);
  let assemblyFormat = "$dst `,` $lhs `,` $rhs attr-dict `:` type($dst) `,` type($lhs) `,` type($rhs)";
}

#endif // SYSY_SYSYOPS_TD_