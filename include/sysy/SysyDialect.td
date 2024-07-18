#ifndef SYSY_SYSYDIALECT_TD_
#define SYSY_SYSYDIALECT_TD_

include "mlir/IR/DialectBase.td"
include "mlir/IR/OpBase.td"

def Sysy_Dialect : Dialect {
  let name = "sysy";
  let summary = "Extended features of SysY2022";
  let description = [{
      A high-level dialect describes the numeric computation
      opration defined in the extended SysY2022 specification.}];
  let cppNamespace = "::mlir::sysy";
}

#endif // SYSY_SYSYDIALECT_TD_