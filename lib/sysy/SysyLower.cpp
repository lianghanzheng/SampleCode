#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h" // for llvm::errs.
#include "llvm/ADT/ArrayRef.h"
#include <cstdint>
#include <memory>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/ValueRange.h>
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Support/TypeID.h"

#include "sysy/SysyLower.h"
#include "sysy/SysyDialect.h"
#include "sysy/SysyOps.h"

namespace mlir {
namespace sysy {

using llvm::cast;
using llvm::dyn_cast;
using llvm::SmallVector;
using llvm::ArrayRef;

#define GEN_PASS_DEF_SYSYLOWER
#include "sysy/SysyLower.h.inc"

class SysyLowerTypeConverter : public TypeConverter {
public:
  SysyLowerTypeConverter(MLIRContext *ctx) {
    addConversion([](Type type) { return type; });
  }
}; // class SysyLowerTypeConverter


//===----------------------------------------------------------------------===//
// Op convertors for SysyTensorToMemRef
//===----------------------------------------------------------------------===//

static MemRefType convertTensorToMemRef(RankedTensorType type) {
  return MemRefType::get(type.getShape(), type.getElementType());
}

static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
  auto alloc = rewriter.create<memref::AllocOp>(loc, type);

  auto *parentBlock = alloc->getBlock();
  alloc->moveBefore(&parentBlock->front());

  auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
  dealloc->moveBefore(&parentBlock->back());
  return alloc;
}

struct ConstantLowering : public OpRewritePattern<sysy::ConstantOp> {

using OpRewritePattern<sysy::ConstantOp>::OpRewritePattern;

LogicalResult matchAndRewrite(
    sysy::ConstantOp op,
    PatternRewriter &rewriter) const override {
  const auto &constantValue = op.getValue();
  Location loc = op.getLoc();
  auto tensorType = llvm::cast<RankedTensorType>(op.getType());
  //llvm::errs() << tensorType << "\n";
  auto memRefType = convertTensorToMemRef(tensorType);
  Value alloc = insertAllocAndDealloc(memRefType, loc, rewriter);
  //llvm::errs() << alloc << "\n";
  
  auto valueShape = memRefType.getShape();
  SmallVector<Value, 8> constantIndices;

  if (!valueShape.empty()) {
    for (auto i : llvm::seq<int64_t>(
             0, *std::max_element(valueShape.begin(), valueShape.end())))
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, i));
  } else {
    // This is the case of a tensor of rank 0.
    constantIndices.push_back(
        rewriter.create<arith::ConstantIndexOp>(loc, 0));
  }

  SmallVector<Value, 2> indices;
  auto valueIt = constantValue.value_begin<FloatAttr>();
  std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
    if (dimension == valueShape.size()) {
      rewriter.create<affine::AffineStoreOp>(
          loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
          llvm::ArrayRef(indices));
      return;
    }

    for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
      indices.push_back(constantIndices[i]);
      storeElements(dimension + 1);
      indices.pop_back();
    }
  };

  storeElements(/*dimension=*/0);

  rewriter.replaceOp(op, alloc);

  return success();
}

}; // ConstantLowering

struct SysyTensorToMemRefLowering
    : public PassWrapper<SysyTensorToMemRefLowering, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SysyTensorToMemRefLowering);

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<SysyDialect,
                    memref::MemRefDialect,
                    arith::ArithDialect,
                    affine::AffineDialect>();
  }

  void runOnOperation() final;

}; // struct SysyTensorToMemRefLowering

void SysyTensorToMemRefLowering::runOnOperation() {
  ConversionTarget target(getContext());
  target.addLegalDialect<
      BuiltinDialect, 
      arith::ArithDialect, 
      affine::AffineDialect,
      memref::MemRefDialect>();

  //target.addIllegalDialect<SysyDialect>();
  //target.addIllegalOp<sysy::ConstantOp>();
  //target.addDynamicallyLegalOp<MatmulOp>([](MatmulOp op) {
  //  return llvm::none_of(op->getOperandTypes(), [](Type type) {
  //    return llvm::isa<TensorType>(type);
  //  });
  //});

  RewritePatternSet patterns(&getContext());
  patterns.add<ConstantLowering>(&getContext());

  if (failed(applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<Pass> createTensorLoweringPass() {
  return std::make_unique<SysyTensorToMemRefLowering>();
}

//===----------------------------------------------------------------------===//
// Op convertors for SysyLowering
//===----------------------------------------------------------------------===//

struct ConvertMatmul : public OpConversionPattern<MatmulOp> {
  ConvertMatmul(MLIRContext *ctx)
      : OpConversionPattern<MatmulOp>(ctx) {}

  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      MatmulOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    const auto &C = adaptor.getDst();
    const auto &A = adaptor.getLhs();
    const auto &B = adaptor.getRhs();
    // Gather the metadata of matmul operation.
    auto dstMemref = cast<MemRefType>(C.getType());    
    auto lhsMemref = cast<MemRefType>(A.getType());
    auto M = dstMemref.getShape()[0];
    auto N = dstMemref.getShape()[1];
    auto K = lhsMemref.getShape()[1];
    auto elemType = dstMemref.getElementType();

    ImplicitLocOpBuilder b(op.getLoc(), rewriter);
    
    // Build the boilerplate of 3D affine for loop.
    auto loopLevelM = b.create<affine::AffineForOp>(0, M, 1);
    b.setInsertionPointToStart(loopLevelM.getBody());
    auto loopLevelN = b.create<affine::AffineForOp>(0, N, 1);
    b.setInsertionPointToStart(loopLevelN.getBody());
    auto loopLevelK = b.create<affine::AffineForOp>(0, K, 1);
    b.setInsertionPointToStart(loopLevelK.getBody());
    
    auto indexI = cast<Value>(loopLevelM.getInductionVar());
    auto indexJ = cast<Value>(loopLevelN.getInductionVar());
    auto indexK = cast<Value>(loopLevelK.getInductionVar());

    using affineLoadIdx = SmallVector<Value, 2>;
    affineLoadIdx indexA { indexI, indexK };
    auto elemA = b.create<affine::AffineLoadOp>(A, indexA);
    affineLoadIdx indexB { indexK, indexJ };
    auto elemB = b.create<affine::AffineLoadOp>(B, indexB);
    affineLoadIdx indexC { indexI, indexJ };
    auto inElemC = b.create<affine::AffineLoadOp>(C, indexC);
    
    if (elemType.isF32()) {
      auto tmp = b.create<arith::MulFOp>(elemA, elemB);
      auto outElemC = b.create<arith::AddFOp>(inElemC, tmp);
      (void)b.create<affine::AffineStoreOp>(
          outElemC, C, indexC);
    } else if (elemType.isInteger(32)) {
      auto tmp = b.create<arith::MulIOp>(elemA, elemB);
      auto outElemC = b.create<arith::AddIOp>(inElemC, tmp);
      (void)b.create<affine::AffineStoreOp>(
          outElemC, C, indexC); 
    } else {
      //llvm::errs() << "The data type of `sysy.matmul` "
      //                 "should be I32 or F32\n";
      op.emitError("The data type of `sysy.matmul` should be I32 or F32\n");
      return failure();
    }

    rewriter.eraseOp(op);    
    return success();
  }
}; // struct ConvertMatmul

struct ConvertAdd : public OpConversionPattern<AddOp> {
  ConvertAdd(MLIRContext *ctx)
      : OpConversionPattern<AddOp>(ctx) {}

  using OpConversionPattern::OpConversionPattern;

  static LogicalResult createAddBody(
      const SmallVector<Value, 3> &idx, 
      const Type &elemType, 
      ImplicitLocOpBuilder &b,
      const Value &dst, 
      const Value &lhs,
      const Value &rhs) {
    auto elemLhs = b.create<affine::AffineLoadOp>(lhs, idx);
    auto elemRhs = b.create<affine::AffineLoadOp>(rhs, idx);
      
    if (elemType.isF32()) {
      auto elemDst = b.create<arith::AddFOp>(elemLhs, elemRhs);
      (void)b.create<affine::AffineStoreOp>(
          elemDst, dst, idx);
    } else if (elemType.isInteger(32)) {
      auto elemDst = b.create<arith::AddIOp>(elemLhs, elemRhs);
      (void)b.create<affine::AffineStoreOp>(
          elemDst, dst, idx);
    } else {
      return failure();
    }

    return success();
  }

  LogicalResult matchAndRewrite(
      AddOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    const auto dst = adaptor.getDst();
    const auto lhs = adaptor.getLhs();
    const auto rhs = adaptor.getRhs();
    const auto dstMemref = cast<MemRefType>(dst.getType());
    const auto elemType = dstMemref.getElementType();
    
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    const auto &dims = dstMemref.getShape();
    using affineLoadIdx = SmallVector<Value, 3>;
    
    if (dims.size() == 0) {
      op.emitError("Occurs the case like memref<f32>\n");
      return failure();
    } else if (1<=dims.size() && dims.size()<=3) {
      affineLoadIdx index;
      for (const auto ub : dims) {
        auto loop = b.create<affine::AffineForOp>(0, ub, 1);
        b.setInsertionPointToStart(loop.getBody());
        auto i = cast<Value>(loop.getInductionVar());
        index.push_back(i);        
      }
      if (failed(createAddBody(index, elemType, b, dst, lhs, rhs))) {
        op.emitError("The data type of `sysy.add` should be I32 or F32\n");
        return failure();
      }
    } else {
      op.emitError("The dimension of sysy tensor "
                   "is too high with current sysy.add implementation.\n");
      return failure();
    }

    rewriter.eraseOp(op);    
    return success();
  }
}; // struct ConvertAdd

struct ConvertSub : public OpConversionPattern<SubOp> {
  ConvertSub(MLIRContext *ctx)
      : OpConversionPattern<SubOp>(ctx) {}

  using OpConversionPattern::OpConversionPattern;

  static LogicalResult createSubBody(
      const SmallVector<Value, 3> &idx, 
      const Type &elemType, 
      ImplicitLocOpBuilder &b,
      const Value &dst, 
      const Value &lhs,
      const Value &rhs) {
    auto elemLhs = b.create<affine::AffineLoadOp>(lhs, idx);
    auto elemRhs = b.create<affine::AffineLoadOp>(rhs, idx);
      
    if (elemType.isF32()) {
      auto elemDst = b.create<arith::SubFOp>(elemLhs, elemRhs);
      (void)b.create<affine::AffineStoreOp>(
          elemDst, dst, idx);
    } else if (elemType.isInteger(32)) {
      auto elemDst = b.create<arith::SubIOp>(elemLhs, elemRhs);
      (void)b.create<affine::AffineStoreOp>(
          elemDst, dst, idx);
    } else {
      return failure();
    }

    return success();
  }

  LogicalResult matchAndRewrite(
      SubOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    const auto dst = adaptor.getDst();
    const auto lhs = adaptor.getLhs();
    const auto rhs = adaptor.getRhs();
    const auto dstMemref = cast<MemRefType>(dst.getType());
    const auto elemType = dstMemref.getElementType();
    
    ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    const auto &dims = dstMemref.getShape();
    using affineLoadIdx = SmallVector<Value, 3>;
    
    if (dims.size() == 0) {
      op.emitError("Occurs the case like memref<f32>\n");
      return failure();
    } else if (1<=dims.size() && dims.size()<=3) {
      affineLoadIdx index;
      for (const auto ub : dims) {
        auto loop = b.create<affine::AffineForOp>(0, ub, 1);
        b.setInsertionPointToStart(loop.getBody());
        auto i = cast<Value>(loop.getInductionVar());
        index.push_back(i);        
      }
      if (failed(createSubBody(index, elemType, b, dst, lhs, rhs))) {
        op.emitError("The data type of `sysy.sub` should be I32 or F32\n");
        return failure();
      }
    } else {
      op.emitError("The dimension of sysy tensor "
                   "is too high with current sysy.sub implementation.\n");
      return failure();
    }

    rewriter.eraseOp(op);    
    return success();
  }
}; // struct ConvertSub


struct SysyLower : impl::SysyLowerBase<SysyLower> {
  using SysyLowerBase::SysyLowerBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    auto *module = getOperation();
    
    ConversionTarget target(*context);
    target.addLegalDialect<affine::AffineDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalDialect<func::FuncDialect>();
    target.addIllegalDialect<SysyDialect>();

    RewritePatternSet patterns(context);
    SysyLowerTypeConverter typeConverter(context);
    patterns.add<ConvertMatmul>(typeConverter, context);
    patterns.add<ConvertAdd>(typeConverter, context);
    patterns.add<ConvertSub>(typeConverter, context);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
}; // struct SysyLower

} // namespace sysy
} // namespace mlir