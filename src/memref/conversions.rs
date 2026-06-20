//! Dialect conversions from the memref dialect.

use pliron::{
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{
            CallOpCallable, OneRegionInterface, OneResultInterface, SymbolOpInterface,
        },
        type_interfaces::{FloatTypeInterface, FunctionTypeInterface},
        types::{IntegerType, Signedness},
    },
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    input_error,
    irbuild::{
        dialect_conversion::{DialectConversion, DialectConversionRewriter, OperandsInfo},
        inserter::{BlockInsertionPoint, Inserter, OpInsertionPoint},
        rewriter::{Rewriter, ScopedRewriter},
    },
    linked_list::LinkedList,
    op::{Op, op_cast, op_impls},
    operation::Operation,
    region::Region,
    result::Result,
    symbol_table::{SymbolTableCollection, nearest_symbol_table},
    r#type::{TypeHandle, Typed, TypedHandle, type_cast, type_impls},
    value::{DefiningEntity, Value},
};
use pliron_common_dialects::{
    cf::{ToCFDialect, op_interfaces::YieldingRegion, ops::NDForOp},
    index::{
        ops::{IndexConstantOp, IndexToIntegerOp, IntegerToIndexOp},
        types::IndexType,
    },
};
use pliron_llvm::{
    ToLLVMType, ToLLVMTypeFn,
    attributes::{FastmathFlagsAttr, IntegerOverflowFlagsAttr},
    function_call_utils::{
        compute_type_size_in_bytes, lookup_or_create_free_fn, lookup_or_create_malloc_fn,
    },
    op_interfaces::{
        BinArithOp, CastOpInterface, FloatBinArithOpWithFastMathFlags,
        IntBinArithOpWithOverflowFlag,
    },
    ops::{BrOp, CallOp, FuncOp, MulOp},
};

use crate::memref::{
    descriptor,
    op_interfaces::ElementWiseBinaryMemrefOpInterface,
    ops::{
        AddOp, AllocOp, CopyOp, DeallocOp, DimOp, DivOp, GenerateOp, LoadOp,
        MatMulOp as MemrefMatMulOp, MulOp as MemrefMulOp, ReshapeOp, SliceParam, StoreOp, SubOp,
        SubviewOp, YieldOp,
    },
    type_interfaces::{MultiDimensionalType, ShapedType},
    types::RankedMemrefType,
};

#[derive(Debug, thiserror::Error)]
pub enum AllocOpRewriteError {
    #[error("Nearest symbol table not found")]
    NearestSymbolTableNotFound,
}

// Replace [AllocOp] with
// * Compute the sizes, strides and total number of elements based on the memref type.
// * A `malloc` that allocates memory for the total number of elements.
// * Build the memref descriptor by storing the allocated pointer, aligned pointer,
//   offset, sizes and strides to the appropriate fields in the descriptor.
// * Replace uses of the original [AllocOp]'s result with the newly built descriptor.
#[op_interface_impl]
impl ToCFDialect for AllocOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_ty = self.result_type(ctx);
        let memref_ty = TypedHandle::<RankedMemrefType>::from_handle(result_ty, ctx)
            .expect("Expected the result type of AllocOp to be a RankedMemrefType");
        let dyn_dimensions: Vec<_> = self.get_dynamic_dimensions(ctx);

        let (sizes, strides, num_elems) =
            descriptor::compute_sizes_strides(ctx, rewriter, memref_ty, dyn_dimensions);

        let element_ty = memref_ty.deref(ctx).element_type();
        let elem_size = compute_type_size_in_bytes(ctx, rewriter, element_ty);
        let alloc_size = MulOp::new_with_overflow_flag(
            ctx,
            elem_size,
            num_elems,
            IntegerOverflowFlagsAttr::default(),
        );
        rewriter.append_op(ctx, &alloc_size);

        let symbol_table_op = nearest_symbol_table(ctx, self.get_operation()).ok_or_else(|| {
            input_error!(
                self.loc(ctx),
                AllocOpRewriteError::NearestSymbolTableNotFound
            )
        })?;
        let malloc = lookup_or_create_malloc_fn(
            ctx,
            &mut SymbolTableCollection::default(),
            symbol_table_op,
        )?;
        let call_malloc_op = CallOp::new(
            ctx,
            CallOpCallable::Direct(malloc.get_symbol_name(ctx)),
            malloc.get_type(ctx),
            vec![alloc_size.get_result(ctx)],
        );
        rewriter.append_op(ctx, &call_malloc_op);
        let allocated_ptr = call_malloc_op.get_result(ctx);

        let offset = IndexConstantOp::new(ctx, 0);
        rewriter.append_op(ctx, &offset);

        // Build the memref descriptor.
        let descriptor = descriptor::pack_descriptor(
            ctx,
            rewriter,
            memref_ty,
            descriptor::Descriptor {
                allocated_ptr,
                aligned_ptr: allocated_ptr,
                offset: offset.get_result(ctx),
                sizes,
                strides,
            },
        )?;

        // Replace uses of the original AllocOp's result with the newly built descriptor.
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![descriptor]);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum DeallocOpRewriteError {
    #[error("Nearest symbol table not found")]
    NearestSymbolTableNotFound,
}

// Replace [DeallocOp] with
// * Extract the allocated pointer from the memref descriptor.
// * A `free` call on the allocated pointer.
#[op_interface_impl]
impl ToCFDialect for DeallocOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref = self.get_memref(ctx);
        let allocated_ptr = descriptor::unpack_allocated_ptr(ctx, rewriter, memref);

        let symbol_table_op = nearest_symbol_table(ctx, self.get_operation()).ok_or_else(|| {
            input_error!(
                self.loc(ctx),
                DeallocOpRewriteError::NearestSymbolTableNotFound
            )
        })?;
        let free_fn =
            lookup_or_create_free_fn(ctx, &mut SymbolTableCollection::default(), symbol_table_op)?;
        let call_free_op = CallOp::new(
            ctx,
            CallOpCallable::Direct(free_fn.get_symbol_name(ctx)),
            free_fn.get_type(ctx),
            vec![allocated_ptr],
        );
        rewriter.append_op(ctx, &call_free_op);
        rewriter.replace_operation(ctx, self.get_operation(), call_free_op.get_operation());
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpConversionErr {
    #[error("Unsupported induction variable type for GenerateOp conversion")]
    UnsupportedIVType,
}
// Replace [GenerateOp] with
// * An [NDForOp], with the loop bounds computed from the memref operand.
// * Inside the innermost loop, the generated value at the current indices,
//   (the [GenerateOp]'s [YieldOp]'s argument) is [StoreOp]'d to the memref.
#[op_interface_impl]
impl ToCFDialect for GenerateOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        // Compute the loop upper bounds based on the memref operand.
        let sizes = descriptor::unpack_sizes(ctx, rewriter, self.get_destination_memref(ctx));

        let const_index_0 = IndexConstantOp::new(ctx, 0);
        let const_index_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, &const_index_0);
        rewriter.append_op(ctx, &const_index_1);

        let lbs = vec![const_index_0.get_result(ctx); sizes.len()];
        let steps = vec![const_index_1.get_result(ctx); sizes.len()];

        let ndforop = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);
            struct State<'a> {
                rewriter: ScopedRewriter<'a>,
                generate_op_region: Ptr<Region>,
                yield_op: YieldOp,
                memref_opd: Value,
                generate_indices: Vec<Value>,
            }
            let mut state = State {
                rewriter: scoped_rewriter,
                generate_op_region: self.get_region(ctx),
                memref_opd: self.get_destination_memref(ctx),
                yield_op: self.get_yield(ctx),
                generate_indices: self.get_entry(ctx).deref(ctx).arguments().collect(),
            };
            NDForOp::new(
                ctx,
                lbs,
                sizes,
                steps,
                |ctx, state, inserter, indices| {
                    // We use the outer rewriter so that newly added ops are
                    // tracked and considered for further rewrites in this pass.
                    let insertion_point = inserter.get_insertion_point();
                    let ndfor_entry = insertion_point
                        .get_insertion_block(ctx)
                        .expect("Failed to get insertion block");

                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(insertion_point);
                    rewriter.inline_region(
                        ctx,
                        state.generate_op_region,
                        BlockInsertionPoint::AfterBlock(ndfor_entry),
                    );
                    // Branch from entry block of our NDForOp to the inlined region's entry block,
                    // passing the induction variables as arguments.
                    let branch_to = ndfor_entry
                        .deref(ctx)
                        .get_next()
                        .expect("Failed to get next block for NDForOp entry block");
                    let branch = BrOp::new(ctx, branch_to, indices);
                    rewriter.append_op(ctx, &branch);

                    // Store the generated value to the memref.
                    let yield_operation = state.yield_op.get_operation();
                    let generated_value = yield_operation.deref(ctx).get_operand(0);
                    let store_op = StoreOp::new(
                        ctx,
                        generated_value,
                        state.memref_opd,
                        state.generate_indices.clone(),
                    );
                    rewriter
                        .set_insertion_point(OpInsertionPoint::BeforeOperation(yield_operation));
                    rewriter.append_op(ctx, &store_op);
                    rewriter.replace_operation(ctx, yield_operation, store_op.get_operation());
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, &ndforop);
        rewriter.replace_operation(ctx, self.get_operation(), ndforop.get_operation());
        Ok(())
    }
}

// Replace [StoreOp] with a sequence of operations that compute the address
// of the element at the given indices in the memref and store the value to that address.
#[op_interface_impl]
impl ToCFDialect for StoreOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let value = self.get_value(ctx);
        let memref = self.get_destination_memref(ctx);
        let elem_ty = value.get_type(ctx);

        let indices = self.get_indices(ctx);
        let ptr = descriptor::get_strided_element_ptr(ctx, rewriter, elem_ty, memref, indices);
        let store_op = pliron_llvm::ops::StoreOp::new(ctx, value, ptr);
        rewriter.append_op(ctx, &store_op);
        rewriter.replace_operation(ctx, self.get_operation(), store_op.get_operation());
        Ok(())
    }
}

// Replace [LoadOp] with a sequence of operations that compute the address
// of the element at the given indices in the memref and load the value from that address.
#[op_interface_impl]
impl ToCFDialect for LoadOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref = self.get_source_memref(ctx);
        let elem_ty = self.get_result(ctx).get_type(ctx);

        let indices = self.get_indices(ctx);
        let ptr = descriptor::get_strided_element_ptr(ctx, rewriter, elem_ty, memref, indices);
        let load_op = pliron_llvm::ops::LoadOp::new(ctx, ptr, elem_ty);
        rewriter.append_op(ctx, &load_op);
        rewriter.replace_operation(ctx, self.get_operation(), load_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToCFDialect for DimOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref = self.get_source_memref(ctx);
        let dim_idx = self.get_dimension_index(ctx);

        let dim_size = match dim_idx.defining_entity() {
            // If the dimension index is a constant, it's more efficient to use `unpack_size`.
            DefiningEntity::Op(op)
                if let Some(index_const_op) = Operation::get_op::<IndexConstantOp>(op, ctx) =>
            {
                let dim = index_const_op
                    .get_attr_constant_index(ctx)
                    .unwrap()
                    .constant_index;
                descriptor::unpack_size(ctx, rewriter, memref, dim)
            }
            _ => descriptor::unpack_size_dynamic_dim_idx(ctx, rewriter, memref, dim_idx),
        };
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![dim_size]);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ElementWiseBinaryMemrefOpConversionErr {
    #[error("Expected memref operand type not found among the previous types of the operand")]
    ExpectedMemrefTypeNotFound,
}

trait ElementWiseBinaryMemrefOpToCF: ElementWiseBinaryMemrefOpInterface {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        // Compute the loop upper bounds based on the memref operand.
        let sizes = descriptor::unpack_sizes(ctx, rewriter, self.get_result_memref(ctx));

        let const_index_0 = IndexConstantOp::new(ctx, 0);
        let const_index_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, &const_index_0);
        rewriter.append_op(ctx, &const_index_1);

        let lbs = vec![const_index_0.get_result(ctx); sizes.len()];
        let steps = vec![const_index_1.get_result(ctx); sizes.len()];

        let ndforop = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);
            struct State<'a> {
                rewriter: ScopedRewriter<'a>,
                memref_result: Value,
                memref_lhs: Value,
                memref_rhs: Value,
                op_fn: fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation>,
                elem_ty: TypeHandle,
            }
            let memref_result = self.get_result_memref(ctx);
            let element_ty = operands_info
                .lookup_most_recent_of_type::<RankedMemrefType>(ctx, memref_result)
                .ok_or_else(|| {
                    input_error!(
                        self.loc(ctx),
                        ElementWiseBinaryMemrefOpConversionErr::ExpectedMemrefTypeNotFound
                    )
                })?
                .element_type();
            let mut state = State {
                rewriter: scoped_rewriter,
                memref_result: self.get_result_memref(ctx),
                memref_lhs: self.get_lhs_memref(ctx),
                memref_rhs: self.get_rhs_memref(ctx),
                op_fn: self.build_llvm_op(),
                elem_ty: element_ty,
            };
            NDForOp::new(
                ctx,
                lbs,
                sizes,
                steps,
                |ctx, state, inserter, indices| {
                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(inserter.get_insertion_point());

                    let element_ty = state.elem_ty;
                    let lhs_loaded =
                        LoadOp::new(ctx, element_ty, state.memref_lhs, indices.clone());
                    let rhs_loaded =
                        LoadOp::new(ctx, element_ty, state.memref_rhs, indices.clone());
                    rewriter.append_op(ctx, &lhs_loaded);
                    rewriter.append_op(ctx, &rhs_loaded);

                    let result = (state.op_fn)(
                        ctx,
                        lhs_loaded.get_result(ctx),
                        rhs_loaded.get_result(ctx),
                        element_ty,
                    );
                    rewriter.append_operation(ctx, result);
                    let result = result.deref(ctx).get_result(0);

                    let store_op = StoreOp::new(ctx, result, state.memref_result, indices);
                    rewriter.append_op(ctx, &store_op);
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, &ndforop);
        rewriter.replace_operation(ctx, self.get_operation(), ndforop.get_operation());
        Ok(())
    }

    fn build_llvm_op(&self) -> fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation>;
}

impl ElementWiseBinaryMemrefOpToCF for AddOp {
    fn build_llvm_op(
        &self,
    ) -> fn(&mut Context, Value, Value, elem_ty: TypeHandle) -> Ptr<Operation> {
        |ctx, lhs, rhs, elem_ty| {
            if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                pliron_llvm::ops::FAddOp::new_with_fast_math_flags(
                    ctx,
                    lhs,
                    rhs,
                    FastmathFlagsAttr::default(),
                )
                .get_operation()
            } else {
                pliron_llvm::ops::AddOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
                .get_operation()
            }
        }
    }
}

#[op_interface_impl]
impl ToCFDialect for AddOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        <Self as ElementWiseBinaryMemrefOpToCF>::rewrite(self, ctx, rewriter, operands_info)
    }
}

impl ElementWiseBinaryMemrefOpToCF for SubOp {
    fn build_llvm_op(
        &self,
    ) -> fn(&mut Context, Value, Value, elem_ty: TypeHandle) -> Ptr<Operation> {
        |ctx, lhs, rhs, elem_ty| {
            if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                pliron_llvm::ops::FSubOp::new_with_fast_math_flags(
                    ctx,
                    lhs,
                    rhs,
                    FastmathFlagsAttr::default(),
                )
                .get_operation()
            } else {
                pliron_llvm::ops::SubOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
                .get_operation()
            }
        }
    }
}

#[op_interface_impl]
impl ToCFDialect for SubOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        <Self as ElementWiseBinaryMemrefOpToCF>::rewrite(self, ctx, rewriter, operands_info)
    }
}

impl ElementWiseBinaryMemrefOpToCF for MemrefMulOp {
    fn build_llvm_op(
        &self,
    ) -> fn(&mut Context, Value, Value, elem_ty: TypeHandle) -> Ptr<Operation> {
        |ctx, lhs, rhs, elem_ty| {
            if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                pliron_llvm::ops::FMulOp::new_with_fast_math_flags(
                    ctx,
                    lhs,
                    rhs,
                    FastmathFlagsAttr::default(),
                )
                .get_operation()
            } else {
                pliron_llvm::ops::MulOp::new_with_overflow_flag(
                    ctx,
                    lhs,
                    rhs,
                    IntegerOverflowFlagsAttr::default(),
                )
                .get_operation()
            }
        }
    }
}

#[op_interface_impl]
impl ToCFDialect for MemrefMulOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        <Self as ElementWiseBinaryMemrefOpToCF>::rewrite(self, ctx, rewriter, operands_info)
    }
}

impl ElementWiseBinaryMemrefOpToCF for DivOp {
    fn build_llvm_op(
        &self,
    ) -> fn(&mut Context, Value, Value, elem_ty: TypeHandle) -> Ptr<Operation> {
        |ctx, lhs, rhs, elem_ty| {
            if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                pliron_llvm::ops::FDivOp::new_with_fast_math_flags(
                    ctx,
                    lhs,
                    rhs,
                    FastmathFlagsAttr::default(),
                )
                .get_operation()
            } else {
                let signedness = {
                    let elem_ty_ref = elem_ty.deref(ctx);
                    elem_ty_ref
                        .downcast_ref::<IntegerType>()
                        .expect("Expected integer element type for integer division")
                        .signedness()
                };
                match signedness {
                    Signedness::Unsigned => {
                        pliron_llvm::ops::UDivOp::new(ctx, lhs, rhs).get_operation()
                    }
                    Signedness::Signed | Signedness::Signless => {
                        pliron_llvm::ops::SDivOp::new(ctx, lhs, rhs).get_operation()
                    }
                }
            }
        }
    }
}

#[op_interface_impl]
impl ToCFDialect for DivOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        <Self as ElementWiseBinaryMemrefOpToCF>::rewrite(self, ctx, rewriter, operands_info)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MemrefMatMulOpConversionErr {
    #[error("Expected memref operand type not found among the previous types of the operand")]
    ExpectedMemrefTypeNotFound,
}

// Lowering for memref::MatMulOp -> 3D NDForOp over (i, j, k):
//   result[i,j] = accum[i,j] + sum_k(lhs[i,k] * rhs[k,j])
// The op result aliases with the accumulator memref.
#[op_interface_impl]
impl ToCFDialect for MemrefMatMulOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref_lhs = self.get_lhs_memref(ctx);
        let memref_rhs = self.get_rhs_memref(ctx);
        let memref_accum = self.get_accum_memref(ctx);

        // Recover the element type from before the memref descriptor conversion.
        let element_ty = operands_info
            .lookup_most_recent_of_type::<RankedMemrefType>(ctx, memref_accum)
            .ok_or_else(|| {
                input_error!(
                    self.loc(ctx),
                    MemrefMatMulOpConversionErr::ExpectedMemrefTypeNotFound
                )
            })?
            .element_type();

        // Unpack loop bounds: M, N from accum sizes; K from lhs dim 1.
        let accum_sizes = descriptor::unpack_sizes(ctx, rewriter, memref_accum);
        let m_size = accum_sizes[0];
        let n_size = accum_sizes[1];

        let lhs_sizes = descriptor::unpack_sizes(ctx, rewriter, memref_lhs);
        let k_size = lhs_sizes[1];

        let const_0 = IndexConstantOp::new(ctx, 0);
        let const_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, &const_0);
        rewriter.append_op(ctx, &const_1);
        let lb0 = const_0.get_result(ctx);
        let step1 = const_1.get_result(ctx);

        // Accumulate into the provided accumulator/result memref.
        let ndfor = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);
            struct AccumState<'a> {
                rewriter: ScopedRewriter<'a>,
                memref_accum: Value,
                memref_lhs: Value,
                memref_rhs: Value,
                elem_ty: TypeHandle,
                mul_fn: fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation>,
                add_fn: fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation>,
            }
            let mul_fn: fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation> =
                |ctx, a, b, elem_ty| {
                    if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                        pliron_llvm::ops::FMulOp::new_with_fast_math_flags(
                            ctx,
                            a,
                            b,
                            FastmathFlagsAttr::default(),
                        )
                        .get_operation()
                    } else {
                        pliron_llvm::ops::MulOp::new_with_overflow_flag(
                            ctx,
                            a,
                            b,
                            IntegerOverflowFlagsAttr::default(),
                        )
                        .get_operation()
                    }
                };
            let add_fn: fn(&mut Context, Value, Value, TypeHandle) -> Ptr<Operation> =
                |ctx, a, b, elem_ty| {
                    if type_impls::<dyn FloatTypeInterface>(&*elem_ty.deref(ctx)) {
                        pliron_llvm::ops::FAddOp::new_with_fast_math_flags(
                            ctx,
                            a,
                            b,
                            FastmathFlagsAttr::default(),
                        )
                        .get_operation()
                    } else {
                        pliron_llvm::ops::AddOp::new_with_overflow_flag(
                            ctx,
                            a,
                            b,
                            IntegerOverflowFlagsAttr::default(),
                        )
                        .get_operation()
                    }
                };
            let mut state = AccumState {
                rewriter: scoped_rewriter,
                memref_accum,
                memref_lhs,
                memref_rhs,
                elem_ty: element_ty,
                mul_fn,
                add_fn,
            };
            NDForOp::new(
                ctx,
                vec![lb0, lb0, lb0],
                vec![m_size, n_size, k_size],
                vec![step1, step1, step1],
                |ctx, state, inserter, indices| {
                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(inserter.get_insertion_point());

                    let i = indices[0];
                    let j = indices[1];
                    let k = indices[2];
                    let elem_ty = state.elem_ty;

                    // Load accumulator: acc = accum[i, j]
                    let load_acc = LoadOp::new(ctx, elem_ty, state.memref_accum, vec![i, j]);
                    rewriter.append_op(ctx, &load_acc);
                    // Load a = lhs[i, k]
                    let load_a = LoadOp::new(ctx, elem_ty, state.memref_lhs, vec![i, k]);
                    rewriter.append_op(ctx, &load_a);
                    // Load b = rhs[k, j]
                    let load_b = LoadOp::new(ctx, elem_ty, state.memref_rhs, vec![k, j]);
                    rewriter.append_op(ctx, &load_b);

                    // mul = a * b
                    let mul = (state.mul_fn)(
                        ctx,
                        load_a.get_result(ctx),
                        load_b.get_result(ctx),
                        elem_ty,
                    );
                    rewriter.append_operation(ctx, mul);

                    // new_acc = acc + mul
                    let mul_result = mul.deref(ctx).get_result(0);
                    let load_acc_result = load_acc.get_result(ctx);
                    let add = (state.add_fn)(ctx, load_acc_result, mul_result, elem_ty);
                    rewriter.append_operation(ctx, add);

                    // Store accum[i, j] = new_acc
                    let add_result = add.deref(ctx).get_result(0);
                    let store = StoreOp::new(ctx, add_result, state.memref_accum, vec![i, j]);
                    rewriter.append_op(ctx, &store);
                },
                &mut state,
            )
        };
        rewriter.append_op(ctx, &ndfor);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![memref_accum]);
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CopyOpConversionErr {
    #[error("Missing operand info for memref.copy source operand")]
    MissingSourceOperandInfo,
}

#[derive(Debug, thiserror::Error)]
pub enum SubviewOpConversionErr {
    #[error("Missing memref_subview_slice_params attribute on memref.subview")]
    MissingSliceParams,
}

fn materialize_slice_params(
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    params: &[SliceParam],
) -> Vec<Value> {
    params
        .iter()
        .map(|param| match param {
            SliceParam::Static(val) => {
                let op = IndexConstantOp::new(ctx, *val);
                rewriter.append_op(ctx, &op);
                op.get_result(ctx)
            }
            SliceParam::Dynamic(value) => *value,
        })
        .collect()
}

/// Cast the given value to the target type if necessary,
/// handling index<->integer casts as needed.
fn cast_value_to_type(
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    value: Value,
    target_ty: TypeHandle,
) -> Value {
    if value.get_type(ctx) == target_ty {
        return value;
    }

    if value.get_type(ctx).deref(ctx).is::<IndexType>() {
        let cast = IndexToIntegerOp::new(ctx, value, target_ty);
        rewriter.append_op(ctx, &cast);
        return cast.get_result(ctx);
    }

    if target_ty.deref(ctx).is::<IndexType>() {
        let cast = IntegerToIndexOp::new(ctx, value, target_ty);
        rewriter.append_op(ctx, &cast);
        return cast.get_result(ctx);
    }

    value
}

#[op_interface_impl]
impl ToCFDialect for CopyOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        let source = self.source(ctx);
        let destination = self.destination(ctx);

        let element_ty = operands_info
            .lookup_most_recent_of_type::<RankedMemrefType>(ctx, source)
            .ok_or_else(|| {
                input_error!(self.loc(ctx), CopyOpConversionErr::MissingSourceOperandInfo)
            })?
            .element_type();
        let sizes = descriptor::unpack_sizes(ctx, rewriter, destination);
        let rank = sizes.len();

        let const_zero = IndexConstantOp::new(ctx, 0);
        let const_one = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, &const_zero);
        rewriter.append_op(ctx, &const_one);
        let lb0 = const_zero.get_result(ctx);
        let step1 = const_one.get_result(ctx);
        let lbs = vec![lb0; rank];
        let loop_steps = vec![step1; rank];

        let ndfor = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);
            struct State<'a> {
                rewriter: ScopedRewriter<'a>,
                memref_dst: Value,
                memref_src: Value,
                elem_ty: TypeHandle,
            }
            let mut state = State {
                rewriter: scoped_rewriter,
                memref_dst: destination,
                memref_src: source,
                elem_ty: element_ty,
            };
            NDForOp::new(
                ctx,
                lbs,
                sizes,
                loop_steps,
                |ctx, state, inserter, dst_indices| {
                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(inserter.get_insertion_point());
                    let load =
                        LoadOp::new(ctx, state.elem_ty, state.memref_src, dst_indices.clone());
                    rewriter.append_op(ctx, &load);
                    let store =
                        StoreOp::new(ctx, load.get_result(ctx), state.memref_dst, dst_indices);
                    rewriter.append_op(ctx, &store);
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, &ndfor);
        rewriter.replace_operation(ctx, self.get_operation(), ndfor.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl ToCFDialect for SubviewOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let source = self.source(ctx);
        let result_ty = TypedHandle::<RankedMemrefType>::from_handle(self.result_type(ctx), ctx)
            .expect("SubviewOp result must be a RankedMemrefType");

        let source_descriptor = descriptor::unpack_descriptor(ctx, rewriter, source);
        let offset_ty = source_descriptor.offset.get_type(ctx);
        let size_ty = source_descriptor
            .sizes
            .first()
            .map(|size| size.get_type(ctx))
            .unwrap_or(offset_ty);

        let offsets = materialize_slice_params(ctx, rewriter, &self.slice_offsets(ctx))
            .into_iter()
            .map(|offset| cast_value_to_type(ctx, rewriter, offset, offset_ty))
            .collect::<Vec<_>>();
        let sizes = materialize_slice_params(ctx, rewriter, &self.slice_sizes(ctx))
            .into_iter()
            .map(|size| cast_value_to_type(ctx, rewriter, size, size_ty))
            .collect::<Vec<_>>();
        let steps = materialize_slice_params(ctx, rewriter, &self.slice_steps(ctx))
            .into_iter()
            .zip(source_descriptor.strides.iter())
            .map(|(step, stride)| cast_value_to_type(ctx, rewriter, step, stride.get_type(ctx)))
            .collect::<Vec<_>>();

        let new_offset = offsets.iter().zip(source_descriptor.strides.iter()).fold(
            source_descriptor.offset,
            |current_offset, (offset, stride)| {
                let scaled_offset = MulOp::new_with_overflow_flag(
                    ctx,
                    *offset,
                    *stride,
                    IntegerOverflowFlagsAttr::default(),
                );
                rewriter.append_op(ctx, &scaled_offset);
                let sum = pliron_llvm::ops::AddOp::new_with_overflow_flag(
                    ctx,
                    current_offset,
                    scaled_offset.get_result(ctx),
                    IntegerOverflowFlagsAttr::default(),
                );
                rewriter.append_op(ctx, &sum);
                sum.get_result(ctx)
            },
        );

        let new_strides = source_descriptor
            .strides
            .iter()
            .zip(steps.iter())
            .map(|(stride, step)| {
                let new_stride = MulOp::new_with_overflow_flag(
                    ctx,
                    *stride,
                    *step,
                    IntegerOverflowFlagsAttr::default(),
                );
                rewriter.append_op(ctx, &new_stride);
                new_stride.get_result(ctx)
            })
            .collect();

        let result_descriptor = descriptor::pack_descriptor(
            ctx,
            rewriter,
            result_ty,
            descriptor::Descriptor {
                allocated_ptr: source_descriptor.allocated_ptr,
                aligned_ptr: source_descriptor.aligned_ptr,
                offset: new_offset,
                sizes,
                strides: new_strides,
            },
        )?;

        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![result_descriptor]);
        Ok(())
    }
}

/// Ranked memref types are converted to LLVM struct types with details.
#[type_interface_impl]
impl ToLLVMType for RankedMemrefType {
    fn converter(&self) -> ToLLVMTypeFn {
        // Compute an LLVM struct type with the following fields:
        // * allocated pointer (ptr)
        // * aligned pointer (ptr)
        // * offset (i64)
        // * sizes (i64 array of length rank)
        // * strides (i64 array of length rank)
        |self_ty: TypeHandle, ctx: &mut Context| -> Result<TypeHandle> {
            let rank: u64 = {
                let self_ty = self_ty.deref(ctx);
                let self_ty = self_ty
                    .downcast_ref::<RankedMemrefType>()
                    .expect("Expected a RankedMemrefType");

                self_ty.rank().try_into().expect("Rank should fit into u64")
            };
            let ptr = pliron_llvm::types::PointerType::get(ctx, 0);
            let i64 = pliron::builtin::types::IntegerType::get(ctx, 64, Signedness::Signless);
            let sizes_array = pliron_llvm::types::ArrayType::get(ctx, i64.into(), rank);
            let strides_array = sizes_array;
            let struct_ty = pliron_llvm::types::StructType::get_unnamed(
                ctx,
                vec![
                    ptr.into(),
                    ptr.into(),
                    i64.into(),
                    sizes_array.into(),
                    strides_array.into(),
                ],
            );
            Ok(struct_ty.into())
        }
    }
}

fn lower_func_op_to_llvm(func_op: &FuncOp, ctx: &mut Context) -> Result<()> {
    // update the function type to convert any tensor types in the signature to LLVM types.
    let func_ty = func_op.get_type(ctx);
    let res_ty = func_ty.deref(ctx).result_type();
    let res_ty_converter =
        type_cast::<dyn ToLLVMType>(&*res_ty.deref(ctx)).map(|to_llvm_ty| to_llvm_ty.converter());
    let res_ty = if let Some(res_ty_converter) = res_ty_converter {
        (res_ty_converter)(res_ty, ctx)?
    } else {
        res_ty
    };
    let arg_tys = func_ty.deref(ctx).arg_types();
    let arg_tys = arg_tys
        .iter()
        .map(|arg_ty| {
            let arg_ty_converter = type_cast::<dyn ToLLVMType>(&*arg_ty.deref(ctx))
                .map(|to_llvm_ty| to_llvm_ty.converter());
            if let Some(arg_ty_converter) = arg_ty_converter {
                (arg_ty_converter)(*arg_ty, ctx)
            } else {
                Ok(*arg_ty)
            }
        })
        .collect::<Result<Vec<_>>>()?;
    let new_func_ty = pliron_llvm::types::FuncType::get(ctx, res_ty, arg_tys, false);
    func_op.set_attr_llvm_func_type(ctx, TypeAttr::new(new_func_ty.into()));

    // Update the argument types in the entry block as well.
    let entry_block = func_op
        .get_entry_block(ctx)
        .expect("FuncOp must have an entry block");
    let args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
    for arg in args {
        let arg_ty = arg.get_type(ctx);
        let arg_ty_converter = type_cast::<dyn ToLLVMType>(&*arg_ty.deref(ctx))
            .map(|to_llvm_ty| to_llvm_ty.converter());
        let arg_ty = if let Some(arg_ty_converter) = arg_ty_converter {
            (arg_ty_converter)(arg_ty, ctx)?
        } else {
            arg_ty
        };
        arg.set_type(ctx, arg_ty);
    }
    Ok(())
}

fn lower_llvm_load_op_to_llvm(
    load_op: &pliron_llvm::ops::LoadOp,
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
) -> Result<()> {
    let loaded_ty = load_op.get_result(ctx).get_type(ctx);
    let to_memref_ty = type_cast::<dyn ToLLVMType>(&*loaded_ty.deref(ctx)).map(|t| t.converter());
    let memref_ty = if let Some(to_memref_ty) = to_memref_ty {
        (to_memref_ty)(loaded_ty, ctx)?
    } else {
        loaded_ty
    };
    rewriter.set_value_type(ctx, load_op.get_result(ctx), memref_ty);
    Ok(())
}

// Replace [ReshapeOp] with descriptor construction over the source storage.
// The resulting descriptor reuses source allocated/aligned pointers and offset,
// while recomputing sizes and strides for the new result shape.
#[op_interface_impl]
impl ToCFDialect for ReshapeOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let source = self.get_source(ctx);
        let dynamic_dimensions = self.get_dynamic_dimensions(ctx);

        let result_memref_ty =
            TypedHandle::<RankedMemrefType>::from_handle(self.result_type(ctx), ctx)
                .expect("ReshapeOp result type must be RankedMemrefType");

        let (sizes, strides, _num_elems) =
            descriptor::compute_sizes_strides(ctx, rewriter, result_memref_ty, dynamic_dimensions);

        let allocated_ptr = descriptor::unpack_allocated_ptr(ctx, rewriter, source);
        let aligned_ptr = descriptor::unpack_aligned_ptr(ctx, rewriter, source);
        let offset = descriptor::unpack_offset(ctx, rewriter, source);

        let reshaped = descriptor::pack_descriptor(
            ctx,
            rewriter,
            result_memref_ty,
            descriptor::Descriptor {
                allocated_ptr,
                aligned_ptr,
                offset,
                sizes,
                strides,
            },
        )?;

        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![reshaped]);
        Ok(())
    }
}

/// Implement [DialectConversion] for control-flow to CF conversion.
pub struct MemrefToCF;

impl DialectConversion for MemrefToCF {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn ToCFDialect>(&*Operation::get_op_dyn(op, ctx))
            || Operation::get_op::<FuncOp>(op, ctx).is_some()
            || Operation::get_op::<pliron_llvm::ops::LoadOp>(op, ctx).is_some()
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_impls::<dyn ToLLVMType>(&*ty.deref(ctx))
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let Some(ty_converter) =
            type_cast::<dyn ToLLVMType>(&*ty.deref(ctx)).map(|t| t.converter())
        else {
            return Ok(ty);
        };
        ty_converter(ty, ctx)
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        operands_info: &OperandsInfo,
    ) -> Result<()> {
        if let Some(func_op) = Operation::get_op::<FuncOp>(op, ctx) {
            return lower_func_op_to_llvm(&func_op, ctx);
        }
        if let Some(load_op) = Operation::get_op::<pliron_llvm::ops::LoadOp>(op, ctx) {
            return lower_llvm_load_op_to_llvm(&load_op, ctx, rewriter);
        }
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let to_cf_op =
            op_cast::<dyn ToCFDialect>(&*op_dyn).expect("Matched Op must implement ToCFDialect");
        to_cf_op.rewrite(ctx, rewriter, operands_info)
    }
}
