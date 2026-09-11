// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Translate tensor to memref

use pliron::{
    basic_block::BasicBlock,
    builtin::{
        attributes::TypeAttr,
        op_interfaces::{
            OneOpdInterface, OneResultInterface, OperandSegmentInterface,
            SingleBlockRegionInterface,
        },
        type_interfaces::FunctionTypeInterface,
    },
    context::{Context, Ptr},
    derive::{op_interface_impl, type_interface_impl},
    identifier::Identifier,
    input_error,
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::{Inserter, OpInsertionPoint},
        rewriter::{Rewriter, ScopedRewriter},
    },
    linked_list::ContainsLinkedList,
    location::Located,
    op::Op,
    operation::Operation,
    result::Result,
    symbol_table::nearest_symbol_table,
    r#type::{TypeHandle, Typed, TypedHandle},
    value::{Use, Value},
};
use pliron_common_dialects::{
    cf::{
        op_interfaces::YieldingRegions,
        ops::{ForOp, NDForOp},
    },
    index::ops::IndexConstantOp,
};
use pliron_llvm::ops::FuncOp;

use crate::{
    memref::{
        self, ToMemrefType,
        attributes::DenseElementsAttr,
        descriptor,
        op_interfaces::ElementWiseBinaryMemrefOpInterface,
        ops::{
            CopyOp as MemrefCopyOp, GetGlobalOp as MemrefGetGlobalOp, GlobalOp as MemrefGlobalOp,
            MatMulOp as MemrefMatMulOp, ReshapeOp as MemrefReshapeOp, SliceParam,
            SubviewOp as MemrefSubviewOp, YieldOp,
        },
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::{
        bufferize::{Alias, AliasKind, BufferRelation, BufferizableOpInterface, BufferizerState},
        op_interfaces::ElementWiseBinaryTensorOpInterface,
        ops::{
            AddOp, BatchMatMulOp, ConstantOp, DivOp, ExtractOp,
            ExtractSliceOp as TensorExtractSliceOp, GenerateOp,
            InsertSliceOp as TensorInsertSliceOp, MatMulOp, MulOp, ReshapeOp as TensorReshapeOp,
            SubOp,
        },
        types::RankedTensorType,
    },
};

#[type_interface_impl]
impl ToMemrefType for RankedTensorType {
    fn convert(&self, ctx: &Context) -> Result<TypeHandle> {
        let memref_ty = RankedMemrefType::get(ctx, self.element_type(), self.shape().clone());
        Ok(memref_ty.into())
    }
}

/// Convert a tensor type (which must implement [ToMemrefType]) to its
/// memref equivalent. Returns an error if it cannot do the conversion.
fn tensor_type_to_memref_type(
    ty: TypeHandle,
    ctx: &Context,
) -> Result<TypedHandle<RankedMemrefType>> {
    TypedHandle::<RankedMemrefType>::from_handle(memref::to_memref_type(ty, ctx)?, ctx)
}

#[derive(thiserror::Error, Debug)]
pub enum ConstantOpConversionErr {
    #[error("Nearest symbol table not found")]
    NearestSymbolTableNotFound,
}

/// Build a name for a constant global from its shaped type. The caller should unique the name.
fn constant_global_name(ctx: &Context, value: &DenseElementsAttr) -> Identifier {
    use core::fmt::Write;

    let mut name = String::from("__constant_");
    let shaped_ty = value.ty();
    let shaped_ty = shaped_ty.deref(ctx);
    for dim in shaped_ty.shape() {
        let Dimension::Static(extent) = dim else {
            panic!("A constant global must have a fully static shape");
        };
        let _ = write!(name, "{extent}x");
    }
    let element_ty = shaped_ty.element_type();
    let element_name = Identifier::from(element_ty.deref(ctx).get_type_id().name.clone());
    let _ = write!(name, "{element_name}_{}B", value.element_size(ctx));
    name.try_into()
        .expect("What we just built must be a valid identifier")
}

/// Create a [MemrefGlobalOp] that holds `value` in the symbol table nearest to `op`.
/// Return its name.
fn create_constant_global(
    ctx: &mut Context,
    bufferizer_state: &mut BufferizerState,
    op: Ptr<Operation>,
    memref_ty: TypedHandle<RankedMemrefType>,
    value: DenseElementsAttr,
) -> Result<Identifier> {
    let symbol_table_op = nearest_symbol_table(ctx, op).ok_or_else(|| {
        input_error!(
            op.deref(ctx).loc(),
            ConstantOpConversionErr::NearestSymbolTableNotFound
        )
    })?;
    let symbol_table = bufferizer_state
        .symbol_tables
        .get_symbol_table(ctx, symbol_table_op);

    // Build a new identifier from an existing one and a counter suffic.
    let append_count = |id, count| {
        id + format!("_{count}")
            .try_into()
            .expect("_i must be an Identifier")
    };

    // Come up with a name for this global and unique it.
    let hint = constant_global_name(ctx, &value);
    let mut name = append_count(hint.clone(), bufferizer_state.name_counter);
    while symbol_table.lookup(&name).is_some() {
        bufferizer_state.name_counter += 1;
        name = append_count(hint.clone(), bufferizer_state.name_counter);
    }
    bufferizer_state.name_counter += 1;

    // Create and insert the global in the symbol table (op).
    let global = MemrefGlobalOp::new(ctx, name.clone(), memref_ty, Some(value), true);
    symbol_table.insert(ctx, Box::new(global), None)?;
    Ok(name)
}

// A constant bufferizes to a [MemrefGlobalOp] at module scope,
// and a [MemrefGetGlobalOp] to access it.
#[op_interface_impl]
impl BufferizableOpInterface for ConstantOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
        vec![]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn is_writable(&self, _ctx: &Context, _value: Value) -> bool {
        // Constant storage isn't writable
        false
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref_ty = tensor_type_to_memref_type(self.get_result(ctx).get_type(ctx), ctx)?;
        // `take_value` here makes `self` incorrect, but we delete `self` right after.
        let mut value = self.take_value(ctx);
        value.set_type(ctx, memref_ty.into())?;
        let name = create_constant_global(
            ctx,
            bufferizer_state,
            self.get_operation(),
            memref_ty,
            value,
        )?;

        let get_global = MemrefGetGlobalOp::new(ctx, name, memref_ty);
        rewriter.append_op(ctx, &get_global);
        rewriter.replace_operation(ctx, self.get_operation(), get_global.get_operation());
        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpConversionErr {
    #[error("Unsupported induction variable type for GenerateOp conversion")]
    UnsupportedIVType,
}

#[op_interface_impl]
impl BufferizableOpInterface for GenerateOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
        vec![]
    }

    fn get_dynamic_dimensions(&self, ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        Some(self.dynamic_dimensions(ctx))
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_ty = tensor_type_to_memref_type(self.get_result(ctx).get_type(ctx), ctx)?;

        let alloc = bufferizer_state.tmm.create_memref_alloc(
            ctx,
            result_ty,
            self.dynamic_dimensions(ctx),
        )?;
        rewriter.append_operation(ctx, alloc.get_operation());

        let yield_op = self.get_yield(ctx, 0);

        struct State<'a> {
            yield_op: YieldOp,
            rewriter: &'a mut DialectConversionRewriter,
            source_body: Ptr<BasicBlock>,
        }
        let generate_op = memref::ops::GenerateOp::new(
            ctx,
            alloc.get_result(ctx),
            |ctx, state, inserter, indices: Vec<Value>| {
                let source_body = state.source_body;

                // Replace uses of the source block's arguments with the
                // memref.generate's induction variables.
                let source_args: Vec<_> = source_body.deref(ctx).arguments().collect();
                for (source_arg, idx) in source_args.iter().zip(indices.iter()) {
                    state
                        .rewriter
                        .replace_value_uses_with(ctx, *source_arg, *idx);
                }

                // Move all of the source block's operations into the memref.generate entry block.
                let entry_block = inserter
                    .get_insertion_block(ctx)
                    .expect("Inserter must be set to entry block");
                let body_ops: Vec<_> = source_body.deref(ctx).iter(ctx).collect();
                for op in body_ops {
                    state.rewriter.move_operation(
                        ctx,
                        op,
                        OpInsertionPoint::AtBlockEnd(entry_block),
                    );
                }

                let yield_value = state.yield_op.get_operand(ctx);
                // Remove the previous yield as the memref GenerateOp will add a new one.
                state
                    .rewriter
                    .erase_operation(ctx, state.yield_op.get_operation());
                yield_value
            },
            State {
                yield_op,
                rewriter,
                source_body: self.get_body(ctx, 0),
            },
        );
        rewriter.append_op(ctx, &generate_op);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());

        Ok(())
    }
}

#[op_interface_impl]
impl BufferizableOpInterface for ExtractOp {
    fn operand_bufferizes_to_memory_read(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(0) == opd
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
        vec![]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let operand = self.get_tensor_operand(ctx);
        let indices = self.get_index_operands(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        // Create a LoadOp to extract the value from the memref.
        let load_op = memref::ops::LoadOp::new(ctx, result_ty, operand, indices.clone());
        rewriter.append_op(ctx, &load_op);
        rewriter.replace_operation(ctx, self.get_operation(), load_op.get_operation());
        Ok(())
    }
}

trait ElementWiseBinaryTensorOpToMemref: ElementWiseBinaryTensorOpInterface {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);

        let result_ty = tensor_type_to_memref_type(self.get_result(ctx).get_type(ctx), ctx)?;
        let elem_ty = result_ty.deref(ctx).element_type();
        // Based on the operand shapes, it is possible that the result shape can be inferred
        // to have more static dimensions than what we know with `result_ty` above.
        let compatible_shape = self.compatible_shape(ctx);
        let dynamic_dim_operands = compatible_shape
            .iter()
            .enumerate()
            .filter_map(|(i, dim)| {
                if let Dimension::Dynamic = dim {
                    // Get the dynamic operands from the memref descriptor of the first operand.
                    Some(descriptor::unpack_size(ctx, rewriter, lhs, i))
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let result_ty = RankedMemrefType::get(ctx, elem_ty, compatible_shape);

        let alloc =
            bufferizer_state
                .tmm
                .create_memref_alloc(ctx, result_ty, dynamic_dim_operands)?;
        rewriter.append_operation(ctx, alloc.get_operation());
        let add = self.build_memref_op(ctx, alloc.get_result(ctx), lhs, rhs);
        rewriter.append_operation(ctx, add);
        rewriter.replace_operation(ctx, self.get_operation(), alloc.get_operation());
        Ok(())
    }

    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation>;
}

impl ElementWiseBinaryTensorOpToMemref for AddOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::AddOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for SubOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::SubOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for MulOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::MulOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

impl ElementWiseBinaryTensorOpToMemref for DivOp {
    fn build_memref_op(
        &self,
        ctx: &mut Context,
        res: Value,
        lhs: Value,
        rhs: Value,
    ) -> Ptr<Operation> {
        memref::ops::DivOp::new(ctx, res, lhs, rhs).get_operation()
    }
}

macro_rules! impl_non_aliasing_bufferizable {
    ($op_ty:ty) => {
        #[op_interface_impl]
        impl BufferizableOpInterface for $op_ty {
            fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
                true
            }

            fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
                false
            }

            fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
                vec![]
            }

            fn get_dynamic_dimensions(
                &self,
                _ctx: &Context,
                _opd: Use<Value>,
            ) -> Option<Vec<Value>> {
                None
            }

            fn rewrite(
                &self,
                ctx: &mut Context,
                rewriter: &mut DialectConversionRewriter,
                bufferizer_state: &mut BufferizerState,
                _operands_info: &OperandsInfo,
            ) -> Result<()> {
                <Self as ElementWiseBinaryTensorOpToMemref>::rewrite(
                    self,
                    ctx,
                    rewriter,
                    bufferizer_state,
                    _operands_info,
                )
            }
        }
    };
}

impl_non_aliasing_bufferizable!(AddOp);
impl_non_aliasing_bufferizable!(SubOp);
impl_non_aliasing_bufferizable!(MulOp);
impl_non_aliasing_bufferizable!(DivOp);

/// Allow [pliron_llvm::ops::LoadOp] to participate in bufferization when it
/// loads a tensor value — the rewrite converts the result type to memref.
#[op_interface_impl]
impl BufferizableOpInterface for pliron_llvm::ops::LoadOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
        vec![]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let loaded_ty = self.get_result(ctx).get_type(ctx);
        let memref_ty = memref::to_memref_type(loaded_ty, ctx)?;
        rewriter.set_value_type(ctx, self.get_result(ctx), memref_ty);
        Ok(())
    }
}

#[op_interface_impl]
impl BufferizableOpInterface for ForOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        true
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        // Tensor iter_args are always considered a write.
        true
    }

    // The i-th `iter_args_init` operand aliases the op's i-th result.
    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        let op = self.get_operation().deref(ctx);
        let iter_args_start = self.segment_size(ctx, 0) as usize;
        let num_iter_args = self.get_num_iter_arg_inits(ctx) as usize;
        (0..num_iter_args)
            .map(|i| Alias {
                operand: op.get_operand_as_use(iter_args_start + i),
                result: op.get_result(i),
                kind: AliasKind::Must,
                relation: BufferRelation::Equivalent,
            })
            .collect()
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn is_writable(&self, _ctx: &Context, _value: Value) -> bool {
        // Loop-carried block arguments can always be viewed as writable from inside the body
        true
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let iter_args_init = self.get_iter_args_init(ctx);
        let loop_carried_vars = self.get_loop_carried_variables(ctx);
        for (block_arg, init_val) in loop_carried_vars.iter().zip(iter_args_init.iter()) {
            rewriter.set_value_type(ctx, *block_arg, init_val.get_type(ctx));
        }
        let results: Vec<Value> = self.get_operation().deref(ctx).results().collect();
        for (result, init_val) in results.iter().zip(iter_args_init.iter()) {
            rewriter.set_value_type(ctx, *result, init_val.get_type(ctx));
        }
        Ok(())
    }
}

// Lowering for tensor::MatMulOp -> memref::MatMulOp with accumulator aliasing.
#[op_interface_impl]
impl BufferizableOpInterface for MatMulOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        true
    }

    fn operand_bufferizes_to_memory_write(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(2) == opd
    }

    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        vec![Alias {
            operand: self.get_operation().deref(ctx).get_operand_as_use(2),
            result: self.get_result(ctx),
            kind: AliasKind::Must,
            relation: BufferRelation::Equivalent,
        }]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);
        let accum = self.get_operation().deref(ctx).get_operand(2);

        let matmul = MemrefMatMulOp::new(ctx, lhs, rhs, accum);
        rewriter.append_operation(ctx, matmul.get_operation());

        rewriter.replace_operation(ctx, self.get_operation(), matmul.get_operation());
        Ok(())
    }
}

// Lowering for tensor::BatchMatMulOp using accumulator aliasing.
// For each batch index, it creates subviews of lhs/rhs/accum and performs 2D memref.matmul.
#[op_interface_impl]
impl BufferizableOpInterface for BatchMatMulOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        true
    }

    fn operand_bufferizes_to_memory_write(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(2) == opd
    }

    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        vec![Alias {
            operand: self.get_operation().deref(ctx).get_operand_as_use(2),
            result: self.get_result(ctx),
            kind: AliasKind::Must,
            relation: BufferRelation::Equivalent,
        }]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;

        let lhs = self.get_operation().deref(ctx).get_operand(0);
        let rhs = self.get_operation().deref(ctx).get_operand(1);
        let accum = self.get_operation().deref(ctx).get_operand(2);

        let lhs_memref_ty = TypedHandle::<RankedMemrefType>::from_handle(lhs.get_type(ctx), ctx)
            .expect("BatchMatMulOp lhs must be a ranked memref after conversion");
        let rhs_memref_ty = TypedHandle::<RankedMemrefType>::from_handle(rhs.get_type(ctx), ctx)
            .expect("BatchMatMulOp rhs must be a ranked memref after conversion");
        let accum_memref_ty =
            TypedHandle::<RankedMemrefType>::from_handle(accum.get_type(ctx), ctx)
                .expect("BatchMatMulOp accum must be a ranked memref after conversion");

        let rank = lhs_memref_ty.deref(ctx).rank();
        let batch_rank = rank - 2;

        let lhs_shape = lhs_memref_ty.deref(ctx).shape().clone();
        let rhs_shape = rhs_memref_ty.deref(ctx).shape().clone();
        let elem_ty = accum_memref_ty.deref(ctx).element_type();

        let lhs_sizes = lhs_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| match dim {
                Dimension::Dynamic => descriptor::unpack_size(ctx, rewriter, lhs, i),
                Dimension::Static(v) => {
                    let c = IndexConstantOp::new(ctx, *v);
                    rewriter.append_op(ctx, &c);
                    c.get_result(ctx)
                }
            })
            .collect::<Vec<_>>();
        let rhs_sizes = rhs_shape
            .iter()
            .enumerate()
            .map(|(i, dim)| match dim {
                Dimension::Dynamic => descriptor::unpack_size(ctx, rewriter, rhs, i),
                Dimension::Static(v) => {
                    let c = IndexConstantOp::new(ctx, *v);
                    rewriter.append_op(ctx, &c);
                    c.get_result(ctx)
                }
            })
            .collect::<Vec<_>>();

        if batch_rank == 0 {
            let matmul = MemrefMatMulOp::new(ctx, lhs, rhs, accum);
            rewriter.append_operation(ctx, matmul.get_operation());
            rewriter.replace_operation(ctx, self.get_operation(), matmul.get_operation());
            return Ok(());
        }

        let const_index_0 = IndexConstantOp::new(ctx, 0);
        let const_index_1 = IndexConstantOp::new(ctx, 1);
        rewriter.append_op(ctx, &const_index_0);
        rewriter.append_op(ctx, &const_index_1);

        let lb0 = const_index_0.get_result(ctx);
        let step1 = const_index_1.get_result(ctx);

        let batch_ubs = (0..batch_rank).map(|i| lhs_sizes[i]).collect::<Vec<_>>();

        let ndfor = {
            let scoped_rewriter = ScopedRewriter::new(rewriter, OpInsertionPoint::Unset);

            struct State<'a> {
                rewriter: ScopedRewriter<'a>,
                lhs: Value,
                rhs: Value,
                accum: Value,
                lhs_shape: Vec<Dimension>,
                rhs_shape: Vec<Dimension>,
                lhs_sizes: Vec<Value>,
                rhs_sizes: Vec<Value>,
                rank: usize,
                batch_rank: usize,
                elem_ty: TypeHandle,
            }

            let mut state = State {
                rewriter: scoped_rewriter,
                lhs,
                rhs,
                accum,
                lhs_shape,
                rhs_shape,
                lhs_sizes,
                rhs_sizes,
                rank,
                batch_rank,
                elem_ty,
            };

            NDForOp::new(
                ctx,
                vec![lb0; batch_rank],
                batch_ubs,
                vec![step1; batch_rank],
                |ctx, state, inserter, indices| {
                    let rewriter = &mut state.rewriter;
                    rewriter.set_insertion_point(inserter.get_insertion_point());

                    let dim_to_size = |dim: &Dimension, size_val: Value| match dim {
                        Dimension::Static(v) => SliceParam::Static(*v),
                        Dimension::Dynamic => SliceParam::Dynamic(size_val),
                    };

                    let one_sizes = vec![SliceParam::Static(1); state.batch_rank];
                    let zero_offsets = vec![SliceParam::Static(0); 2];
                    let unit_steps = vec![SliceParam::Static(1); state.rank];

                    let lhs_m_dim = dim_to_size(
                        &state.lhs_shape[state.rank - 2],
                        state.lhs_sizes[state.rank - 2],
                    );
                    let lhs_k_dim = dim_to_size(
                        &state.lhs_shape[state.rank - 1],
                        state.lhs_sizes[state.rank - 1],
                    );
                    let rhs_k_dim = dim_to_size(
                        &state.rhs_shape[state.rank - 2],
                        state.rhs_sizes[state.rank - 2],
                    );
                    let rhs_n_dim = dim_to_size(
                        &state.rhs_shape[state.rank - 1],
                        state.rhs_sizes[state.rank - 1],
                    );

                    let mut lhs_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    lhs_offsets.extend(zero_offsets.clone());
                    let mut lhs_sizes = one_sizes.clone();
                    lhs_sizes.push(lhs_m_dim.clone());
                    lhs_sizes.push(lhs_k_dim.clone());

                    let lhs_subview = MemrefSubviewOp::new(
                        ctx,
                        state.lhs,
                        lhs_offsets,
                        lhs_sizes,
                        unit_steps.clone(),
                    );
                    rewriter.append_op(ctx, &lhs_subview);

                    let mut rhs_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    rhs_offsets.extend(zero_offsets.clone());
                    let mut rhs_sizes = one_sizes.clone();
                    rhs_sizes.push(rhs_k_dim.clone());
                    rhs_sizes.push(rhs_n_dim.clone());

                    let rhs_subview = MemrefSubviewOp::new(
                        ctx,
                        state.rhs,
                        rhs_offsets,
                        rhs_sizes,
                        unit_steps.clone(),
                    );
                    rewriter.append_op(ctx, &rhs_subview);

                    let mut accum_offsets = indices
                        .iter()
                        .copied()
                        .map(SliceParam::Dynamic)
                        .collect::<Vec<_>>();
                    accum_offsets.extend(zero_offsets);
                    let mut accum_sizes = one_sizes;
                    accum_sizes.push(lhs_m_dim.clone());
                    accum_sizes.push(rhs_n_dim.clone());

                    let accum_subview = MemrefSubviewOp::new(
                        ctx,
                        state.accum,
                        accum_offsets,
                        accum_sizes,
                        unit_steps,
                    );
                    rewriter.append_op(ctx, &accum_subview);

                    let m_dim = match &lhs_m_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };
                    let k_dim = match &lhs_k_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };
                    let n_dim = match &rhs_n_dim {
                        SliceParam::Static(v) => Dimension::Static(*v),
                        SliceParam::Dynamic(_) => Dimension::Dynamic,
                    };

                    let lhs_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![m_dim.clone(), k_dim.clone()],
                    );
                    let rhs_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![k_dim.clone(), n_dim.clone()],
                    );
                    let accum_2d_ty = RankedMemrefType::get(
                        ctx,
                        state.elem_ty,
                        vec![m_dim.clone(), n_dim.clone()],
                    );

                    let dyn_value = |p: &SliceParam| match p {
                        SliceParam::Dynamic(v) => Some(*v),
                        SliceParam::Static(_) => None,
                    };

                    let mut lhs_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&lhs_m_dim) {
                        lhs_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&lhs_k_dim) {
                        lhs_2d_dyn.push(v);
                    }

                    let mut rhs_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&rhs_k_dim) {
                        rhs_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&rhs_n_dim) {
                        rhs_2d_dyn.push(v);
                    }

                    let mut accum_2d_dyn = Vec::new();
                    if let Some(v) = dyn_value(&lhs_m_dim) {
                        accum_2d_dyn.push(v);
                    }
                    if let Some(v) = dyn_value(&rhs_n_dim) {
                        accum_2d_dyn.push(v);
                    }

                    let lhs_2d = MemrefReshapeOp::new(
                        ctx,
                        lhs_subview.get_result(ctx),
                        lhs_2d_dyn,
                        lhs_2d_ty,
                    );
                    rewriter.append_op(ctx, &lhs_2d);

                    let rhs_2d = MemrefReshapeOp::new(
                        ctx,
                        rhs_subview.get_result(ctx),
                        rhs_2d_dyn,
                        rhs_2d_ty,
                    );
                    rewriter.append_op(ctx, &rhs_2d);

                    let accum_2d = MemrefReshapeOp::new(
                        ctx,
                        accum_subview.get_result(ctx),
                        accum_2d_dyn,
                        accum_2d_ty,
                    );
                    rewriter.append_op(ctx, &accum_2d);

                    let matmul = MemrefMatMulOp::new(
                        ctx,
                        lhs_2d.get_result(ctx),
                        rhs_2d.get_result(ctx),
                        accum_2d.get_result(ctx),
                    );
                    rewriter.append_operation(ctx, matmul.get_operation());
                },
                &mut state,
            )
        };

        rewriter.append_op(ctx, &ndfor);
        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![accum]);
        Ok(())
    }
}

/// Update a [FuncOp]'s type signature and entry block argument types,
/// converting any tensor types to their memref equivalents.
pub fn lower_func_op_to_llvm(func_op: &FuncOp, ctx: &mut Context) -> Result<()> {
    // update the function type to convert any tensor types in the signature to memref types.
    let func_ty = func_op.get_type(ctx);
    let res_ty = func_ty.deref(ctx).result_type();
    let res_ty = memref::to_memref_type(res_ty, ctx)?;
    let arg_tys = func_ty.deref(ctx).arg_types();
    let arg_tys = arg_tys
        .iter()
        .map(|arg_ty| memref::to_memref_type(*arg_ty, ctx))
        .collect::<Result<Vec<_>>>()?;
    let new_func_ty = pliron_llvm::types::FuncType::get(ctx, res_ty, arg_tys, false);
    func_op.set_attr_llvm_func_type(ctx, TypeAttr::new(new_func_ty.into()));

    // Update all arguments in the entry block to use the new memref types.
    let entry_block = func_op
        .get_entry_block(ctx)
        .expect("FuncOp must have an entry block");

    let args = entry_block.deref(ctx).arguments().collect::<Vec<_>>();
    for arg in args {
        let arg_ty = memref::to_memref_type(arg.get_type(ctx), ctx)?;
        arg.set_type(ctx, arg_ty);
    }

    Ok(())
}

/// Allow [FuncOp] to participate in bufferization. `FuncOp` itself has no operands
/// or results, so the only thing of interest here is [Self::is_writable] on its
/// (entry-block) arguments (the function's parameters).
///
/// Function arguments are writable by default. It is up to the caller (i.e. whatever
/// bufferizes a call to this function) to insert a copy if it still needs the
/// original tensor after the call.
#[op_interface_impl]
impl BufferizableOpInterface for FuncOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, _ctx: &Context) -> Vec<Alias> {
        vec![]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn is_writable(&self, _ctx: &Context, _value: Value) -> bool {
        true
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        _rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        lower_func_op_to_llvm(self, ctx)
    }
}

#[op_interface_impl]
impl BufferizableOpInterface for TensorExtractSliceOp {
    fn operand_bufferizes_to_memory_read(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(0) == opd
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        let operand = self.get_operation().deref(ctx).get_operand_as_use(0);
        vec![Alias {
            operand,
            result: self.get_result(ctx),
            kind: AliasKind::Must,
            relation: BufferRelation::Contains,
        }]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let subview = MemrefSubviewOp::new(
            ctx,
            self.source(ctx),
            self.slice_offsets(ctx),
            self.slice_sizes(ctx),
            self.slice_steps(ctx),
        );
        rewriter.append_op(ctx, &subview);
        rewriter.replace_operation(ctx, self.get_operation(), subview.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl BufferizableOpInterface for TensorInsertSliceOp {
    fn operand_bufferizes_to_memory_read(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        true
    }

    fn operand_bufferizes_to_memory_write(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(1) == opd
    }

    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        let operand = self.get_operation().deref(ctx).get_operand_as_use(1);
        vec![Alias {
            operand,
            result: self.get_result(ctx),
            kind: AliasKind::Must,
            relation: BufferRelation::Equivalent,
        }]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let destination = self.destination(ctx);
        let view = MemrefSubviewOp::new(
            ctx,
            destination,
            self.slice_offsets(ctx),
            self.slice_sizes(ctx),
            self.slice_steps(ctx),
        );
        rewriter.append_op(ctx, &view);

        let copy_source = MemrefCopyOp::new(ctx, view.get_result(ctx), self.source(ctx));
        rewriter.append_op(ctx, &copy_source);

        rewriter.replace_operation_with_values(ctx, self.get_operation(), vec![destination]);
        Ok(())
    }
}

#[op_interface_impl]
impl BufferizableOpInterface for TensorReshapeOp {
    fn operand_bufferizes_to_memory_read(&self, ctx: &Context, opd: Use<Value>) -> bool {
        self.get_operation().deref(ctx).get_operand_as_use(0) == opd
    }

    fn operand_bufferizes_to_memory_write(&self, _ctx: &Context, _opd: Use<Value>) -> bool {
        false
    }

    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias> {
        let operand = self.get_operation().deref(ctx).get_operand_as_use(0);
        vec![Alias {
            operand,
            result: self.get_result(ctx),
            kind: AliasKind::Must,
            relation: BufferRelation::Equivalent,
        }]
    }

    fn get_dynamic_dimensions(&self, _ctx: &Context, _opd: Use<Value>) -> Option<Vec<Value>> {
        None
    }

    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _bufferizer_state: &mut BufferizerState,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_ty = tensor_type_to_memref_type(self.get_result(ctx).get_type(ctx), ctx)?;
        let memref_reshape = MemrefReshapeOp::new(
            ctx,
            self.get_source(ctx),
            TensorReshapeOp::get_dynamic_dimensions(self, ctx),
            result_ty,
        );
        rewriter.append_op(ctx, &memref_reshape);
        rewriter.replace_operation(ctx, self.get_operation(), memref_reshape.get_operation());
        Ok(())
    }
}
