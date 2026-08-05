// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor ops and related functionality.

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, AtLeastNOpdsInterface, NOpdsInterface,
        NRegionsInterface, NResultsInterface, OneRegionInterface, OneResultInterface,
        OperandNOfType, OperandSegmentInterface, ResultNOfType, SingleBlockRegionInterface,
    },
    combine::{
        Parser,
        parser::char::{self, spaces},
    },
    common_traits::Verify,
    context::Context,
    derive::pliron_op,
    identifier::Identifier,
    irbuild::{
        inserter::{BlockInsertionPoint, IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
    },
    irfmt::{
        parsers::{
            delimited_list_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser, type_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{self, IntoParseResult, Parsable},
    printable::{self, ListSeparator, Printable},
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast},
    value::Value,
    verify_err, verify_error,
};

use pliron_common_dialects::{cf::op_interfaces::YieldingRegions, index::types::IndexType};

use crate::memref::{
    op_interfaces::{CompatibleShapesOp, GenerateOpInterface},
    ops::{SliceParam, YieldOp},
    type_interfaces::{MultiDimensionalType, ShapedType},
};

use super::{
    attributes::{SliceParamAttr, SliceParamsAttr},
    op_interfaces::ElementWiseBinaryTensorOpInterface,
    types::RankedTensorType,
};

/// Op to generate a tensor by applying a function to generate the value at each index.
/// See MLIR's [GenerateOp](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorgenerate-tensorgenerateop).
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The generated tensor of the specified type. |
///
/// ### Regions
///   - A single region containing the body that computes the values of the tensor.
///   The region takes as many arguments as the rank of the result tensor type,
///   each representing an index along the corresponding dimension. The body should
///   yield a single value that matches the element type of the tensor.
#[pliron_op(
    name = "tensor.generate",
    format = "operands(CharSpace(`,`)) ` : ` type($0) region($0)",
    interfaces = [
        SingleBlockRegionInterface,
        OneRegionInterface,
        NRegionsInterface<1>,
        OneResultInterface,
        NResultsInterface<1>,
        YieldingRegions<YieldOp>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<IndexType>,
    ],
)]
pub struct GenerateOp;

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpVerifyErr {
    #[error(
        "GenerateOp number of operands {expected} does not match number of dynamic dimensions {got}"
    )]
    NumOperandsMismatch { expected: usize, got: usize },
}

impl Verify for GenerateOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let result_shape = self.get_generated_shape(ctx);
        let num_dynamic_dims = result_shape.num_dynamic_dimensions();

        let dynamic_dim_operands = self
            .get_operation()
            .deref(ctx)
            .operands()
            .collect::<Vec<_>>();
        let num_operands = dynamic_dim_operands.len();
        if num_operands != num_dynamic_dims {
            return verify_err!(
                loc,
                GenerateOpVerifyErr::NumOperandsMismatch {
                    expected: num_dynamic_dims,
                    got: num_operands
                }
            );
        }
        Ok(())
    }
}

impl GenerateOpInterface for GenerateOp {
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let result_ty = self.result_type(ctx).deref(ctx);
        Ref::map(result_ty, |result_ty| {
            type_cast::<dyn ShapedType>(result_ty).expect("The result type must be a shaped type")
        })
    }
}

impl GenerateOp {
    /// Creates a new dynamically sized tensor.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        dynamic_dimensions: Vec<Value>,
        result_type: TypedHandle<RankedTensorType>,
        body_builder: fn(
            ctx: &mut Context,
            state: State,
            inserter: &mut dyn Inserter,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: State,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            dynamic_dimensions,
            vec![],
            1,
        );
        let opop = GenerateOp { op };
        let rank = result_type.deref(ctx).rank();

        // Create the initializer region.
        let index_ty = IndexType::get(ctx);
        let region = opop.get_region(ctx);
        let op_inserter = &mut IRInserter::<DummyListener>::default();
        let entry_block = op_inserter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionStart(region),
            Some("entry".try_into().unwrap()),
            vec![index_ty.into(); rank],
        );
        // Build the body.
        let indices = entry_block.deref(ctx).arguments().collect();
        let yield_value = body_builder(ctx, body_builder_state, op_inserter, indices);
        let yield_op = YieldOp::new(ctx, yield_value);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(opop.get_exit(ctx, 0)));
        op_inserter.append_op(ctx, &yield_op);

        opop
    }

    /// Get the dynamic dimension operands of this op.
    pub fn dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

/// Extract an element from a tensor at the given indices.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `tensor` | The tensor to extract from. |
/// | `indices` | One [Index](IndexType) operand per dimension, indicating the index to extract along that dimension. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The extracted element, with the same type as the element type of the operand tensor. |
#[pliron_op(
    name = "tensor.extract",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        OperandSegmentInterface,
        OperandNOfType<0, RankedTensorType>
    ],
)]
pub struct ExtractOp;

impl Printable for ExtractOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let tensor = self.get_tensor_operand(ctx);
        let indices = self.get_index_operands(ctx);
        write!(
            f,
            "{} {}[{}] : {}",
            Self::get_opid_static(),
            tensor.disp(ctx),
            iter_with_sep(
                indices.iter(),
                pliron::printable::ListSeparator::CharSpace(',')
            )
            .disp(ctx),
            self.result_type(ctx).disp(ctx)
        )
    }
}

impl Parsable for ExtractOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (memref, indices, res_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', ssa_opd_parser()),
            spaced(char::string(":")).with(type_parser()),
        );

        let ((memref, indices, res_ty), _) = (memref, indices, res_ty)
            .parse_stream(state_stream)
            .into_result()?;

        let op = ExtractOp::new(state_stream.state.ctx, res_ty, memref, indices);

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

impl ExtractOp {
    /// Create a new ExtractOp with the given operand and result type.
    pub fn new(ctx: &mut Context, res_ty: TypeHandle, tensor: Value, indices: Vec<Value>) -> Self {
        let (operands, operand_segments) = Self::compute_segment_sizes(vec![vec![tensor], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![res_ty],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, operand_segments);
        op
    }

    /// Get the operand representing the tensor to extract from.
    pub fn get_tensor_operand(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the operands representing the indices to extract at.
    pub fn get_index_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractOpVerifyErr {
    #[error("The result type of ExtractOp must match the element type of the operand tensor")]
    ResultTypeMismatch,
    #[error("The number of operands must match the rank of the operand tensor")]
    NumOperandsMismatch { expected: usize, got: usize },
    #[error("All operands except the first one must be of IndexType")]
    NonIndexOperand { index: usize },
}

impl Verify for ExtractOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let mut operand_tys = op_ref.operands().map(|opd| opd.get_type(ctx));
        let tensor_operand_ty = operand_tys.next().expect("Must have at least one operand");
        let tensor_operand_ty_ref = tensor_operand_ty.deref(ctx);
        let ranked_tensor_ty = tensor_operand_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("The first operand must be a ranked tensor type");
        let element_ty = ranked_tensor_ty.element_type();
        let result_ty = self.result_type(ctx);
        if result_ty != element_ty {
            return verify_err!(loc, ExtractOpVerifyErr::ResultTypeMismatch);
        }
        let expected_num_indices = ranked_tensor_ty.rank();
        let mut num_indices = 0;
        for (i, index_ty) in operand_tys.enumerate() {
            let index_ty_ref = index_ty.deref(ctx);
            if !index_ty_ref.is::<IndexType>() {
                return verify_err!(loc, ExtractOpVerifyErr::NonIndexOperand { index: i });
            }
            num_indices += 1;
        }
        if num_indices != expected_num_indices {
            return verify_err!(
                loc,
                ExtractOpVerifyErr::NumOperandsMismatch {
                    expected: expected_num_indices,
                    got: num_indices
                }
            );
        }
        Ok(())
    }
}

/// Add two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.add",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct AddOp;

impl AddOp {
    /// Create a new AddOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Subtract two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.sub",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct SubOp;

impl SubOp {
    /// Create a new SubOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Multiply two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.mul",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct MulOp;

impl MulOp {
    /// Create a new MulOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Divide two tensors.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | The left-hand side tensor. |
/// | `rhs` | The right-hand side tensor. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor, with same shape as the operands. |
#[pliron_op(
    name = "tensor.div",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        ElementWiseBinaryTensorOpInterface,
        NResultsInterface<1>,
        NOpdsInterface<2>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
        CompatibleShapesOp<RankedTensorType>,
    ],
    verifier = "succ"
)]
pub struct DivOp;

impl DivOp {
    /// Create a new DivOp with the given operands and result type.
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![lhs.get_type(ctx)],
            vec![lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Matrix multiplication with accumulation for 2D tensors.
/// Computes `lhs * rhs + accum`.
/// `lhs` has shape [M, K], `rhs` has shape [K, N], `accum` has shape [M, N],
/// and the result has shape [M, N].
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | Left-hand side 2D tensor of shape [M, K]. |
/// | `rhs` | Right-hand side 2D tensor of shape [K, N]. |
/// | `accum` | Accumulator 2D tensor of shape [M, N]. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The product tensor of shape [M, N]. |
#[pliron_op(
    name = "tensor.matmul",
    format = "$0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        NOpdsInterface<3>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
    ],
)]
pub struct MatMulOp;

#[derive(Debug, thiserror::Error)]
pub enum MatMulOpVerifyErr {
    #[error("MatMulOp operands must be 2D ranked tensors")]
    OperandNot2DTensor,
    #[error("MatMulOp result must be a 2D ranked tensor")]
    ResultNot2DTensor,
    #[error("MatMulOp lhs inner dimension K ({lhs_k}) must match rhs outer dimension K ({rhs_k})")]
    InnerDimMismatch { lhs_k: usize, rhs_k: usize },
    #[error("MatMulOp result dimension {dim} (={result_d}) does not match expected {expected}")]
    ResultDimMismatch {
        dim: usize,
        result_d: usize,
        expected: usize,
    },
    #[error("MatMulOp operands and result must have the same element type")]
    ElementTypeMismatch,
}

impl Verify for MatMulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let accum = op_ref.get_operand(2);
        let result = op_ref.get_result(0);

        let lhs_ty_deref = lhs.get_type(ctx);
        let rhs_ty_deref = rhs.get_type(ctx);
        let accum_ty_deref = accum.get_type(ctx);
        let result_ty_deref = result.get_type(ctx);

        let lhs_binding = lhs_ty_deref.deref(ctx);
        let lhs_ty = lhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DTensor))?;
        let rhs_binding = rhs_ty_deref.deref(ctx);
        let rhs_ty = rhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DTensor))?;
        let accum_binding = accum_ty_deref.deref(ctx);
        let accum_ty = accum_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DTensor))?;
        let result_binding = result_ty_deref.deref(ctx);
        let result_ty = result_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::ResultNot2DTensor))?;

        if lhs_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DTensor);
        }
        if rhs_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DTensor);
        }
        if accum_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DTensor);
        }
        if result_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::ResultNot2DTensor);
        }

        let elem_ty = lhs_ty.element_type();
        if rhs_ty.element_type() != elem_ty
            || accum_ty.element_type() != elem_ty
            || result_ty.element_type() != elem_ty
        {
            return verify_err!(loc, MatMulOpVerifyErr::ElementTypeMismatch);
        }

        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let accum_shape = accum_ty.shape();
        let result_shape = result_ty.shape();

        // K: lhs[1] must match rhs[0]
        if let (Dimension::Static(lhs_k), Dimension::Static(rhs_k)) = (&lhs_shape[1], &rhs_shape[0])
            && lhs_k != rhs_k
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::InnerDimMismatch {
                    lhs_k: *lhs_k,
                    rhs_k: *rhs_k
                }
            );
        }
        // M: lhs[0] must match accum[0]
        if let (Dimension::Static(lhs_m), Dimension::Static(accum_m)) =
            (&lhs_shape[0], &accum_shape[0])
            && lhs_m != accum_m
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 0,
                    result_d: *accum_m,
                    expected: *lhs_m
                }
            );
        }
        // N: rhs[1] must match accum[1]
        if let (Dimension::Static(rhs_n), Dimension::Static(accum_n)) =
            (&rhs_shape[1], &accum_shape[1])
            && rhs_n != accum_n
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 1,
                    result_d: *accum_n,
                    expected: *rhs_n
                }
            );
        }

        // Result shape must match accum shape.
        if let (Dimension::Static(accum_m), Dimension::Static(result_m)) =
            (&accum_shape[0], &result_shape[0])
            && accum_m != result_m
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 0,
                    result_d: *result_m,
                    expected: *accum_m
                }
            );
        }
        if let (Dimension::Static(accum_n), Dimension::Static(result_n)) =
            (&accum_shape[1], &result_shape[1])
            && accum_n != result_n
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 1,
                    result_d: *result_n,
                    expected: *accum_n
                }
            );
        }

        // Kept for diagnostics consistency when result shape carries stricter static info.
        // M: lhs[0] must match result[0]
        if let (Dimension::Static(lhs_m), Dimension::Static(result_m)) =
            (&lhs_shape[0], &result_shape[0])
            && lhs_m != result_m
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 0,
                    result_d: *result_m,
                    expected: *lhs_m
                }
            );
        }
        // N: rhs[1] must match result[1]
        if let (Dimension::Static(rhs_n), Dimension::Static(result_n)) =
            (&rhs_shape[1], &result_shape[1])
            && rhs_n != result_n
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 1,
                    result_d: *result_n,
                    expected: *rhs_n
                }
            );
        }

        Ok(())
    }
}

impl MatMulOp {
    /// Create a new [MatMulOp]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, accum: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![accum.get_type(ctx)],
            vec![lhs, rhs, accum],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Batched matrix multiplication with accumulation over tensors of rank >= 2.
///
/// For operands of shape `[..., M, K]`, `[..., K, N]`, and `[..., M, N]`, this
/// computes batched matrix products for each batch index in `...`, producing
/// `lhs * rhs + accum` with result shape `[..., M, N]`.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `lhs` | Left-hand side ranked tensor of shape `[..., M, K]`. |
/// | `rhs` | Right-hand side ranked tensor of shape `[..., K, N]`. |
/// | `accum` | Accumulator ranked tensor of shape `[..., M, N]`. |
///
/// ## Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | Ranked tensor with shape `[..., M, N]`. |
#[pliron_op(
    name = "tensor.batch_matmul",
    format = "$0 `, ` $1 `, ` $2 ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        NOpdsInterface<3>,
        AllResultsOfType<RankedTensorType>,
        AllOperandsOfType<RankedTensorType>,
    ],
)]
pub struct BatchMatMulOp;

#[derive(Debug, thiserror::Error)]
pub enum BatchMatMulOpVerifyErr {
    #[error("BatchMatMulOp operands and result must be ranked tensors with rank >= 2")]
    InvalidRank,
    #[error("BatchMatMulOp lhs and rhs must have the same rank")]
    OperandRankMismatch,
    #[error("BatchMatMulOp operands and result must have the same element type")]
    ElementTypeMismatch,
    #[error("BatchMatMulOp batch dim {dim} mismatch: lhs={lhs_d}, rhs={rhs_d}")]
    BatchDimMismatch {
        dim: usize,
        lhs_d: usize,
        rhs_d: usize,
    },
    #[error("BatchMatMulOp lhs K dim ({lhs_k}) must match rhs K dim ({rhs_k})")]
    InnerDimMismatch { lhs_k: usize, rhs_k: usize },
    #[error("BatchMatMulOp result dim {dim} (={result_d}) does not match expected {expected}")]
    ResultDimMismatch {
        dim: usize,
        result_d: usize,
        expected: usize,
    },
}

impl Verify for BatchMatMulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;

        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let lhs = op_ref.get_operand(0);
        let rhs = op_ref.get_operand(1);
        let accum = op_ref.get_operand(2);
        let result = op_ref.get_result(0);

        let lhs_ty_deref = lhs.get_type(ctx);
        let rhs_ty_deref = rhs.get_type(ctx);
        let accum_ty_deref = accum.get_type(ctx);
        let result_ty_deref = result.get_type(ctx);

        let lhs_binding = lhs_ty_deref.deref(ctx);
        let lhs_ty = lhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), BatchMatMulOpVerifyErr::InvalidRank))?;
        let rhs_binding = rhs_ty_deref.deref(ctx);
        let rhs_ty = rhs_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), BatchMatMulOpVerifyErr::InvalidRank))?;
        let accum_binding = accum_ty_deref.deref(ctx);
        let accum_ty = accum_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), BatchMatMulOpVerifyErr::InvalidRank))?;
        let result_binding = result_ty_deref.deref(ctx);
        let result_ty = result_binding
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), BatchMatMulOpVerifyErr::InvalidRank))?;

        if lhs_ty.rank() < 2 || rhs_ty.rank() < 2 || accum_ty.rank() < 2 || result_ty.rank() < 2 {
            return verify_err!(loc, BatchMatMulOpVerifyErr::InvalidRank);
        }
        if lhs_ty.rank() != rhs_ty.rank() {
            return verify_err!(loc, BatchMatMulOpVerifyErr::OperandRankMismatch);
        }
        if accum_ty.rank() != lhs_ty.rank() {
            return verify_err!(loc, BatchMatMulOpVerifyErr::OperandRankMismatch);
        }
        if result_ty.rank() != lhs_ty.rank() {
            return verify_err!(loc, BatchMatMulOpVerifyErr::InvalidRank);
        }

        let elem_ty = lhs_ty.element_type();
        if rhs_ty.element_type() != elem_ty
            || accum_ty.element_type() != elem_ty
            || result_ty.element_type() != elem_ty
        {
            return verify_err!(loc, BatchMatMulOpVerifyErr::ElementTypeMismatch);
        }

        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();
        let accum_shape = accum_ty.shape();
        let result_shape = result_ty.shape();
        let rank = lhs_ty.rank();

        for i in 0..(rank - 2) {
            if let (Dimension::Static(lhs_d), Dimension::Static(rhs_d)) =
                (&lhs_shape[i], &rhs_shape[i])
                && lhs_d != rhs_d
            {
                return verify_err!(
                    loc,
                    BatchMatMulOpVerifyErr::BatchDimMismatch {
                        dim: i,
                        lhs_d: *lhs_d,
                        rhs_d: *rhs_d
                    }
                );
            }
            if let (Dimension::Static(lhs_d), Dimension::Static(accum_d)) =
                (&lhs_shape[i], &accum_shape[i])
                && lhs_d != accum_d
            {
                return verify_err!(
                    loc,
                    BatchMatMulOpVerifyErr::ResultDimMismatch {
                        dim: i,
                        result_d: *accum_d,
                        expected: *lhs_d
                    }
                );
            }
            if let (Dimension::Static(lhs_d), Dimension::Static(res_d)) =
                (&lhs_shape[i], &result_shape[i])
                && lhs_d != res_d
            {
                return verify_err!(
                    loc,
                    BatchMatMulOpVerifyErr::ResultDimMismatch {
                        dim: i,
                        result_d: *res_d,
                        expected: *lhs_d
                    }
                );
            }
        }

        // K: lhs[-1] must match rhs[-2]
        if let (Dimension::Static(lhs_k), Dimension::Static(rhs_k)) =
            (&lhs_shape[rank - 1], &rhs_shape[rank - 2])
            && lhs_k != rhs_k
        {
            return verify_err!(
                loc,
                BatchMatMulOpVerifyErr::InnerDimMismatch {
                    lhs_k: *lhs_k,
                    rhs_k: *rhs_k
                }
            );
        }

        // M: lhs[-2] must match accum[-2]
        if let (Dimension::Static(lhs_m), Dimension::Static(accum_m)) =
            (&lhs_shape[rank - 2], &accum_shape[rank - 2])
            && lhs_m != accum_m
        {
            return verify_err!(
                loc,
                BatchMatMulOpVerifyErr::ResultDimMismatch {
                    dim: rank - 2,
                    result_d: *accum_m,
                    expected: *lhs_m
                }
            );
        }

        // N: rhs[-1] must match accum[-1]
        if let (Dimension::Static(rhs_n), Dimension::Static(accum_n)) =
            (&rhs_shape[rank - 1], &accum_shape[rank - 1])
            && rhs_n != accum_n
        {
            return verify_err!(
                loc,
                BatchMatMulOpVerifyErr::ResultDimMismatch {
                    dim: rank - 1,
                    result_d: *accum_n,
                    expected: *rhs_n
                }
            );
        }

        // Result shape must match accum shape.
        if let (Dimension::Static(accum_m), Dimension::Static(result_m)) =
            (&accum_shape[rank - 2], &result_shape[rank - 2])
            && accum_m != result_m
        {
            return verify_err!(
                loc,
                BatchMatMulOpVerifyErr::ResultDimMismatch {
                    dim: rank - 2,
                    result_d: *result_m,
                    expected: *accum_m
                }
            );
        }

        if let (Dimension::Static(accum_n), Dimension::Static(result_n)) =
            (&accum_shape[rank - 1], &result_shape[rank - 1])
            && accum_n != result_n
        {
            return verify_err!(
                loc,
                BatchMatMulOpVerifyErr::ResultDimMismatch {
                    dim: rank - 1,
                    result_d: *result_n,
                    expected: *accum_n
                }
            );
        }

        Ok(())
    }
}

impl BatchMatMulOp {
    /// Create a new [BatchMatMulOp]
    pub fn new(ctx: &mut Context, lhs: Value, rhs: Value, accum: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![accum.get_type(ctx)],
            vec![lhs, rhs, accum],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Extract a narrow slice from a tensor.
///
/// This operation extracts a contiguous slice from a tensor, specified by
/// offsets, sizes, and strides per dimension. Each parameter can be static
/// or dynamic. The result has the same rank as the source tensor.
///
/// Similar to (but not the same as) MLIR's
/// [tensor.extract_slice](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorextract_slice-tensorextractsliceop)
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `source` | The source tensor to extract from (ranked tensor). |
/// | `dynamic_offsets` | Zero or more [Index](IndexType) operands for dynamic offsets. |
/// | `dynamic_sizes` | Zero or more [Index](IndexType) operands for dynamic sizes. |
/// | `dynamic_steps` | Zero or more [Index](IndexType) operands for dynamic steps. |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The extracted slice, a ranked tensor with the same rank as the source. |
#[pliron_op(
    name = "tensor.extract_slice",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        AllResultsOfType<RankedTensorType>,
        OperandNOfType<0, RankedTensorType>,
    ],
    attributes = (tensor_slice_params: SliceParamsAttr)
)]
pub struct ExtractSliceOp;

impl Printable for ExtractSliceOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let source = self.source(ctx);
        write!(f, "{} {}", Self::get_opid_static(), source.disp(ctx))?;

        let offsets = self.slice_offsets(ctx);
        let sizes = self.slice_sizes(ctx);
        let steps = self.slice_steps(ctx);

        write!(
            f,
            " [{}]",
            list_with_sep(&offsets, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&sizes, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&steps, ListSeparator::CharSpace(',')).disp(ctx)
        )?;

        write!(f, " : {}", self.result_type(ctx).disp(ctx))?;

        Ok(())
    }
}

impl Parsable for ExtractSliceOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (source, offsets, sizes, steps, result_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())),
            spaced(char::string(":")).with(TypedHandle::<RankedTensorType>::parser(())),
        );

        let ((source, offsets, sizes, steps, result_ty), _) =
            (source, offsets, sizes, steps, result_ty)
                .parse_stream(state_stream)
                .into_result()?;

        let op = ExtractSliceOp::new_with_result_type(
            state_stream.state.ctx,
            source,
            offsets,
            sizes,
            steps,
            result_ty,
        );

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ExtractSliceOpVerifyErr {
    #[error(
        "The result type of ExtractSliceOp must be a RankedTensorType with same rank as source"
    )]
    ResultTypeMismatch,
    #[error(
        "ExtractSliceOp: All dynamic operands must be of IndexType, but operand {index} is {ty}"
    )]
    NonIndexOperand { index: usize, ty: String },
    #[error(
        "ExtractSliceOp: Number of dynamic operands ({got}) does not match number of dynamic parameters ({expected})"
    )]
    NumDynamicOperandsMismatch { expected: usize, got: usize },
    #[error(
        "ExtractSliceOp: Number of offsets ({got}) does not match rank of source tensor ({expected})"
    )]
    NumOffsetsMismatch { expected: usize, got: usize },
    #[error(
        "ExtractSliceOp: Number of sizes ({got}) does not match rank of source tensor ({expected})"
    )]
    NumSizesMismatch { expected: usize, got: usize },
    #[error(
        "ExtractSliceOp: Number of steps ({got}) does not match rank of source tensor ({expected})"
    )]
    NumStepsMismatch { expected: usize, got: usize },
    #[error("ExtractSliceOp: Missing tensor.slice_params attribute")]
    MissingSliceParamsAttr,
    #[error("ExtractSliceOp: Static step values must be non-zero (got 0 at dimension {dim})")]
    InvalidStaticStep { dim: usize },
}

impl Verify for ExtractSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let mut operands = op_ref.operands();

        let source_operand = operands
            .next()
            .expect("ExtractSliceOp must have at least one operand");

        let source_ty_ptr = source_operand.get_type(ctx);
        let source_ty_ref = source_ty_ptr.deref(ctx);
        let source_ty = source_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("ExtractSliceOp source must be a RankedTensorType");

        let rank = source_ty.rank();

        // Get result type and verify it's a ranked tensor with same rank as source
        let result_ty_ptr = self.result_type(ctx);
        let result_ty_ref = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("ExtractSliceOp result must be a RankedTensorType");
        if result_ty.rank() != rank {
            return verify_err!(loc, ExtractSliceOpVerifyErr::ResultTypeMismatch);
        }

        // Get the slice params attribute
        let slice_params = self.get_attr_tensor_slice_params(ctx).ok_or_else(|| {
            verify_error!(loc.clone(), ExtractSliceOpVerifyErr::MissingSliceParamsAttr)
        })?;

        // Verify that the number of offsets/sizes/steps match the rank
        if slice_params.offsets.len() != rank {
            return verify_err!(
                loc,
                ExtractSliceOpVerifyErr::NumOffsetsMismatch {
                    expected: rank,
                    got: slice_params.offsets.len()
                }
            );
        }
        if slice_params.sizes.len() != rank {
            return verify_err!(
                loc,
                ExtractSliceOpVerifyErr::NumSizesMismatch {
                    expected: rank,
                    got: slice_params.sizes.len()
                }
            );
        }
        if slice_params.steps.len() != rank {
            return verify_err!(
                loc,
                ExtractSliceOpVerifyErr::NumStepsMismatch {
                    expected: rank,
                    got: slice_params.steps.len()
                }
            );
        }

        // Verify all step values are non-zero
        for (dim, step) in slice_params.steps.iter().enumerate() {
            if let SliceParamAttr::Static(0) = step {
                return verify_err!(loc, ExtractSliceOpVerifyErr::InvalidStaticStep { dim });
            }
        }

        // Count dynamic parameters
        let num_dynamic_offsets = slice_params
            .offsets
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_sizes = slice_params
            .sizes
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_steps = slice_params
            .steps
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();

        let total_dynamic = num_dynamic_offsets + num_dynamic_sizes + num_dynamic_steps;
        let remaining_operands: Vec<_> = operands.collect();

        // Verify that all remaining operands are Index type
        for (i, opd) in remaining_operands.iter().enumerate() {
            let opd_ty = opd.get_type(ctx);
            let opd_ty_ref = opd_ty.deref(ctx);
            if !opd_ty_ref.is::<IndexType>() {
                let ty_name = format!("{:?}", opd_ty_ref);
                return verify_err!(
                    loc,
                    ExtractSliceOpVerifyErr::NonIndexOperand {
                        index: i + 1,
                        ty: ty_name
                    }
                );
            }
        }

        // Verify the count of dynamic operands matches
        if remaining_operands.len() != total_dynamic {
            return verify_err!(
                loc,
                ExtractSliceOpVerifyErr::NumDynamicOperandsMismatch {
                    expected: total_dynamic,
                    got: remaining_operands.len()
                }
            );
        }

        Ok(())
    }
}

impl ExtractSliceOp {
    /// Create a new ExtractSliceOp with the given parameters.
    ///
    /// # Arguments
    /// * `ctx` - The context
    /// * `source` - The source tensor to slice from
    /// * `offsets` - The offset for each dimension (static or dynamic)
    /// * `sizes` - The size for each dimension (static or dynamic)
    /// * `steps` - The step for each dimension (static or dynamic)
    ///
    /// The result type is inferred from the source element type and `sizes`:
    /// static sizes map to static dimensions, and dynamic sizes map to dynamic dimensions.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
    ) -> Self {
        let source_element_type = {
            let source_ty_ptr = source.get_type(ctx);
            let source_ty_ref = source_ty_ptr.deref(ctx);
            let source_ty = source_ty_ref
                .downcast_ref::<RankedTensorType>()
                .expect("ExtractSliceOp source must be a RankedTensorType");
            source_ty.element_type()
        };

        let result_shape = sizes
            .iter()
            .map(|s| match s {
                SliceParam::Static(v) => crate::memref::type_interfaces::Dimension::Static(*v),
                SliceParam::Dynamic(_) => crate::memref::type_interfaces::Dimension::Dynamic,
            })
            .collect();
        let result_type = RankedTensorType::get(ctx, source_element_type, result_shape);

        Self::new_with_result_type(ctx, source, offsets, sizes, steps, result_type)
    }

    /// Create a new ExtractSliceOp with an explicitly provided result type.
    ///
    /// This is useful when constructing from parsed text IR where the result type is
    /// syntactically present.
    pub fn new_with_result_type(
        ctx: &mut Context,
        source: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        let mut operands = vec![source];
        let mut offset_attrs = Vec::new();
        let mut size_attrs = Vec::new();
        let mut step_attrs = Vec::new();

        // Process offsets
        for offset in offsets {
            match offset {
                SliceParam::Static(val) => {
                    offset_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    offset_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        // Process sizes
        for size in sizes {
            match size {
                SliceParam::Static(val) => {
                    size_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    size_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        // Process steps
        for step in steps {
            match step {
                SliceParam::Static(val) => {
                    step_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    step_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            operands,
            vec![],
            0,
        );

        let op = Self { op };
        let slice_params = SliceParamsAttr {
            offsets: offset_attrs,
            sizes: size_attrs,
            steps: step_attrs,
        };
        op.set_attr_tensor_slice_params(ctx, slice_params);
        op
    }

    /// Get the source tensor operand.
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    fn slice_attr_to_params(&self, ctx: &Context, attrs: &[SliceParamAttr]) -> Vec<SliceParam> {
        let op_ref = self.get_operation().deref(ctx);
        attrs
            .iter()
            .map(|a| match a {
                SliceParamAttr::Static(val) => SliceParam::Static(*val),
                SliceParamAttr::OperandIdx(idx) => SliceParam::Dynamic(op_ref.get_operand(*idx)),
            })
            .collect()
    }

    /// Get the per-dimension offsets as [SliceParam] values.
    pub fn slice_offsets(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_tensor_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.offsets)
    }

    /// Get the per-dimension sizes as [SliceParam] values.
    pub fn slice_sizes(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_tensor_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.sizes)
    }

    /// Get the per-dimension steps as [SliceParam] values.
    pub fn slice_steps(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_tensor_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.steps)
    }

    /// Get all dynamic operands (excluding the source tensor).
    pub fn dynamic_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

/// Insert a tensor slice into another tensor of the same rank.
///
/// This operation inserts the source tensor into the destination tensor at the
/// slice defined by offsets, sizes, and strides per dimension. Each parameter
/// can be static or dynamic. The source, destination, and result must all have
/// the same rank.
///
/// Similar to MLIR's
/// [tensor.insert_slice](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorinsert_slice-tensorinsertsliceop),
/// but rank-altering insertion is intentionally not supported here.
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `source` | The source tensor to insert (ranked tensor). |
/// | `destination` | The destination tensor to insert into (ranked tensor). |
/// | `dynamic_offsets` | Zero or more [Index](IndexType) operands for dynamic offsets. |
/// | `dynamic_sizes` | Zero or more [Index](IndexType) operands for dynamic sizes. |
/// | `dynamic_steps` | Zero or more [Index](IndexType) operands for dynamic steps. |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The resulting tensor after insertion, with same shape as the destination. |
#[pliron_op(
    name = "tensor.insert_slice",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        AllResultsOfType<RankedTensorType>,
        AtLeastNOpdsInterface<2>,
        ResultNOfType<0, RankedTensorType>,
        OperandNOfType<0, RankedTensorType>,
        OperandNOfType<1, RankedTensorType>,
    ],
    attributes = (insert_slice_params: SliceParamsAttr)
)]
pub struct InsertSliceOp;

impl Printable for InsertSliceOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let source = self.source(ctx);
        let destination = self.destination(ctx);
        write!(
            f,
            "{} {} into {}",
            Self::get_opid_static(),
            source.disp(ctx),
            destination.disp(ctx)
        )?;

        let offsets = self.slice_offsets(ctx);
        let sizes = self.slice_sizes(ctx);
        let steps = self.slice_steps(ctx);

        write!(
            f,
            " [{}]",
            list_with_sep(&offsets, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&sizes, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&steps, ListSeparator::CharSpace(',')).disp(ctx)
        )?;

        write!(f, " : {}", self.result_type(ctx).disp(ctx))?;

        Ok(())
    }
}

impl Parsable for InsertSliceOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (source, destination, offsets, sizes, steps, result_ty) = (
            ssa_opd_parser().skip(spaced(char::string("into"))),
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())),
            spaced(char::string(":")).with(TypedHandle::<RankedTensorType>::parser(())),
        );

        let ((source, destination, offsets, sizes, steps, result_ty), _) =
            (source, destination, offsets, sizes, steps, result_ty)
                .parse_stream(state_stream)
                .into_result()?;

        let op = InsertSliceOp::new_with_result_type(
            state_stream.state.ctx,
            source,
            destination,
            offsets,
            sizes,
            steps,
            result_ty,
        );

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum InsertSliceOpVerifyErr {
    #[error("InsertSliceOp source and destination tensors must have the same rank")]
    SourceDestinationRankMismatch,
    #[error("InsertSliceOp source, destination, and result element types must match")]
    ElementTypeMismatch,
    #[error("InsertSliceOp result type must match the destination tensor type")]
    ResultTypeMismatch,
    #[error(
        "InsertSliceOp: All dynamic operands must be of IndexType, but operand {index} is {ty}"
    )]
    NonIndexOperand { index: usize, ty: String },
    #[error(
        "InsertSliceOp: Number of dynamic operands ({got}) does not match number of dynamic parameters ({expected})"
    )]
    NumDynamicOperandsMismatch { expected: usize, got: usize },
    #[error(
        "InsertSliceOp: Number of offsets ({got}) does not match rank of source tensor ({expected})"
    )]
    NumOffsetsMismatch { expected: usize, got: usize },
    #[error(
        "InsertSliceOp: Number of sizes ({got}) does not match rank of source tensor ({expected})"
    )]
    NumSizesMismatch { expected: usize, got: usize },
    #[error(
        "InsertSliceOp: Number of steps ({got}) does not match rank of source tensor ({expected})"
    )]
    NumStepsMismatch { expected: usize, got: usize },
    #[error("InsertSliceOp: Missing tensor.slice_params attribute")]
    MissingSliceParamsAttr,
    #[error("InsertSliceOp: Static step values must be non-zero (got 0 at dimension {dim})")]
    InvalidStaticStep { dim: usize },
    #[error(
        "InsertSliceOp: Static size {size} at dimension {dim} does not match static source dimension {source_dim}"
    )]
    StaticSizeSourceMismatch {
        dim: usize,
        size: usize,
        source_dim: usize,
    },
}

impl Verify for InsertSliceOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let mut operands = op_ref.operands();

        let source_operand = operands
            .next()
            .expect("InsertSliceOp must have at least two operands");
        let destination_operand = operands
            .next()
            .expect("InsertSliceOp must have at least two operands");

        let source_ty_ptr = source_operand.get_type(ctx);
        let source_ty_ref = source_ty_ptr.deref(ctx);
        let source_ty = source_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("InsertSliceOp source must be a RankedTensorType");

        let destination_ty_ptr = destination_operand.get_type(ctx);
        let destination_ty_ref = destination_ty_ptr.deref(ctx);
        let destination_ty = destination_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("InsertSliceOp destination must be a RankedTensorType");

        let rank = source_ty.rank();
        if destination_ty.rank() != rank {
            return verify_err!(loc, InsertSliceOpVerifyErr::SourceDestinationRankMismatch);
        }

        if source_ty.element_type() != destination_ty.element_type() {
            return verify_err!(loc, InsertSliceOpVerifyErr::ElementTypeMismatch);
        }

        let result_ty_ptr = self.result_type(ctx);
        let result_ty_ref = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| {
                verify_error!(loc.clone(), InsertSliceOpVerifyErr::ResultTypeMismatch)
            })?;

        if result_ty.rank() != rank
            || result_ty.shape() != destination_ty.shape()
            || result_ty.element_type() != destination_ty.element_type()
        {
            return verify_err!(loc, InsertSliceOpVerifyErr::ResultTypeMismatch);
        }

        let slice_params = self.get_attr_insert_slice_params(ctx).ok_or_else(|| {
            verify_error!(loc.clone(), InsertSliceOpVerifyErr::MissingSliceParamsAttr)
        })?;

        if slice_params.offsets.len() != rank {
            return verify_err!(
                loc,
                InsertSliceOpVerifyErr::NumOffsetsMismatch {
                    expected: rank,
                    got: slice_params.offsets.len()
                }
            );
        }
        if slice_params.sizes.len() != rank {
            return verify_err!(
                loc,
                InsertSliceOpVerifyErr::NumSizesMismatch {
                    expected: rank,
                    got: slice_params.sizes.len()
                }
            );
        }
        if slice_params.steps.len() != rank {
            return verify_err!(
                loc,
                InsertSliceOpVerifyErr::NumStepsMismatch {
                    expected: rank,
                    got: slice_params.steps.len()
                }
            );
        }

        for (dim, step) in slice_params.steps.iter().enumerate() {
            if let SliceParamAttr::Static(0) = step {
                return verify_err!(loc, InsertSliceOpVerifyErr::InvalidStaticStep { dim });
            }
        }

        for (dim, (size, source_dim)) in slice_params
            .sizes
            .iter()
            .zip(source_ty.shape().iter())
            .enumerate()
        {
            if let (
                SliceParamAttr::Static(size),
                crate::memref::type_interfaces::Dimension::Static(source_dim),
            ) = (size, source_dim)
                && size != source_dim
            {
                return verify_err!(
                    loc,
                    InsertSliceOpVerifyErr::StaticSizeSourceMismatch {
                        dim,
                        size: *size,
                        source_dim: *source_dim
                    }
                );
            }
        }

        let num_dynamic_offsets = slice_params
            .offsets
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_sizes = slice_params
            .sizes
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_steps = slice_params
            .steps
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();

        let total_dynamic = num_dynamic_offsets + num_dynamic_sizes + num_dynamic_steps;
        let remaining_operands: Vec<_> = operands.collect();

        for (i, opd) in remaining_operands.iter().enumerate() {
            let opd_ty = opd.get_type(ctx);
            let opd_ty_ref = opd_ty.deref(ctx);
            if !opd_ty_ref.is::<IndexType>() {
                let ty_name = format!("{:?}", opd_ty_ref);
                return verify_err!(
                    loc,
                    InsertSliceOpVerifyErr::NonIndexOperand {
                        index: i + 2,
                        ty: ty_name
                    }
                );
            }
        }

        if remaining_operands.len() != total_dynamic {
            return verify_err!(
                loc,
                InsertSliceOpVerifyErr::NumDynamicOperandsMismatch {
                    expected: total_dynamic,
                    got: remaining_operands.len()
                }
            );
        }

        Ok(())
    }
}

impl InsertSliceOp {
    /// Create a new InsertSliceOp with the given parameters.
    ///
    /// The result type is inferred from the destination tensor type.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        destination: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
    ) -> Self {
        let result_type =
            TypedHandle::<RankedTensorType>::from_handle(destination.get_type(ctx), ctx)
                .expect("InsertSliceOp destination must be a RankedTensorType");

        Self::new_with_result_type(ctx, source, destination, offsets, sizes, steps, result_type)
    }

    /// Create a new InsertSliceOp with an explicitly provided result type.
    pub fn new_with_result_type(
        ctx: &mut Context,
        source: Value,
        destination: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        let mut operands = vec![source, destination];
        let mut offset_attrs = Vec::new();
        let mut size_attrs = Vec::new();
        let mut step_attrs = Vec::new();

        for offset in offsets {
            match offset {
                SliceParam::Static(val) => {
                    offset_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    offset_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        for size in sizes {
            match size {
                SliceParam::Static(val) => {
                    size_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    size_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        for step in steps {
            match step {
                SliceParam::Static(val) => {
                    step_attrs.push(SliceParamAttr::Static(val));
                }
                SliceParam::Dynamic(val) => {
                    step_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            operands,
            vec![],
            0,
        );

        let op = Self { op };
        let slice_params = SliceParamsAttr {
            offsets: offset_attrs,
            sizes: size_attrs,
            steps: step_attrs,
        };
        op.set_attr_insert_slice_params(ctx, slice_params);
        op
    }

    /// Get the source tensor operand.
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the destination tensor operand.
    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    fn slice_attr_to_params(&self, ctx: &Context, attrs: &[SliceParamAttr]) -> Vec<SliceParam> {
        let op_ref = self.get_operation().deref(ctx);
        attrs
            .iter()
            .map(|a| match a {
                SliceParamAttr::Static(val) => SliceParam::Static(*val),
                SliceParamAttr::OperandIdx(idx) => SliceParam::Dynamic(op_ref.get_operand(*idx)),
            })
            .collect()
    }

    /// Get the per-dimension offsets as [SliceParam] values.
    pub fn slice_offsets(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_insert_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.offsets)
    }

    /// Get the per-dimension sizes as [SliceParam] values.
    pub fn slice_sizes(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_insert_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.sizes)
    }

    /// Get the per-dimension steps as [SliceParam] values.
    pub fn slice_steps(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_insert_slice_params(ctx)
            .expect("Failed to get tensor slice params");
        self.slice_attr_to_params(ctx, &attrs.steps)
    }

    /// Get all dynamic operands (excluding the source and destination tensors).
    pub fn dynamic_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(2).collect()
    }
}

/// Reshape a tensor to a new shape.
/// The operation takes a source tensor and a set of dynamic dimension operands
/// (one per dynamic dimension in the result type), and produces a result tensor
/// with the specified shape. The total number of elements must be the same.
///
/// See MLIR's [tensor.reshape](https://mlir.llvm.org/docs/Dialects/TensorOps/#tensorreshape-tensorreshapeop).
/// Unlike MLIR, only ranked tensors are supported (no unranked tensors).
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `source` | The source ranked tensor to reshape. |
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension in the result type. |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The reshaped tensor with the new shape. |
#[pliron_op(
    name = "tensor.reshape",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        AtLeastNOpdsInterface<1>,
        AllResultsOfType<RankedTensorType>,
    ],
)]
pub struct ReshapeOp;

impl Printable for ReshapeOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let source = self.get_source(ctx);
        let dyn_dims = self.get_dynamic_dimensions(ctx);
        write!(f, "{} {}(", Self::get_opid_static(), source.disp(ctx))?;
        write!(
            f,
            "{}",
            iter_with_sep(dyn_dims.iter(), ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(f, ") : {}", self.result_type(ctx).disp(ctx))
    }
}

impl Parsable for ReshapeOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (source, dyn_dims, result_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('(', ')', ',', ssa_opd_parser()),
            spaced(char::string(":")).with(TypedHandle::<RankedTensorType>::parser(())),
        );

        let ((source, dyn_dims, result_ty), _) = (source, dyn_dims, result_ty)
            .parse_stream(state_stream)
            .into_result()?;

        let op = ReshapeOp::new(state_stream.state.ctx, source, dyn_dims, result_ty);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ReshapeOpVerifyErr {
    #[error("ReshapeOp source must be a RankedTensorType")]
    SourceNotRankedTensor,
    #[error("ReshapeOp source and result element types must match")]
    ElementTypeMismatch,
    #[error(
        "ReshapeOp: number of dynamic dimension operands ({got}) must match \
        number of dynamic dimensions in result type ({expected})"
    )]
    DynDimCountMismatch { expected: usize, got: usize },
    #[error("ReshapeOp: all dynamic dimension operands must be of IndexType")]
    DynDimNotIndex,
    #[error(
        "ReshapeOp: total element count of source ({src_count}) must match result ({result_count})"
    )]
    ElementCountMismatch {
        src_count: usize,
        result_count: usize,
    },
}

impl Verify for ReshapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;
        let loc = self.loc(ctx);
        let source_ty_ptr = self.get_source(ctx).get_type(ctx);
        let source_ty_ref = source_ty_ptr.deref(ctx);
        let source_ty = source_ty_ref
            .downcast_ref::<RankedTensorType>()
            .ok_or_else(|| verify_error!(loc.clone(), ReshapeOpVerifyErr::SourceNotRankedTensor))?;

        let result_ty_ptr = self.result_type(ctx);
        let result_ty_ref = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<RankedTensorType>()
            .expect("AllResultsOfType<RankedTensorType> ensures result is RankedTensorType");

        if source_ty.element_type() != result_ty.element_type() {
            return verify_err!(loc, ReshapeOpVerifyErr::ElementTypeMismatch);
        }

        let dyn_dims = self.get_dynamic_dimensions(ctx);
        let num_dynamic_in_result = result_ty
            .shape()
            .iter()
            .filter(|d| matches!(d, Dimension::Dynamic))
            .count();

        if dyn_dims.len() != num_dynamic_in_result {
            return verify_err!(
                loc,
                ReshapeOpVerifyErr::DynDimCountMismatch {
                    expected: num_dynamic_in_result,
                    got: dyn_dims.len()
                }
            );
        }

        for dyn_dim in &dyn_dims {
            if !dyn_dim.get_type(ctx).deref(ctx).is::<IndexType>() {
                return verify_err!(loc, ReshapeOpVerifyErr::DynDimNotIndex);
            }
        }

        // If both shapes are fully static, verify element count equality.
        let source_shape = source_ty.shape();
        let result_shape = result_ty.shape();
        if source_shape
            .iter()
            .all(|d| matches!(d, Dimension::Static(_)))
            && result_shape
                .iter()
                .all(|d| matches!(d, Dimension::Static(_)))
        {
            let source_count: usize = source_shape
                .iter()
                .map(|d| match d {
                    Dimension::Static(s) => *s,
                    _ => unreachable!(),
                })
                .product();
            let result_count: usize = result_shape
                .iter()
                .map(|d| match d {
                    Dimension::Static(s) => *s,
                    _ => unreachable!(),
                })
                .product();
            if source_count != result_count {
                return verify_err!(
                    loc,
                    ReshapeOpVerifyErr::ElementCountMismatch {
                        src_count: source_count,
                        result_count
                    }
                );
            }
        }

        Ok(())
    }
}

impl ReshapeOp {
    /// Create a new `ReshapeOp`.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        dynamic_dimensions: Vec<Value>,
        result_type: TypedHandle<RankedTensorType>,
    ) -> Self {
        let mut operands = vec![source];
        operands.extend(dynamic_dimensions);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            operands,
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the source tensor operand.
    pub fn get_source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the dynamic dimension operands.
    pub fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}
