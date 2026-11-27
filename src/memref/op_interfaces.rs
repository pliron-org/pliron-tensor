// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Memref op interfaces

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface, OneOpdInterface,
        OneResultInterface, OperandNOfType, ResultNImplsTy, ResultNOfType,
        SingleBlockRegionInterface,
    },
    context::Context,
    derive::op_interface,
    dict_key,
    op::{Op, op_cast},
    operation::Operation,
    result::Result,
    r#type::{Typed, type_cast},
    value::Value,
    verify_err,
};
use pliron_common_dialects::{cf::op_interfaces::YieldingRegions, index::types::IndexType};

use crate::memref::{
    attributes::{PermutationAttr, ReductionDimsAttr, ReductionKindAttr, ShapedTypeHandle},
    layout::{is_valid_permutation, permute},
    ops::YieldOp,
    type_interfaces::{Dimension, ShapedType},
    types::RankedMemrefType,
};

#[derive(thiserror::Error, Debug)]
pub enum GenerateOpInterfaceVerifyErr {
    #[error("GenerateInterface entry block must have {0} arguments, found {1}")]
    EntryBlockArgMismatch(usize, usize),
    #[error("GenerateInterface entry block arguments must be of index type")]
    EntryBlockArgTypeMismatch,
    #[error("GenerateInterface yield operand type does not match result element type")]
    YieldOperandTypeMismatch,
}

#[op_interface]
pub trait GenerateOpInterface: SingleBlockRegionInterface + YieldingRegions<YieldOp> {
    /// Get the shape of the memref/tensor we're generating.
    fn get_generated_shape(&self, ctx: &Context) -> ShapedTypeHandle;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op = op_cast::<dyn GenerateOpInterface>(op)
            .expect("Operation does not implement GenerateOpInterface");

        let entry_block = op.get_body(ctx, 0);
        let result_shape = op.get_generated_shape(ctx);
        let rank = result_shape.deref(ctx).rank();
        let entry_block = &*entry_block.deref(ctx);
        if entry_block.get_num_arguments() != rank {
            return verify_err!(
                loc,
                GenerateOpInterfaceVerifyErr::EntryBlockArgMismatch(
                    rank,
                    entry_block.get_num_arguments()
                )
            );
        }

        if entry_block
            .arguments()
            .any(|arg| !arg.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(
                loc.clone(),
                GenerateOpInterfaceVerifyErr::EntryBlockArgTypeMismatch
            );
        }

        let yield_op = op.get_yield(ctx, 0);
        if yield_op.get_operand(ctx).get_type(ctx) != result_shape.deref(ctx).element_type() {
            return verify_err!(loc, GenerateOpInterfaceVerifyErr::YieldOperandTypeMismatch);
        }

        Ok(())
    }
}

#[derive(thiserror::Error, Debug)]
pub enum CompatibleShapesOpErr {
    #[error(
        "Expected all operands and results to have the same shape (for non-dynamic dimensions) and rank"
    )]
    IncompatibleShapes,
    #[error("Expected all operands and results to have the same element type")]
    ElementTypeMismatch,
}

/// Tensor and Memref ops that have operands and results of the same
/// element type, shape (for non-dynamic dimensions) and rank.
#[op_interface]
pub trait CompatibleShapesOp<T: ShapedType>: AllResultsOfType<T> + AllOperandsOfType<T> {
    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op_ref = op.get_operation().deref(ctx);
        let mut element_type = None;
        let mut shapes = Vec::new();
        for value in op_ref.results().chain(op_ref.operands()) {
            let ty = value.get_type(ctx);
            let ty = ty.deref(ctx);
            let shaped = ty
                .downcast_ref::<T>()
                .expect("AllResultsOfType<T> and AllOperandsOfType<T> ensure the type is T");
            if element_type.is_some_and(|el| el != shaped.element_type()) {
                return verify_err!(op.loc(ctx), CompatibleShapesOpErr::ElementTypeMismatch);
            }
            element_type = Some(shaped.element_type());
            shapes.push(shaped.shape().clone());
        }

        let mut cur_shape: Option<Vec<Dimension>> = None;
        for shape in shapes {
            if let Some(ref mut cur_shape) = cur_shape {
                if cur_shape.len() != shape.len() {
                    return verify_err!(op.loc(ctx), CompatibleShapesOpErr::IncompatibleShapes);
                }
                for (cur_dim, next_dim) in cur_shape.iter_mut().zip(shape.iter()) {
                    match (&cur_dim.clone(), next_dim) {
                        (Dimension::Static(d1), Dimension::Static(d2)) if d1 != d2 => {
                            return verify_err!(
                                op.loc(ctx),
                                CompatibleShapesOpErr::IncompatibleShapes
                            );
                        }
                        (Dimension::Dynamic, Dimension::Static(d)) => {
                            *cur_dim = Dimension::Static(*d);
                        }
                        _ => {}
                    }
                }
            } else {
                cur_shape = Some(shape);
            }
        }
        Ok(())
    }

    /// Get the compatible shape of all operands and results, where dynamic dimensions
    /// are replaced with static dimensions if possible.
    fn compatible_shape(&self, ctx: &Context) -> Vec<Dimension> {
        let op_ref = self.get_operation().deref(ctx);
        let shapes = op_ref.results().chain(op_ref.operands()).map(|v| {
            let ty = v.get_type(ctx);
            let ty_ref = ty.deref(ctx);
            let shaped_ty = type_cast::<dyn ShapedType>(&*ty_ref)
                .expect("Expected operand and result types to be of the specified ShapedType");
            shaped_ty.shape().clone()
        });
        // Compute the compatible shape by merging all shapes.
        let mut compatible_shape: Option<Vec<Dimension>> = None;
        for shape in shapes {
            if let Some(ref mut comp_shape) = compatible_shape {
                if comp_shape.len() != shape.len() {
                    panic!("Incompatible shapes");
                }
                for (comp_dim, dim) in comp_shape.iter_mut().zip(shape.iter()) {
                    match (&comp_dim.clone(), dim) {
                        (Dimension::Static(d1), Dimension::Static(d2)) if d1 != d2 => {
                            panic!("Incompatible shapes");
                        }
                        (Dimension::Dynamic, Dimension::Static(d)) => {
                            *comp_dim = Dimension::Static(*d)
                        }
                        _ => {}
                    }
                }
            } else {
                compatible_shape = Some(shape);
            }
        }
        compatible_shape.expect("Op has 0 results and 0 operands to determine compatible shape")
    }
}

#[derive(thiserror::Error, Debug)]
pub enum DynamicDimensionOperandsOpVerifyErr {
    #[error(
        "The number of dynamic dimension operands ({got}) must match \
        the number of dynamic dimensions in the result type ({expected})"
    )]
    CountMismatch { expected: usize, got: usize },
    #[error("Dynamic dimension operands must be of index type")]
    NonIndexOperand,
}

/// Ops with a shaped result whose dynamic dimensions are provided by [Index](IndexType)
/// operands, one each for every dynamic dimension of the result type.
#[op_interface]
pub trait DynamicDimensionOperandsOp: ResultNImplsTy<0, dyn ShapedType> {
    /// Get the dynamic dimension operands, in result-dimension order.
    fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value>;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let dynamic_dims = op
            .downcast_ref::<Self>()
            .expect("Failed to downcast op to Self")
            .get_dynamic_dimensions(ctx);

        let result = op.get_operation().deref(ctx).get_result(0);
        let result_ty_handle = result.get_type(ctx);
        let result_ty_ref = result_ty_handle.deref(ctx);
        let result_ty = type_cast::<dyn ShapedType>(&*result_ty_ref)
            .expect("ResultNImplsTy<0, dyn ShapedType> ensures the result is a ShapedType");

        let expected = result_ty.num_dynamic_dimensions();
        let got = dynamic_dims.len();
        if got != expected {
            return verify_err!(
                op.loc(ctx),
                DynamicDimensionOperandsOpVerifyErr::CountMismatch { expected, got }
            );
        }

        if dynamic_dims
            .iter()
            .any(|dim| !dim.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(
                op.loc(ctx),
                DynamicDimensionOperandsOpVerifyErr::NonIndexOperand
            );
        }

        Ok(())
    }
}

/// Try to get `value`'s dynamic `dim` size from
/// its defining op's [DynamicDimensionOperandsOp] impl.
pub fn dynamic_dim_size(ctx: &Context, value: Value, dim: usize) -> Option<Value> {
    let def_op = value.defining_op()?;
    if def_op.deref(ctx).get_result(0) != value {
        // [DynamicDimensionOperandsOp] gives the dimension sizes of result 0 only.
        return None;
    }
    let def_op = Operation::get_op_dyn(def_op, ctx);
    let dynamic_dims =
        op_cast::<dyn DynamicDimensionOperandsOp>(&*def_op)?.get_dynamic_dimensions(ctx);

    let value_ty = value.get_type(ctx);
    let value_ty = value_ty.deref(ctx);
    let shape = type_cast::<dyn ShapedType>(&*value_ty)?.shape();
    if *shape.get(dim)? != Dimension::Dynamic {
        return None;
    }
    // The dynamic dimension operands are in result-dimension order.
    let position = shape[..dim]
        .iter()
        .filter(|dim| **dim == Dimension::Dynamic)
        .count();
    Some(
        *dynamic_dims.get(position).expect(
            "DynamicDimensionOperandsOp's verifier ensures one operand per dynamic dimension",
        ),
    )
}

#[derive(thiserror::Error, Debug)]
pub enum ReshapeOpInterfaceVerifyErr {
    #[error("Reshape source and result element types must match")]
    ElementTypeMismatch,
    #[error(
        "Reshape: total element count of the source ({src_count}) must match the result ({result_count})"
    )]
    ElementCountMismatch {
        src_count: usize,
        result_count: usize,
    },
}

/// Tensor and memref ops that view a source value under a new shape.
///
/// - Operand 0: The source operand
/// - Operand(s) 1+: [Index](IndexType) operands, one each for every dynamic dimension of the result type.
/// - The source and the result must have
///   - The same element type
///   - Equal number of elements when both their shapes are static
#[op_interface]
pub trait ReshapeOpInterface<T: ShapedType>:
    OperandNOfType<0, T> + ResultNOfType<0, T> + DynamicDimensionOperandsOp
{
    /// Get the source operand that is reshaped.
    fn get_source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let (source, result) = {
            let op_ref = op.get_operation().deref(ctx);
            (op_ref.get_operand(0), op_ref.get_result(0))
        };

        let src_ty_handle = source.get_type(ctx);
        let src_ty_ref = src_ty_handle.deref(ctx);
        let src_ty = src_ty_ref
            .downcast_ref::<T>()
            .expect("OperandNOfType<0, T> ensures the source is of type T");

        let result_ty_handle = result.get_type(ctx);
        let result_ty_ref = result_ty_handle.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<T>()
            .expect("ResultNOfType<0, T> ensures the result is of type T");

        if src_ty.element_type() != result_ty.element_type() {
            return verify_err!(
                loc.clone(),
                ReshapeOpInterfaceVerifyErr::ElementTypeMismatch
            );
        }

        // The element counts are comparable only when neither shape is dynamic.
        if let (Some(src_count), Some(result_count)) =
            (src_ty.num_elements(), result_ty.num_elements())
            && src_count != result_count
        {
            return verify_err!(
                loc,
                ReshapeOpInterfaceVerifyErr::ElementCountMismatch {
                    src_count,
                    result_count
                }
            );
        }

        Ok(())
    }
}

dict_key!(
    /// Attribute key for the [PermutationAttr] of a [TransposeOpInterface] op.
    ATTR_KEY_TRANSPOSE_PERMUTATION,
    "memref_transpose_permutation"
);

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum TransposeOpInterfaceVerifyErr {
    #[error("Transpose: Missing {ATTR_KEY_TRANSPOSE_PERMUTATION} attribute")]
    MissingPermutationAttr,
    #[error("Transpose: {permutation:?} is not a permutation of the {rank} source dimensions")]
    InvalidPermutation {
        permutation: Vec<usize>,
        rank: usize,
    },
    #[error("Transpose result must have the source element type and the permuted source shape")]
    ResultTypeMismatch,
}

/// Tensor and memref ops that permute the dimensions of their source
/// by a [PermutationAttr], stored at [ATTR_KEY_TRANSPOSE_PERMUTATION].
///
/// - Operand 0: The source operand
/// - The result has
///   - The element type of the source
///   - The permuted source shape
#[op_interface]
pub trait TransposeOpInterface<T: ShapedType>:
    OneOpdInterface + OneResultInterface + OperandNOfType<0, T> + ResultNOfType<0, T>
{
    /// Get the source operand.
    fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the source dimension of each result dimension.
    fn permutation(&self, ctx: &Context) -> Vec<usize> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<PermutationAttr>(&ATTR_KEY_TRANSPOSE_PERMUTATION)
            .expect("Transpose permutation is missing or has an incorrect type")
            .0
            .clone()
    }

    /// Set the source dimension of each result dimension.
    fn set_permutation(&self, ctx: &Context, permutation: Vec<usize>) {
        self.get_operation().deref_mut(ctx).attributes.set(
            ATTR_KEY_TRANSPOSE_PERMUTATION.clone(),
            PermutationAttr(permutation),
        );
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op_ref = op.get_operation().deref(ctx);
        let Some(PermutationAttr(permutation)) = op_ref
            .attributes
            .get::<PermutationAttr>(&ATTR_KEY_TRANSPOSE_PERMUTATION)
        else {
            return verify_err!(loc, TransposeOpInterfaceVerifyErr::MissingPermutationAttr);
        };

        let src_ty_handle = op_ref.get_operand(0).get_type(ctx);
        let src_ty_ref = src_ty_handle.deref(ctx);
        let src_ty = src_ty_ref
            .downcast_ref::<T>()
            .expect("OperandNOfType<0, T> ensures the source is of type T");

        let rank = src_ty.rank();
        if !is_valid_permutation(permutation, rank) {
            return verify_err!(
                loc,
                TransposeOpInterfaceVerifyErr::InvalidPermutation {
                    permutation: permutation.clone(),
                    rank
                }
            );
        }

        let result_ty_handle = op_ref.get_result(0).get_type(ctx);
        let result_ty_ref = result_ty_handle.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<T>()
            .expect("ResultNOfType<0, T> ensures the result is of type T");

        if result_ty.element_type() != src_ty.element_type()
            || *result_ty.shape() != permute(src_ty.shape(), permutation)
        {
            return verify_err!(loc, TransposeOpInterfaceVerifyErr::ResultTypeMismatch);
        }

        Ok(())
    }
}

/// Interface for element-wise binary arithmetic memref ops.
#[op_interface]
pub trait ElementWiseBinaryMemrefOpInterface:
    NResultsInterface<0>
    + AllOperandsOfType<RankedMemrefType>
    + NOpdsInterface<3>
    + CompatibleShapesOp<RankedMemrefType>
{
    /// Create a new instance of the element-wise binary memref op with the specified operands.
    fn new(ctx: &mut Context, res: Value, lhs: Value, rhs: Value) -> Self
    where
        Self: Sized,
    {
        Operation::new_op::<Self>(ctx, vec![], vec![res, lhs, rhs], vec![], 0)
    }

    /// Get the result memref operand.
    fn get_result_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the left-hand side memref operand.
    fn get_lhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the right-hand side memref operand.
    fn get_rhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Interface for element-wise unary memref ops.
#[op_interface]
pub trait ElementWiseUnaryMemrefOpInterface:
    NResultsInterface<0>
    + AllOperandsOfType<RankedMemrefType>
    + NOpdsInterface<2>
    + CompatibleShapesOp<RankedMemrefType>
{
    /// Create a new instance of the element-wise unary memref op with the specified operands.
    fn new(ctx: &mut Context, res: Value, input: Value) -> Self
    where
        Self: Sized,
    {
        Operation::new_op::<Self>(ctx, vec![], vec![res, input], vec![], 0)
    }

    /// Get the result memref operand.
    fn get_result_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the input memref operand.
    fn get_input_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

dict_key!(
    /// Attribute key for the [ReductionKindAttr] of a [ReductionOpInterface] op.
    ATTR_KEY_REDUCTION_KIND,
    "memref_reduction_kind"
);

dict_key!(
    /// Attribute key for the [ReductionDimsAttr] of a [ReductionOpInterface] op.
    ATTR_KEY_REDUCTION_DIMS,
    "memref_reduction_dims"
);

#[derive(thiserror::Error, Debug, PartialEq, Eq)]
pub enum ReductionOpInterfaceVerifyErr {
    #[error("Reduction: Missing {ATTR_KEY_REDUCTION_KIND} attribute")]
    MissingKindAttr,
    #[error("Reduction: Missing {ATTR_KEY_REDUCTION_DIMS} attribute")]
    MissingDimsAttr,
    #[error("Reduction: {dims:?} must be strictly increasing dimensions of the rank {rank} input")]
    InvalidDims { dims: Vec<usize>, rank: usize },
    #[error("Reduction: The input and the accumulator must have the same element type")]
    ElementTypeMismatch,
    #[error(
        "Reduction: The accumulator shape must be the input shape with each reduced dimension set to 1"
    )]
    AccumShapeMismatch,
}

/// Compute the accumulator shape when `input_shape` is reduced over `reduced_dims`.
///
/// Each dimension of `input_shape` that is in `reduced_dims` is set to 1.
pub fn reduced_shape(input_shape: &[Dimension], reduced_dims: &[usize]) -> Vec<Dimension> {
    let mut shape = input_shape.to_vec();
    for &dim in reduced_dims {
        shape[dim] = Dimension::Static(1);
    }
    shape
}

/// Tensor and memref ops that reduce their input over [ReductionDimsAttr]
/// with [ReductionKindAttr].
///
/// - Operand 0: The input.
/// - Operand 1: The accumulator, with the element type and [reduced_shape] of the input.
///
/// Post reduction, each reduced dimension has size 1, thus preserving the rank.
#[op_interface]
pub trait ReductionOpInterface<T: ShapedType>: NOpdsInterface<2> + AllOperandsOfType<T> {
    /// Get the input operand.
    fn input(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the accumulator operand.
    fn accum(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the function used by the reduction to combine the elements.
    fn reduction_kind(&self, ctx: &Context) -> ReductionKindAttr {
        *self
            .get_operation()
            .deref(ctx)
            .attributes
            .get::<ReductionKindAttr>(&ATTR_KEY_REDUCTION_KIND)
            .expect("Reduction kind is missing or has an incorrect type")
    }

    /// Get the reduced dimensions.
    fn reduction_dims(&self, ctx: &Context) -> Vec<usize> {
        self.get_operation()
            .deref(ctx)
            .attributes
            .get::<ReductionDimsAttr>(&ATTR_KEY_REDUCTION_DIMS)
            .expect("Reduction dimensions are missing or have an incorrect type")
            .0
            .clone()
    }

    /// Set [ReductionKindAttr] and [ReductionDimsAttr].
    fn set_reduction(&self, ctx: &Context, kind: ReductionKindAttr, reduced_dims: Vec<usize>) {
        let attributes = &mut self.get_operation().deref_mut(ctx).attributes;
        attributes.set(ATTR_KEY_REDUCTION_KIND.clone(), kind);
        attributes.set(
            ATTR_KEY_REDUCTION_DIMS.clone(),
            ReductionDimsAttr(reduced_dims),
        );
    }

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let loc = op.loc(ctx);
        let op_ref = op.get_operation().deref(ctx);
        if op_ref
            .attributes
            .get::<ReductionKindAttr>(&ATTR_KEY_REDUCTION_KIND)
            .is_none()
        {
            return verify_err!(loc, ReductionOpInterfaceVerifyErr::MissingKindAttr);
        }
        let Some(ReductionDimsAttr(reduced_dims)) = op_ref
            .attributes
            .get::<ReductionDimsAttr>(&ATTR_KEY_REDUCTION_DIMS)
        else {
            return verify_err!(loc, ReductionOpInterfaceVerifyErr::MissingDimsAttr);
        };

        let input_ty = op_ref.get_operand(0).get_type(ctx);
        let input_ty = input_ty.deref(ctx);
        let input_ty = input_ty
            .downcast_ref::<T>()
            .expect("AllOperandsOfType<T> ensures the input is of type T");
        let accum_ty = op_ref.get_operand(1).get_type(ctx);
        let accum_ty = accum_ty.deref(ctx);
        let accum_ty = accum_ty
            .downcast_ref::<T>()
            .expect("AllOperandsOfType<T> ensures the accumulator is of type T");

        let rank = input_ty.rank();
        let increasing = reduced_dims.windows(2).all(|pair| pair[0] < pair[1]);
        if !increasing || reduced_dims.last().is_some_and(|&last| last >= rank) {
            return verify_err!(
                loc,
                ReductionOpInterfaceVerifyErr::InvalidDims {
                    dims: reduced_dims.clone(),
                    rank
                }
            );
        }

        if input_ty.element_type() != accum_ty.element_type() {
            return verify_err!(loc, ReductionOpInterfaceVerifyErr::ElementTypeMismatch);
        }

        let expected = reduced_shape(input_ty.shape(), reduced_dims);
        let accum_shape = accum_ty.shape();
        let compatible = expected.len() == accum_shape.len()
            && expected.iter().zip(accum_shape.iter()).enumerate().all(
                |(i, (expected, got))| match (expected, got) {
                    // A reduced dimension must be a static 1.
                    _ if reduced_dims.contains(&i) => *got == Dimension::Static(1),
                    (Dimension::Static(e), Dimension::Static(g)) => e == g,
                    _ => true,
                },
            );
        if !compatible {
            return verify_err!(loc, ReductionOpInterfaceVerifyErr::AccumShapeMismatch);
        }

        Ok(())
    }
}
