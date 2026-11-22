// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Implementation of various interfaces for the tensor dialect.

use std::num::NonZero;

use pliron::{
    attribute::{AttrObj, Attribute, attr_cast},
    builtin::{
        attr_interfaces::{FloatAttr, MaterializableAttr},
        attributes::{FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr},
        op_interfaces::OneResultInterface,
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    context::{Context, Ptr},
    derive::{attr_interface_impl, op_interface_impl},
    dyn_clone::clone_box,
    irbuild::{IRStatus, rewriter::Rewriter},
    op::Op,
    operation::Operation,
    opts::constants::ConstFoldInterface,
    r#type::{TypeHandle, Typed},
    utils::{
        apfloat::{
            Double, Float, FloatConvert, Half, Round, Single, Status, double_to_f64, f32_to_single,
            f64_to_double, single_to_f32,
        },
        apint::APInt,
    },
};
use pliron_common_dialects::index::attributes::known_constant_index;

use crate::{
    memref::{
        attributes::ShapedTypeHandle,
        interface_impls::fold_dim,
        layout::{identity_strides, permute},
        op_interfaces::{ReductionOpInterface, TransposeOpInterface},
        type_interfaces::{DenseElementTypeHandle, Dimension},
    },
    tensor::{
        attributes::{DenseElementsAttr, ReductionKindAttr},
        conversions::{ScalarCastKind, scalar_cast_kind},
        ops::{
            AddOp, BatchMatMulOp, BroadcastOp, ConstantOp, DimOp, DivOp, ElementwiseCastOp, ExpOp,
            ExtractOp, ExtractSliceOp, GenerateOp, InsertSliceOp, MatMulOp, MaxOp, MinOp, MulOp,
            ReduceOp, ReshapeOp, SplatOp, SubOp, TransposeOp,
        },
        types::RankedTensorType,
    },
};

/// The maximum number of elements in a non-splat folded constant.
pub const MAX_FOLD_ELEMENTS: usize = 1024;

// Tensors have value semantics, so their ops have no side effects of their own.
impl_no_self_side_effects!(
    ConstantOp,
    BroadcastOp,
    SplatOp,
    ElementwiseCastOp,
    GenerateOp,
    ExtractOp,
    DimOp,
    AddOp,
    SubOp,
    MulOp,
    DivOp,
    MaxOp,
    MinOp,
    ExpOp,
    ReduceOp,
    MatMulOp,
    BatchMatMulOp,
    ExtractSliceOp,
    InsertSliceOp,
    ReshapeOp,
    TransposeOp,
);

#[op_interface_impl]
impl ConstFoldInterface for DimOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [_tensor, dim_index] = operand_attrs else {
            panic!("DimOp must have two operands");
        };
        let source_ty =
            ShapedTypeHandle::from_handle(self.get_source_tensor(ctx).get_type(ctx), ctx)
                .expect("OperandNOfType<0, RankedTensorType> ensures the source is shaped");
        vec![fold_dim(ctx, source_ty, dim_index)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[attr_interface_impl]
impl MaterializableAttr for DenseElementsAttr {
    fn materialize(&self, ctx: &mut Context) -> Ptr<Operation> {
        assert!(
            TypeHandle::from(self.ty())
                .deref(ctx)
                .is::<RankedTensorType>(),
            "Only a constant of a tensor type materializes as a tensor.constant"
        );
        ConstantOp::new(ctx, self.clone()).get_operation()
    }
}

/// Get `attr` as a [DenseElementsAttr], if it is one.
fn dense_operand(attr: &Option<AttrObj>) -> Option<&DenseElementsAttr> {
    attr.as_ref()?.downcast_ref::<DenseElementsAttr>()
}

/// Get `ty` as a [ShapedTypeHandle], if all of its dimensions are static.
fn static_shaped_type(ctx: &Context, ty: TypeHandle) -> Option<ShapedTypeHandle> {
    let shaped_ty = ShapedTypeHandle::from_handle(ty, ctx).ok()?;
    shaped_ty.deref(ctx).has_static_shape().then_some(shaped_ty)
}

/// Fold (`combine`) `operands` elementwise.
/// The element count of each operand must equal `result_ty`'s
/// [num_elements](crate::memref::type_interfaces::ShapedType::num_elements).
///
/// - The result is a splat if all `operands` are splats.
/// - The result is `None` if
///   - `combine` gives `None` for an element, or
///   - `result_ty` is not a static shape, or
///   - a non-splat result has more than [MAX_FOLD_ELEMENTS] elements.
fn fold_elementwise(
    ctx: &Context,
    result_ty: ShapedTypeHandle,
    operands: &[&DenseElementsAttr],
    combine: impl Fn(&[AttrObj]) -> Option<AttrObj>,
) -> Option<DenseElementsAttr> {
    let element_ty =
        DenseElementTypeHandle::from_handle(result_ty.deref(ctx).element_type(), ctx).ok()?;
    let num_elements = if operands.iter().all(|operand| operand.is_splat(ctx)) {
        1
    } else {
        let num_elements = result_ty.deref(ctx).num_elements()?;
        if num_elements > MAX_FOLD_ELEMENTS {
            return None;
        }
        num_elements
    };

    let mut data = Vec::new();
    for idx in 0..num_elements {
        let elements: Vec<AttrObj> = operands
            .iter()
            .map(|operand| operand.element_attr(ctx, idx))
            .collect();
        let result = combine(&elements)?;
        data.extend(element_ty.deref(ctx).element_bytes(&*result)?);
    }
    DenseElementsAttr::new(ctx, result_ty, data).ok()
}

/// The elementwise binary arithmetic operations.
#[derive(Clone, Copy)]
enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Max,
    Min,
}

/// Compute `lhs <bin_op> rhs` for two scalar constants, as per `bin_op`'s LLVM lowering.
fn fold_scalar_binary(
    ctx: &Context,
    bin_op: BinaryOpKind,
    lhs: &dyn Attribute,
    rhs: &dyn Attribute,
) -> Option<AttrObj> {
    if let (Some(lhs), Some(rhs)) = (
        lhs.downcast_ref::<IntegerAttr>(),
        rhs.downcast_ref::<IntegerAttr>(),
    ) {
        let (lhs_value, rhs_value) = (lhs.value(), rhs.value());
        let is_unsigned = lhs.get_type().deref(ctx).signedness() == Signedness::Unsigned;
        let result = match bin_op {
            BinaryOpKind::Add => lhs_value.add(&rhs_value),
            BinaryOpKind::Sub => lhs_value.sub(&rhs_value),
            BinaryOpKind::Mul => lhs_value.mul(&rhs_value),
            BinaryOpKind::Div if is_unsigned => {
                if rhs_value.is_zero() {
                    return None;
                }
                lhs_value.udiv(&rhs_value)
            }
            BinaryOpKind::Div => {
                let bw =
                    NonZero::new(rhs_value.bw()).expect("An integer must have a non-zero bitwidth");
                // LLVM does not define signed `x / 0` or `imin / -1` (`-1` is `umax`).
                if rhs_value.is_zero()
                    || (lhs_value == APInt::imin(bw) && rhs_value == APInt::umax(bw))
                {
                    return None;
                }
                lhs_value.sdiv(&rhs_value)
            }
            BinaryOpKind::Max | BinaryOpKind::Min => {
                let lhs_is_less = if is_unsigned {
                    lhs_value.ult(&rhs_value)
                } else {
                    lhs_value.slt(&rhs_value)
                };
                if lhs_is_less == matches!(bin_op, BinaryOpKind::Max) {
                    rhs_value
                } else {
                    lhs_value
                }
            }
        };
        return Some(Box::new(IntegerAttr::new(lhs.get_type(), result)));
    }

    let lhs = attr_cast::<dyn FloatAttr>(lhs)?;
    let rhs = attr_cast::<dyn FloatAttr>(rhs)?;
    let result = match bin_op {
        BinaryOpKind::Add => lhs.add(rhs).value,
        BinaryOpKind::Sub => lhs.sub(rhs).value,
        BinaryOpKind::Mul => lhs.mul(rhs).value,
        BinaryOpKind::Div => lhs.div(rhs).value,
        // IEEE 754-2019 `maximum` and `minimum`, as `llvm.maximum` and `llvm.minimum`.
        BinaryOpKind::Max => lhs.maximum(rhs),
        BinaryOpKind::Min => lhs.minimum(rhs),
    };
    Some(result)
}

/// Get `ty` as an [IntegerType], if it is one.
fn integer_type(ctx: &Context, ty: TypeHandle) -> Option<IntegerType> {
    ty.deref(ctx).downcast_ref::<IntegerType>().cloned()
}

/// Convert the float constant `value` to the float type `result_ty`,
/// rounding to nearest-even.
fn convert_float(ctx: &Context, value: &dyn Attribute, result_ty: TypeHandle) -> Option<AttrObj> {
    fn convert<S>(ctx: &Context, value: S, result_ty: TypeHandle) -> Option<AttrObj>
    where
        S: FloatConvert<Half> + FloatConvert<Single> + FloatConvert<Double>,
    {
        let result_ty = result_ty.deref(ctx);
        if result_ty.is::<FP16Type>() {
            Some(Box::new(FPHalfAttr(value.convert(&mut false).value)))
        } else if result_ty.is::<FP32Type>() {
            Some(Box::new(FPSingleAttr(value.convert(&mut false).value)))
        } else if result_ty.is::<FP64Type>() {
            Some(Box::new(FPDoubleAttr(value.convert(&mut false).value)))
        } else {
            None
        }
    }

    if let Some(value) = value.downcast_ref::<FPHalfAttr>() {
        convert(ctx, value.0, result_ty)
    } else if let Some(value) = value.downcast_ref::<FPSingleAttr>() {
        convert(ctx, value.0, result_ty)
    } else if let Some(value) = value.downcast_ref::<FPDoubleAttr>() {
        convert(ctx, value.0, result_ty)
    } else {
        None
    }
}

/// Convert the float constant `value` to the integer type `result_ty`, rounding toward zero.
/// The result is `None` if the rounded value does not fit `result_ty`.
fn float_to_int(
    ctx: &Context,
    value: &dyn Attribute,
    result_ty: TypeHandle,
    signed: bool,
) -> Option<AttrObj> {
    let value = attr_cast::<dyn FloatAttr>(value)?;
    let int_ty = integer_type(ctx, result_ty)?;
    let width = int_ty.width() as usize;
    if width > 128 {
        return None;
    }
    let bw = NonZero::new(width).expect("An integer must have a non-zero bitwidth");

    let mut is_exact = false;
    let result = if signed {
        // rustc_apfloat does not support a conversion to a signed integer of one bit.
        let converted = value.to_i128_r(width.max(2), Round::TowardZero, &mut is_exact);
        if converted.status.contains(Status::INVALID_OP)
            || (width == 1 && converted.value != -1 && converted.value != 0)
        {
            return None;
        }
        APInt::from_i128(converted.value, bw)
    } else {
        let converted = value.to_u128_r(width, Round::TowardZero, &mut is_exact);
        if converted.status.contains(Status::INVALID_OP) {
            return None;
        }
        APInt::from_u128(converted.value, bw)
    };
    let int_ty = IntegerType::get(ctx, int_ty.width(), int_ty.signedness());
    Some(Box::new(IntegerAttr::new(int_ty, result)))
}

/// Convert the integer constant `value` to the float type `result_ty`.
fn int_to_float(
    ctx: &Context,
    value: &dyn Attribute,
    result_ty: TypeHandle,
    signed: bool,
) -> Option<AttrObj> {
    let value = value.downcast_ref::<IntegerAttr>()?.value();
    if value.bw() > 128 {
        return None;
    }
    let result_ty = result_ty.deref(ctx);
    let zero: Box<dyn FloatAttr> = if result_ty.is::<FP16Type>() {
        Box::new(FPHalfAttr(Half::ZERO))
    } else if result_ty.is::<FP32Type>() {
        Box::new(FPSingleAttr(Single::ZERO))
    } else if result_ty.is::<FP64Type>() {
        Box::new(FPDoubleAttr(Double::ZERO))
    } else {
        return None;
    };
    let result = if signed {
        zero.build_from_i128(value.to_i128())
    } else {
        zero.build_from_u128(value.to_u128())
    };
    Some(result.value)
}

/// Cast the scalar constant `value` to `result_ty`, as per `cast_op`'s LLVM lowering.
fn fold_scalar_cast(
    ctx: &Context,
    cast_op: ScalarCastKind,
    value: &dyn Attribute,
    result_ty: TypeHandle,
) -> Option<AttrObj> {
    match cast_op {
        ScalarCastKind::Identity => Some(clone_box(value)),
        ScalarCastKind::Trunc | ScalarCastKind::SExt | ScalarCastKind::ZExt => {
            let value = value.downcast_ref::<IntegerAttr>()?.value();
            let int_ty = integer_type(ctx, result_ty)?;
            let bw = NonZero::new(int_ty.width() as usize)
                .expect("An integer must have a non-zero bitwidth");
            let result = match cast_op {
                ScalarCastKind::Trunc => value.trunc(bw),
                ScalarCastKind::SExt => value.sext(bw),
                _ => value.zext(bw),
            };
            let int_ty = IntegerType::get(ctx, int_ty.width(), int_ty.signedness());
            Some(Box::new(IntegerAttr::new(int_ty, result)))
        }
        ScalarCastKind::FPTrunc | ScalarCastKind::FPExt => convert_float(ctx, value, result_ty),
        ScalarCastKind::FPToSI => float_to_int(ctx, value, result_ty, true),
        ScalarCastKind::FPToUI => float_to_int(ctx, value, result_ty, false),
        ScalarCastKind::SIToFP => int_to_float(ctx, value, result_ty, true),
        ScalarCastKind::UIToFP => int_to_float(ctx, value, result_ty, false),
    }
}

#[op_interface_impl]
impl ConstFoldInterface for ConstantOp {
    fn check_fold(
        &self,
        ctx: &Context,
        _operand_attrs: &[Option<AttrObj>],
    ) -> Vec<Option<AttrObj>> {
        vec![Some(Box::new(self.value(ctx).clone()) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        _ctx: &mut Context,
        _operand_attrs: &[Option<AttrObj>],
        _rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        IRStatus::Unchanged
    }
}

#[op_interface_impl]
impl ConstFoldInterface for ExtractOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [tensor, indices @ ..] = operand_attrs else {
            panic!("ExtractOp must have a tensor operand");
        };
        let Some(tensor) = dense_operand(tensor) else {
            // The tensor is not a constant.
            return vec![None];
        };
        let tensor_ty = tensor.ty();
        let tensor_ty = tensor_ty.deref(ctx);
        let mut linear_idx = 0;
        for (index, dim) in indices.iter().zip(tensor_ty.shape()) {
            let Some(index) =
                known_constant_index(index).and_then(|index| usize::try_from(index).ok())
            else {
                // The index is not a constant.
                return vec![None];
            };
            let Dimension::Static(size) = *dim else {
                panic!("A constant tensor must have a static shape");
            };
            if index >= size {
                // The index is out of bounds.
                return vec![None];
            }
            linear_idx = linear_idx * size + index;
        }
        vec![Some(tensor.element_attr(ctx, linear_idx))]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

/// Fold an elementwise binary op of `bin_op` with the result type `result_ty`.
fn fold_binary(
    ctx: &Context,
    bin_op: BinaryOpKind,
    result_ty: TypeHandle,
    operand_attrs: &[Option<AttrObj>],
) -> Option<AttrObj> {
    let [lhs, rhs] = operand_attrs else {
        panic!("An elementwise binary op must have two operands");
    };
    let (lhs, rhs) = (dense_operand(lhs)?, dense_operand(rhs)?);
    let result_ty = static_shaped_type(ctx, result_ty)?;
    let folded = fold_elementwise(ctx, result_ty, &[lhs, rhs], |elements| {
        fold_scalar_binary(ctx, bin_op, &*elements[0], &*elements[1])
    })?;
    Some(Box::new(folded))
}

/// Implement [ConstFoldInterface] for elementwise binary ops.
macro_rules! impl_binary_const_fold {
    ($($op:ty => $bin_op:expr),+ $(,)?) => {
        $(
            #[op_interface_impl]
            impl ConstFoldInterface for $op {
                fn check_fold(
                    &self,
                    ctx: &Context,
                    operand_attrs: &[Option<AttrObj>],
                ) -> Vec<Option<AttrObj>> {
                    vec![fold_binary(ctx, $bin_op, self.result_type(ctx), operand_attrs)]
                }

                fn fold_in_place(
                    &self,
                    ctx: &mut Context,
                    operand_attrs: &[Option<AttrObj>],
                    rewriter: &mut dyn Rewriter,
                ) -> IRStatus {
                    self.fold_with_materialization(ctx, operand_attrs, rewriter)
                }
            }
        )+
    };
}

impl_binary_const_fold!(
    AddOp => BinaryOpKind::Add,
    SubOp => BinaryOpKind::Sub,
    MulOp => BinaryOpKind::Mul,
    DivOp => BinaryOpKind::Div,
    MaxOp => BinaryOpKind::Max,
    MinOp => BinaryOpKind::Min,
);

#[op_interface_impl]
impl ConstFoldInterface for ExpOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [input] = operand_attrs else {
            panic!("ExpOp must have one operand");
        };
        let Some(input) = dense_operand(input) else {
            // The input is not a constant.
            return vec![None];
        };
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        let folded = fold_elementwise(ctx, result_ty, &[input], |elements| {
            // Compute `exp` with [f32::exp] or [f64::exp].
            let element = &*elements[0];
            if let Some(value) = element.downcast_ref::<FPSingleAttr>() {
                let result = single_to_f32(value.0).exp();
                Some(Box::new(FPSingleAttr(f32_to_single(result))) as AttrObj)
            } else if let Some(value) = element.downcast_ref::<FPDoubleAttr>() {
                let result = double_to_f64(value.0).exp();
                Some(Box::new(FPDoubleAttr(f64_to_double(result))) as AttrObj)
            } else {
                None
            }
        });
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

/// Fold `reduce` with the constant operands `operand_attrs`.
///
/// The elements are combined in the same order as the unfolded op would at run time.
///
/// The result is `None` if
/// - an operand is not a constant, or
/// - the input or the result has more than [MAX_FOLD_ELEMENTS] elements, or
/// - an element does not fold.
fn fold_reduce(
    ctx: &Context,
    reduce: &ReduceOp,
    operand_attrs: &[Option<AttrObj>],
) -> Option<DenseElementsAttr> {
    let [input, accum] = operand_attrs else {
        panic!("ReduceOp must have two operands");
    };
    let (input, accum) = (dense_operand(input)?, dense_operand(accum)?);
    let result_ty = static_shaped_type(ctx, reduce.result_type(ctx))?;
    let element_ty =
        DenseElementTypeHandle::from_handle(result_ty.deref(ctx).element_type(), ctx).ok()?;
    let bin_op = match reduce.reduction_kind(ctx) {
        ReductionKindAttr::Sum => BinaryOpKind::Add,
        ReductionKindAttr::Prod => BinaryOpKind::Mul,
        ReductionKindAttr::Max => BinaryOpKind::Max,
        ReductionKindAttr::Min => BinaryOpKind::Min,
    };
    let reduced_dims = reduce.reduction_dims(ctx);

    let input_ty = input.ty();
    let input_shape: Vec<usize> = input_ty
        .deref(ctx)
        .shape()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(size) => *size,
            Dimension::Dynamic => panic!("A constant tensor must have a static shape"),
        })
        .collect();
    let num_input = input_shape.iter().product::<usize>();
    let num_accum = result_ty.deref(ctx).num_elements()?;
    if num_input > MAX_FOLD_ELEMENTS || num_accum > MAX_FOLD_ELEMENTS {
        return None;
    }

    if input.is_splat(ctx) && accum.is_splat(ctx) {
        // All accumulator elements combine the same value the same number of times.
        let element = input.element_attr(ctx, 0);
        let mut result = accum.element_attr(ctx, 0);
        for _ in 0..num_input.checked_div(num_accum).unwrap_or(0) {
            result = fold_scalar_binary(ctx, bin_op, &*result, &*element)?;
        }
        let bytes = element_ty.deref(ctx).element_bytes(&*result)?;
        return DenseElementsAttr::new_splat(ctx, result_ty, bytes).ok();
    }

    let accum_shape: Vec<usize> = result_ty
        .deref(ctx)
        .shape()
        .iter()
        .map(|dim| match dim {
            Dimension::Static(size) => *size,
            Dimension::Dynamic => panic!("A folded result must have a static shape"),
        })
        .collect();
    assert_eq!(
        input_shape.len(),
        accum_shape.len(),
        "The verifier ensures that a reduction does not change the rank"
    );

    let mut results: Vec<AttrObj> = (0..num_accum)
        .map(|idx| accum.element_attr(ctx, idx))
        .collect();

    // The input indices, advanced in row-major order.
    let mut indices = vec![0; input_shape.len()];

    // Combine each input element into the accumulator element that it maps to.
    for _ in 0..num_input {
        // The row-major input position.
        let input_pos = indices
            .iter()
            .zip(&input_shape)
            .fold(0, |pos, (&index, size)| pos * size + index);
        // The row-major accumulator position. The accumulator indices are derived
        // from the input indices, by setting the indices of the reduced dimensions to 0.
        let accum_pos =
            indices
                .iter()
                .zip(&accum_shape)
                .enumerate()
                .fold(0, |pos, (dim, (&index, size))| {
                    pos * size
                        + if reduced_dims.contains(&dim) {
                            0
                        } else {
                            index
                        }
                });
        let element = input.element_attr(ctx, input_pos);
        results[accum_pos] = fold_scalar_binary(ctx, bin_op, &*results[accum_pos], &*element)?;

        // Advance to the next input indices, the way an odometer does.
        for (index, &size) in indices.iter_mut().zip(&input_shape).rev() {
            *index += 1;
            if *index < size {
                break;
            }
            *index = 0;
        }
    }

    let mut data = Vec::new();
    for result in &results {
        data.extend(element_ty.deref(ctx).element_bytes(&**result)?);
    }
    DenseElementsAttr::new(ctx, result_ty, data).ok()
}

#[op_interface_impl]
impl ConstFoldInterface for ReduceOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let folded = fold_reduce(ctx, self, operand_attrs);
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for SplatOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some(Some(value)) = operand_attrs.first() else {
            // The value is not a constant.
            return vec![None];
        };
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        let Ok(element_ty) =
            DenseElementTypeHandle::from_handle(result_ty.deref(ctx).element_type(), ctx)
        else {
            // The element type has no dense encoding.
            return vec![None];
        };
        let Some(element) = element_ty.deref(ctx).element_bytes(&**value) else {
            // The value is not a scalar of the element type.
            return vec![None];
        };
        let folded = DenseElementsAttr::new_splat(ctx, result_ty, element).ok();
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for BroadcastOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some(source) = operand_attrs.first().and_then(dense_operand) else {
            // The source is not a constant.
            return vec![None];
        };
        if !source.is_splat(ctx) {
            // Only a splat source folds.
            return vec![None];
        }
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        let folded = DenseElementsAttr::new_splat(ctx, result_ty, source.raw_data().to_vec()).ok();
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for ElementwiseCastOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [input] = operand_attrs else {
            panic!("ElementwiseCastOp must have one operand");
        };
        let Some(input) = dense_operand(input) else {
            // The input is not a constant.
            return vec![None];
        };
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        let from = input.ty().deref(ctx).element_type();
        let to = result_ty.deref(ctx).element_type();
        let signedness = self.signedness(ctx);
        let Ok(cast_op) = scalar_cast_kind(
            ctx,
            from,
            to,
            signedness.input_is_signed,
            signedness.result_is_signed,
        ) else {
            // The lowering does not support this cast.
            return vec![None];
        };
        let folded = fold_elementwise(ctx, result_ty, &[input], |elements| {
            fold_scalar_cast(ctx, cast_op, &*elements[0], to)
        });
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for ReshapeOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some(source) = operand_attrs.first().and_then(dense_operand) else {
            // The source is not a constant.
            return vec![None];
        };
        if !source.is_splat(ctx)
            && source
                .num_elements(ctx)
                .is_none_or(|num_elements| num_elements > MAX_FOLD_ELEMENTS)
        {
            // The source is too large to copy.
            return vec![None];
        }
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        let mut folded = source.clone();
        if folded.set_type(ctx, result_ty).is_err() {
            return vec![None];
        }
        vec![Some(Box::new(folded))]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}

#[op_interface_impl]
impl ConstFoldInterface for TransposeOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let Some(source) = operand_attrs.first().and_then(dense_operand) else {
            // The source is not a constant.
            return vec![None];
        };
        let Some(result_ty) = static_shaped_type(ctx, self.result_type(ctx)) else {
            // The result shape is not static.
            return vec![None];
        };
        if source.is_splat(ctx) {
            // The source is a splat, so the result is the same splat, but with type `result_ty`.
            let folded =
                DenseElementsAttr::new_splat(ctx, result_ty, source.raw_data().to_vec()).ok();
            return vec![folded.map(|folded| Box::new(folded) as AttrObj)];
        }
        let Some(num_elements) = source
            .num_elements(ctx)
            .filter(|num_elements| *num_elements <= MAX_FOLD_ELEMENTS)
        else {
            // The source is too large to copy.
            return vec![None];
        };
        let static_sizes = |shape: &[Dimension]| -> Option<Vec<usize>> {
            shape
                .iter()
                .map(|dim| match dim {
                    Dimension::Static(size) => Some(*size),
                    Dimension::Dynamic => None,
                })
                .collect()
        };
        let permutation = self.permutation(ctx);
        let (Some(source_strides), Some(result_sizes)) = (
            static_sizes(&identity_strides(source.ty().deref(ctx).shape())),
            static_sizes(result_ty.deref(ctx).shape()),
        ) else {
            return vec![None];
        };
        // The source stride of each result dimension.
        let strides = permute(&source_strides, &permutation);
        let mut data = Vec::with_capacity(num_elements * source.element_size(ctx));
        for result_idx in 0..num_elements {
            // Convert the row-major result position to a source position.
            let mut remaining = result_idx;
            let mut source_idx = 0;
            for (size, stride) in result_sizes.iter().zip(&strides).rev() {
                source_idx += (remaining % size) * stride;
                remaining /= size;
            }
            data.extend_from_slice(source.element_data(ctx, source_idx));
        }
        let folded = DenseElementsAttr::new(ctx, result_ty, data).ok();
        vec![folded.map(|folded| Box::new(folded) as AttrObj)]
    }

    fn fold_in_place(
        &self,
        ctx: &mut Context,
        operand_attrs: &[Option<AttrObj>],
        rewriter: &mut dyn Rewriter,
    ) -> IRStatus {
        self.fold_with_materialization(ctx, operand_attrs, rewriter)
    }
}
