// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Implementation of various interfaces for the memref dialect.

use pliron::{
    attribute::AttrObj,
    context::{Context, Ptr},
    derive::op_interface_impl,
    irbuild::{IRStatus, rewriter::Rewriter},
    op::Op,
    operation::Operation,
    opts::{canonicalize::CanonicalizeInterface, constants::ConstFoldInterface},
    r#type::Typed,
    value::Value,
};
use pliron_common_dialects::index::{
    attributes::{ConstantIndexAttr, known_constant_index},
    ops::try_match_constant_index,
};

use crate::memref::{
    attributes::ShapedTypeHandle,
    op_interfaces::dynamic_dim_size,
    ops::{DimOp, GetGlobalOp, MemrefCastOp, ReshapeOp, SubviewOp, TransposeOp, YieldOp},
    type_interfaces::Dimension,
};

impl_no_self_side_effects!(
    YieldOp,
    DimOp,
    SubviewOp,
    TransposeOp,
    ReshapeOp,
    MemrefCastOp,
    GetGlobalOp
);

#[op_interface_impl]
impl ConstFoldInterface for DimOp {
    fn check_fold(&self, ctx: &Context, operand_attrs: &[Option<AttrObj>]) -> Vec<Option<AttrObj>> {
        let [_memref, dim_index] = operand_attrs else {
            panic!("DimOp must have two operands");
        };
        let source_ty =
            ShapedTypeHandle::from_handle(self.get_source_memref(ctx).get_type(ctx), ctx)
                .expect("OperandNOfType<0, RankedMemrefType> ensures the source is shaped");
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

/// Try to fold the size of dimension `dim_index` of `source_ty` to a constant.
pub(crate) fn fold_dim(
    ctx: &Context,
    source_ty: ShapedTypeHandle,
    dim_index: &Option<AttrObj>,
) -> Option<AttrObj> {
    let dim_index =
        known_constant_index(dim_index).and_then(|index| usize::try_from(index).ok())?;
    match source_ty.deref(ctx).shape().get(dim_index) {
        Some(Dimension::Static(size)) => Some(Box::new(ConstantIndexAttr {
            constant_index: *size,
        }) as AttrObj),
        // A dynamic dimension, or an index out of range.
        Some(Dimension::Dynamic) | None => None,
    }
}

/// `memref.dim %m, %c` with a constant `%c` simplifies to the size obtained from `%m`'s
/// [DynamicDimensionOperandsOp] impl.
///
/// [DynamicDimensionOperandsOp]: crate::memref::op_interfaces::DynamicDimensionOperandsOp
#[op_interface_impl]
impl CanonicalizeInterface for DimOp {
    fn canonicalize(&self, ctx: &mut Context, rewriter: &mut dyn Rewriter) -> IRStatus {
        let source = self.get_source_memref(ctx);
        let dim_index = self.get_dimension_index(ctx);
        forward_dim_size(ctx, rewriter, self.get_operation(), source, dim_index)
    }
}

/// Try to canonicalize `dim_op` by forwarding the size of `dim` from `source`'s
/// [DynamicDimensionOperandsOp] impl.
///
/// [DynamicDimensionOperandsOp]: crate::memref::op_interfaces::DynamicDimensionOperandsOp
pub(crate) fn forward_dim_size(
    ctx: &mut Context,
    rewriter: &mut dyn Rewriter,
    dim_op: Ptr<Operation>,
    source: Value,
    dim: Value,
) -> IRStatus {
    let Some(dim) = try_match_constant_index(ctx, dim).and_then(|dim| usize::try_from(dim).ok())
    else {
        return IRStatus::Unchanged;
    };
    let Some(size) = dynamic_dim_size(ctx, source, dim) else {
        return IRStatus::Unchanged;
    };
    rewriter.replace_operation_with_values(ctx, dim_op, vec![size]);
    IRStatus::Changed
}
