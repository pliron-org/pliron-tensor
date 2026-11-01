// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Implementation of various interfaces for the memref dialect.

use pliron::{
    attribute::AttrObj,
    context::Context,
    derive::op_interface_impl,
    irbuild::{IRStatus, rewriter::Rewriter},
    opts::constants::ConstFoldInterface,
    r#type::Typed,
};
use pliron_common_dialects::index::attributes::{ConstantIndexAttr, known_constant_index};

use crate::memref::{
    attributes::ShapedTypeHandle,
    ops::{DimOp, GetGlobalOp, MemrefCastOp, ReshapeOp, SubviewOp, YieldOp},
    type_interfaces::Dimension,
};

impl_no_self_side_effects!(
    YieldOp,
    DimOp,
    SubviewOp,
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

        let Some(dim_index) =
            known_constant_index(dim_index).and_then(|index| usize::try_from(index).ok())
        else {
            // Cannot determine the index statically.
            return vec![None];
        };

        let memref_ty = self.get_source_memref(ctx).get_type(ctx);
        let memref_ty = ShapedTypeHandle::from_handle(memref_ty, ctx)
            .expect("DimOp's source must be a ranked memref (and hence, shaped) type");
        let folded = match memref_ty.deref(ctx).shape().get(dim_index) {
            Some(Dimension::Static(extent)) => Some(Box::new(ConstantIndexAttr {
                constant_index: *extent,
            }) as AttrObj),
            // A dynamic extent, or an index out of range.
            Some(Dimension::Dynamic) | None => None,
        };
        vec![folded]
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
