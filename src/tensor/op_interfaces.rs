// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor op interfaces

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface, OneResultInterface,
    },
    context::Context,
    derive::op_interface,
    op::Op,
    operation::Operation,
    result::Result,
    value::Value,
};

use crate::memref::op_interfaces::CompatibleShapesOp;

use super::types::RankedTensorType;

/// Error for element-wise binary arithmetic tensor ops verification.
#[derive(thiserror::Error, Debug)]
pub enum ElementWiseBinaryTensorOpErr {
    #[error("Element-wise binary tensor op must have exactly 2 operands")]
    InvalidNumOperands,
    #[error("Element-wise binary tensor op result type must be a RankedTensorType")]
    InvalidResult,
}

/// Interface for element-wise binary arithmetic tensor ops (e.g., AddOp).
/// These ops must have exactly 2 operands and 1 result,
/// and the operand and result types must all be the RankedTensorType
/// with the same rank, shape (for non-dynamic dimensions) and element type.
#[op_interface]
pub trait ElementWiseBinaryTensorOpInterface:
    OneResultInterface
    + NResultsInterface<1>
    + AllResultsOfType<RankedTensorType>
    + AllOperandsOfType<RankedTensorType>
    + NOpdsInterface<2>
    + CompatibleShapesOp<RankedTensorType>
{
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn new(ctx: &mut Context, lhs: Value, rhs: Value) -> Self
    where
        Self: Sized,
    {
        use pliron::r#type::Typed;
        let operands = vec![lhs, rhs];
        let result_types = vec![lhs.get_type(ctx)];
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            result_types,
            operands,
            vec![],
            0,
        );
        Self::from_operation(op)
    }
}
