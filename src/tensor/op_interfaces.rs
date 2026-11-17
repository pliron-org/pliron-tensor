// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor op interfaces

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, NOpdsInterface, NResultsInterface, OneOpdInterface,
        OneResultInterface,
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

/// Interface for element-wise binary arithmetic tensor ops.
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
        Operation::new_op::<Self>(ctx, result_types, operands, vec![], 0)
    }
}

/// Interface for element-wise unary tensor ops.
#[op_interface]
pub trait ElementWiseUnaryTensorOpInterface:
    OneResultInterface
    + NResultsInterface<1>
    + AllResultsOfType<RankedTensorType>
    + AllOperandsOfType<RankedTensorType>
    + OneOpdInterface
    + CompatibleShapesOp<RankedTensorType>
{
    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }

    fn new(ctx: &mut Context, input: Value) -> Self
    where
        Self: Sized,
    {
        use pliron::r#type::Typed;
        let result_types = vec![input.get_type(ctx)];
        Operation::new_op::<Self>(ctx, result_types, vec![input], vec![], 0)
    }
}
