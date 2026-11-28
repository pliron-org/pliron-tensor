// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor attribute interfaces

use pliron::{
    attribute::Attribute, context::Context, derive::attr_interface, result::Result,
    r#type::TypeHandle,
};

use crate::memref::type_interfaces::Dimension;

/// Verification for a tensor encoding.
///
/// Pliron's verifiers only provide `self` and `&Context`.
/// An encoding requires additional details of the tensor type.
#[attr_interface]
pub trait VerifiableTensorEncoding {
    /// Verify this encoding for a tensor with `shape` and `element_type`.
    fn verify_encoding(
        &self,
        ctx: &Context,
        shape: &[Dimension],
        element_type: TypeHandle,
    ) -> Result<()>;

    fn verify(_attr: &dyn Attribute, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
