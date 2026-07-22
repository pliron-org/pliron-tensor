// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Memref dialect for pliron.

pub mod attributes;
pub mod conversions;
pub mod descriptor;
pub mod op_interfaces;
pub mod ops;
pub mod type_interfaces;
pub mod types;

use pliron::{
    context::Context,
    derive::type_interface,
    result::Result,
    r#type::{Type, TypeHandle},
};

/// A function pointer type for the [ToMemrefType] interface.
pub type ToMemrefTypeFn = fn(self_ty: TypeHandle, &mut Context) -> Result<TypeHandle>;

/// Interface for converting to a Memref type.
#[type_interface]
pub trait ToMemrefType {
    /// Get a function to convert [self] to a Memref type.
    // We don't directly specify a conversion function here because
    // the caller cannot get `&dyn ToMemrefType` (&self) while also
    // passing `&mut Context` to the conversion function.
    fn converter(&self) -> ToMemrefTypeFn;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}
