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
    r#type::{Type, TypeHandle, type_cast},
};

/// Interface for converting to a Memref type.
#[type_interface]
pub trait ToMemrefType {
    /// Convert [self] to a Memref type.
    fn convert(&self, ctx: &Context) -> Result<TypeHandle>;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Convert `ty` to its memref equivalent.
///
/// A type that does not implement [ToMemrefType] is simply returned.
pub fn to_memref_type(ty: TypeHandle, ctx: &Context) -> Result<TypeHandle> {
    match type_cast::<dyn ToMemrefType>(&*ty.deref(ctx)) {
        Some(converter) => converter.convert(ctx),
        None => Ok(ty),
    }
}
