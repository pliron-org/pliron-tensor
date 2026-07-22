// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Types for the memref dialect.

use pliron::{
    derive::{pliron_type, type_interface_impl},
    r#type::TypeHandle,
};

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Ranked memref type.
#[pliron_type(
    name = "memref.ranked",
    format = "`<` vec($shape, Char(`x`)) ` : ` $element_type `>`",
    verifier = "succ",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedMemrefType {
    element_type: TypeHandle,
    shape: Vec<Dimension>,
}

#[type_interface_impl]
impl MultiDimensionalType for RankedMemrefType {
    fn element_type(&self) -> TypeHandle {
        self.element_type
    }
}

#[type_interface_impl]
impl ShapedType for RankedMemrefType {
    /// Get the shape of the ranked memref.
    fn shape(&self) -> &Vec<Dimension> {
        &self.shape
    }
}

/// Unranked memref type.
#[pliron_type(
    name = "memref.unranked",
    format = "`<` $element_type `>`",
    verifier = "succ"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedMemrefType {
    element_type: TypeHandle,
}

#[type_interface_impl]
impl MultiDimensionalType for UnrankedMemrefType {
    fn element_type(&self) -> TypeHandle {
        self.element_type
    }
}
