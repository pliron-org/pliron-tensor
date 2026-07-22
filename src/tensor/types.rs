// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor types and related functionality.

use pliron::derive::{pliron_type, type_interface_impl};
use pliron::r#type::TypeHandle;

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Ranked tensor type.
#[pliron_type(
    name = "tensor.ranked",
    format = "`<` vec($shape, Char(`x`)) ` : ` $element_type `>`",
    verifier = "succ",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedTensorType {
    element_type: TypeHandle,
    shape: Vec<Dimension>,
}

#[type_interface_impl]
impl MultiDimensionalType for RankedTensorType {
    fn element_type(&self) -> TypeHandle {
        self.element_type
    }
}

#[type_interface_impl]
impl ShapedType for RankedTensorType {
    /// Get the shape of the ranked tensor.
    fn shape(&self) -> &Vec<Dimension> {
        &self.shape
    }
}

/// Unranked tensor type.
#[pliron_type(
    name = "tensor.unranked",
    format = "`<` $element_type `>`",
    verifier = "succ",
    generate_get = true
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct UnrankedTensorType {
    element_type: TypeHandle,
}

#[type_interface_impl]
impl MultiDimensionalType for UnrankedTensorType {
    fn element_type(&self) -> TypeHandle {
        self.element_type
    }
}
