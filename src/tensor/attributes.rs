// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor attributes and related functionality.

use pliron::derive::pliron_attr;

/// Signedness information for element-wise casts. Integer types in Pliron are
/// signless, so conversions involving integers need this semantic information
/// to select the appropriate LLVM operation.
#[pliron_attr(name = "tensor.cast_signedness", format, verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Default, Hash)]
pub struct CastSignednessAttr {
    pub input_is_signed: bool,
    pub result_is_signed: bool,
}

// These attributes live in the memref dialect since they are shared between
// memref and tensor ops
pub use crate::memref::attributes::{
    DenseElementsAttr, DenseElementsErr, SliceParamAttr, SliceParamsAttr,
};
