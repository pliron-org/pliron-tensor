// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Memref attributes.

use pliron::derive::{format, pliron_attr};

/// Represents a slice parameter (offset, size, or step) that can be either
/// a static usize constant or a dynamic Index value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[format]
pub enum SliceParamAttr {
    /// A static usize constant value
    Static(usize),
    /// Refers to the operand at the given index in the containing operation
    OperandIdx(usize),
}

/// Stores the slice parameters (offsets, sizes, steps) for slice-like operations.
/// Each dimension can have static or dynamic offsets, sizes, and steps.
///
/// Prints/parses as: `[offset0, offset1, ...] [size0, size1, ...] [step0, step1, ...]`
/// where each element is either `Static(n)` or `OperandIdx(n)`.
#[pliron_attr(
    name = "memref.slice_params",
    format = "`[` vec($offsets, CharSpace(`,`)) `]` `[` vec($sizes, CharSpace(`,`)) `]` `[` vec($steps, CharSpace(`,`)) `]`",
    verifier = "succ"
)]
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SliceParamsAttr {
    /// For each dimension: the offset (static or dynamic)
    pub offsets: Vec<SliceParamAttr>,
    /// For each dimension: the size (static or dynamic)
    pub sizes: Vec<SliceParamAttr>,
    /// For each dimension: the step (static or dynamic)
    pub steps: Vec<SliceParamAttr>,
}

#[pliron_attr(name = "memref.const_pointer", format = "$0", verifier = "succ")]
#[derive(PartialEq, Eq, Clone, Copy, Debug, Hash)]
pub struct ConstPointerAttr(pub *const ());
unsafe impl Send for ConstPointerAttr {}
unsafe impl Sync for ConstPointerAttr {}
