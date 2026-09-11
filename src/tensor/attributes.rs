// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor attributes and related functionality.

// These attributes live in the memref dialect since they are shared between
// memref and tensor ops
pub use crate::memref::attributes::{
    DenseElementsAttr, DenseElementsErr, SliceParamAttr, SliceParamsAttr,
};
