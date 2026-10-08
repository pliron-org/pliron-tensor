// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Types for the memref dialect.

use pliron::{
    common_traits::Verify,
    context::Context,
    derive::{pliron_type, type_interface_impl},
    result::Result,
    r#type::{Type, TypeHandle, TypedHandle},
    verify_err_noloc,
};

use crate::memref::layout::{MemrefLayout, StridedLayout, canonicalize};
use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Ranked memref type.
#[pliron_type(
    name = "memref.ranked",
    format = "`<` vec($shape, Char(`x`)) ` : ` $element_type opt($layout, delimiters(`, strided<`, `>`)) `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedMemrefType {
    element_type: TypeHandle,
    shape: Vec<Dimension>,
    layout: Option<StridedLayout>,
}

impl RankedMemrefType {
    /// Get a memref with a canonical layout.
    pub fn get(
        ctx: &Context,
        element_type: TypeHandle,
        shape: Vec<Dimension>,
        layout: MemrefLayout,
    ) -> TypedHandle<Self> {
        let layout = canonicalize(layout, &shape);
        Self::instantiate(
            Self {
                element_type,
                shape,
                layout,
            },
            ctx,
        )
    }

    /// Get the layout information.
    pub fn layout(&self) -> &MemrefLayout {
        &self.layout
    }

    /// Get the same element type and shape with identity layout.
    pub fn with_identity_layout(&self, ctx: &Context) -> TypedHandle<Self> {
        Self::get(ctx, self.element_type, self.shape.clone(), None)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RankedMemrefTypeVerifyErr {
    #[error("The number of strides ({got}) must equal the rank ({expected})")]
    StrideCountMismatch { expected: usize, got: usize },
    #[error(
        "An identity layout (offset zero, static identity strides) must not be written as an explicit strided layout"
    )]
    NonCanonicalLayout,
}

impl Verify for RankedMemrefType {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        if let Some(StridedLayout { strides, .. }) = &self.layout
            && strides.len() != self.shape.len()
        {
            return verify_err_noloc!(RankedMemrefTypeVerifyErr::StrideCountMismatch {
                expected: self.shape.len(),
                got: strides.len()
            });
        }
        // The parser does not use `get`, and hence does not canonicalize.
        // TODO: Come up with a way to have the parser canonicalize
        // (we could always implement manual formatting).
        if canonicalize(self.layout.clone(), &self.shape) != self.layout {
            return verify_err_noloc!(RankedMemrefTypeVerifyErr::NonCanonicalLayout);
        }
        Ok(())
    }
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
