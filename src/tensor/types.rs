// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor types and related functionality.

use pliron::attribute::{AttrObj, attr_cast};
use pliron::common_traits::Verify;
use pliron::context::Context;
use pliron::derive::{pliron_type, type_interface_impl};
use pliron::result::Result;
use pliron::r#type::{Type, TypeHandle, TypedHandle};

use crate::tensor::attr_interfaces::VerifiableTensorEncoding;

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Compute the broadcast shape of two ranked values using
/// [NumPy-style rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules).
/// A dynamic dimension is compatible with any dimension, and the resulting dimension is dynamic.
pub fn broadcast_shapes(lhs: &[Dimension], rhs: &[Dimension]) -> Option<Vec<Dimension>> {
    let rank = lhs.len().max(rhs.len());
    let aligned = |shape: &[Dimension], i: usize| {
        (i + shape.len())
            .checked_sub(rank)
            .map(|index| shape[index].clone())
    };

    (0..rank)
        .map(|i| match (aligned(lhs, i), aligned(rhs, i)) {
            (Some(dim), None) | (None, Some(dim)) => Some(dim),
            (Some(Dimension::Static(1)), Some(dim)) | (Some(dim), Some(Dimension::Static(1))) => {
                Some(dim)
            }
            (Some(Dimension::Static(lhs)), Some(Dimension::Static(rhs))) => {
                (lhs == rhs).then_some(Dimension::Static(lhs))
            }
            (Some(Dimension::Dynamic), Some(_)) | (Some(_), Some(Dimension::Dynamic)) => {
                Some(Dimension::Dynamic)
            }
            (None, None) => unreachable!(),
        })
        .collect()
}

/// Can `source` be broadcast to `result`.
/// See [BroadcastOp](crate::tensor::ops::BroadcastOp) rules.
pub fn can_broadcast_to(source: &[Dimension], result: &[Dimension]) -> bool {
    source.len() <= result.len()
        && source
            .iter()
            .rev()
            .zip(result.iter().rev())
            .all(|(src, dst)| {
                matches!(src, Dimension::Static(1) | Dimension::Dynamic)
                    || matches!(dst, Dimension::Dynamic)
                    || src == dst
            })
}

/// Ranked tensor type.
#[pliron_type(
    name = "tensor.ranked",
    format = "`<` vec($shape, Char(`x`)) ` : ` $element_type opt($encoding, delimiters(`, `, ``)) `>`"
)]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RankedTensorType {
    element_type: TypeHandle,
    shape: Vec<Dimension>,
    encoding: Option<AttrObj>,
}

impl RankedTensorType {
    /// Get a ranked tensor type.
    pub fn get(
        ctx: &Context,
        element_type: TypeHandle,
        shape: Vec<Dimension>,
    ) -> TypedHandle<Self> {
        Self::get_with_encoding(ctx, element_type, shape, None)
    }

    /// Get a ranked tensor type with an optional encoding.
    pub fn get_with_encoding(
        ctx: &Context,
        element_type: TypeHandle,
        shape: Vec<Dimension>,
        encoding: Option<AttrObj>,
    ) -> TypedHandle<Self> {
        Self::instantiate(
            Self {
                element_type,
                shape,
                encoding,
            },
            ctx,
        )
    }

    /// Get this tensor's encoding.
    pub fn encoding(&self) -> Option<&AttrObj> {
        self.encoding.as_ref()
    }
}

impl Verify for RankedTensorType {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if let Some(encoding) = &self.encoding {
            encoding.verify(ctx)?;
            if let Some(encoding) = attr_cast::<dyn VerifiableTensorEncoding>(&**encoding) {
                encoding.verify_encoding(ctx, &self.shape, self.element_type)?;
            }
        }
        Ok(())
    }
}

#[type_interface_impl]
impl MultiDimensionalType for RankedTensorType {
    fn element_type(&self) -> TypeHandle {
        self.element_type
    }
}

#[type_interface_impl]
impl ShapedType for RankedTensorType {
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

#[cfg(test)]
mod tests {
    use super::{broadcast_shapes, can_broadcast_to};
    use crate::memref::type_interfaces::Dimension::{Dynamic, Static};

    #[test]
    fn broadcast_shape_rules() {
        assert_eq!(
            broadcast_shapes(
                &[Static(1), Static(2), Static(3), Static(4)],
                &[Static(1), Static(1), Static(1), Static(4)]
            ),
            Some(vec![Static(1), Static(2), Static(3), Static(4)])
        );
        assert_eq!(
            broadcast_shapes(&[Static(3), Static(1)], &[Static(2), Static(1), Static(4)]),
            Some(vec![Static(2), Static(3), Static(4)])
        );
        assert_eq!(broadcast_shapes(&[Static(2)], &[Static(3)]), None);
        assert_eq!(
            broadcast_shapes(&[Dynamic, Static(4)], &[Static(2), Static(4)]),
            Some(vec![Dynamic, Static(4)])
        );
    }

    #[test]
    fn broadcast_to_rules() {
        assert!(can_broadcast_to(
            &[Static(1), Static(4)],
            &[Static(1), Static(2), Static(3), Static(4)]
        ));
        assert!(can_broadcast_to(
            &[Dynamic, Static(4)],
            &[Static(3), Static(4)]
        ));
        assert!(!can_broadcast_to(
            &[Static(2), Static(4)],
            &[Static(3), Static(4)]
        ));
        assert!(!can_broadcast_to(
            &[Static(1), Static(2), Static(3)],
            &[Static(2), Static(3)]
        ));
    }
}
