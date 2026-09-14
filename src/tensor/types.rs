// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor types and related functionality.

use pliron::derive::{pliron_type, type_interface_impl};
use pliron::r#type::TypeHandle;

use crate::memref::type_interfaces::{Dimension, MultiDimensionalType, ShapedType};

/// Compute the broadcast shape of two ranked values using
/// [NumPy-style rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules).
/// Dynamic dimensions are compatible with any dimension and remain dynamic
/// unless the other dimension is statically one.
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

/// Return whether `source` can be broadcast to `result`.
/// See [NumPy broadcast rules](https://numpy.org/doc/stable/user/basics.broadcasting.html#general-broadcasting-rules).
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
