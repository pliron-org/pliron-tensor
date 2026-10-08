// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Compile-time information about memref offsets and strides.

use pliron::derive::format;

use super::type_interfaces::Dimension;
use Dimension::{Dynamic, Static};

/// An offset and one stride per dimension, in element units.
/// Element position: `offset + (index[0] * stride[0]) + ... + (index[n-1] * stride[n-1])`.
#[format("`[` vec($strides, CharSpace(`,`)) `]` `, ` `offset` `: ` $offset")]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct StridedLayout {
    pub offset: Dimension,
    pub strides: Vec<Dimension>,
}

/// The layout of a ranked memref.
/// `None` denotes the identity layout: offset zero and [identity strides](identity_strides).
pub type MemrefLayout = Option<StridedLayout>;

/// Multiply two dimensions. A dynamic operand or an overflow gives a dynamic result.
fn multiply(a: &Dimension, b: &Dimension) -> Dimension {
    match (a, b) {
        (Static(a), Static(b)) => a.checked_mul(*b).map(Static).unwrap_or(Dynamic),
        _ => Dynamic,
    }
}

/// Add two dimensions. A dynamic operand or an overflow gives a dynamic result.
fn add(a: &Dimension, b: &Dimension) -> Dimension {
    match (a, b) {
        (Static(a), Static(b)) => a.checked_add(*b).map(Static).unwrap_or(Dynamic),
        _ => Dynamic,
    }
}

/// Compute the identity strides of `shape`: `stride[i] = size[i+1] * ... * size[n-1]`.
/// A dynamic size or an overflow gives a dynamic stride: `3x?x4` gives `[?, 4, 1]`.
pub fn identity_strides(shape: &[Dimension]) -> Vec<Dimension> {
    let mut stride = Static(1);
    let mut strides = Vec::with_capacity(shape.len());
    for size in shape.iter().rev() {
        strides.push(stride.clone());
        stride = multiply(&stride, size);
    }
    strides.reverse();
    strides
}

/// Get the offset and strides of a [MemrefLayout].
/// `shape` is used for computing the [identity strides](identity_strides).
pub fn as_strided(layout: &MemrefLayout, shape: &[Dimension]) -> (Dimension, Vec<Dimension>) {
    match layout {
        None => (Static(0), identity_strides(shape)),
        Some(StridedLayout { offset, strides }) => (offset.clone(), strides.clone()),
    }
}

/// Canonicalize a layout with offset zero and static [identity strides](identity_strides)
/// of `shape` to `None`.
pub fn canonicalize(layout: MemrefLayout, shape: &[Dimension]) -> MemrefLayout {
    if let Some(StridedLayout {
        offset: Static(0),
        strides,
    }) = &layout
        && strides.iter().all(|s| matches!(s, Static(_)))
        && *strides == identity_strides(shape)
    {
        None
    } else {
        layout
    }
}

/// Get the most static layout that both inputs can use.
pub fn merge(a: &MemrefLayout, b: &MemrefLayout, shape: &[Dimension]) -> MemrefLayout {
    // Two identity layouts merge to identity.
    if a.is_none() && b.is_none() {
        return None;
    }
    let (a_offset, a_strides) = as_strided(a, shape);
    let (b_offset, b_strides) = as_strided(b, shape);
    let field = |a: Dimension, b: Dimension| if a == b { a } else { Dynamic };
    canonicalize(
        Some(StridedLayout {
            offset: field(a_offset, b_offset),
            strides: a_strides
                .into_iter()
                .zip(b_strides)
                .map(|(a, b)| field(a, b))
                .collect(),
        }),
        shape,
    )
}

/// Get a layout with no static offset or stride information.
pub fn fully_dynamic(rank: usize) -> MemrefLayout {
    Some(StridedLayout {
        offset: Dynamic,
        strides: vec![Dynamic; rank],
    })
}

/// Can we cast from `from` to `to`?
/// A cast can remove static information, but cannot add it.
pub fn is_cast_compatible(from: &MemrefLayout, to: &MemrefLayout, shape: &[Dimension]) -> bool {
    let (from_offset, from_strides) = as_strided(from, shape);
    let (to_offset, to_strides) = as_strided(to, shape);
    let field = |from: &Dimension, to: &Dimension| matches!(to, Dynamic) || from == to;
    from_strides.len() == shape.len()
        && to_strides.len() == shape.len()
        && field(&from_offset, &to_offset)
        && from_strides
            .iter()
            .zip(&to_strides)
            .all(|(from, to)| field(from, to))
}

/// Compute a view's offset and strides from the source descriptor fields.
///
/// Dropped dimensions:
///   - Must have size one. Thus their index in the view is always zero.
///   - Add their offset to the view offset.
///   - Have no size or stride in the result.
pub fn subview_layout(
    source_shape: &[Dimension],
    source_layout: &MemrefLayout,
    offsets: &[Dimension],
    sizes: &[Dimension],
    steps: &[Dimension],
    dropped_dims: &[usize],
) -> MemrefLayout {
    let (offset, strides) = as_strided(source_layout, source_shape);
    let offset = offsets
        .iter()
        .zip(&strides)
        .fold(offset, |offset, (index, stride)| {
            add(&offset, &multiply(index, stride))
        });
    let kept = |dim: &usize| !dropped_dims.contains(dim);
    let strides = strides
        .iter()
        .zip(steps)
        .enumerate()
        .filter(|(dim, _)| kept(dim))
        .map(|(_, (stride, step))| multiply(stride, step))
        .collect();
    let sizes = sizes
        .iter()
        .enumerate()
        .filter(|(dim, _)| kept(dim))
        .map(|(_, size)| size.clone())
        .collect::<Vec<_>>();
    canonicalize(Some(StridedLayout { offset, strides }), &sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each identity stride must depend only on the later sizes, and must be dynamic
    /// if a later size is dynamic or the product overflows.
    #[test]
    fn strides() {
        assert_eq!(identity_strides(&[]), vec![]);
        assert_eq!(
            identity_strides(&[Static(3), Static(4)]),
            vec![Static(4), Static(1)]
        );
        assert_eq!(
            identity_strides(&[Dynamic, Static(4)]),
            vec![Static(4), Static(1)]
        );
        assert_eq!(
            identity_strides(&[Static(3), Dynamic, Static(4)]),
            vec![Dynamic, Static(4), Static(1)]
        );
        assert_eq!(
            identity_strides(&[Static(2), Static(usize::MAX), Static(2)]),
            vec![Dynamic, Static(2), Static(1)]
        );
    }

    /// A strided layout equal to identity must canonicalize to identity. A merge must keep
    /// only the static fields common to both inputs, and casts must only remove static fields.
    #[test]
    fn canonical_forms_and_merges() {
        let shape = [Static(2), Static(2)];
        let identity = None;
        assert_eq!(
            as_strided(&identity, &shape),
            (Static(0), vec![Static(2), Static(1)])
        );
        assert_eq!(
            canonicalize(
                Some(StridedLayout {
                    offset: Static(0),
                    strides: vec![Static(2), Static(1)]
                }),
                &shape
            ),
            identity
        );
        assert_eq!(
            canonicalize(
                Some(StridedLayout {
                    offset: Static(0),
                    strides: vec![]
                }),
                &[]
            ),
            identity
        );
        let view = Some(StridedLayout {
            offset: Static(1),
            strides: vec![Static(4), Static(1)],
        });
        let merged = merge(&identity, &view, &shape);
        assert_eq!(
            merged,
            Some(StridedLayout {
                offset: Dynamic,
                strides: vec![Dynamic, Static(1)]
            })
        );
        assert!(is_cast_compatible(&identity, &merged, &shape));
        assert!(is_cast_compatible(&view, &merged, &shape));
        assert!(!is_cast_compatible(&merged, &view, &shape));
        assert!(!is_cast_compatible(&view, &identity, &shape));
        assert!(is_cast_compatible(&view, &fully_dynamic(2), &shape));
        assert_eq!(merge(&identity, &identity, &[Static(2), Dynamic]), identity);
    }

    /// A view must compose the offset and strides of its source, and a dynamic input must
    /// give a dynamic field.
    #[test]
    fn views() {
        let shape = [Static(3), Static(4)];
        let sizes = [Static(2), Static(2)];
        let view = subview_layout(
            &shape,
            &None,
            &[Static(0), Static(1)],
            &sizes,
            &[Static(1), Static(1)],
            &[],
        );
        assert_eq!(
            view,
            Some(StridedLayout {
                offset: Static(1),
                strides: vec![Static(4), Static(1)]
            })
        );
        assert_eq!(
            subview_layout(
                &sizes,
                &view,
                &[Static(1), Static(0)],
                &[Static(1), Static(2)],
                &[Static(2), Static(1)],
                &[]
            ),
            Some(StridedLayout {
                offset: Static(5),
                strides: vec![Static(8), Static(1)]
            })
        );
        assert_eq!(
            subview_layout(
                &shape,
                &None,
                &[Dynamic, Static(0)],
                &sizes,
                &[Static(1), Dynamic],
                &[]
            ),
            Some(StridedLayout {
                offset: Dynamic,
                strides: vec![Static(4), Dynamic]
            })
        );
    }

    /// A rank-reducing view must add the offset of a dropped dimension, and must remove
    /// its stride.
    #[test]
    fn rank_reducing_views() {
        let shape = [Static(2), Static(3), Static(4)];
        let strided = Some(StridedLayout {
            offset: Static(5),
            strides: vec![Static(16), Static(4), Static(1)],
        });
        assert_eq!(
            subview_layout(
                &shape,
                &strided,
                &[Static(1), Static(0), Static(0)],
                &[Static(1), Static(2), Static(3)],
                &[Static(1), Static(1), Static(1)],
                &[0]
            ),
            Some(StridedLayout {
                offset: Static(21),
                strides: vec![Static(4), Static(1)]
            })
        );
        assert_eq!(
            subview_layout(
                &shape,
                &None,
                &[Dynamic, Static(0), Static(0)],
                &[Static(1), Static(3), Static(4)],
                &[Static(1), Static(1), Static(1)],
                &[0]
            ),
            Some(StridedLayout {
                offset: Dynamic,
                strides: vec![Static(4), Static(1)]
            })
        );
        // The first batch of an identity buffer must be an identity view.
        assert_eq!(
            subview_layout(
                &shape,
                &None,
                &[Static(0), Static(0), Static(0)],
                &[Static(1), Static(3), Static(4)],
                &[Static(1), Static(1), Static(1)],
                &[0]
            ),
            None
        );
        // A dropped middle dimension must remove the middle stride.
        assert_eq!(
            subview_layout(
                &shape,
                &None,
                &[Static(0), Static(2), Static(0)],
                &[Static(2), Static(1), Static(4)],
                &[Static(1), Static(1), Static(1)],
                &[1]
            ),
            Some(StridedLayout {
                offset: Static(8),
                strides: vec![Static(12), Static(1)]
            })
        );
    }
}
