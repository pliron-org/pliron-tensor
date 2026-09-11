// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Type interfaces for the memref dialect.

use pliron::{
    builtin::attr_interfaces::TypedAttrInterface,
    builtin::{type_interfaces::FloatTypeInterface, types::IntegerType},
    combine::{self, Parser},
    context::Context,
    derive::type_interface,
    parsable::{Parsable, StateStream},
    printable::Printable,
    result::Result,
    r#type::{Type, TypeHandle, TypeInterfaceHandle, type_cast, type_impls},
    verify_err_noloc,
};

/// Each dimension in a shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Dimension {
    Dynamic,
    Static(usize),
}

impl Printable for Dimension {
    fn fmt(
        &self,
        _ctx: &pliron::context::Context,
        _state: &pliron::printable::State,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            Dimension::Dynamic => write!(f, "?"),
            Dimension::Static(size) => write!(f, "{}", size),
        }
    }
}

impl Parsable for Dimension {
    type Arg = ();
    type Parsed = Self;
    fn parse<'a>(
        state_stream: &mut pliron::parsable::StateStream<'a>,
        _arg: Self::Arg,
    ) -> pliron::parsable::ParseResult<'a, Self::Parsed> {
        combine::parser::char::char('?')
            .map(|_| Dimension::Dynamic)
            .or(usize::parser(()).map(Dimension::Static))
            .parse_stream(state_stream)
            .into_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum MultiDimensionalTypeErr {
    #[error("Invalid element type {0} for multi-dimensional type")]
    InvalidElementType(String),
}

#[type_interface]
pub trait MultiDimensionalType {
    /// Get the element type of the multi-dimensional type.
    fn element_type(&self) -> TypeHandle;

    /// Verify the invariants of the multi-dimensional type.
    fn verify(ty: &dyn Type, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let ty = type_cast::<dyn MultiDimensionalType>(ty)
            .expect("We cannot be here if the type does not implement MultiDimensionalType");
        let el_ty = &*ty.element_type().deref(ctx);
        // TODO: Expand later as necessary.
        if !(el_ty.is::<IntegerType>() || type_impls::<dyn FloatTypeInterface>(el_ty)) {
            let ty_str = format!("{}", el_ty.disp(ctx));
            return verify_err_noloc!(MultiDimensionalTypeErr::InvalidElementType(ty_str));
        }
        Ok(())
    }
}

#[type_interface]
pub trait ShapedType: MultiDimensionalType {
    /// Get the shape of the shaped type.
    fn shape(&self) -> &Vec<Dimension>;

    /// Are all dimensions static?
    fn has_static_shape(&self) -> bool {
        self.shape()
            .iter()
            .all(|dim| matches!(dim, Dimension::Static(_)))
    }

    /// Number of dynamic dimensions.
    fn num_dynamic_dimensions(&self) -> usize {
        self.shape()
            .iter()
            .filter(|dim| matches!(dim, Dimension::Dynamic))
            .count()
    }

    /// Number of dimensions (Rank).
    fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Number of elements: the product of the extents of all dimensions.
    ///
    /// The result is `None` if any dimension is dynamic.
    fn num_elements(&self) -> Option<usize> {
        self.shape()
            .iter()
            .map(|dim| match dim {
                Dimension::Static(extent) => Some(*extent),
                Dimension::Dynamic => None,
            })
            .product()
    }

    /// Verify the invariants of the shaped type.
    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// The type of an element in a [DenseElementsAttr](crate::memref::attributes::DenseElementsAttr).
#[type_interface]
pub trait DenseElementType {
    /// Get the number of bytes that one element uses in the raw buffer:
    /// `ceildiv(bitwidth, 8)`.
    fn element_size(&self) -> usize;

    /// Decode one element and get the builtin scalar attribute for it.
    fn element_attr(&self, ctx: &Context, bytes: &[u8]) -> Box<dyn TypedAttrInterface>;

    /// Write one element as a literal.
    fn print_element(&self, bytes: &[u8], f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result;

    /// Get a parser for one element literal. The parser output is the raw bytes.
    fn element_parser<'a>(
        &self,
    ) -> Box<dyn Parser<StateStream<'a>, Output = Vec<u8>, PartialState = ()> + 'a>;

    fn verify(_ty: &dyn Type, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// Alias for a handle around [DenseElementType].
pub type DenseElementTypeHandle = TypeInterfaceHandle<dyn DenseElementType>;
