// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Memref attributes.

use std::num::NonZero;

use pliron::{
    arg_err_noloc,
    builtin::{
        attr_interfaces::{OutlinedAttr, PrintOnceAttr, TypedAttrInterface},
        attributes::{FPDoubleAttr, FPHalfAttr, FPSingleAttr, IntegerAttr},
        types::{FP16Type, FP32Type, FP64Type, IntegerType, Signedness},
    },
    combine::{self, Parser, parser::char},
    common_traits::Verify,
    context::Context,
    derive::{attr_interface_impl, format, pliron_attr, type_interface_impl},
    input_err, input_error,
    irfmt::parsers::{
        delimited_list_parser, number_as_string_parser, quoted_string_parser, spaced,
    },
    location::Located,
    parsable::{IntoParseResult, Parsable, ParseResult, StateStream, parser_combinator},
    printable::{self, Printable},
    result::{Error, ErrorKind, Result},
    r#type::{TypeHandle, TypeInterfaceHandle},
    utils::{
        apfloat::{Double, Float, Half, Single, float_parser},
        apint::APInt,
    },
    verify_err_noloc,
};

use crate::memref::type_interfaces::{DenseElementType, DenseElementTypeHandle, ShapedType};

/// Alias for a handle around [ShapedType].
pub type ShapedTypeHandle = TypeInterfaceHandle<dyn ShapedType>;

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

/// The maximum number of elements that the printer writes as literals.
/// Above this number, the printer writes the same elements as one hex
/// string of the raw bytes. All elements are written in either case.
const MAX_ELEMENTS_AS_LITERALS: usize = 100;

/// The number of elements above which a [DenseElementsAttr] is printed outlined.
const MAX_INLINE_ELEMENTS: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum DenseElementsErr {
    #[error("{0} is not a supported element type for memref.dense_elements")]
    UnsupportedElementType(String),
    #[error("memref.dense_elements of type {0} needs a fully static shape")]
    DynamicShape(String),
    #[error(
        "memref.dense_elements of type {ty} needs {expected} byte(s) of data \
         (or {element} for a splat), but has {got}"
    )]
    DataSize {
        ty: String,
        expected: usize,
        element: usize,
        got: usize,
    },
    #[error(
        "a splat of memref.dense_elements of type {ty} needs the {expected} byte(s) of one \
         element, but has {got}"
    )]
    SplatSize {
        ty: String,
        expected: usize,
        got: usize,
    },
    #[error("Invalid hex data: {0}")]
    InvalidHex(String),
    #[error("{value} is out of range for the element type {ty} of memref.dense_elements")]
    ElementOutOfRange { ty: String, value: String },
}

/// Get the number of bytes that an integer of `width` bits uses in the buffer.
fn int_element_size(width: u32) -> usize {
    width.div_ceil(8) as usize
}

/// Get the bit-width of an integer element
fn int_element_width(ty: &IntegerType) -> NonZero<usize> {
    NonZero::new(ty.width() as usize).expect("the verifier rejects an element type of no bits")
}

/// Decode an integer element of type `ty` from its raw bytes.
fn decode_int(bytes: &[u8], ty: &IntegerType) -> APInt {
    APInt::from_u8_slice(bytes, int_element_width(ty))
}

fn decode_half(bytes: &[u8]) -> Half {
    Half::from_bits(u16::from_ne_bytes(bytes.try_into().unwrap()) as u128)
}

fn decode_single(bytes: &[u8]) -> Single {
    Single::from_bits(u32::from_ne_bytes(bytes.try_into().unwrap()) as u128)
}

fn decode_double(bytes: &[u8]) -> Double {
    Double::from_bits(u64::from_ne_bytes(bytes.try_into().unwrap()) as u128)
}

/// Parse one `f16` literal and get its raw bytes.
fn parse_half_element<'a>(
    state_stream: &mut StateStream<'a>,
    _arg: (),
) -> ParseResult<'a, Vec<u8>> {
    float_parser::<Half>(())
        .map(|v| (v.to_bits() as u16).to_ne_bytes().to_vec())
        .parse_stream(state_stream)
        .into_result()
}

/// Parse one `f32` literal and get its raw bytes.
fn parse_single_element<'a>(
    state_stream: &mut StateStream<'a>,
    _arg: (),
) -> ParseResult<'a, Vec<u8>> {
    float_parser::<Single>(())
        .map(|v| (v.to_bits() as u32).to_ne_bytes().to_vec())
        .parse_stream(state_stream)
        .into_result()
}

/// Parse one `f64` literal and get its raw bytes.
fn parse_double_element<'a>(
    state_stream: &mut StateStream<'a>,
    _arg: (),
) -> ParseResult<'a, Vec<u8>> {
    float_parser::<Double>(())
        .map(|v| (v.to_bits() as u64).to_ne_bytes().to_vec())
        .parse_stream(state_stream)
        .into_result()
}

/// Parse one integer literal of type `ty` and get its raw bytes.
fn parse_int_element<'a>(
    state_stream: &mut StateStream<'a>,
    ty: IntegerType,
) -> ParseResult<'a, Vec<u8>> {
    let loc = state_stream.loc();
    let (literal, committed) = number_as_string_parser()
        .parse_stream(state_stream)
        .into_result()?;

    let signed = ty.signedness() == Signedness::Signed;
    let out_of_range = || {
        let ty = ty.disp(state_stream.state.ctx).to_string();
        input_err!(
            loc,
            DenseElementsErr::ElementOutOfRange {
                ty,
                value: literal.clone(),
            }
        )
        .into_parse_result()
    };

    let Ok(value) = APInt::from_str(&literal, ty.width() as usize, 10) else {
        return out_of_range();
    };

    // A literal with no sign must not become a negative element.
    if signed && !literal.starts_with('-') && value.to_string_decimal(true).starts_with('-') {
        return out_of_range();
    }

    let mut bytes = vec![0u8; int_element_size(ty.width())];
    value.to_u8_slice(&mut bytes);
    Ok((bytes, committed))
}

#[type_interface_impl]
impl DenseElementType for FP16Type {
    fn element_size(&self) -> usize {
        2
    }

    fn element_attr(&self, _ctx: &Context, bytes: &[u8]) -> Box<dyn TypedAttrInterface> {
        Box::new(FPHalfAttr(decode_half(bytes)))
    }

    fn print_element(&self, bytes: &[u8], f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", decode_half(bytes))
    }

    fn element_parser<'a>(
        &self,
    ) -> Box<dyn Parser<StateStream<'a>, Output = Vec<u8>, PartialState = ()> + 'a> {
        parser_combinator(parse_half_element, ())
    }
}

#[type_interface_impl]
impl DenseElementType for FP32Type {
    fn element_size(&self) -> usize {
        4
    }

    fn element_attr(&self, _ctx: &Context, bytes: &[u8]) -> Box<dyn TypedAttrInterface> {
        Box::new(FPSingleAttr(decode_single(bytes)))
    }

    fn print_element(&self, bytes: &[u8], f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", decode_single(bytes))
    }

    fn element_parser<'a>(
        &self,
    ) -> Box<dyn Parser<StateStream<'a>, Output = Vec<u8>, PartialState = ()> + 'a> {
        parser_combinator(parse_single_element, ())
    }
}

#[type_interface_impl]
impl DenseElementType for FP64Type {
    fn element_size(&self) -> usize {
        8
    }

    fn element_attr(&self, _ctx: &Context, bytes: &[u8]) -> Box<dyn TypedAttrInterface> {
        Box::new(FPDoubleAttr(decode_double(bytes)))
    }

    fn print_element(&self, bytes: &[u8], f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", decode_double(bytes))
    }

    fn element_parser<'a>(
        &self,
    ) -> Box<dyn Parser<StateStream<'a>, Output = Vec<u8>, PartialState = ()> + 'a> {
        parser_combinator(parse_double_element, ())
    }
}

#[type_interface_impl]
impl DenseElementType for IntegerType {
    fn element_size(&self) -> usize {
        int_element_size(self.width())
    }

    fn element_attr(&self, ctx: &Context, bytes: &[u8]) -> Box<dyn TypedAttrInterface> {
        let int_ty = IntegerType::get(ctx, self.width(), self.signedness());
        Box::new(IntegerAttr::new(int_ty, decode_int(bytes, self)))
    }

    fn print_element(&self, bytes: &[u8], f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{}",
            decode_int(bytes, self).to_string_decimal(self.prints_as_signed())
        )
    }

    fn element_parser<'a>(
        &self,
    ) -> Box<dyn Parser<StateStream<'a>, Output = Vec<u8>, PartialState = ()> + 'a> {
        parser_combinator(parse_int_element, self.clone())
    }
}

/// Elements of a constant in a raw byte buffer.
///
/// - Elements are held in row-major order.
/// - Each element uses `ceildiv(bitwidth, 8)` bytes in the native byte order.
/// - A buffer with the width of one element is a *splat*.
#[pliron_attr(name = "memref.dense_elements")]
#[derive(PartialEq, Eq, Clone, Debug, Hash)]
pub struct DenseElementsAttr {
    ty: ShapedTypeHandle,
    data: Vec<u8>,
}

#[attr_interface_impl]
impl TypedAttrInterface for DenseElementsAttr {
    fn get_type(&self, _ctx: &Context) -> TypeHandle {
        self.ty.into()
    }
}

#[attr_interface_impl]
impl OutlinedAttr for DenseElementsAttr {
    fn outline(&self, ctx: &Context) -> bool {
        // A splat prints as one element, whatever the shape of its type.
        if self.is_splat(ctx) {
            return false;
        }
        // A type with no element count has no known printed length either.
        self.num_elements(ctx)
            .is_none_or(|elements| elements > MAX_INLINE_ELEMENTS)
    }
}

/// Two operations that hold the same constant share one outlined entry.
#[attr_interface_impl]
impl PrintOnceAttr for DenseElementsAttr {}

/// Verify `attr` and return any failure as an argument error.
fn verify_arguments(attr: &DenseElementsAttr, ctx: &Context) -> Result<()> {
    if let Err(e @ Error { .. }) = attr.verify(ctx) {
        return Err(Error {
            kind: ErrorKind::InvalidArgument,
            // We reset the error origin to be from here
            backtrace: pliron::std_deps::backtrace::Backtrace::capture(),
            ..e
        });
    }
    Ok(())
}

impl DenseElementsAttr {
    /// Make a constant of type `ty` from `data`. `ty` must have a fully static shape.
    pub fn new(ctx: &Context, ty: ShapedTypeHandle, data: Vec<u8>) -> Result<Self> {
        let attr = DenseElementsAttr { ty, data };
        verify_arguments(&attr, ctx)?;
        Ok(attr)
    }

    /// Make a constant of type `ty` in which every element is `element`.
    /// The bytes of only one element is stored. `ty` must have a fully static shape.
    pub fn new_splat(ctx: &Context, ty: ShapedTypeHandle, element: Vec<u8>) -> Result<Self> {
        let element_type = ty.deref(ctx).element_type();
        let Ok(element_ty) = DenseElementTypeHandle::from_handle(element_type, ctx) else {
            return arg_err_noloc!(DenseElementsErr::UnsupportedElementType(
                element_type.disp(ctx).to_string()
            ));
        };
        let element_size = element_ty.deref(ctx).element_size();
        if element.len() != element_size {
            return arg_err_noloc!(DenseElementsErr::SplatSize {
                ty: ty.disp(ctx).to_string(),
                expected: element_size,
                got: element.len(),
            });
        }
        Self::new(ctx, ty, element)
    }

    /// Make a zero splat of type `ty`.
    pub fn new_zeroed(ctx: &Context, ty: ShapedTypeHandle) -> Result<Self> {
        let element_ty = ty.deref(ctx).element_type();
        let element_size = DenseElementTypeHandle::from_handle(element_ty, ctx)?
            .deref(ctx)
            .element_size();
        Self::new_splat(ctx, ty, vec![0u8; element_size])
    }

    /// The type of this constant.
    pub fn ty(&self) -> ShapedTypeHandle {
        self.ty
    }

    /// Get the raw buffer. Splats are not expanded.
    pub fn raw_data(&self) -> &[u8] {
        &self.data
    }

    /// Get the type of one element of this constant.
    pub fn element_type(&self, ctx: &Context) -> DenseElementTypeHandle {
        self.try_element_type(ctx)
            .expect("The verifier establishes the element type of memref.dense_elements")
    }

    /// Get the type of one element, if the element type has the
    /// [DenseElementType] interface.
    fn try_element_type(&self, ctx: &Context) -> Option<DenseElementTypeHandle> {
        DenseElementTypeHandle::from_handle(self.ty.deref(ctx).element_type(), ctx).ok()
    }

    /// Get the number of bytes that one element uses in the raw buffer.
    pub fn element_size(&self, ctx: &Context) -> usize {
        self.element_type(ctx).deref(ctx).element_size()
    }

    /// Get the number of elements, if all dimensions of the type are static.
    pub fn num_elements(&self, ctx: &Context) -> Option<usize> {
        self.ty.deref(ctx).num_elements()
    }

    /// Do all elements have the same value?
    pub fn is_splat(&self, ctx: &Context) -> bool {
        self.data.len() == self.element_size(ctx)
    }

    /// Are all elements zero?
    pub fn is_all_zero(&self) -> bool {
        self.data.iter().all(|byte| *byte == 0)
    }

    /// Get a clone of the buffer, expanding a splat to all elements.
    pub fn expanded_data(&self, ctx: &Context) -> Vec<u8> {
        match (self.is_splat(ctx), self.num_elements(ctx)) {
            (true, Some(n)) => self.data.repeat(n),
            _ => self.data.clone(),
        }
    }

    /// Get the buffer, expanding a splat to all elements.
    pub fn into_expanded_data(self, ctx: &Context) -> Vec<u8> {
        match (self.is_splat(ctx), self.num_elements(ctx)) {
            (true, Some(n)) => self.data.repeat(n),
            _ => self.data,
        }
    }

    /// Set this constant's type to `ty`, with no changes to the buffer.
    /// The original type is retained on a failure.
    pub fn set_type(&mut self, ctx: &Context, ty: ShapedTypeHandle) -> Result<()> {
        let old_ty = core::mem::replace(&mut self.ty, ty);
        if let Err(err) = verify_arguments(self, ctx) {
            self.ty = old_ty;
            return Err(err);
        }
        Ok(())
    }
}

impl Verify for DenseElementsAttr {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let Some(element_ty) = self.try_element_type(ctx) else {
            let element_type = self.ty.deref(ctx).element_type();
            return verify_err_noloc!(DenseElementsErr::UnsupportedElementType(
                element_type.disp(ctx).to_string()
            ));
        };
        let element_size = element_ty.deref(ctx).element_size();
        if element_size == 0 {
            let element_type = self.ty.deref(ctx).element_type();
            return verify_err_noloc!(DenseElementsErr::UnsupportedElementType(
                element_type.disp(ctx).to_string()
            ));
        }
        let Some(num_elements) = self.ty.deref(ctx).num_elements() else {
            return verify_err_noloc!(DenseElementsErr::DynamicShape(
                self.ty.disp(ctx).to_string()
            ));
        };

        // The data must hold all elements, or one element for a splat.
        let full = num_elements * element_size;
        if self.data.len() != full && self.data.len() != element_size {
            return verify_err_noloc!(DenseElementsErr::DataSize {
                ty: self.ty.disp(ctx).to_string(),
                expected: full,
                element: element_size,
                got: self.data.len(),
            });
        }
        Ok(())
    }
}

impl Printable for DenseElementsAttr {
    fn fmt(
        &self,
        ctx: &Context,
        state: &printable::State,
        f: &mut core::fmt::Formatter<'_>,
    ) -> core::fmt::Result {
        write!(f, "<{} = ", self.ty.print(ctx, state))?;

        let Some(element_ty) = self.try_element_type(ctx) else {
            return write!(f, "{}>", HexData(&self.data));
        };
        let element_ty = element_ty.deref(ctx);
        let size = element_ty.element_size();

        if self.data.len() == size {
            write!(f, "splat ")?;
            element_ty.print_element(&self.data, f)?;
        } else if self.data.len() / size > MAX_ELEMENTS_AS_LITERALS {
            write!(f, "{}", HexData(&self.data))?;
        } else {
            write!(f, "[")?;
            for (i, bytes) in self.data.chunks(size).enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                element_ty.print_element(bytes, f)?;
            }
            write!(f, "]")?;
        }
        write!(f, ">")
    }
}

impl Parsable for DenseElementsAttr {
    type Arg = ();
    type Parsed = Self;

    fn parse<'a>(state_stream: &mut StateStream<'a>, _arg: ()) -> ParseResult<'a, Self> {
        let attr_loc = state_stream.loc();

        // The type comes first so that it can pick the element parser.
        let (ty, _) = combine::between(
            char::char('<').skip(char::spaces()),
            spaced(char::char('=')),
            ShapedTypeHandle::parser(()),
        )
        .parse_stream(state_stream)
        .into_result()?;

        let element_type = ty.deref(state_stream.state.ctx).element_type();
        let Ok(element_ty) =
            DenseElementTypeHandle::from_handle(element_type, state_stream.state.ctx)
        else {
            let msg = element_type.disp(state_stream.state.ctx).to_string();
            return input_err!(attr_loc, DenseElementsErr::UnsupportedElementType(msg))
                .into_parse_result();
        };

        if element_ty.deref(state_stream.state.ctx).element_size() == 0 {
            let msg = element_type.disp(state_stream.state.ctx).to_string();
            return input_err!(attr_loc, DenseElementsErr::UnsupportedElementType(msg))
                .into_parse_result();
        }

        let splat = char::string("splat")
            .skip(char::spaces())
            .with(element_ty.deref(state_stream.state.ctx).element_parser());

        let elements = delimited_list_parser(
            '[',
            ']',
            ',',
            element_ty.deref(state_stream.state.ctx).element_parser(),
        )
        .map(|v| v.concat());

        let hex = combine::parser(|state_stream: &mut StateStream<'a>| {
            let loc = state_stream.loc();
            let (text, _) = quoted_string_parser()
                .parse_stream(state_stream)
                .into_result()?;
            decode_hex(&text)
                .map_err(|err| input_error!(loc, DenseElementsErr::InvalidHex(err.to_string())))
                .into_parse_result()
        });

        let (data, _) = combine::choice((combine::attempt(splat), combine::attempt(elements), hex))
            .skip(spaced(char::char('>')))
            .parse_stream(state_stream)
            .into_result()?;

        DenseElementsAttr::new(state_stream.state.ctx, ty, data)
            .map_err(|e| Error {
                kind: ErrorKind::InvalidInput,
                loc: attr_loc,
                ..e
            })
            .into_parse_result()
    }
}

/// A raw buffer; its format is a quoted hexadecimal string with the prefix `0x`.
struct HexData<'a>(&'a [u8]);

impl core::fmt::Display for HexData<'_> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "\"0x")?;
        for byte in self.0 {
            write!(f, "{byte:02X}")?;
        }
        write!(f, "\"")
    }
}

fn decode_hex(s: &str) -> std::result::Result<Vec<u8>, String> {
    let digits = s
        .strip_prefix("0x")
        .ok_or_else(|| "expected a 0x prefix".to_string())?;
    if digits.len() % 2 != 0 {
        return Err("odd number of hex digits".to_string());
    }
    (0..digits.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&digits[i..i + 2], 16).map_err(|e| e.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use pliron::{
        builtin::types::{FP32Type, IntegerType, Signedness},
        parsable::parse_from_str,
        result::ExpectOk,
    };

    use super::*;
    use crate::{
        memref::{type_interfaces::Dimension, types::RankedMemrefType},
        tensor::types::RankedTensorType,
    };

    /// Make a ranked tensor type with static dimensions.
    fn tensor_of(ctx: &Context, element_type: TypeHandle, shape: Vec<usize>) -> ShapedTypeHandle {
        RankedTensorType::get(
            ctx,
            element_type,
            shape.into_iter().map(Dimension::Static).collect(),
        )
        .into()
    }

    /// Make a ranked tensor type with one dynamic dimension.
    fn dynamic_tensor_of(ctx: &Context, element_type: TypeHandle) -> ShapedTypeHandle {
        RankedTensorType::get(
            ctx,
            element_type,
            vec![Dimension::Dynamic, Dimension::Static(2)],
        )
        .into()
    }

    /// Print `attr`, parse the text again, and compare the two attributes.
    fn round_trip(ctx: &mut Context, attr: &DenseElementsAttr) -> String {
        let printed = attr.disp(ctx).to_string();
        let parsed = parse_from_str(DenseElementsAttr::parser(()), ctx, &printed).expect_ok(ctx);
        assert_eq!(&parsed, attr, "the parsed attribute is different");
        assert_eq!(
            parsed.disp(ctx).to_string(),
            printed,
            "the second print is different"
        );
        printed
    }

    #[test]
    fn elements_round_trip() {
        let ctx = &mut Context::new();

        let ty = tensor_of(ctx, FP32Type::get(ctx).into(), vec![2, 2]);
        let data = [1.0f32, -2.5, 3.25, 0.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        assert!(!attr.is_splat(ctx));
        assert_eq!(attr.num_elements(ctx), Some(4));
        expect!["<tensor.ranked <2x2 : builtin.fp32 > = [1, -2.5, 3.25, 0]>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A signed element keeps its sign.
        let i64_ty = IntegerType::get(ctx, 64, Signedness::Signed).into();
        let ty = tensor_of(ctx, i64_ty, vec![3]);
        let data = [-1i64, 0, 5].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        expect!["<tensor.ranked <3 : builtin.integer si64> = [-1, 0, 5]>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A memref type is a shaped type, and so it can also type a constant.
        let ty = RankedMemrefType::get(
            ctx,
            FP32Type::get(ctx).into(),
            vec![Dimension::Static(2), Dimension::Static(2)],
        )
        .into();
        let data = [1.0f32, -2.5, 3.25, 0.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        expect!["<memref.ranked <2x2 : builtin.fp32 > = [1, -2.5, 3.25, 0]>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A 128 bits element must keep its value.
        let ty = tensor_of(
            ctx,
            IntegerType::get(ctx, 128, Signedness::Signed).into(),
            vec![2],
        );
        let data = [-1i128, 42].iter().flat_map(|v| v.to_ne_bytes()).collect();
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        assert_eq!(attr.raw_data().len(), 32);
        expect!["<tensor.ranked <2 : builtin.integer si128> = [-1, 42]>"]
            .assert_eq(&round_trip(ctx, &attr));

        let ty = tensor_of(
            ctx,
            IntegerType::get(ctx, 64, Signedness::Signless).into(),
            vec![2],
        );
        let mut data = vec![0xFFu8; 8];
        data.extend_from_slice(&7u64.to_ne_bytes());
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        // A signless element prints as a signed element.
        expect!["<tensor.ranked <2 : builtin.integer i64> = [-1, 7]>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A signless i8 must be able to hold 200 (or equivalently -56).
        let from_unsigned = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            "<tensor.ranked <1 : builtin.integer i8> = [200]>",
        )
        .expect_ok(ctx);
        let from_signed = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            "<tensor.ranked <1 : builtin.integer i8> = [-56]>",
        )
        .expect_ok(ctx);
        assert_eq!(from_unsigned.raw_data(), from_signed.raw_data());
        expect!["<tensor.ranked <1 : builtin.integer i8> = splat -56>"]
            .assert_eq(&round_trip(ctx, &from_unsigned));

        // A signless element of one bit holds a boolean. It prints as unsigned.
        let ty = tensor_of(
            ctx,
            IntegerType::get(ctx, 1, Signedness::Signless).into(),
            vec![2],
        );
        let attr = DenseElementsAttr::new(ctx, ty, vec![1u8, 0u8]).expect_ok(ctx);
        expect!["<tensor.ranked <2 : builtin.integer i1> = [1, 0]>"]
            .assert_eq(&round_trip(ctx, &attr));

        // The buffer holds an element of more bits than that of any Rust integer.
        // Check that 2^200 retains every digit.
        let two_pow_200 = "1606938044258990275541962092341162602522202993782792835301376";
        let text = format!("<tensor.ranked <2 : builtin.integer si256> = [{two_pow_200}, -3]>");
        let attr = parse_from_str(DenseElementsAttr::parser(()), ctx, &text).expect_ok(ctx);
        assert_eq!(attr.raw_data().len(), 64);
        assert_eq!(round_trip(ctx, &attr), text);
    }

    /// A splat keeps the bytes of only one element
    #[test]
    fn splat_round_trip() {
        let ctx = &mut Context::new();
        let element = 7.5f32.to_ne_bytes().to_vec();

        let ty = tensor_of(ctx, FP32Type::get(ctx).into(), vec![1000, 1000]);
        let attr = DenseElementsAttr::new_splat(ctx, ty, element.clone()).expect_ok(ctx);
        assert!(attr.is_splat(ctx));
        assert_eq!(attr.raw_data().len(), 4);
        assert_eq!(attr.num_elements(ctx), Some(1_000_000));
        assert_eq!(attr.expanded_data(ctx).len(), 4_000_000);
        expect!["<tensor.ranked <1000x1000 : builtin.fp32 > = splat 7.5>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A zeroed constant is a splat of one zero element.
        let ty = tensor_of(ctx, FP32Type::get(ctx).into(), vec![64, 64]);
        let attr = DenseElementsAttr::new_zeroed(ctx, ty).expect_ok(ctx);
        assert!(attr.is_splat(ctx));
        assert!(attr.is_all_zero());
        assert_eq!(attr.raw_data().len(), 4);
        expect!["<tensor.ranked <64x64 : builtin.fp32 > = splat 0>"]
            .assert_eq(&round_trip(ctx, &attr));

        // A type of one element is a splat: it must be trivially equal to each element.
        let ty = tensor_of(ctx, FP32Type::get(ctx).into(), vec![1]);
        let attr = DenseElementsAttr::new_splat(ctx, ty, element.clone()).expect_ok(ctx);
        assert!(attr.is_splat(ctx));
        assert_eq!(attr.num_elements(ctx), Some(1));
        assert_eq!(attr.expanded_data(ctx), element);
        expect!["<tensor.ranked <1 : builtin.fp32 > = splat 7.5>"]
            .assert_eq(&round_trip(ctx, &attr));
    }

    /// Above [MAX_ELEMENTS_AS_LITERALS], the printer writes hexadecimal.
    #[test]
    fn large_data_prints_as_hex() {
        let ctx = &mut Context::new();
        let count = MAX_ELEMENTS_AS_LITERALS + 1;
        let ty = tensor_of(ctx, FP32Type::get(ctx).into(), vec![count]);
        let data = (0..count as u32)
            .flat_map(|i| (i as f32).to_ne_bytes())
            .collect();
        let attr = DenseElementsAttr::new(ctx, ty, data).expect_ok(ctx);
        let printed = round_trip(ctx, &attr);
        assert!(printed.contains("= \"0x"), "{printed}");
    }

    /// Data that does not agree with the type is rejected.
    #[test]
    fn invalid_constants_fail() {
        let ctx = &mut Context::new();
        let f32_ty: TypeHandle = FP32Type::get(ctx).into();
        let static_ty = tensor_of(ctx, f32_ty, vec![2, 2]);
        let dynamic_ty = dynamic_tensor_of(ctx, f32_ty);
        let element = 7.5f32.to_ne_bytes().to_vec();
        let four_elements: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();

        // The data holds neither every element nor one element.
        let err = DenseElementsAttr::new(ctx, static_ty, vec![0u8; 12])
            .expect_err("data of the wrong size must fail");
        assert!(matches!(err.kind, ErrorKind::InvalidArgument));
        assert!(matches!(
            err.err.downcast_ref::<DenseElementsErr>(),
            Some(DenseElementsErr::DataSize {
                expected: 16,
                element: 4,
                got: 12,
                ..
            })
        ));

        // A splat takes the bytes of one element only.
        let err = DenseElementsAttr::new_splat(ctx, static_ty, four_elements)
            .expect_err("a splat of more than one element must fail");
        assert!(matches!(err.kind, ErrorKind::InvalidArgument));
        assert!(matches!(
            err.err.downcast_ref::<DenseElementsErr>(),
            Some(DenseElementsErr::SplatSize {
                expected: 4,
                got: 16,
                ..
            })
        ));

        // Dynamic shapes aren't supported.
        let err = DenseElementsAttr::new(ctx, dynamic_ty, vec![0u8; 8])
            .expect_err("a dynamic shape must fail");
        assert!(matches!(err.kind, ErrorKind::InvalidArgument));
        assert!(matches!(
            err.err.downcast_ref::<DenseElementsErr>(),
            Some(DenseElementsErr::DynamicShape(_))
        ));

        // A splat cannot have dynamic shapes either.
        let err = DenseElementsAttr::new_splat(ctx, dynamic_ty, element)
            .expect_err("a dynamic shape must fail for a splat too");
        assert!(matches!(err.kind, ErrorKind::InvalidArgument));
        assert!(matches!(
            err.err.downcast_ref::<DenseElementsErr>(),
            Some(DenseElementsErr::DynamicShape(_))
        ));

        // An element with no bits must be rejected.
        let zero_width_ty = tensor_of(
            ctx,
            IntegerType::get(ctx, 0, Signedness::Signless).into(),
            vec![2],
        );
        let err = DenseElementsAttr::new(ctx, zero_width_ty, vec![])
            .expect_err("an element of no bits must fail");
        assert!(matches!(err.kind, ErrorKind::InvalidArgument));
        assert!(matches!(
            err.err.downcast_ref::<DenseElementsErr>(),
            Some(DenseElementsErr::UnsupportedElementType(_))
        ));
    }

    /// Text that does not describe a constant is rejected, and nothing is
    /// quietly cut down to fit.
    #[test]
    fn invalid_text_fails_to_parse() {
        let ctx = &mut Context::new();

        // A type with no shape cannot be a constant value.
        let err = parse_from_str(DenseElementsAttr::parser(()), ctx, "<builtin.fp32 = [1]>")
            .expect_err("an unshaped type must fail");
        expect![[r#"
            Compilation error: invalid input program.
            Parse error at line: 1, column: 2
            TypeInterfaceHandle mismatch: builtin.fp32  does not implement interface dyn pliron_tensor::memref::type_interfaces::ShapedType
        "#]]
        .assert_eq(&err.to_string());

        // A buffer of the wrong size must fail.
        let err = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            r#"<tensor.ranked <2x2 : builtin.fp32> = "0x0102030405">"#,
        )
        .expect_err("a buffer that holds neither every element nor one element must fail");
        expect![[r#"
            Compilation error: invalid input program.
            Parse error at line: 1, column: 1
            memref.dense_elements of type tensor.ranked <2x2 : builtin.fp32 > needs 16 byte(s) of data (or 4 for a splat), but has 5
        "#]]
        .assert_eq(&err.to_string());

        // An i8 must reject 300, which needs more than 8 bits.
        let err = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            "<tensor.ranked <2 : builtin.integer i8> = [1, 300]>",
        )
        .expect_err("a literal wider than the element must fail");
        expect![[r#"
            Compilation error: invalid input program.
            Parse error at line: 1, column: 47
            Unexpected `[`
            Unexpected `<`
            Expected splat
            300 is out of range for the element type i8 of memref.dense_elements
        "#]]
        .assert_eq(&err.to_string());

        // An si8 must reject 200, which needs 8 bits without a sign.
        let err = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            "<tensor.ranked <1 : builtin.integer si8> = [200]>",
        )
        .expect_err("a literal that the sign of the element cannot hold must fail");
        assert!(
            err.to_string()
                .contains("200 is out of range for the element type si8"),
            "{err}"
        );

        // An si136 must reject 2^136, which needs one bit more.
        let two_pow_136 = "87112285931760246646623899502532662132736";
        let err = parse_from_str(
            DenseElementsAttr::parser(()),
            ctx,
            &format!("<tensor.ranked <1 : builtin.integer si136> = [{two_pow_136}]>"),
        )
        .expect_err("a literal wider than the element must fail");
        assert!(
            err.to_string()
                .contains("is out of range for the element type si136"),
            "{err}"
        );
    }
}
