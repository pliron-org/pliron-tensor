// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Helpers that the tests of tensor IR share.

use std::fmt::Debug;

use pliron::{
    context::{Context, Ptr},
    location::erase_locations,
    op::verify_op,
    operation::Operation,
    printable::Printable,
    result::ExpectOk,
};
use pliron_llvm::llvm_sys::lljit::SimpleJIT;

use pliron_tensor::tensor::{
    bufferize::bufferize, memory_management::TensorMemoryManager, runtime_utils::TensorDesciptor,
};

use super::lookup_fn;

/// Bufferize the module with `tmm` and return the bufferized IR as text.
pub fn bufferize_module<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    parsed_op: Ptr<Operation>,
) -> String {
    bufferize(tmm, parsed_op, ctx).expect_ok(ctx);
    verify_op(Operation::get_op_dyn(parsed_op, ctx).as_ref(), ctx).expect_ok(ctx);
    erase_locations(ctx, parsed_op);
    let after_bufferization = parsed_op.disp(ctx).to_string();
    log::debug!("pliron module after bufferization {}", after_bufferization);
    after_bufferization
}

/// A descriptor for an input tensor with shape `dims` and elements `data`.
pub fn input_tensor<T>(dims: &[usize], data: &[T]) -> TensorDesciptor {
    TensorDesciptor::new(dims.to_vec(), size_of::<T>(), data.as_ptr() as *const u8)
}

/// A descriptor for a result tensor with shape `dims`.
pub fn output_tensor<T>(dims: &[usize]) -> TensorDesciptor {
    TensorDesciptor::new(dims.to_vec(), size_of::<T>(), std::ptr::null::<u8>())
}

/// Read back the tensor that an executed function wrote into `out_ir_descr`.
///
/// # Safety
/// `out_ir_descr` must hold a tensor of rank `rank` with elements of type `T`.
pub unsafe fn output_data<T: Copy>(out_ir_descr: &[u8], rank: usize) -> Vec<T> {
    let descr =
        unsafe { TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), rank, size_of::<T>()) };
    let mut data = Vec::new();
    unsafe { descr.copy_to_vec(&mut data) };
    data
}

/// Element comparison helper for tests.
pub trait TestElement: Copy + Debug {
    /// Check that `self` and `other` are the same value.
    fn same(self, other: Self) -> bool;
}

impl TestElement for i32 {
    fn same(self, other: Self) -> bool {
        self == other
    }
}

impl TestElement for i64 {
    fn same(self, other: Self) -> bool {
        self == other
    }
}

impl TestElement for u32 {
    fn same(self, other: Self) -> bool {
        self == other
    }
}

impl TestElement for f32 {
    fn same(self, other: Self) -> bool {
        // Bitwise equality. NaNs always compare equal.
        (self.is_nan() && other.is_nan()) || self.to_bits() == other.to_bits()
    }
}

/// bf16 value, as its raw bits.
#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct Bf16(pub u16);

impl Bf16 {
    /// Convert `value` to bf16 by taking the upper 16 bits of its f32 representation.
    ///
    /// Panics if no exact bf16 representation exists.
    pub fn from_f32(value: f32) -> Self {
        assert_eq!(
            value.to_bits() & 0xffff,
            0,
            "{value} has no exact bf16 value"
        );
        Bf16((value.to_bits() >> 16) as u16)
    }

    /// Convert to f32 by placing the bits in the upper 16 bits of an f32.
    pub fn to_f32(self) -> f32 {
        f32::from_bits(u32::from(self.0) << 16)
    }
}

impl TestElement for Bf16 {
    fn same(self, other: Self) -> bool {
        self.to_f32().same(other.to_f32())
    }
}

/// Operands of elementwise binary ops and the expected result of each op.
pub struct BinaryCase<T> {
    pub lhs: [T; 8],
    pub rhs: [T; 8],
    /// Op name and expected result, in the order that the test function stores the results.
    pub expected: Vec<(&'static str, [T; 8])>,
}

impl<T: Copy> BinaryCase<T> {
    /// Apply `f` to all operands and expected results.
    pub fn map<U>(&self, f: impl Fn(T) -> U) -> BinaryCase<U> {
        BinaryCase {
            lhs: self.lhs.map(&f),
            rhs: self.rhs.map(&f),
            expected: self
                .expected
                .iter()
                .map(|(op, expected)| (*op, expected.map(&f)))
                .collect(),
        }
    }
}

/// Source of the [BinaryCase] operands in a test function.
#[derive(Clone, Copy)]
pub enum Operands {
    /// As its first two arguments.
    Arguments,
    /// As constants in its body.
    Constants,
}

/// Call `name` and check its results against `case`.
///
/// `name` takes the following arguments:
/// - `case.lhs` and `case.rhs`, when `operands` is [Operands::Arguments]
/// - one output pointer for each result
pub fn check_binary_case<T: TestElement>(
    jit: &SimpleJIT,
    name: &str,
    operands: Operands,
    case: &BinaryCase<T>,
) {
    let lhs = input_tensor(&[8], &case.lhs).build_ir_descriptor();
    let rhs = input_tensor(&[8], &case.rhs).build_ir_descriptor();
    let inputs = match operands {
        Operands::Arguments => vec![lhs.as_ptr(), rhs.as_ptr()],
        Operands::Constants => vec![],
    };
    let mut outputs: Vec<_> = case
        .expected
        .iter()
        .map(|_| output_tensor::<T>(&[8]).build_ir_descriptor())
        .collect();
    let out: Vec<_> = outputs
        .iter_mut()
        .map(|output| output.as_mut_ptr())
        .collect();

    type P = *const u8;
    type M = *mut u8;
    unsafe {
        match (inputs.as_slice(), out.as_slice()) {
            (&[], &[a, b]) => lookup_fn::<extern "C" fn(M, M)>(jit, name)(a, b),
            (&[], &[a, b, c, d]) => lookup_fn::<extern "C" fn(M, M, M, M)>(jit, name)(a, b, c, d),
            (&[l, r], &[a, b]) => lookup_fn::<extern "C" fn(P, P, M, M)>(jit, name)(l, r, a, b),
            (&[l, r], &[a, b, c, d]) => {
                lookup_fn::<extern "C" fn(P, P, M, M, M, M)>(jit, name)(l, r, a, b, c, d)
            }
            _ => panic!(
                "No function type for {} inputs and {} outputs",
                inputs.len(),
                out.len()
            ),
        }
    }

    for (output, (op, expected)) in outputs.iter().zip(&case.expected) {
        let actual = unsafe { output_data::<T>(output, 1) };
        for (idx, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                a.same(*e),
                "{name}, {op}, element {idx}: got {a:?}, expected {e:?}"
            );
        }
    }
}

/// Operands for `max` and `min` on f32.
pub fn max_min_f32() -> BinaryCase<f32> {
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    BinaryCase {
        lhs: [1.0, -2.0, nan, 3.0, -0.0, 0.0, inf, -inf],
        rhs: [2.0, -3.0, 1.0, nan, 0.0, -0.0, 5.0, 1.0],
        expected: vec![
            ("max", [2.0, -2.0, nan, nan, 0.0, 0.0, inf, 1.0]),
            ("min", [1.0, -3.0, nan, nan, -0.0, -0.0, 5.0, -inf]),
        ],
    }
}

/// Operands for `max` and `min` on i32.
pub fn max_min_i32() -> BinaryCase<i32> {
    BinaryCase {
        lhs: [1, -2, -5, 7, i32::MIN, i32::MAX, 0, -1],
        rhs: [2, -3, 5, -7, 0, -1, 0, 1],
        expected: vec![
            ("max", [2, -2, 5, 7, 0, i32::MAX, 0, 1]),
            ("min", [1, -3, -5, -7, i32::MIN, -1, 0, -1]),
        ],
    }
}

/// Operands for `max` and `min` on ui32.
pub fn max_min_ui32() -> BinaryCase<u32> {
    let top = 1u32 << 31;
    BinaryCase {
        lhs: [1, top, 5, u32::MAX, 0, 7, 3, u32::MAX - 1],
        rhs: [2, 1, top + 1, 0, 0, 7, top, u32::MAX],
        expected: vec![
            ("max", [2, top, top + 1, u32::MAX, 0, 7, top, u32::MAX]),
            ("min", [1, 1, 5, 0, 0, 7, 3, u32::MAX - 1]),
        ],
    }
}

/// Operands for `add`, `sub`, `mul` and `div` on f32.
pub fn arith_f32() -> BinaryCase<f32> {
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    let (a, b) = (0.1f32, 0.2f32);
    BinaryCase {
        lhs: [1.0, inf, nan, 0.0, -0.0, a, 3.0, -inf],
        rhs: [0.0, inf, 1.0, -0.0, -0.0, b, -0.0, inf],
        expected: vec![
            ("add", [1.0, inf, nan, 0.0, -0.0, a + b, 3.0, nan]),
            ("sub", [1.0, nan, nan, 0.0, 0.0, a - b, 3.0, -inf]),
            ("mul", [0.0, inf, nan, -0.0, 0.0, a * b, -0.0, -inf]),
            ("div", [inf, nan, nan, nan, nan, a / b, -inf, nan]),
        ],
    }
}

/// Operands for `add`, `sub`, `mul` and `div` on i32.
pub fn arith_i32() -> BinaryCase<i32> {
    BinaryCase {
        lhs: [i32::MAX, i32::MIN, -7, 7, -7, 100, 0, 1 << 16],
        rhs: [1, 1, 2, -2, -2, 7, -5, 1 << 16],
        expected: vec![
            ("add", [i32::MIN, i32::MIN + 1, -5, 5, -9, 107, -5, 1 << 17]),
            ("sub", [i32::MAX - 1, i32::MAX, -9, 9, -5, 93, 5, 0]),
            ("mul", [i32::MAX, i32::MIN, -14, -14, 14, 700, 0, 0]),
            ("div", [i32::MAX, i32::MIN, -3, -3, 3, 14, 0, 1]),
        ],
    }
}

/// Operands for `add`, `sub`, `mul` and `div` on ui32.
pub fn arith_ui32() -> BinaryCase<u32> {
    let top = 1u32 << 31;
    BinaryCase {
        lhs: [0, u32::MAX, top + 2, top, 7, 100, 1 << 16, u32::MAX],
        rhs: [1, 1, 2, 2, 2, 7, 1 << 16, u32::MAX],
        expected: vec![
            (
                "add",
                [1, 0, top + 4, top + 2, 9, 107, 1 << 17, u32::MAX - 1],
            ),
            ("sub", [u32::MAX, u32::MAX - 1, top, top - 2, 5, 93, 0, 0]),
            ("mul", [0, u32::MAX, 4, 0, 14, 700, 0, 1]),
            ("div", [0, u32::MAX, (top >> 1) + 1, top >> 1, 3, 14, 1, 1]),
        ],
    }
}

/// Reference implementation of `tensor.reduce`.
///
/// Reduce the row-major `input` (of `shape`) over `dims` with `combine`,
/// starting from the accumulator `init`.
pub fn reduce_reference<T: Copy>(
    shape: &[usize],
    input: &[T],
    dims: &[usize],
    init: &[T],
    combine: impl Fn(T, T) -> T,
) -> Vec<T> {
    let mut accum = init.to_vec();
    for (flat_index, &element) in input.iter().enumerate() {
        // The row-major accumulator index, with each reduced dimension set to 0.
        let (mut rest, mut accum_index, mut stride) = (flat_index, 0, 1);
        for dim in (0..shape.len()).rev() {
            let index = rest % shape[dim];
            rest /= shape[dim];
            if !dims.contains(&dim) {
                accum_index += index * stride;
                stride *= shape[dim];
            }
        }
        accum[accum_index] = combine(accum[accum_index], element);
    }
    accum
}
