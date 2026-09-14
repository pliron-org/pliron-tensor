// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test conversions of memref operations to CF / LLVM dialect.

use pliron::{
    builtin::ops::ModuleOp, context::Context, location::erase_locations, op::Op,
    printable::Printable,
};

use pliron_llvm::llvm_sys::{
    core::{LLVMContext, LLVMModule},
    lljit::SimpleJIT,
};

use expect_test::expect_file;

mod common;
use common::{assert_module_round_trips, lookup_fn, lower_to_llvm_ir, parse_module};

/// Run `input_ir` through Memref -> CF -> LLVM dialect. The converted values are returned.
fn compile(ctx: &mut Context, input_ir: &str) -> (LLVMContext, LLVMModule, ModuleOp) {
    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    erase_locations(ctx, parsed_op);
    (llvm_ctx, llvm_ir, module_op)
}

/// Calls [compile] and returns the converted module and its JIT object.
fn compile_and_jit(ctx: &mut Context, input_ir: &str) -> (SimpleJIT, ModuleOp) {
    let (llvm_ctx, llvm_ir, module_op) = compile(ctx, input_ir);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    (jit, module_op)
}

/// Every memref operation must print in a form that parses back.
#[test]
fn test_memref_ops_round_trip() {
    assert_module_round_trips(include_str!(
        "resources/test_memref_ops_round_trip.input.plir"
    ));
}

#[test]
fn test_alloc_generate() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_alloc_generate.input.plir");

    let (llvm_ctx, llvm_ir, module_op) = compile(ctx, input_ir);

    expect_file!["resources/test_alloc_generate.expect.plir"]
        .assert_eq(&module_op.get_operation().disp(ctx).to_string());

    expect_file!["resources/test_alloc_generate.expect.ll"].assert_eq(&llvm_ir.to_string());

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let f = unsafe { lookup_fn::<fn(i64, i64) -> i64>(&jit, "test_alloc_generate") };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, i + j);
        }
    }
}

/// `memref.dim` with a dynamic and with a constant dimension index.
#[test]
fn test_memref_dim() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_memref_dim.input.plir");

    let (jit, module_op) = compile_and_jit(ctx, input_ir);

    // A dynamic index must go through the descriptor in memory, a constant one is
    // an extract_value of the sizes array.
    expect_file!["resources/test_memref_dim.expect.plir"]
        .assert_eq(&module_op.get_operation().disp(ctx).to_string());

    let dynamic_index =
        unsafe { lookup_fn::<fn(i64) -> i64>(&jit, "test_memref_dim_dynamic_index") };
    assert_eq!(dynamic_index(0), 16);
    assert_eq!(dynamic_index(1), 32);

    let const_index = unsafe { lookup_fn::<fn() -> i64>(&jit, "test_memref_dim_const_index") };
    // Encoded return value = dim0 * 1000 + dim1 = 16 * 1000 + 32
    assert_eq!(const_index(), 16032);
}

/// `memref.subview`, `memref.copy`, and the `memref.copy + memref.subview +
/// memref.copy` insertion sequence, lowered to CF / LLVM.
///
/// Every function fills its memrefs with `memref.generate` and returns the element
/// at the index it is called with.
#[test]
fn test_subview_copy_and_insert_slice() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_subview_copy_and_insert_slice.input.plir");

    let (jit, _) = compile_and_jit(ctx, input_ir);

    // src is a 2x3 memref with src[i][j] = i*3 + j, and the view has offsets [0, 1]:
    // view[i][j] = src[i][1 + j] = i*3 + j + 1.
    let subview = unsafe { lookup_fn::<fn(i64, i64) -> i64>(&jit, "test_subview") };
    for i in 0..2_i64 {
        for j in 0..2_i64 {
            let result = subview(i, j);
            assert_eq!(result, i * 3 + j + 1, "test_subview({i}, {j}) = {result}");
        }
    }

    // dst is a copy of src, which has src[i][j] = i*10 + j.
    let copy = unsafe { lookup_fn::<fn(i64, i64) -> i64>(&jit, "test_copy") };
    for i in 0..2_i64 {
        for j in 0..2_i64 {
            let result = copy(i, j);
            assert_eq!(result, i * 10 + j, "test_copy({i}, {j}) = {result}");
        }
    }

    // res is dst, with dst[i][j] = 100 + i*4 + j, with the 2x2 src inserted at
    // offsets [1, 1].
    let insert_slice = unsafe { lookup_fn::<fn(i64, i64) -> i64>(&jit, "test_insert_slice") };
    for i in 0..3_i64 {
        for j in 0..4_i64 {
            let result = insert_slice(i, j);
            let expected = if (1..3).contains(&i) && (1..3).contains(&j) {
                (i - 1) * 10 + (j - 1)
            } else {
                100 + i * 4 + j
            };
            assert_eq!(result, expected, "test_insert_slice({i}, {j}) = {result}");
        }
    }
}

#[test]
fn test_globals() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_globals.input.plir");

    let (llvm_ctx, llvm_ir, _) = compile(ctx, input_ir);

    expect_file!["resources/test_globals.expect.ll"].assert_eq(&llvm_ir.to_string());

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    // The mutable global must keep its value between calls.
    let bump = unsafe { lookup_fn::<extern "C" fn(i64) -> i64>(&jit, "bump") };
    assert_eq!(bump(5), 5);
    assert_eq!(bump(7), 12);
    assert_eq!(bump(0), 12);

    // Read all elements of @odd
    let read = unsafe { lookup_fn::<extern "C" fn(i64) -> i32>(&jit, "read") };
    assert_eq!(read(0), 11);
    assert_eq!(read(1), 22);
    assert_eq!(read(2), 33);
}
