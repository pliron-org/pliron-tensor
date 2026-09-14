// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test the tensor pipeline
//! - bufferization to memref
//! - Memref -> CF -> LLVM dialect
//! - execution of the result in a JIT.

use expect_test::expect_file;
use pliron::{
    context::{Context, Ptr},
    location::erase_locations,
    operation::Operation,
    printable::Printable,
    result::ExpectOk,
};

use pliron_llvm::llvm_sys::{
    core::{LLVMContext, LLVMModule},
    lljit::{LLVMLLJIT, SimpleJIT},
    target::initialize_native,
};

use pliron_tensor::tensor::{
    bufferize::bufferize,
    memory_management::{MallocFreeTMM, TensorMemoryManager},
    runtime_utils::TensorDesciptor,
    tracked_tmm::TrackedTMM,
};

mod common;
use common::{assert_module_round_trips, lookup_fn, lower_to_llvm_ir, parse_module};

/// Bufferize the module with `tmm` and return the bufferized IR as text.
fn bufferize_module<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    parsed_op: Ptr<Operation>,
) -> String {
    bufferize(tmm, parsed_op, ctx).expect_ok(ctx);
    erase_locations(ctx, parsed_op);
    let after_bufferization = parsed_op.disp(ctx).to_string();
    log::debug!("pliron module after bufferization {}", after_bufferization);
    after_bufferization
}

/// Run `input_ir` through the full pipeline and return
/// (LLVMContext, lowered LLVM module, bufferized pliron IR)
fn compile<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    input_ir: &str,
) -> (LLVMContext, LLVMModule, String) {
    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let after_bufferization = bufferize_module(ctx, tmm, parsed_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    (llvm_ctx, llvm_ir, after_bufferization)
}

/// Calls [compile] and returns the JIT object for the LLVM module.
/// The bufferized pliron IR is also returned as-is.
fn compile_and_jit<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    input_ir: &str,
) -> (SimpleJIT, String) {
    let (llvm_ctx, llvm_ir, after_bufferization) = compile(ctx, tmm, input_ir);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    (jit, after_bufferization)
}

/// The same as [compile_and_jit], but the runtime symbols of `tmm` are also
/// registered with the JIT. The bufferized pliron IR is also returned as-is.
fn compile_and_jit_with_runtime<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    input_ir: &str,
) -> (LLVMLLJIT, String) {
    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let after_bufferization = bufferize_module(ctx, tmm, parsed_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    tmm.register_runtime_symbols(&jit)
        .expect("Failed to register runtime symbols");
    jit.add_module(llvm_ctx, llvm_ir)
        .expect("Failed to add module to JIT");
    (jit, after_bufferization)
}

/// Look up `name` in `jit` and interpret it as a function of type `F`.
///
/// # Safety
/// `F` must be the type of the compiled function.
unsafe fn lookup_lljit_fn<F: Copy>(jit: &LLVMLLJIT, name: &str) -> F {
    const { assert!(size_of::<F>() == size_of::<u64>()) };
    let symbol_addr = jit.lookup_symbol(name).expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    unsafe { std::mem::transmute_copy::<u64, F>(&symbol_addr) }
}

/// A descriptor for an input tensor with shape `dims` and elements `data`.
fn input_tensor<T>(dims: &[usize], data: &[T]) -> TensorDesciptor {
    TensorDesciptor::new(dims.to_vec(), size_of::<T>(), data.as_ptr() as *const u8)
}

/// A descriptor for a result tensor with shape `dims`.
fn output_tensor<T>(dims: &[usize]) -> TensorDesciptor {
    TensorDesciptor::new(dims.to_vec(), size_of::<T>(), std::ptr::null::<u8>())
}

/// Read back the tensor that an executed function wrote into `out_ir_descr`.
///
/// # Safety
/// `out_ir_descr` must hold a tensor of rank `rank` with elements of type `T`.
unsafe fn output_data<T: Copy>(out_ir_descr: &[u8], rank: usize) -> Vec<T> {
    let descr =
        unsafe { TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), rank, size_of::<T>()) };
    let mut data = Vec::new();
    unsafe { descr.copy_to_vec(&mut data) };
    data
}

/// Every tensor operation must print in a form that parses back.
#[test]
fn test_tensor_ops_round_trip() {
    assert_module_round_trips(include_str!(
        "resources/test_tensor_ops_round_trip.input.plir"
    ));
}

#[test]
fn test_broadcast_splat_and_elementwise_cast_from_rust() {
    let ctx = &mut Context::new();
    let input_ir = include_str!("resources/test_broadcast_splat_cast.input.plir");
    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    expect_file!["resources/test_broadcast_splat_cast.expect.plir"].assert_eq(&after_bufferization);

    let input_data = [1.25f64, 2.5, 3.75, 4.0];
    let input = input_tensor(&[1, 1, 1, 4], &input_data);
    let mut output = output_tensor::<f32>(&[1, 2, 3, 4]).build_ir_descriptor();
    let function = unsafe {
        lookup_fn::<extern "C" fn(*const u8, f32, *mut u8) -> ()>(&jit, "test_broadcast_splat_cast")
    };
    function(
        input.build_ir_descriptor().as_ptr(),
        2.0,
        output.as_mut_ptr(),
    );

    assert_eq!(
        unsafe { output_data::<f32>(&output, 4) },
        [
            3.25, 4.5, 5.75, 6.0, 3.25, 4.5, 5.75, 6.0, 3.25, 4.5, 5.75, 6.0, 3.25, 4.5, 5.75, 6.0,
            3.25, 4.5, 5.75, 6.0, 3.25, 4.5, 5.75, 6.0
        ]
    );
}

/// `tensor.generate`, `tensor.extract` and the elementwise binary ops.
#[test]
fn test_elementwise_ops_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_elementwise_ops_from_rust.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    expect_file!["resources/test_elementwise_ops_from_rust.expect.plir"]
        .assert_eq(&after_bufferization);

    let generate_add = unsafe { lookup_fn::<fn(i64, i64) -> i64>(&jit, "test_generate_add") };
    for i in 0..16 {
        for j in 0..16 {
            assert_eq!(generate_add(i, j), (i + j) * 2);
        }
    }

    let int_lhs_data = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let int_rhs_data = [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let int_lhs = input_tensor(&[4, 4], &int_lhs_data);
    let int_rhs = input_tensor(&[4, 4], &int_rhs_data);
    let mut int_res_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();

    let add_int = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(&jit, "test_tensor_add_int")
    };
    add_int(
        int_lhs.build_ir_descriptor().as_ptr(),
        int_rhs.build_ir_descriptor().as_ptr(),
        int_res_ir_descr.as_mut_ptr(),
    );
    assert_eq!(
        unsafe { output_data::<u64>(&int_res_ir_descr, 2) },
        [17; 16]
    );

    let lhs_data = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let rhs_data = [
        16.0f64, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];
    // The elementwise ops always allocate a new buffer for their result, so
    // `lhs` and `rhs` stay unchanged and both functions below can use them.
    let lhs = input_tensor(&[4, 4], &lhs_data);
    let rhs = input_tensor(&[4, 4], &rhs_data);

    let add_float = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_tensor_add_float",
        )
    };
    let mut add_res_ir_descr = output_tensor::<f64>(&[4, 4]).build_ir_descriptor();
    add_float(
        lhs.build_ir_descriptor().as_ptr(),
        rhs.build_ir_descriptor().as_ptr(),
        add_res_ir_descr.as_mut_ptr(),
    );
    assert_eq!(
        unsafe { output_data::<f64>(&add_res_ir_descr, 2) },
        [17.0; 16]
    );

    let all_binops = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_tensor_all_binops_float",
        )
    };
    let mut binops_res_ir_descr = output_tensor::<f64>(&[4, 4]).build_ir_descriptor();
    all_binops(
        lhs.build_ir_descriptor().as_ptr(),
        rhs.build_ir_descriptor().as_ptr(),
        binops_res_ir_descr.as_mut_ptr(),
    );
    let binops_res = unsafe { output_data::<f64>(&binops_res_ir_descr, 2) };
    for ((&a, &b), &c) in lhs_data.iter().zip(rhs_data.iter()).zip(binops_res.iter()) {
        let expected = ((a + b) * b) / a;
        assert!((c - expected).abs() < 1e-12);
    }
}

/// `tensor.matmul`, with static and with dynamic operand shapes, and
/// `tensor.batch_matmul`.
#[test]
fn test_matmul_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_matmul_from_rust.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    expect_file!["resources/test_matmul_from_rust.expect.plir"].assert_eq(&after_bufferization);

    let lhs_data = [1u64; 16];
    let rhs_data = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let lhs = input_tensor(&[4, 4], &lhs_data);
    let rhs = input_tensor(&[4, 4], &rhs_data);

    // The three matmul functions differ only in how static their operand shapes are.
    for name in [
        "test_matmul_all_static",
        "test_matmul_inner_dynamic",
        "test_matmul_all_dynamic",
    ] {
        let f = unsafe {
            lookup_fn::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(&jit, name)
        };

        // The accumulator may be written in place, so it is fresh for every call.
        let accum_data = [1u64; 16];
        let accum = input_tensor(&[4, 4], &accum_data);
        let mut res_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();

        f(
            lhs.build_ir_descriptor().as_ptr(),
            rhs.build_ir_descriptor().as_ptr(),
            accum.build_ir_descriptor().as_ptr(),
            res_ir_descr.as_mut_ptr(),
        );

        assert_eq!(
            unsafe { output_data::<u64>(&res_ir_descr, 2) },
            [
                29u64, 33, 37, 41, 29, 33, 37, 41, 29, 33, 37, 41, 29, 33, 37, 41
            ],
            "{name} computed the wrong result"
        );
    }

    // Batch 0 lhs: [[1,2,3],[4,5,6]], rhs: [[1,2],[3,4],[5,6]]
    // result: [[22,28],[49,64]]
    // Batch 1 lhs: [[7,8,9],[10,11,12]], rhs: [[7,8],[9,10],[11,12]]
    // result: [[220,244],[301,334]]
    let batch_data = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let batch_lhs = input_tensor(&[2, 2, 3], &batch_data);
    let batch_rhs = input_tensor(&[2, 3, 2], &batch_data);
    let batch_accum_data = [1u64, 1, 1, 1, 2, 2, 2, 2];
    let batch_accum = input_tensor(&[2, 2, 2], &batch_accum_data);
    let mut batch_res_ir_descr = output_tensor::<u64>(&[2, 2, 2]).build_ir_descriptor();

    let batch_matmul = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_batch_matmul",
        )
    };
    batch_matmul(
        batch_lhs.build_ir_descriptor().as_ptr(),
        batch_rhs.build_ir_descriptor().as_ptr(),
        batch_accum.build_ir_descriptor().as_ptr(),
        batch_res_ir_descr.as_mut_ptr(),
    );

    assert_eq!(
        unsafe { output_data::<u64>(&batch_res_ir_descr, 3) },
        [23u64, 29, 50, 65, 222, 246, 303, 336]
    );
}

/// [TrackedTMM] must account for every tensor that the IR allocates.
#[test]
fn test_tracked_tmm_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_tracked_tmm_from_rust.input.plir");

    let mut tmm = TrackedTMM::new();
    let tmm_address = format!("{:p}", &tmm);
    let (jit, after_bufferization) = compile_and_jit_with_runtime(ctx, &mut tmm, input_ir);

    // Replace the actual address (to work around ASLR) with a deterministic string.
    expect_file!["resources/test_tracked_tmm_from_rust.expect.plir"]
        .assert_eq(&after_bufferization.replace(&tmm_address, "<tracked-tmm>"));

    // No tensor is allocated by the IR yet.
    assert_eq!(tmm.tracked_allocations().len(), 0);

    let add_lhs_data = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let add_rhs_data = [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let add_lhs = input_tensor(&[4, 4], &add_lhs_data);
    let add_rhs = input_tensor(&[4, 4], &add_rhs_data);
    let mut add_res_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();

    let add = unsafe {
        lookup_lljit_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_tensor_add_tracked",
        )
    };
    add(
        add_lhs.build_ir_descriptor().as_ptr(),
        add_rhs.build_ir_descriptor().as_ptr(),
        add_res_ir_descr.as_mut_ptr(),
    );

    // We have one tensor allocated for the result.
    assert_eq!(tmm.tracked_allocations().len(), 1);
    assert_eq!(
        unsafe { output_data::<u64>(&add_res_ir_descr, 2) },
        [17; 16]
    );

    let lhs_data = [1i64, 2, 3, 4, 5, 6, 7, 8, 2, 1, 0, 3, 4, 2, 1, 5];
    let rhs_data = [2i64, 1, 0, 1, 3, 2, 1, 0, 4, 1, 2, 3, 1, 0, 2, 1];
    let accum_data = [0i64; 16];
    let lhs = input_tensor(&[4, 4], &lhs_data);
    let rhs = input_tensor(&[4, 4], &rhs_data);
    let accum = input_tensor(&[4, 4], &accum_data);
    let mut res_ir_descr = output_tensor::<i64>(&[4, 4]).build_ir_descriptor();

    let complex = unsafe {
        lookup_lljit_fn::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_tensor_complex_tracked",
        )
    };
    complex(
        lhs.build_ir_descriptor().as_ptr(),
        rhs.build_ir_descriptor().as_ptr(),
        accum.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    assert!(
        tmm.tracked_allocations().len() >= 4,
        "expected tracked allocations for intermediates and final result"
    );

    let mut expected = [0i64; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut mat = 0i64;
            for k in 0..4 {
                mat += lhs_data[i * 4 + k] * rhs_data[k * 4 + j];
            }
            let sum = mat + lhs_data[i * 4 + j];
            let diff = sum - rhs_data[i * 4 + j];
            expected[i * 4 + j] = diff * lhs_data[i * 4 + j];
        }
    }
    assert_eq!(unsafe { output_data::<i64>(&res_ir_descr, 2) }, expected);

    tmm.free_all();
    assert_eq!(tmm.tracked_allocations().len(), 0);
}

#[test]
fn test_successor_operand_aliasing_needs_copy() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_successor_operand_aliasing_needs_copy.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    expect_file!["resources/test_successor_operand_aliasing_needs_copy.expect.plir"]
        .assert_eq(&after_bufferization);

    // Expected with correct bufferization:
    //   z is original x = [1, 2, 3, 4]
    //   y is x with index 0 updated to 10 => [10, 2, 3, 4]
    //   sum[0] = 1 + 10 = 11
    for name in ["test_aliasing_br", "test_aliasing_cond_br"] {
        let f = unsafe { lookup_fn::<fn(bool) -> i64>(&jit, name) };
        assert_eq!(f(false), 11, "{name} computed the wrong result");
    }
}

/// `tensor.extract_slice` lowered to `memref.subview`
#[test]
fn test_extract_slice() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_extract_slice.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // extract_slice only reads its source, so no slice needs a private buffer.
    expect_file!["resources/test_extract_slice.expect.plir"].assert_eq(&after_bufferization);

    let src_data: Vec<u64> = (0..200_u64).collect();
    let src = input_tensor(&[10, 20], &src_data);

    let extract_slice =
        unsafe { lookup_fn::<extern "C" fn(*const u8, *mut u8) -> ()>(&jit, "test_extract_slice") };
    let mut out_ir_descr = output_tensor::<u64>(&[5, 10]).build_ir_descriptor();
    extract_slice(
        src.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let mut expected = Vec::with_capacity(5 * 10);
    for i in 0..5_u64 {
        for j in 0..10_u64 {
            // src[i][2 + 2*j] for offsets [0, 2], sizes [5, 10], strides [1, 2].
            expected.push(i * 20 + 2 + 2 * j);
        }
    }
    assert_eq!(unsafe { output_data::<u64>(&out_ir_descr, 2) }, expected);

    let sequential = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *mut u8) -> ()>(&jit, "test_extract_slice_sequential")
    };
    let mut out_ir_descr = output_tensor::<u64>(&[3, 4]).build_ir_descriptor();
    sequential(
        src.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let mut expected = Vec::with_capacity(3 * 4);
    for i in 0..3_u64 {
        for j in 0..4_u64 {
            // first[i1, j1] = src[1 + i1, 2 + 2*j1]
            // second[i, j] = first[1 + 2*i, 1 + 2*j] = src[2 + 2*i, 4 + 4*j]
            expected.push((2 + 2 * i) * 20 + (4 + 4 * j));
        }
    }
    assert_eq!(unsafe { output_data::<u64>(&out_ir_descr, 2) }, expected);

    let live_source = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *mut u8, *mut u8) -> ()>(
            &jit,
            "test_extract_slice_live_source",
        )
    };
    let mut out_first_ir_descr = output_tensor::<u64>(&[5, 10]).build_ir_descriptor();
    let mut out_second_ir_descr = output_tensor::<u64>(&[5, 10]).build_ir_descriptor();
    live_source(
        src.build_ir_descriptor().as_ptr(),
        out_first_ir_descr.as_mut_ptr(),
        out_second_ir_descr.as_mut_ptr(),
    );

    let mut expected_first = Vec::with_capacity(5 * 10);
    let mut expected_second = Vec::with_capacity(5 * 10);
    for i in 0..5_u64 {
        for j in 0..10_u64 {
            expected_first.push(i * 20 + j);
            expected_second.push((5 + i) * 20 + (10 + j));
        }
    }
    assert_eq!(
        unsafe { output_data::<u64>(&out_first_ir_descr, 2) },
        expected_first
    );
    assert_eq!(
        unsafe { output_data::<u64>(&out_second_ir_descr, 2) },
        expected_second
    );
}

#[test]
fn test_insert_slice() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_insert_slice.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // `tensor.insert_slice` writes in place only when destination buffer isn't seen later.
    // Only the two functions whose destination stays visible allocate a buffer.
    expect_file!["resources/test_insert_slice.expect.plir"].assert_eq(&after_bufferization);

    let src_data: Vec<u64> = (100..150_u64).collect();
    let dst_data: Vec<u64> = (0..200_u64).collect();
    let src = input_tensor(&[5, 10], &src_data);
    let dst = input_tensor(&[10, 20], &dst_data);

    // The destination, with the source inserted and all other elements unchanged.
    let mut expected_updated = dst_data.clone();
    for i in 0..5_usize {
        for j in 0..10_usize {
            expected_updated[i * 20 + (2 + 2 * j)] = src_data[i * 10 + j];
        }
    }

    let insert_slice = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(&jit, "test_insert_slice")
    };
    let mut out_ir_descr = output_tensor::<u64>(&[10, 20]).build_ir_descriptor();
    insert_slice(
        src.build_ir_descriptor().as_ptr(),
        dst.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );
    assert_eq!(
        unsafe { output_data::<u64>(&out_ir_descr, 2) },
        expected_updated
    );

    // The in-place write above may have updated dst_data.
    let dst_data: Vec<u64> = (0..200_u64).collect();
    let dst = input_tensor(&[10, 20], &dst_data);

    let dest_live_after = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            &jit,
            "test_insert_slice_dest_live_after",
        )
    };
    let mut out_updated_ir_descr = output_tensor::<u64>(&[10, 20]).build_ir_descriptor();
    let mut out_dst_ir_descr = output_tensor::<u64>(&[10, 20]).build_ir_descriptor();
    dest_live_after(
        src.build_ir_descriptor().as_ptr(),
        dst.build_ir_descriptor().as_ptr(),
        out_updated_ir_descr.as_mut_ptr(),
        out_dst_ir_descr.as_mut_ptr(),
    );

    assert_eq!(
        unsafe { output_data::<u64>(&out_updated_ir_descr, 2) },
        expected_updated,
        "updated tensor does not reflect the inserted slice"
    );
    assert_eq!(
        unsafe { output_data::<u64>(&out_dst_ir_descr, 2) },
        dst_data,
        "dst was mutated in place even though it was still live after the insert"
    );

    let t_data: Vec<u64> = (0..16_u64).collect();
    let small_data: Vec<u64> = vec![900, 901, 902, 903];
    let t = input_tensor(&[4, 4], &t_data);
    let small = input_tensor(&[2, 2], &small_data);

    let write_through_slice = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            &jit,
            "test_write_through_slice_of_live_tensor",
        )
    };
    let mut out_u_ir_descr = output_tensor::<u64>(&[2, 2]).build_ir_descriptor();
    let mut out_t_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();
    write_through_slice(
        t.build_ir_descriptor().as_ptr(),
        small.build_ir_descriptor().as_ptr(),
        out_u_ir_descr.as_mut_ptr(),
        out_t_ir_descr.as_mut_ptr(),
    );

    assert_eq!(
        unsafe { output_data::<u64>(&out_u_ir_descr, 2) },
        small_data,
        "the inserted slice is wrong"
    );
    assert_eq!(
        unsafe { output_data::<u64>(&out_t_ir_descr, 2) },
        t_data,
        "`t` was clobbered by a write through its slice"
    );
}

#[test]
fn test_tensor_reshape_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_tensor_reshape_from_rust.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);
    expect_file!["resources/test_tensor_reshape_from_rust.expect.plir"]
        .assert_eq(&after_bufferization);

    let input_data = [1u64, 2, 3, 4, 5, 6];
    let input = input_tensor(&[2, 3], &input_data);
    let f = unsafe {
        lookup_fn::<extern "C" fn(*const u8, i64, i64) -> i64>(&jit, "test_tensor_reshape_extract")
    };

    // 2x3 row-major [1,2,3,4,5,6] reshaped to 3x2 is:
    // [[1,2], [3,4], [5,6]]
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 0, 0), 1);
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 1, 0), 3);
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 2, 1), 6);
}

/// Tiled matmul, in control-flow form and with `cf.for`.
///
/// 4x4 matrices, 2x2 tiles. An outer loop over row tiles of C and an inner loop
/// over column tiles, with the accumulator threaded through both loops as a
/// loop-carried value.
#[test]
fn test_tiled_matmul() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_tiled_matmul.input.plir");

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // Both forms bufferize the same way: `a` and `b` are only read, so their tiles
    // are plain subviews (no copies) even though both stay live across every
    // iteration. The accumulator tile of matmul gets its own buffer.
    expect_file!["resources/test_tiled_matmul.expect.plir"].assert_eq(&after_bufferization);

    let a_data: Vec<u64> = (1..=16_u64).collect();
    let b_data: Vec<u64> = (17..=32_u64).collect();
    let a = input_tensor(&[4, 4], &a_data);
    let b = input_tensor(&[4, 4], &b_data);

    // tensor.matmul accumulates, so the tiled nest computes C + A*B.
    let c_data: Vec<u64> = (0..16_u64).map(|x| x * 100).collect();
    let mut expected = c_data.clone();
    for i in 0..4_usize {
        for j in 0..4_usize {
            for k in 0..4_usize {
                expected[i * 4 + j] += a_data[i * 4 + k] * b_data[k * 4 + j];
            }
        }
    }

    for name in ["test_tiled_matmul_cf", "test_tiled_matmul_scf"] {
        let f = unsafe {
            lookup_fn::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(&jit, name)
        };

        // `c` is the loop-carried accumulator and is written in place.
        let c_data: Vec<u64> = (0..16_u64).map(|x| x * 100).collect();
        let c = input_tensor(&[4, 4], &c_data);
        let mut out_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();

        f(
            a.build_ir_descriptor().as_ptr(),
            b.build_ir_descriptor().as_ptr(),
            c.build_ir_descriptor().as_ptr(),
            out_ir_descr.as_mut_ptr(),
        );

        assert_eq!(
            unsafe { output_data::<u64>(&out_ir_descr, 2) },
            expected,
            "{name} produced wrong values"
        );
    }
}

#[test]
fn test_constant_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_constant_from_rust.input.plir");

    let (llvm_ctx, llvm_ir, after_bufferization) = compile(ctx, &mut MallocFreeTMM, input_ir);

    // - Each constant gets its own `memref.global`
    // - The accumulator of the matmul is written in place
    // - The constant's buffer is read-only, so its bufferization is not in-place.
    expect_file!["resources/test_constant_from_rust.expect.plir"].assert_eq(&after_bufferization);

    expect_file!["resources/test_constant_from_rust.expect.ll"].assert_eq(&llvm_ir.to_string());

    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let extract =
        unsafe { lookup_fn::<extern "C" fn(i64, i64) -> f64>(&jit, "test_constant_extract") };
    for i in 0..2i64 {
        for j in 0..3i64 {
            assert_eq!(extract(i, j), (i * 3 + j + 1) as f64);
        }
    }

    let add =
        unsafe { lookup_fn::<extern "C" fn(*const u8, *mut u8) -> ()>(&jit, "test_constant_add") };
    let arg_data = [10.0f64, 20.0, 30.0, 40.0, 50.0, 60.0];
    let arg = input_tensor(&[2, 3], &arg_data);
    let mut add_res_ir_descr = output_tensor::<f64>(&[2, 3]).build_ir_descriptor();
    add(
        arg.build_ir_descriptor().as_ptr(),
        add_res_ir_descr.as_mut_ptr(),
    );
    assert_eq!(
        unsafe { output_data::<f64>(&add_res_ir_descr, 2) },
        [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    );

    // The matmul writes to its accumulator, which is a constant here. It must write to
    // a copy (inserted by the bufferizer), and thus two calls must give the same result.
    let accumulate = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            &jit,
            "test_constant_accumulator",
        )
    };
    let lhs_data = [1i64, 2, 3, 4];
    let rhs_data = [5i64, 6, 7, 8];
    let lhs = input_tensor(&[2, 2], &lhs_data);
    let rhs = input_tensor(&[2, 2], &rhs_data);
    // [[1,2],[3,4]] * [[5,6],[7,8]] = [[19,22],[43,50]], plus [[10,20],[30,40]].
    for call in 0..2 {
        let mut res_ir_descr = output_tensor::<i64>(&[2, 2]).build_ir_descriptor();
        accumulate(
            lhs.build_ir_descriptor().as_ptr(),
            rhs.build_ir_descriptor().as_ptr(),
            res_ir_descr.as_mut_ptr(),
        );
        assert_eq!(
            unsafe { output_data::<i64>(&res_ir_descr, 2) },
            [29i64, 42, 73, 90],
            "call {call} of test_constant_accumulator wrote to the constant"
        );
    }

    let splat = unsafe { lookup_fn::<extern "C" fn(*mut u8) -> ()>(&jit, "test_constant_splat") };
    let mut splat_res_ir_descr = output_tensor::<f64>(&[4, 4]).build_ir_descriptor();
    splat(splat_res_ir_descr.as_mut_ptr());
    assert_eq!(
        unsafe { output_data::<f64>(&splat_res_ir_descr, 2) },
        [1.0f64; 16]
    );
}
