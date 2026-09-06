// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test the tensor pipeline
//! - bufferization to memref
//! - Memref -> CF -> LLVM dialect
//! - execution of the result in a JIT.

use expect_test::expect;
use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::{Context, Ptr},
    init_env_logger_for_tests, input_error_noloc,
    irbuild::dialect_conversion::apply_dialect_conversion,
    irfmt::parsers::spaced,
    location,
    op::verify_op,
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{
    core::{LLVMContext, LLVMModule},
    lljit::{LLVMLLJIT, SimpleJIT},
    target::initialize_native,
};

use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{
        bufferize::bufferize,
        memory_management::{MallocFreeTMM, TensorMemoryManager},
        runtime_utils::TensorDesciptor,
        tracked_tmm::TrackedTMM,
    },
};

/// Parse `input_ir` into a module and verify it.
fn parse_module(ctx: &mut Context, input_ir: &str) -> (Ptr<Operation>, ModuleOp) {
    init_env_logger_for_tests!();

    let state_stream = state_stream_from_iterator(
        input_ir.chars(),
        parsable::State::new(ctx, location::Source::InMemory),
    );
    let parsed = spaced(Operation::top_level_parser())
        .parse(state_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));

    let parsed_op = parsed.expect_ok(ctx);
    let module_op = Operation::get_op::<ModuleOp>(parsed_op, ctx).unwrap();
    log::debug!("pliron module parsed {}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);
    (parsed_op, module_op)
}

/// Bufferize the module with `tmm` and return the bufferized IR as text.
fn bufferize_module<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    parsed_op: Ptr<Operation>,
    module_op: ModuleOp,
) -> String {
    bufferize(tmm, parsed_op, ctx).expect_ok(ctx);
    let after_bufferization = module_op.disp(ctx).to_string();
    log::debug!("pliron module after bufferization {}", after_bufferization);
    after_bufferization
}

/// Lower the bufferized module Memref -> CF -> LLVM dialect and emit its LLVM-IR.
fn lower_to_llvm_ir(
    ctx: &mut Context,
    parsed_op: Ptr<Operation>,
    module_op: ModuleOp,
    llvm_ctx: &LLVMContext,
) -> LLVMModule {
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx_module =
        pliron_llvm::to_llvm_ir::convert_module(ctx, llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ctx_module);
    llvm_ctx_module
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();
    llvm_ctx_module
}

/// Run `input_ir` through the full pipeline and JIT compile the result. The
/// bufferized IR is returned as text.
fn compile_and_jit<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    input_ir: &str,
) -> (SimpleJIT, String) {
    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let after_bufferization = bufferize_module(ctx, tmm, parsed_op, module_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");
    (jit, after_bufferization)
}

/// The same as [compile_and_jit], but the runtime symbols of `tmm` are also
/// registered with the JIT.
fn compile_and_jit_with_runtime<TMM: TensorMemoryManager>(
    ctx: &mut Context,
    tmm: &mut TMM,
    input_ir: &str,
) -> LLVMLLJIT {
    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    bufferize_module(ctx, tmm, parsed_op, module_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    tmm.register_runtime_symbols(&jit)
        .expect("Failed to register runtime symbols");
    jit.add_module(llvm_ctx, llvm_ir)
        .expect("Failed to add module to JIT");
    jit
}

/// Look up `name` in `jit` and interpret it as a function of type `F`.
///
/// # Safety
/// `F` must be the type of the compiled function.
unsafe fn lookup_fn<F: Copy>(jit: &LLVMLLJIT, name: &str) -> F {
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

/// `tensor.generate`, `tensor.extract` and the elementwise binary ops.
#[test]
fn test_elementwise_ops_from_rust() {
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_generate_add: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_res: builtin.integer i64, j_res: builtin.integer i64):
                    input1 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_1 : index.index, j_1 : index.index):
                        i_int_1 = index.to_integer i_1 to builtin.integer i64;
                        j_int_1 = index.to_integer j_1 to builtin.integer i64;
                        sum_1 = llvm.add i_int_1, j_int_1 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_1
                    };
                    input2 = tensor.generate : tensor.ranked<16x16:builtin.integer i64> {
                      ^entry(i_2 : index.index, j_2 : index.index):
                        i_int_2 = index.to_integer i_2 to builtin.integer i64;
                        j_int_2 = index.to_integer j_2 to builtin.integer i64;
                        sum_2 = llvm.add i_int_2, j_int_2 <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum_2
                    };
                    res_tensor = tensor.add input1, input2 : tensor.ranked<16x16:builtin.integer i64>;
                    i_res_index = index.from_integer i_res : index.index;
                    j_res_index = index.from_integer j_res : index.index;
                    res = tensor.extract res_tensor[i_res_index, j_res_index]: builtin.integer i64;
                    llvm.return res
                };
                llvm.func @test_tensor_add_int: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_tensor_add_float: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.fp64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.fp64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.fp64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_tensor_all_binops_float: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.fp64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.fp64>;
                    zero = tensor.sub arg2, arg2 : tensor.ranked<4x4:builtin.fp64>;
                    sum = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.fp64>;
                    sum_norm = tensor.add sum, zero : tensor.ranked<4x4:builtin.fp64>;
                    prod = tensor.mul sum_norm, arg2 : tensor.ranked<4x4:builtin.fp64>;
                    res = tensor.div prod, arg1 : tensor.ranked<4x4:builtin.fp64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

    let (jit, _) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    let generate_add = unsafe { jit.lookup_symbol::<fn(i64, i64) -> i64>("test_generate_add") }
        .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            "test_tensor_add_int",
        )
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            "test_tensor_add_float",
        )
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
            "test_tensor_all_binops_float",
        )
    }
    .expect("Failed to lookup symbol");
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
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_matmul_all_static: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_matmul_inner_dynamic: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x4:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_matmul_all_dynamic: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_batch_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<2x2x3:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<2x3x2:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<2x2x2:builtin.integer i64>;
                    res = tensor.batch_matmul arg1, arg2, arg3 : tensor.ranked<2x2x2:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

    let (jit, _) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

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
            jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(name)
        }
        .expect("Failed to lookup symbol");

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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            "test_batch_matmul",
        )
    }
    .expect("Failed to lookup symbol");
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
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_add_tracked: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                };
                llvm.func @test_tensor_complex_tracked: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    mat = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    sum = tensor.add mat, arg1 : tensor.ranked<4x4:builtin.integer i64>;
                    diff = tensor.sub sum, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.mul diff, arg1 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

    let mut tmm = TrackedTMM::new();
    let jit = compile_and_jit_with_runtime(ctx, &mut tmm, input_ir);

    // No tensor is allocated by the IR yet.
    assert_eq!(tmm.tracked_allocations().len(), 0);

    let add_lhs_data = [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let add_rhs_data = [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
    let add_lhs = input_tensor(&[4, 4], &add_lhs_data);
    let add_rhs = input_tensor(&[4, 4], &add_rhs_data);
    let mut add_res_ir_descr = output_tensor::<u64>(&[4, 4]).build_ir_descriptor();

    let add = unsafe {
        lookup_fn::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(
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
        lookup_fn::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
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

    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_aliasing_br: llvm.func <builtin.integer i64 (builtin.integer i1) variadic = false> [] {
                    ^entry(flag: builtin.integer i1):
                        x = tensor.generate : tensor.ranked<4:builtin.integer i64> {
                            ^entry(i_1 : index.index):
                                i_int = index.to_integer i_1 to builtin.integer i64;
                                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                                x_elem = llvm.add i_int, one <{nsw = false, nuw = false}> : builtin.integer i64;
                                memref.yield x_elem
                        };
                        llvm.br ^block_b(x)

                    ^block_b(z: tensor.ranked<4:builtin.integer i64>):
                        src = tensor.generate : tensor.ranked<1:builtin.integer i64> {
                            ^entry(i_2 : index.index):
                                ten = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64;
                                memref.yield ten
                        };
                        y = tensor.insert_slice src into z [0] [1] [1] : tensor.ranked<4:builtin.integer i64>;
                        sum = tensor.add x, y : tensor.ranked<4:builtin.integer i64>;
                        zero_idx_i64 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                        zero_idx = index.from_integer zero_idx_i64 : index.index;
                        res = tensor.extract sum[zero_idx]: builtin.integer i64;
                        llvm.return res
                };
                llvm.func @test_aliasing_cond_br: llvm.func <builtin.integer i64 (builtin.integer i1) variadic = false> [] {
                    ^entry(flag: builtin.integer i1):
                        x = tensor.generate : tensor.ranked<4:builtin.integer i64> {
                            ^entry(i_1 : index.index):
                                i_int = index.to_integer i_1 to builtin.integer i64;
                                one = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                                x_elem = llvm.add i_int, one <{nsw = false, nuw = false}> : builtin.integer i64;
                                memref.yield x_elem
                        };
                        llvm.cond_br if flag ^block_b(x, x) else ^block_c(x, x)

                    ^block_c(z_c: tensor.ranked<4:builtin.integer i64>, x_c: tensor.ranked<4:builtin.integer i64>):
                        src = tensor.generate : tensor.ranked<1:builtin.integer i64> {
                            ^entry(i_2 : index.index):
                                ten = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64;
                                memref.yield ten
                        };
                        y = tensor.insert_slice src into x_c [0] [1] [1] : tensor.ranked<4:builtin.integer i64>;
                        llvm.br ^block_b(z_c, y)

                    ^block_b(z: tensor.ranked<4:builtin.integer i64>, y_b: tensor.ranked<4:builtin.integer i64>):
                        sum = tensor.add z, y_b : tensor.ranked<4:builtin.integer i64>;
                        zero_idx_i64 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                        zero_idx = index.from_integer zero_idx_i64 : index.index;
                        res = tensor.extract sum[zero_idx]: builtin.integer i64;
                        llvm.return res
                }
        }
        "#;

    let (jit, _) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // Expected with correct bufferization:
    //   z is original x = [1, 2, 3, 4]
    //   y is x with index 0 updated to 10 => [10, 2, 3, 4]
    //   sum[0] = 1 + 10 = 11
    for name in ["test_aliasing_br", "test_aliasing_cond_br"] {
        let f =
            unsafe { jit.lookup_symbol::<fn(bool) -> i64>(name) }.expect("Failed to lookup symbol");
        assert_eq!(f(false), 11, "{name} computed the wrong result");
    }
}

/// `tensor.extract_slice` lowered to `memref.subview`
#[test]
fn test_extract_slice() {
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_extract_slice: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(src_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                        src = llvm.load src_p : tensor.ranked<10x20:builtin.integer i64>;
                        slice = tensor.extract_slice src [0, 2] [5, 10] [1, 2] : tensor.ranked<5x10:builtin.integer i64>;
                        llvm.store *out_p <- slice;
                        llvm.return
                };
                llvm.func @test_extract_slice_sequential: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(src_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                        src = llvm.load src_p : tensor.ranked<10x20:builtin.integer i64>;
                        first = tensor.extract_slice src [1, 2] [6, 8] [1, 2] : tensor.ranked<6x8:builtin.integer i64>;
                        second = tensor.extract_slice first [1, 1] [3, 4] [2, 2] : tensor.ranked<3x4:builtin.integer i64>;
                        llvm.store *out_p <- second;
                        llvm.return
                };
                llvm.func @test_extract_slice_live_source: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(src_p: llvm.ptr(0), out_first_p: llvm.ptr(0), out_second_p: llvm.ptr(0)):
                        src = llvm.load src_p : tensor.ranked<10x20:builtin.integer i64>;
                        first = tensor.extract_slice src [0, 0] [5, 10] [1, 1] : tensor.ranked<5x10:builtin.integer i64>;
                        second = tensor.extract_slice src [5, 10] [5, 10] [1, 1] : tensor.ranked<5x10:builtin.integer i64>;
                        llvm.store *out_first_p <- first;
                        llvm.store *out_second_p <- second;
                        llvm.return
                }
        }
        "#;

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // extract_slice only reads its source, so no slice needs a private buffer.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_extract_slice: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block2v1(src_p_v0: llvm.ptr (0), out_p_v1: llvm.ptr (0)) !1:
                src_v2 = llvm.load src_p_v0  : memref.ranked <10x20 : builtin.integer i64> !2;
                $slice_v15 = memref.subview src_v2 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64> !3;
                llvm.store *out_p_v1 <- slice_v15  !4;
                llvm.return  !5
            } !6;
            llvm.func @test_extract_slice_sequential: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block3v1(src_p_v4: llvm.ptr (0), out_p_v5: llvm.ptr (0)) !7:
                src_v6 = llvm.load src_p_v4  : memref.ranked <10x20 : builtin.integer i64> !8;
                $first_v16 = memref.subview src_v6 [1, 2] [6, 8] [1, 2] : memref.ranked <6x8 : builtin.integer i64> !9;
                $second_v17 = memref.subview first_v16 [1, 1] [3, 4] [2, 2] : memref.ranked <3x4 : builtin.integer i64> !10;
                llvm.store *out_p_v5 <- second_v17  !11;
                llvm.return  !12
            } !13;
            llvm.func @test_extract_slice_live_source: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block4v1(src_p_v9: llvm.ptr (0), out_first_p_v10: llvm.ptr (0), out_second_p_v11: llvm.ptr (0)) !14:
                src_v12 = llvm.load src_p_v9  : memref.ranked <10x20 : builtin.integer i64> !15;
                $first_v18 = memref.subview src_v12 [0, 0] [5, 10] [1, 1] : memref.ranked <5x10 : builtin.integer i64> !16;
                $second_v19 = memref.subview src_v12 [5, 10] [5, 10] [1, 1] : memref.ranked <5x10 : builtin.integer i64> !17;
                llvm.store *out_first_p_v10 <- first_v18  !18;
                llvm.store *out_second_p_v11 <- second_v19  !19;
                llvm.return  !20
            } !21
        }"#]].assert_eq(&after_bufferization);

    let src_data: Vec<u64> = (0..200_u64).collect();
    let src = input_tensor(&[10, 20], &src_data);

    let extract_slice = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const u8, *mut u8) -> ()>("test_extract_slice")
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *mut u8) -> ()>(
            "test_extract_slice_sequential",
        )
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *mut u8, *mut u8) -> ()>(
            "test_extract_slice_live_source",
        )
    }
    .expect("Failed to lookup symbol");
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

    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_insert_slice: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(src_p: llvm.ptr(0), dst_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                        src = llvm.load src_p : tensor.ranked<5x10:builtin.integer i64>;
                        dst = llvm.load dst_p : tensor.ranked<10x20:builtin.integer i64>;
                        updated = tensor.insert_slice src into dst [0, 2] [5, 10] [1, 2] : tensor.ranked<10x20:builtin.integer i64>;
                        llvm.store *out_p <- updated;
                        llvm.return
                };
                llvm.func @test_insert_slice_dest_live_after: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(src_p: llvm.ptr(0), dst_p: llvm.ptr(0), out_updated_p: llvm.ptr(0), out_dst_p: llvm.ptr(0)):
                        src = llvm.load src_p : tensor.ranked<5x10:builtin.integer i64>;
                        dst = llvm.load dst_p : tensor.ranked<10x20:builtin.integer i64>;
                        updated = tensor.insert_slice src into dst [0, 2] [5, 10] [1, 2] : tensor.ranked<10x20:builtin.integer i64>;
                        llvm.store *out_updated_p <- updated;
                        llvm.store *out_dst_p <- dst;
                        llvm.return
                };
                llvm.func @test_write_through_slice_of_live_tensor: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                    ^entry(t_p: llvm.ptr(0), small_p: llvm.ptr(0), out_u_p: llvm.ptr(0), out_t_p: llvm.ptr(0)):
                        t = llvm.load t_p : tensor.ranked<4x4:builtin.integer i64>;
                        small = llvm.load small_p : tensor.ranked<2x2:builtin.integer i64>;
                        s = tensor.extract_slice t [0, 0] [2, 2] [1, 1] : tensor.ranked<2x2:builtin.integer i64>;
                        u = tensor.insert_slice small into s [0, 0] [2, 2] [1, 1] : tensor.ranked<2x2:builtin.integer i64>;
                        llvm.store *out_u_p <- u;
                        llvm.store *out_t_p <- t;
                        llvm.return
                }
        }
        "#;

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // `tensor.insert_slice` writes in place only when destination buffer isn't seen later.
    // Only the two functions whose destination stays visible allocate a buffer.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_insert_slice: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block2v1(src_p_v0: llvm.ptr (0), dst_p_v1: llvm.ptr (0), out_p_v2: llvm.ptr (0)) !1:
                src_v3 = llvm.load src_p_v0  : memref.ranked <5x10 : builtin.integer i64> !2;
                dst_v4 = llvm.load dst_p_v1  : memref.ranked <10x20 : builtin.integer i64> !3;
                $v21 = memref.subview dst_v4 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64>;
                memref.copy v21 <- src_v3;
                llvm.store *out_p_v2 <- dst_v4  !4;
                llvm.return  !5
            } !6;
            llvm.func @test_insert_slice_dest_live_after: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block3v1(src_p_v6: llvm.ptr (0), dst_p_v7: llvm.ptr (0), out_updated_p_v8: llvm.ptr (0), out_dst_p_v9: llvm.ptr (0)) !7:
                src_v10 = llvm.load src_p_v6  : memref.ranked <5x10 : builtin.integer i64> !8;
                dst_v11 = llvm.load dst_p_v7  : memref.ranked <10x20 : builtin.integer i64> !9;
                updated_v22 = memref.alloc  : memref.ranked <10x20 : builtin.integer i64> !10;
                memref.copy updated_v22 <- dst_v11;
                $v23 = memref.subview updated_v22 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64>;
                memref.copy v23 <- src_v10;
                llvm.store *out_updated_p_v8 <- updated_v22  !11;
                llvm.store *out_dst_p_v9 <- dst_v11  !12;
                llvm.return  !13
            } !14;
            llvm.func @test_write_through_slice_of_live_tensor: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block4v1(t_p_v13: llvm.ptr (0), small_p_v14: llvm.ptr (0), out_u_p_v15: llvm.ptr (0), out_t_p_v16: llvm.ptr (0)) !15:
                t_v17 = llvm.load t_p_v13  : memref.ranked <4x4 : builtin.integer i64> !16;
                small_v18 = llvm.load small_p_v14  : memref.ranked <2x2 : builtin.integer i64> !17;
                $s_v24 = memref.subview t_v17 [0, 0] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64> !18;
                u_v25 = memref.alloc  : memref.ranked <2x2 : builtin.integer i64> !19;
                memref.copy u_v25 <- s_v24;
                $v26 = memref.subview u_v25 [0, 0] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64>;
                memref.copy v26 <- small_v18;
                llvm.store *out_u_p_v15 <- u_v25  !20;
                llvm.store *out_t_p_v16 <- t_v17  !21;
                llvm.return  !22
            } !23
        }"#]].assert_eq(&after_bufferization);

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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8) -> ()>("test_insert_slice")
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            "test_insert_slice_dest_live_after",
        )
    }
    .expect("Failed to lookup symbol");
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
        jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            "test_write_through_slice_of_live_tensor",
        )
    }
    .expect("Failed to lookup symbol");
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
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_reshape_extract: llvm.func <builtin.integer i64 (llvm.ptr(0), builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(arg_p: llvm.ptr(0), i_res: builtin.integer i64, j_res: builtin.integer i64):
                    arg = llvm.load arg_p : tensor.ranked<2x3:builtin.integer i64>;
                    reshaped = tensor.reshape arg() : tensor.ranked<3x2:builtin.integer i64>;
                    i_idx = index.from_integer i_res : index.index;
                    j_idx = index.from_integer j_res : index.index;
                    res = tensor.extract reshaped[i_idx, j_idx]: builtin.integer i64;
                    llvm.return res
                }
            }
            "#;

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_tensor_reshape_extract: llvm.func <builtin.integer i64(llvm.ptr (0), builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(arg_p_v0: llvm.ptr (0), i_res_v1: builtin.integer i64, j_res_v2: builtin.integer i64) !1:
                arg_v3 = llvm.load arg_p_v0  : memref.ranked <2x3 : builtin.integer i64> !2;
                reshaped_v8 = memref.reshape arg_v3 : memref.ranked <3x2 : builtin.integer i64> !3;
                i_idx_v5 = index.from_integer i_res_v1 : index.index  !4;
                j_idx_v6 = index.from_integer j_res_v2 : index.index  !5;
                res_v9 = memref.load reshaped_v8[i_idx_v5, j_idx_v6] : builtin.integer i64 !6;
                llvm.return res_v9 !7
            } !8
        }"#]].assert_eq(&after_bufferization);

    let input_data = [1u64, 2, 3, 4, 5, 6];
    let input = input_tensor(&[2, 3], &input_data);
    let f = unsafe {
        jit.lookup_symbol::<extern "C" fn(*const u8, i64, i64) -> i64>(
            "test_tensor_reshape_extract",
        )
    }
    .expect("Failed to lookup symbol");

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

    let input_ir = r#"
        builtin.module @test_module {
        ^entry():
            llvm.func @test_tiled_matmul_cf: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
            ^entry(a_p: llvm.ptr(0), b_p: llvm.ptr(0), c_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                a = llvm.load a_p : tensor.ranked<4x4:builtin.integer i64>;
                b = llvm.load b_p : tensor.ranked<4x4:builtin.integer i64>;
                c = llvm.load c_p : tensor.ranked<4x4:builtin.integer i64>;
                i_init = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                llvm.br ^outer_header(i_init, c)

            ^outer_header(i: builtin.integer i64, iv_c: tensor.ranked<4x4:builtin.integer i64>):
                n_i = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
                i_lt = llvm.icmp i <SLT> n_i : builtin.integer i1;
                llvm.cond_br if i_lt ^outer_body(i, iv_c) else ^done(iv_c)

            ^outer_body(i_b: builtin.integer i64, iv_c_b: tensor.ranked<4x4:builtin.integer i64>):
                j_init = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                llvm.br ^inner_header(i_b, j_init, iv_c_b)

            ^inner_header(i_h: builtin.integer i64, j_h: builtin.integer i64, jv_c: tensor.ranked<4x4:builtin.integer i64>):
                n_j = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
                j_lt = llvm.icmp j_h <SLT> n_j : builtin.integer i1;
                llvm.cond_br if j_lt ^inner_body(i_h, j_h, jv_c) else ^outer_latch(i_h, jv_c)

            ^inner_body(i_n: builtin.integer i64, j_n: builtin.integer i64, jv_c_n: tensor.ranked<4x4:builtin.integer i64>):
                i_idx = index.from_integer i_n : index.index;
                j_idx = index.from_integer j_n : index.index;
                slice_a = tensor.extract_slice a [i_idx, 0] [2, 4] [1, 1] : tensor.ranked<2x4:builtin.integer i64>;
                slice_b = tensor.extract_slice b [0, j_idx] [4, 2] [1, 1] : tensor.ranked<4x2:builtin.integer i64>;
                slice_c = tensor.extract_slice jv_c_n [i_idx, j_idx] [2, 2] [1, 1] : tensor.ranked<2x2:builtin.integer i64>;
                tiled = tensor.matmul slice_a, slice_b, slice_c : tensor.ranked<2x2:builtin.integer i64>;
                updated = tensor.insert_slice tiled into jv_c_n [i_idx, j_idx] [2, 2] [1, 1] : tensor.ranked<4x4:builtin.integer i64>;
                step_j = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64;
                j_next = llvm.add j_n, step_j <{nsw = false, nuw = false}> : builtin.integer i64;
                llvm.br ^inner_header(i_n, j_next, updated)

            ^outer_latch(i_l: builtin.integer i64, jv_c_l: tensor.ranked<4x4:builtin.integer i64>):
                step_i = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64;
                i_next = llvm.add i_l, step_i <{nsw = false, nuw = false}> : builtin.integer i64;
                llvm.br ^outer_header(i_next, jv_c_l)

            ^done(result: tensor.ranked<4x4:builtin.integer i64>):
                llvm.store *out_p <- result;
                llvm.return
            };
            llvm.func @test_tiled_matmul_scf: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
            ^entry(a_p: llvm.ptr(0), b_p: llvm.ptr(0), c_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                a = llvm.load a_p : tensor.ranked<4x4:builtin.integer i64>;
                b = llvm.load b_p : tensor.ranked<4x4:builtin.integer i64>;
                c = llvm.load c_p : tensor.ranked<4x4:builtin.integer i64>;
                c0 = index.constant <index.constant 0> : index.index;
                c2 = index.constant <index.constant 2> : index.index;
                c4 = index.constant <index.constant 4> : index.index;
                result = cf.for c0 to c4 step c2 (c) {
                    ^entry(i: index.index, iv_c: tensor.ranked<4x4:builtin.integer i64>):
                        inner_res = cf.for c0 to c4 step c2 (iv_c) {
                            ^entry(j: index.index, jv_c: tensor.ranked<4x4:builtin.integer i64>):
                                slice_a = tensor.extract_slice a [i, 0] [2, 4] [1, 1] : tensor.ranked<2x4:builtin.integer i64>;
                                slice_b = tensor.extract_slice b [0, j] [4, 2] [1, 1] : tensor.ranked<4x2:builtin.integer i64>;
                                slice_c = tensor.extract_slice jv_c [i, j] [2, 2] [1, 1] : tensor.ranked<2x2:builtin.integer i64>;
                                tiled = tensor.matmul slice_a, slice_b, slice_c : tensor.ranked<2x2:builtin.integer i64>;
                                updated = tensor.insert_slice tiled into jv_c [i, j] [2, 2] [1, 1] : tensor.ranked<4x4:builtin.integer i64>;
                                cf.yield updated
                        };
                        cf.yield inner_res
                };
                llvm.store *out_p <- result;
                llvm.return
            }
        }
        "#;

    let (jit, after_bufferization) = compile_and_jit(ctx, &mut MallocFreeTMM, input_ir);

    // Both forms bufferize the same way: `a` and `b` are only read, so their tiles
    // are plain subviews, never copied, even though both stay live across every
    // iteration. Only the accumulator tile of matmul gets a private buffer; the
    // loop-carried accumulator itself is never copied.
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_tiled_matmul_cf: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block2v1(a_p_v0: llvm.ptr (0), b_p_v1: llvm.ptr (0), c_p_v2: llvm.ptr (0), out_p_v3: llvm.ptr (0)) !1:
                a_v4 = llvm.load a_p_v0  : memref.ranked <4x4 : builtin.integer i64> !2;
                b_v5 = llvm.load b_p_v1  : memref.ranked <4x4 : builtin.integer i64> !3;
                c_v6 = llvm.load c_p_v2  : memref.ranked <4x4 : builtin.integer i64> !4;
                i_init_v7 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64 !5;
                llvm.br ^outer_header_block4v1(i_init_v7, c_v6) !6

              ^outer_header_block4v1(i_v8: builtin.integer i64, iv_c_v9: memref.ranked <4x4 : builtin.integer i64>) !7:
                n_i_v10 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !8;
                i_lt_v11 = llvm.icmp i_v8 <SLT> n_i_v10 : builtin.integer i1 !9;
                llvm.cond_br if i_lt_v11 ^outer_body_block6v1(i_v8, iv_c_v9) else ^done_block8v3(iv_c_v9) !10

              ^outer_body_block6v1(i_b_v12: builtin.integer i64, iv_c_b_v13: memref.ranked <4x4 : builtin.integer i64>) !11:
                j_init_v14 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64 !12;
                llvm.br ^inner_header_block7v1(i_b_v12, j_init_v14, iv_c_b_v13) !13

              ^inner_header_block7v1(i_h_v15: builtin.integer i64, j_h_v16: builtin.integer i64, jv_c_v17: memref.ranked <4x4 : builtin.integer i64>) !14:
                n_j_v18 = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64 !15;
                j_lt_v19 = llvm.icmp j_h_v16 <SLT> n_j_v18 : builtin.integer i1 !16;
                llvm.cond_br if j_lt_v19 ^inner_body_block9v1(i_h_v15, j_h_v16, jv_c_v17) else ^outer_latch_block3v9(i_h_v15, jv_c_v17) !17

              ^inner_body_block9v1(i_n_v20: builtin.integer i64, j_n_v21: builtin.integer i64, jv_c_n_v22: memref.ranked <4x4 : builtin.integer i64>) !18:
                i_idx_v23 = index.from_integer i_n_v20 : index.index  !19;
                j_idx_v24 = index.from_integer j_n_v21 : index.index  !20;
                $slice_a_v58 = memref.subview a_v4 [i_idx_v23, 0] [2, 4] [1, 1] : memref.ranked <2x4 : builtin.integer i64> !21;
                $slice_b_v59 = memref.subview b_v5 [0, j_idx_v24] [4, 2] [1, 1] : memref.ranked <4x2 : builtin.integer i64> !22;
                $slice_c_v60 = memref.subview jv_c_n_v22 [i_idx_v23, j_idx_v24] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64> !23;
                v61 = memref.alloc  : memref.ranked <2x2 : builtin.integer i64>;
                memref.copy v61 <- slice_c_v60;
                tiled_v62 = memref.matmul slice_a_v58, slice_b_v59, v61 : memref.ranked <2x2 : builtin.integer i64> !24;
                $v63 = memref.subview jv_c_n_v22 [i_idx_v23, j_idx_v24] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64>;
                memref.copy v63 <- tiled_v62;
                step_j_v30 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !25;
                j_next_v31 = llvm.add j_n_v21, step_j_v30 <{nsw=false,nuw=false}>: builtin.integer i64 !26;
                llvm.br ^inner_header_block7v1(i_n_v20, j_next_v31, jv_c_n_v22) !27

              ^outer_latch_block3v9(i_l_v32: builtin.integer i64, jv_c_l_v33: memref.ranked <4x4 : builtin.integer i64>) !28:
                step_i_v34 = builtin.constant <builtin.integer <2: i64>> : builtin.integer i64 !29;
                i_next_v35 = llvm.add i_l_v32, step_i_v34 <{nsw=false,nuw=false}>: builtin.integer i64 !30;
                llvm.br ^outer_header_block4v1(i_next_v35, jv_c_l_v33) !31

              ^done_block8v3(result_v36: memref.ranked <4x4 : builtin.integer i64>) !32:
                llvm.store *out_p_v3 <- result_v36  !33;
                llvm.return  !34
            } !35;
            llvm.func @test_tiled_matmul_scf: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block5v3(a_p_v37: llvm.ptr (0), b_p_v38: llvm.ptr (0), c_p_v39: llvm.ptr (0), out_p_v40: llvm.ptr (0)) !36:
                a_v41 = llvm.load a_p_v37  : memref.ranked <4x4 : builtin.integer i64> !37;
                b_v42 = llvm.load b_p_v38  : memref.ranked <4x4 : builtin.integer i64> !38;
                c_v43 = llvm.load c_p_v39  : memref.ranked <4x4 : builtin.integer i64> !39;
                c0_v44 = index.constant <index.constant 0> : index.index  !40;
                c2_v45 = index.constant <index.constant 2> : index.index  !41;
                c4_v46 = index.constant <index.constant 4> : index.index  !42;
                result_v47 = cf.for c0_v44 to c4_v46 step c2_v45 (c_v43) 
                {
                  ^entry_block10v1(i_v48: index.index , iv_c_v49: memref.ranked <4x4 : builtin.integer i64>) !43:
                    inner_res_v50 = cf.for c0_v44 to c4_v46 step c2_v45 (iv_c_v49) 
                    {
                      ^entry_block11v1(j_v51: index.index , jv_c_v52: memref.ranked <4x4 : builtin.integer i64>) !44:
                        $slice_a_v64 = memref.subview a_v41 [i_v48, 0] [2, 4] [1, 1] : memref.ranked <2x4 : builtin.integer i64> !45;
                        $slice_b_v65 = memref.subview b_v42 [0, j_v51] [4, 2] [1, 1] : memref.ranked <4x2 : builtin.integer i64> !46;
                        $slice_c_v66 = memref.subview jv_c_v52 [i_v48, j_v51] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64> !47;
                        v67 = memref.alloc  : memref.ranked <2x2 : builtin.integer i64>;
                        memref.copy v67 <- slice_c_v66;
                        tiled_v68 = memref.matmul slice_a_v64, slice_b_v65, v67 : memref.ranked <2x2 : builtin.integer i64> !48;
                        $v69 = memref.subview jv_c_v52 [i_v48, j_v51] [2, 2] [1, 1] : memref.ranked <2x2 : builtin.integer i64>;
                        memref.copy v69 <- tiled_v68;
                        cf.yield jv_c_v52 !49
                    }
         !50;
                    cf.yield inner_res_v50 !51
                }
         !52;
                llvm.store *out_p_v40 <- result_v47  !53;
                llvm.return  !54
            } !55
        }"#]].assert_eq(&after_bufferization);

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
            jit.lookup_symbol::<extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(name)
        }
        .expect("Failed to lookup symbol");

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
