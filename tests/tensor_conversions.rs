// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test conversions of memref operations to Memref -> CF -> LLVM dialect.

use expect_test::expect;
use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
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
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

use pliron_tensor::{
    memref::conversions::MemrefToCF,
    tensor::{
        bufferize::bufferize,
        memory_management::{MallocFreeTMM, TensorMemoryManager},
        runtime_utils::TensorDesciptor,
        tracked_tmm::TrackedTMM,
    },
};

#[test]
fn test_tensor_to_memref_conversion() {
    init_env_logger_for_tests!();
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
                }
            }
            "#;

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

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_generate_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, ((i + j) * 2));
        }
    }
}

fn test_successor_operand_aliasing_needs_copy_helper(input_ir: &str) {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_successor_operand_aliasing")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe { std::mem::transmute::<u64, fn(bool) -> i64>(symbol_addr) };

    let result = f(false);

    // Expected with correct bufferization:
    //   z is original x = [1, 2, 3, 4]
    //   y is x with index 0 updated to 10 => [10, 2, 3, 4]
    //   sum[0] = 1 + 10 = 11
    assert_eq!(result, 11);
}

#[test]
fn test_successor_operand_aliasing_needs_copy_0() {
    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_successor_operand_aliasing: llvm.func <builtin.integer i64 (builtin.integer i1) variadic = false> [] {
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
                }
        }
        "#;
    test_successor_operand_aliasing_needs_copy_helper(input_ir);
}

#[test]
fn test_successor_operand_aliasing_needs_copy_1() {
    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_successor_operand_aliasing: llvm.func <builtin.integer i64 (builtin.integer i1) variadic = false> [] {
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
    test_successor_operand_aliasing_needs_copy_helper(input_ir);
}

#[test]
fn test_int_tensor_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_add: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;

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

    let mut tmm = TrackedTMM::new();
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    log::debug!("pliron module after bufferization {}", module_op.disp(ctx));
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after Memref to CF conversion {}",
        module_op.disp(ctx)
    );
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    tmm.register_runtime_symbols(&jit)
        .expect("Failed to register runtime symbols for TrackedTMM");

    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_add")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [16u64, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1].as_ptr() as *const u8,
    );

    // We build the result descriptor to build the result IR descriptor, where the executed
    // function will write the result descriptor of the addition.
    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    // No tensor is allocated by the IR yet
    assert_eq!(tmm.tracked_allocations().len(), 0);

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    // We have one tensor allocated for the result.
    assert_eq!(tmm.tracked_allocations().len(), 1);

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[17; 16]);
}

#[test]
fn test_matmul_all_statics_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

#[test]
fn test_matmul_inner_dynamic_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x4:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

#[test]
fn test_matmul_all_dynamic_from_rust() {
    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), arg3_p: llvm.ptr(0), res_p: llvm.ptr(0)):
                    arg1 = llvm.load arg1_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg3 = llvm.load arg3_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2, arg3 : tensor.ranked<4x4:builtin.integer i64>;
                    llvm.store *res_p <- res;
                    llvm.return
                }
            }
            "#;
    test_int_tensor_matmul_from_rust(input_ir);
}

fn test_int_tensor_matmul_from_rust(input_ir: &str) {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

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

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_matmul")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let mut accum_data = [1u64; 16];
    let t3 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        accum_data.as_mut_ptr() as *const u8,
    );

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        t3.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(
        res_slice,
        &[
            29u64, 33, 37, 41, 29, 33, 37, 41, 29, 33, 37, 41, 29, 33, 37, 41
        ]
    );
}

#[test]
fn test_batch_matmul_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_tensor_batch_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir.verify().unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_batch_matmul")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    // Batch 0 lhs: [[1,2,3],[4,5,6]], rhs: [[1,2],[3,4],[5,6]]
    // result: [[22,28],[49,64]]
    // Batch 1 lhs: [[7,8,9],[10,11,12]], rhs: [[7,8],[9,10],[11,12]]
    // result: [[220,244],[301,334]]
    let t1 = TensorDesciptor::new(
        [2, 2, 3].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [2, 3, 2].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [2, 2, 2].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let mut accum_data = [1u64, 1, 1, 1, 2, 2, 2, 2];
    let t3 = TensorDesciptor::new(
        [2, 2, 2].to_vec(),
        std::mem::size_of::<u64>(),
        accum_data.as_mut_ptr() as *const u8,
    );

    let mut res_ir_descr = res_descr.build_ir_descriptor();
    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        t3.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 3, std::mem::size_of::<u64>())
    };
    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const u64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[23u64, 29, 50, 65, 222, 246, 303, 336]);
}

#[test]
fn test_float_tensor_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
        llvm.func @test_tensor_add_float: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
          ^entry(arg1_p: llvm.ptr(0), arg2_p: llvm.ptr(0), res_p: llvm.ptr(0)):
          arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.fp64>;
          arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.fp64>;
          res = tensor.add arg1, arg2 : tensor.ranked<4x4:builtin.fp64>;
          llvm.store *res_p <- res;
          llvm.return
        }
      }
      "#;

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

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_add_float")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        [
            1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            16.0,
        ]
        .as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        [
            16.0f64, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0,
            1.0,
        ]
        .as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<f64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const f64,
            res_tensor_descr.num_elements(),
        )
    };

    assert_eq!(res_slice, &[17.0; 16]);
}

#[test]
fn test_float_tensor_all_binary_ops_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
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

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_all_binops_float")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let lhs_data = [
        1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
    ];
    let rhs_data = [
        16.0f64, 15.0, 14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,
    ];

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        lhs_data.as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        rhs_data.as_ptr() as *const u8,
    );

    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<f64>(),
        std::ptr::null::<u8>(),
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<f64>())
    };

    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const f64,
            res_tensor_descr.num_elements(),
        )
    };

    for ((&a, &b), &c) in lhs_data.iter().zip(rhs_data.iter()).zip(res_slice.iter()) {
        let expected = ((a + b) * b) / a;
        assert!((c - expected).abs() < 1e-12);
    }
}

/// Test that `tensor.extract_slice` is correctly lowered to `memref.subview`
/// plus an explicit `memref.copy`
/// by the TensorToMemref conversion pass.
#[test]
fn test_extract_slice_tensor_to_memref() {
    init_env_logger_for_tests!();
    // Build and execute a pliron function that extracts a slice from a tensor passed in from Rust,
    // and writes the slice as an output tensor descriptor so we can validate exact values.
    let exec_ctx = &mut Context::new();
    let exec_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_extract_slice_runtime: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                            ^entry(src_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                                src = llvm.load src_p : tensor.ranked<10x20:builtin.integer i64>;
                                slice = tensor.extract_slice src [0, 2] [5, 10] [1, 2] : tensor.ranked<5x10:builtin.integer i64>;
                                llvm.store *out_p <- slice;
                                llvm.return
                        }
                }
        "#;

    let exec_stream = state_stream_from_iterator(
        exec_ir.chars(),
        parsable::State::new(exec_ctx, location::Source::InMemory),
    );
    let exec_parsed = spaced(Operation::top_level_parser())
        .parse(exec_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));
    let exec_parsed_op = exec_parsed.expect_ok(exec_ctx);
    let exec_module_op = Operation::get_op::<ModuleOp>(exec_parsed_op, exec_ctx).unwrap();
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, exec_parsed_op, exec_ctx).expect_ok(exec_ctx);
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_extract_slice_runtime: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block2v1(src_p_v0: llvm.ptr (0), out_p_v1: llvm.ptr (0)) !1:
                src_v2 = llvm.load src_p_v0  : memref.ranked <10x20 : builtin.integer i64> !2;
                $v4 = memref.subview src_v2 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64> !3;
                llvm.store *out_p_v1 <- v4  !4;
                llvm.return  !5
            } !6
        }"#]].assert_eq(&exec_module_op.disp(exec_ctx).to_string());
    apply_dialect_conversion(exec_ctx, &mut MemrefToCF, exec_parsed_op).expect_ok(exec_ctx);
    apply_dialect_conversion(exec_ctx, &mut CFToLLVM, exec_parsed_op).expect_ok(exec_ctx);
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(exec_ctx, &llvm_ctx, exec_module_op)
        .expect_ok(exec_ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_extract_slice_runtime")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f =
        unsafe { std::mem::transmute::<u64, extern "C" fn(*const u8, *mut u8) -> ()>(symbol_addr) };

    let src_data: Vec<u64> = (0..200_u64).collect();
    let src_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        src_data.as_ptr() as *const u8,
    );
    let out_descr = TensorDesciptor::new(
        [5, 10].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_ir_descr = out_descr.build_ir_descriptor();
    f(
        src_descr.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let out_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };
    let mut actual: Vec<u64> = Vec::new();
    unsafe { out_tensor_descr.copy_to_vec(&mut actual) };

    let mut expected = Vec::with_capacity(5 * 10);
    for i in 0..5_u64 {
        for j in 0..10_u64 {
            // src[i][2 + 2*j] for offsets [0, 2], sizes [5, 10], strides [1, 2].
            expected.push(i * 20 + 2 + 2 * j);
        }
    }
    assert_eq!(actual, expected);
}

/// Test that two sequential `tensor.extract_slice` operations are lowered and
/// executed correctly end-to-end.
#[test]
fn test_extract_slice_tensor_to_memref_sequential() {
    init_env_logger_for_tests!();
    // Build and execute a pliron function that extracts a slice from a tensor and then
    // extracts another slice from the first slice. The final slice is returned through
    // an output descriptor so we can validate exact values.
    let exec_ctx = &mut Context::new();
    let exec_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_extract_slice_runtime_sequential: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                            ^entry(src_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                                src = llvm.load src_p : tensor.ranked<10x20:builtin.integer i64>;
                                first = tensor.extract_slice src [1, 2] [6, 8] [1, 2] : tensor.ranked<6x8:builtin.integer i64>;
                                second = tensor.extract_slice first [1, 1] [3, 4] [2, 2] : tensor.ranked<3x4:builtin.integer i64>;
                                llvm.store *out_p <- second;
                                llvm.return
                        }
                }
        "#;

    let exec_stream = state_stream_from_iterator(
        exec_ir.chars(),
        parsable::State::new(exec_ctx, location::Source::InMemory),
    );
    let exec_parsed = spaced(Operation::top_level_parser())
        .parse(exec_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));
    let exec_parsed_op = exec_parsed.expect_ok(exec_ctx);
    let exec_module_op = Operation::get_op::<ModuleOp>(exec_parsed_op, exec_ctx).unwrap();
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, exec_parsed_op, exec_ctx).expect_ok(exec_ctx);
    let after_tensor_to_memref = format!("{}", exec_module_op.disp(exec_ctx));
    assert!(
        !after_tensor_to_memref.contains("tensor.extract_slice"),
        "both tensor.extract_slice ops should be lowered by TensorToMemref"
    );
    assert!(
        after_tensor_to_memref.matches("memref.subview").count() >= 2,
        "expected at least two memref.subview ops after lowering"
    );

    apply_dialect_conversion(exec_ctx, &mut MemrefToCF, exec_parsed_op).expect_ok(exec_ctx);
    apply_dialect_conversion(exec_ctx, &mut CFToLLVM, exec_parsed_op).expect_ok(exec_ctx);
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(exec_ctx, &llvm_ctx, exec_module_op)
        .expect_ok(exec_ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_extract_slice_runtime_sequential")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f =
        unsafe { std::mem::transmute::<u64, extern "C" fn(*const u8, *mut u8) -> ()>(symbol_addr) };

    let src_data: Vec<u64> = (0..200_u64).collect();
    let src_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        src_data.as_ptr() as *const u8,
    );
    let out_descr = TensorDesciptor::new(
        [3, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_ir_descr = out_descr.build_ir_descriptor();
    f(
        src_descr.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let out_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };
    let mut actual: Vec<u64> = Vec::new();
    unsafe { out_tensor_descr.copy_to_vec(&mut actual) };

    let mut expected = Vec::with_capacity(3 * 4);
    for i in 0..3_u64 {
        for j in 0..4_u64 {
            // first[i1, j1] = src[1 + i1, 2 + 2*j1]
            // second[i, j] = first[1 + 2*i, 1 + 2*j] = src[2 + 2*i, 4 + 4*j]
            expected.push((2 + 2 * i) * 20 + (4 + 4 * j));
        }
    }
    assert_eq!(actual, expected);
}

/// Test that a read-only aliasing operand is bufferized in place even when it stays
/// live afterwards. Here `src` is live across the first `tensor.extract_slice` (the
/// second one reads it again), but `extract_slice` only reads through it, so both
/// slices may share `src`'s buffer with no copy at all.
#[test]
fn test_extract_slice_live_source_bufferizes_in_place() {
    init_env_logger_for_tests!();
    let exec_ctx = &mut Context::new();
    let exec_ir = r#"
        builtin.module @test_module {
        ^entry():
            llvm.func @test_extract_slice_live_source_runtime: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
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

    let exec_stream = state_stream_from_iterator(
        exec_ir.chars(),
        parsable::State::new(exec_ctx, location::Source::InMemory),
    );
    let exec_parsed = spaced(Operation::top_level_parser())
        .parse(exec_stream)
        .map(|(op, _)| op)
        .map_err(|err| input_error_noloc!(err));
    let exec_parsed_op = exec_parsed.expect_ok(exec_ctx);
    let exec_module_op = Operation::get_op::<ModuleOp>(exec_parsed_op, exec_ctx).unwrap();
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, exec_parsed_op, exec_ctx).expect_ok(exec_ctx);
    let after_tensor_to_memref = format!("{}", exec_module_op.disp(exec_ctx));

    // Both extract_slices only read `src`, so neither needs a private buffer,
    // even though `src` is live across the first of them.
    assert!(
        !after_tensor_to_memref.contains("memref.alloc"),
        "read-only aliasing operands must not be copied to a new buffer, got:\n{after_tensor_to_memref}"
    );
    assert!(
        !after_tensor_to_memref.contains("memref.copy"),
        "read-only aliasing operands must not be copied, got:\n{after_tensor_to_memref}"
    );
    assert_eq!(
        after_tensor_to_memref.matches("memref.subview").count(),
        2,
        "expected exactly two memref.subview ops, got:\n{after_tensor_to_memref}"
    );

    apply_dialect_conversion(exec_ctx, &mut MemrefToCF, exec_parsed_op).expect_ok(exec_ctx);
    apply_dialect_conversion(exec_ctx, &mut CFToLLVM, exec_parsed_op).expect_ok(exec_ctx);
    verify_op(&exec_module_op, exec_ctx).expect_ok(exec_ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(exec_ctx, &llvm_ctx, exec_module_op)
        .expect_ok(exec_ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_extract_slice_live_source_runtime")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *mut u8, *mut u8) -> ()>(symbol_addr)
    };

    let src_data: Vec<u64> = (0..200_u64).collect();
    let src_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        src_data.as_ptr() as *const u8,
    );
    let out_first_descr = TensorDesciptor::new(
        [5, 10].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );
    let out_second_descr = TensorDesciptor::new(
        [5, 10].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_first_ir_descr = out_first_descr.build_ir_descriptor();
    let mut out_second_ir_descr = out_second_descr.build_ir_descriptor();
    f(
        src_descr.build_ir_descriptor().as_ptr(),
        out_first_ir_descr.as_mut_ptr(),
        out_second_ir_descr.as_mut_ptr(),
    );

    let first_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(
            out_first_ir_descr.as_ptr(),
            2,
            std::mem::size_of::<u64>(),
        )
    };
    let second_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(
            out_second_ir_descr.as_ptr(),
            2,
            std::mem::size_of::<u64>(),
        )
    };
    let mut actual_first: Vec<u64> = Vec::new();
    let mut actual_second: Vec<u64> = Vec::new();
    unsafe { first_descr.copy_to_vec(&mut actual_first) };
    unsafe { second_descr.copy_to_vec(&mut actual_second) };

    let mut expected_first = Vec::with_capacity(5 * 10);
    let mut expected_second = Vec::with_capacity(5 * 10);
    for i in 0..5_u64 {
        for j in 0..10_u64 {
            expected_first.push(i * 20 + j);
            expected_second.push((5 + i) * 20 + (10 + j));
        }
    }
    assert_eq!(actual_first, expected_first);
    assert_eq!(actual_second, expected_second);
}

/// Test that `tensor.insert_slice` is lowered and executed correctly end-to-end:
/// TensorToMemref -> MemrefToCF -> CFToLLVM -> JIT.
#[test]
fn test_insert_slice_tensor_to_memref() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_insert_slice_runtime: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
                            ^entry(src_p: llvm.ptr(0), dst_p: llvm.ptr(0), out_p: llvm.ptr(0)):
                                src = llvm.load src_p : tensor.ranked<5x10:builtin.integer i64>;
                                dst = llvm.load dst_p : tensor.ranked<10x20:builtin.integer i64>;
                                updated = tensor.insert_slice src into dst [0, 2] [5, 10] [1, 2] : tensor.ranked<10x20:builtin.integer i64>;
                                llvm.store *out_p <- updated;
                                llvm.return
                        }
                }
        "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after_tensor_to_memref = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_insert_slice_runtime: llvm.func <llvm.void (llvm.ptr (0), llvm.ptr (0), llvm.ptr (0)) variadic = false>
              [] 
            {
              ^entry_block2v1(src_p_v0: llvm.ptr (0), dst_p_v1: llvm.ptr (0), out_p_v2: llvm.ptr (0)) !1:
                src_v3 = llvm.load src_p_v0  : memref.ranked <5x10 : builtin.integer i64> !2;
                dst_v4 = llvm.load dst_p_v1  : memref.ranked <10x20 : builtin.integer i64> !3;
                $v6 = memref.subview dst_v4 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64>;
                memref.copy v6 <- src_v3;
                llvm.store *out_p_v2 <- dst_v4  !4;
                llvm.return  !5
            } !6
        }"#]].assert_eq(&after_tensor_to_memref);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_insert_slice_runtime")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let src_data: Vec<u64> = (100..150_u64).collect();
    let dst_data: Vec<u64> = (0..200_u64).collect();

    let src_descr = TensorDesciptor::new(
        [5, 10].to_vec(),
        std::mem::size_of::<u64>(),
        src_data.as_ptr() as *const u8,
    );
    let dst_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        dst_data.as_ptr() as *const u8,
    );
    let out_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_ir_descr = out_descr.build_ir_descriptor();
    f(
        src_descr.build_ir_descriptor().as_ptr(),
        dst_descr.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let out_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };
    let out_slice = unsafe {
        std::slice::from_raw_parts(
            out_tensor_descr.aligned_ptr() as *const u64,
            out_tensor_descr.num_elements(),
        )
    };

    let mut expected = dst_data.clone();
    let mut inserted_positions = vec![false; expected.len()];
    for i in 0..5_usize {
        for j in 0..10_usize {
            let src_idx = i * 10 + j;
            let dst_idx = i * 20 + (2 + 2 * j);
            expected[dst_idx] = src_data[src_idx];
            inserted_positions[dst_idx] = true;
        }
    }

    // Validate every destination element: inserted cells must match source,
    // all other cells must retain original destination data.
    for idx in 0..expected.len() {
        if inserted_positions[idx] {
            assert_eq!(
                out_slice[idx], expected[idx],
                "inserted cell mismatch at {idx}"
            );
        } else {
            assert_eq!(
                out_slice[idx], dst_data[idx],
                "untouched cell mismatch at {idx}"
            );
        }
    }
}

/// Test that when the destination operand of `tensor.insert_slice` is still live
/// after the op (i.e. used again afterwards), bufferization allocates a fresh
/// buffer and copies the destination into it before writing, instead of mutating
/// the original destination buffer in place.
///
/// Without this, the (still-live) original `dst` value would incorrectly observe
/// the in-place write performed for `updated`.
#[test]
fn test_insert_slice_dest_live_after_needs_copy() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
        ^entry():
            llvm.func @test_insert_slice_dest_live_after_runtime: llvm.func
                <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
            ^entry(src_p: llvm.ptr(0), dst_p: llvm.ptr(0), out_updated_p: llvm.ptr(0), out_dst_p: llvm.ptr(0)):
                src = llvm.load src_p : tensor.ranked<5x10:builtin.integer i64>;
                dst = llvm.load dst_p : tensor.ranked<10x20:builtin.integer i64>;
                updated = tensor.insert_slice src into dst [0, 2] [5, 10] [1, 2] : tensor.ranked<10x20:builtin.integer i64>;
                llvm.store *out_updated_p <- updated;
                llvm.store *out_dst_p <- dst;
                llvm.return
            }
        }
        "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after_tensor_to_memref = format!("{}", module_op.disp(ctx));

    // `dst` is live after the insert (it's stored to `out_dst_p`), so a new buffer
    // must be allocated and `dst` copied into it before the in-place write.
    assert!(
        after_tensor_to_memref.contains("memref.alloc"),
        "expected a memref.alloc for the copied destination buffer, got:\n{after_tensor_to_memref}"
    );
    assert_eq!(
        after_tensor_to_memref.matches("memref.copy").count(),
        2,
        "expected one memref.copy for the destination buffer and one for the inserted slice, got:\n{after_tensor_to_memref}"
    );

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_insert_slice_dest_live_after_runtime")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let src_data: Vec<u64> = (100..150_u64).collect();
    let dst_data: Vec<u64> = (0..200_u64).collect();

    let src_descr = TensorDesciptor::new(
        [5, 10].to_vec(),
        std::mem::size_of::<u64>(),
        src_data.as_ptr() as *const u8,
    );
    let dst_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        dst_data.as_ptr() as *const u8,
    );
    let out_updated_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );
    let out_dst_descr = TensorDesciptor::new(
        [10, 20].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_updated_ir_descr = out_updated_descr.build_ir_descriptor();
    let mut out_dst_ir_descr = out_dst_descr.build_ir_descriptor();
    f(
        src_descr.build_ir_descriptor().as_ptr(),
        dst_descr.build_ir_descriptor().as_ptr(),
        out_updated_ir_descr.as_mut_ptr(),
        out_dst_ir_descr.as_mut_ptr(),
    );

    let out_updated_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(
            out_updated_ir_descr.as_ptr(),
            2,
            std::mem::size_of::<u64>(),
        )
    };
    let out_dst_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(
            out_dst_ir_descr.as_ptr(),
            2,
            std::mem::size_of::<u64>(),
        )
    };
    let out_updated_slice = unsafe {
        std::slice::from_raw_parts(
            out_updated_tensor_descr.aligned_ptr() as *const u64,
            out_updated_tensor_descr.num_elements(),
        )
    };
    let out_dst_slice = unsafe {
        std::slice::from_raw_parts(
            out_dst_tensor_descr.aligned_ptr() as *const u64,
            out_dst_tensor_descr.num_elements(),
        )
    };

    // `out_dst` must retain the ORIGINAL destination data: the insert must not
    // have mutated the buffer backing the still-live `dst` value.
    assert_eq!(
        out_dst_slice,
        &dst_data[..],
        "dst was mutated in place even though it was still live after the insert"
    );

    let mut expected_updated = dst_data.clone();
    for i in 0..5_usize {
        for j in 0..10_usize {
            let src_idx = i * 10 + j;
            let dst_idx = i * 20 + (2 + 2 * j);
            expected_updated[dst_idx] = src_data[src_idx];
        }
    }
    assert_eq!(
        out_updated_slice,
        &expected_updated[..],
        "updated tensor does not reflect the inserted slice"
    );
}

/// End-to-end test for tensor.reshape lowering:
/// tensor.reshape -> memref.alloc + memref.copy + memref.reshape (TensorToMemref), then
/// memref.reshape -> descriptor construction (MemrefToCF), then LLVM.
#[test]
fn test_tensor_reshape_to_memref_cf_from_rust() {
    init_env_logger_for_tests!();
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

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after_tensor_to_memref = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block1v1() !0:
            llvm.func @test_tensor_reshape_extract: llvm.func <builtin.integer i64(llvm.ptr (0), builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(arg_p_v0: llvm.ptr (0), i_res_v1: builtin.integer i64, j_res_v2: builtin.integer i64) !1:
                arg_v3 = llvm.load arg_p_v0  : memref.ranked <2x3 : builtin.integer i64> !2;
                v8 = memref.reshape arg_v3 : memref.ranked <3x2 : builtin.integer i64> !3;
                i_idx_v5 = index.from_integer i_res_v1 : index.index  !4;
                j_idx_v6 = index.from_integer j_res_v2 : index.index  !5;
                v9 = memref.load v8[i_idx_v5, j_idx_v6] : builtin.integer i64 !6;
                llvm.return v9 !7
            } !8
        }"#]].assert_eq(&after_tensor_to_memref);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_reshape_extract")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let input = TensorDesciptor::new(
        [2, 3].to_vec(),
        std::mem::size_of::<u64>(),
        [1u64, 2, 3, 4, 5, 6].as_ptr() as *const u8,
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, i64, i64) -> i64>(symbol_addr)
    };

    // 2x3 row-major [1,2,3,4,5,6] reshaped to 3x2 is:
    // [[1,2], [3,4], [5,6]]
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 0, 0), 1);
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 1, 0), 3);
    assert_eq!(f(input.build_ir_descriptor().as_ptr(), 2, 1), 6);
}

#[test]
fn test_tracked_tmm_complex_tensor_computation_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
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

    let mut tmm = TrackedTMM::new();
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM {}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    tmm.register_runtime_symbols(&jit)
        .expect("Failed to register runtime symbols for TrackedTMM");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tensor_complex_tracked")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let lhs_data = [1i64, 2, 3, 4, 5, 6, 7, 8, 2, 1, 0, 3, 4, 2, 1, 5];
    let rhs_data = [2i64, 1, 0, 1, 3, 2, 1, 0, 4, 1, 2, 3, 1, 0, 2, 1];

    let t1 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<i64>(),
        lhs_data.as_ptr() as *const u8,
    );
    let t2 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<i64>(),
        rhs_data.as_ptr() as *const u8,
    );
    let res_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<i64>(),
        std::ptr::null::<u8>(),
    );
    let mut accum_data = [0i64; 16];
    let t3 = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<i64>(),
        accum_data.as_mut_ptr() as *const u8,
    );

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();
    assert_eq!(tmm.tracked_allocations().len(), 0);

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        t3.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    assert!(
        tmm.tracked_allocations().len() >= 3,
        "expected tracked allocations for intermediates and final result"
    );

    let res_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(res_ir_descr.as_ptr(), 2, std::mem::size_of::<i64>())
    };
    let res_slice = unsafe {
        std::slice::from_raw_parts(
            res_tensor_descr.aligned_ptr() as *const i64,
            res_tensor_descr.num_elements(),
        )
    };

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

    assert_eq!(res_slice, &expected);

    tmm.free_all();
    assert_eq!(tmm.tracked_allocations().len(), 0);
}

/// tiled matmul in control-flow form
#[test]
fn test_tiled_matmul_cf() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    // 4x4 matrices, 2x2 tiles. An outer loop over row tiles of C and
    // an inner loop over column tiles, with the accumulator threaded
    // through both loops as a loop-carried value.
    let input_ir = r#"
        builtin.module @test_module {
        ^entry():
            llvm.func @test_tiled_matmul: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
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
            }
        }
        "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after = format!("{}", module_op.disp(ctx));

    // The point of the issue: `a` and `b` are only read, so their tiles are plain
    // subviews. Neither is copied, even though both stay live across every iteration.
    assert!(
        after.contains("memref.subview a_v4"),
        "slice of `a` should be a plain subview, got:\n{after}"
    );
    assert!(
        after.contains("memref.subview b_v5"),
        "slice of `b` should be a plain subview, got:\n{after}"
    );
    // No buffer as large as an input matrix is allocated for a read-only tile.
    assert!(
        !after.contains("memref.alloc  : memref.ranked <2x4"),
        "no buffer should be allocated for the read-only `a` tile, got:\n{after}"
    );
    assert!(
        !after.contains("memref.alloc  : memref.ranked <4x2"),
        "no buffer should be allocated for the read-only `b` tile, got:\n{after}"
    );

    // Only matmul's accumulator tile gets a private buffer.
    assert_eq!(
        after.matches("memref.alloc").count(),
        1,
        "only matmul's accumulator tile should be allocated, got:\n{after}"
    );
    assert_eq!(
        after.matches("memref.copy").count(),
        2,
        "expected one copy seeding matmul's private accumulator and one copy writing \
         the result back into the loop-carried accumulator, got:\n{after}"
    );
    assert!(
        !after.contains("memref.alloc  : memref.ranked <4x4"),
        "the loop-carried accumulator should not be copied, got:\n{after}"
    );

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tiled_matmul")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let a_data: Vec<u64> = (1..=16_u64).collect();
    let b_data: Vec<u64> = (17..=32_u64).collect();
    let c_data: Vec<u64> = (0..16_u64).map(|x| x * 100).collect();

    let a_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        a_data.as_ptr() as *const u8,
    );
    let b_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        b_data.as_ptr() as *const u8,
    );
    let c_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        c_data.as_ptr() as *const u8,
    );
    let out_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    // tensor.matmul accumulates, so the tiled nest computes C + A*B. Computed before
    // the call: `c` is the loop-carried accumulator and is now written in place.
    let mut expected = c_data.clone();
    for i in 0..4_usize {
        for j in 0..4_usize {
            for k in 0..4_usize {
                expected[i * 4 + j] += a_data[i * 4 + k] * b_data[k * 4 + j];
            }
        }
    }

    let mut out_ir_descr = out_descr.build_ir_descriptor();
    f(
        a_descr.build_ir_descriptor().as_ptr(),
        b_descr.build_ir_descriptor().as_ptr(),
        c_descr.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let out_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };
    let mut actual: Vec<u64> = Vec::new();
    unsafe { out_tensor_descr.copy_to_vec(&mut actual) };

    assert_eq!(actual, expected, "tiled matmul produced wrong values");
}

/// The same tiled matmul as [test_tiled_matmul_cf], but expressed with `cf.for`
#[test]
fn test_tiled_matmul_scf_for() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
        ^entry():
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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after = format!("{}", module_op.disp(ctx));

    // Same expectations as the CF-form test: `a` and `b` are read-only, so their
    // tiles are plain subviews and no buffer is allocated for them.
    assert!(
        after.contains("memref.subview a_v4"),
        "slice of `a` should be a plain subview, got:\n{after}"
    );
    assert!(
        after.contains("memref.subview b_v5"),
        "slice of `b` should be a plain subview, got:\n{after}"
    );
    assert!(
        !after.contains("memref.alloc  : memref.ranked <2x4"),
        "no buffer should be allocated for the read-only `a` tile, got:\n{after}"
    );
    assert!(
        !after.contains("memref.alloc  : memref.ranked <4x2"),
        "no buffer should be allocated for the read-only `b` tile, got:\n{after}"
    );

    // Only matmul's accumulator tile gets a private buffer.
    assert_eq!(
        after.matches("memref.alloc").count(),
        1,
        "only matmul's accumulator tile should be allocated, got:\n{after}"
    );
    assert_eq!(
        after.matches("memref.copy").count(),
        2,
        "expected one copy seeding matmul's private accumulator and one copy writing \
         the result back into the loop-carried accumulator, got:\n{after}"
    );
    assert!(
        !after.contains("memref.alloc  : memref.ranked <4x4"),
        "the loop-carried accumulator should not be copied, got:\n{after}"
    );

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_tiled_matmul_scf")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *const u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let a_data: Vec<u64> = (1..=16_u64).collect();
    let b_data: Vec<u64> = (17..=32_u64).collect();
    let c_data: Vec<u64> = (0..16_u64).map(|x| x * 100).collect();

    let a_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        a_data.as_ptr() as *const u8,
    );
    let b_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        b_data.as_ptr() as *const u8,
    );
    let c_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        c_data.as_ptr() as *const u8,
    );
    let out_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut expected = c_data.clone();
    for i in 0..4_usize {
        for j in 0..4_usize {
            for k in 0..4_usize {
                expected[i * 4 + j] += a_data[i * 4 + k] * b_data[k * 4 + j];
            }
        }
    }

    let mut out_ir_descr = out_descr.build_ir_descriptor();
    f(
        a_descr.build_ir_descriptor().as_ptr(),
        b_descr.build_ir_descriptor().as_ptr(),
        c_descr.build_ir_descriptor().as_ptr(),
        out_ir_descr.as_mut_ptr(),
    );

    let out_tensor_descr = unsafe {
        TensorDesciptor::from_ir_descriptor(out_ir_descr.as_ptr(), 2, std::mem::size_of::<u64>())
    };
    let mut actual: Vec<u64> = Vec::new();
    unsafe { out_tensor_descr.copy_to_vec(&mut actual) };

    assert_eq!(actual, expected, "tiled matmul produced wrong values");
}

/// A write through a slice of a tensor that is still live must not touch that
/// tensor's buffer.
///
/// `s` is a slice of `t`, and `s` itself is dead right after the insert, so looking at
/// `s` alone would wrongly allow the write to go in place. It's only via `s` and `t`
/// sharing a buffer that `t`'s liveness is seen and the copy inserted.
#[test]
fn test_write_through_slice_of_live_tensor_needs_copy() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
            ^entry():
                llvm.func @test_write_through_slice: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {
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
    verify_op(&module_op, ctx).expect_ok(ctx);

    let mut tmm = MallocFreeTMM;
    bufferize(&mut tmm, parsed_op, ctx).expect_ok(ctx);
    let after = format!("{}", module_op.disp(ctx));
    assert!(
        after.contains("memref.alloc"),
        "writing through a slice of the live `t` must allocate a private buffer, got:\n{after}"
    );

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_write_through_slice")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8, *mut u8) -> ()>(
            symbol_addr,
        )
    };

    let t_data: Vec<u64> = (0..16_u64).collect();
    let small_data: Vec<u64> = vec![900, 901, 902, 903];

    let t_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        t_data.as_ptr() as *const u8,
    );
    let small_descr = TensorDesciptor::new(
        [2, 2].to_vec(),
        std::mem::size_of::<u64>(),
        small_data.as_ptr() as *const u8,
    );
    let out_u_descr = TensorDesciptor::new(
        [2, 2].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );
    let out_t_descr = TensorDesciptor::new(
        [4, 4].to_vec(),
        std::mem::size_of::<u64>(),
        std::ptr::null::<u8>(),
    );

    let mut out_u_ir = out_u_descr.build_ir_descriptor();
    let mut out_t_ir = out_t_descr.build_ir_descriptor();
    f(
        t_descr.build_ir_descriptor().as_ptr(),
        small_descr.build_ir_descriptor().as_ptr(),
        out_u_ir.as_mut_ptr(),
        out_t_ir.as_mut_ptr(),
    );

    let u_descr =
        unsafe { TensorDesciptor::from_ir_descriptor(out_u_ir.as_ptr(), 2, size_of::<u64>()) };
    let t_out_descr =
        unsafe { TensorDesciptor::from_ir_descriptor(out_t_ir.as_ptr(), 2, size_of::<u64>()) };
    let mut actual_u: Vec<u64> = Vec::new();
    let mut actual_t: Vec<u64> = Vec::new();
    unsafe { u_descr.copy_to_vec(&mut actual_u) };
    unsafe { t_out_descr.copy_to_vec(&mut actual_t) };

    assert_eq!(actual_u, small_data, "the inserted slice is wrong");
    assert_eq!(
        actual_t, t_data,
        "`t` was clobbered by a write through its slice"
    );
}
