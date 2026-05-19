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
        bufferize::{MallocFreeTMM, TensorMemoryManager, bufferize},
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
                                one = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                                x_elem = llvm.add i_int, one <{nsw = false, nuw = false}> : builtin.integer i64;
                                memref.yield x_elem
                        };
                        llvm.br ^block_b(x)

                    ^block_b(z: tensor.ranked<4:builtin.integer i64>):
                        src = tensor.generate : tensor.ranked<1:builtin.integer i64> {
                            ^entry(i_2 : index.index):
                                ten = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64;
                                memref.yield ten
                        };
                        y = tensor.insert_slice src into z [0] [1] [1] : tensor.ranked<4:builtin.integer i64>;
                        sum = tensor.add x, y : tensor.ranked<4:builtin.integer i64>;
                        zero_idx_i64 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
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
                                one = llvm.constant <builtin.integer <1: i64>> : builtin.integer i64;
                                x_elem = llvm.add i_int, one <{nsw = false, nuw = false}> : builtin.integer i64;
                                memref.yield x_elem
                        };
                        llvm.cond_br if flag ^block_b(x, x) else ^block_c(x, x)

                    ^block_c(z_c: tensor.ranked<4:builtin.integer i64>, x_c: tensor.ranked<4:builtin.integer i64>):
                        src = tensor.generate : tensor.ranked<1:builtin.integer i64> {
                            ^entry(i_2 : index.index):
                                ten = llvm.constant <builtin.integer <10: i64>> : builtin.integer i64;
                                memref.yield ten
                        };
                        y = tensor.insert_slice src into x_c [0] [1] [1] : tensor.ranked<4:builtin.integer i64>;
                        llvm.br ^block_b(z_c, y)

                    ^block_b(z: tensor.ranked<4:builtin.integer i64>, y_b: tensor.ranked<4:builtin.integer i64>):
                        sum = tensor.add z, y_b : tensor.ranked<4:builtin.integer i64>;
                        zero_idx_i64 = llvm.constant <builtin.integer <0: i64>> : builtin.integer i64;
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
                llvm.func @test_tensor_add: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
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
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
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
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x4:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
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
                llvm.func @test_tensor_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<?x?:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<?x?:builtin.integer i64>;
                    res = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
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
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
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
            28u64, 32, 36, 40, 28, 32, 36, 40, 28, 32, 36, 40, 28, 32, 36, 40
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
                llvm.func @test_tensor_batch_matmul: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<2x2x3:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<2x3x2:builtin.integer i64>;
                    res = tensor.batch_matmul arg1, arg2 : tensor.ranked<2x2x2:builtin.integer i64>;
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
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();
    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
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

    assert_eq!(res_slice, &[22u64, 28, 49, 64, 220, 244, 301, 334]);
}

#[test]
fn test_float_tensor_from_rust() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::default();

    let input_ir = r#"
      builtin.module @test_module {
        ^entry():
        llvm.func @test_tensor_add_float: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
          ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
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
        llvm.func @test_tensor_all_binops_float: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
          ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
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
                        llvm.func @test_extract_slice_runtime: llvm.func <llvm.void (llvm.ptr, llvm.ptr) variadic = false> [] {
                            ^entry(src_p: llvm.ptr, out_p: llvm.ptr):
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
          ^entry_block2v1() !0:
            llvm.func @test_extract_slice_runtime: llvm.func <llvm.void (llvm.ptr , llvm.ptr ) variadic = false>
              [] 
            {
              ^entry_block1v1(src_p_v4: llvm.ptr , out_p_v5: llvm.ptr ) !1:
                src_v1 = llvm.load src_p_v4  : memref.ranked <10x20 : builtin.integer i64> !2;
                $v6 = memref.subview src_v1 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64> !3;
                llvm.store *out_p_v5 <- v6  !4;
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
                        llvm.func @test_extract_slice_runtime_sequential: llvm.func <llvm.void (llvm.ptr, llvm.ptr) variadic = false> [] {
                            ^entry(src_p: llvm.ptr, out_p: llvm.ptr):
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

/// Test that `tensor.insert_slice` is lowered and executed correctly end-to-end:
/// TensorToMemref -> MemrefToCF -> CFToLLVM -> JIT.
#[test]
fn test_insert_slice_tensor_to_memref() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
                builtin.module @test_module {
                    ^entry():
                        llvm.func @test_insert_slice_runtime: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                            ^entry(src_p: llvm.ptr, dst_p: llvm.ptr, out_p: llvm.ptr):
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
          ^entry_block2v1() !0:
            llvm.func @test_insert_slice_runtime: llvm.func <llvm.void (llvm.ptr , llvm.ptr , llvm.ptr ) variadic = false>
              [] 
            {
              ^entry_block1v1(src_p_v6: llvm.ptr , dst_p_v7: llvm.ptr , out_p_v8: llvm.ptr ) !1:
                src_v1 = llvm.load src_p_v6  : memref.ranked <5x10 : builtin.integer i64> !2;
                dst_v3 = llvm.load dst_p_v7  : memref.ranked <10x20 : builtin.integer i64> !3;
                $v9 = memref.subview dst_v3 [0, 2] [5, 10] [1, 2] : memref.ranked <5x10 : builtin.integer i64>;
                memref.copy v9 <- src_v1;
                llvm.store *out_p_v8 <- dst_v3  !4;
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
                llvm.func @test_tensor_reshape_extract: llvm.func <builtin.integer i64 (llvm.ptr, builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(arg_p: llvm.ptr, i_res: builtin.integer i64, j_res: builtin.integer i64):
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
          ^entry_block2v1() !0:
            llvm.func @test_tensor_reshape_extract: llvm.func <builtin.integer i64(llvm.ptr , builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block1v1(arg_p_v8: llvm.ptr , i_res_v9: builtin.integer i64, j_res_v10: builtin.integer i64) !1:
                arg_v1 = llvm.load arg_p_v8  : memref.ranked <2x3 : builtin.integer i64> !2;
                v11 = memref.reshape arg_v1 : memref.ranked <3x2 : builtin.integer i64> !3;
                i_idx_v4 = index.from_integer i_res_v9 : index.index  !4;
                j_idx_v6 = index.from_integer j_res_v10 : index.index  !5;
                v12 = memref.load v11[i_idx_v4, j_idx_v6] : builtin.integer i64 !6;
                llvm.return v12 !7
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
                llvm.func @test_tensor_complex_tracked: llvm.func <llvm.void (llvm.ptr, llvm.ptr, llvm.ptr) variadic = false> [] {
                  ^entry(arg1_p: llvm.ptr, arg2_p: llvm.ptr, res_p: llvm.ptr):
                    arg1 = llvm.load arg1_p : tensor.ranked<4x4:builtin.integer i64>;
                    arg2 = llvm.load arg2_p : tensor.ranked<4x4:builtin.integer i64>;
                    mat = tensor.matmul arg1, arg2 : tensor.ranked<4x4:builtin.integer i64>;
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

    let f = unsafe {
        std::mem::transmute::<u64, extern "C" fn(*const u8, *const u8, *mut u8) -> ()>(symbol_addr)
    };

    let mut res_ir_descr = res_descr.build_ir_descriptor();
    assert_eq!(tmm.tracked_allocations().len(), 0);

    f(
        t1.build_ir_descriptor().as_ptr(),
        t2.build_ir_descriptor().as_ptr(),
        res_ir_descr.as_mut_ptr(),
    );

    assert!(
        tmm.tracked_allocations().len() >= 4,
        "expected tracked allocations for matmul result, intermediates, and final result"
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
