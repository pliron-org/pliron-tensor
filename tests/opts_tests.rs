// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test pliron's optimizations (canonicalization, SCCP and DCE) on tensor and memref IR.
//! The optimized IR is then lowered and executed in a JIT.

use expect_test::{expect, expect_file};
use pliron::{
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    location::erase_locations,
    op::verify_op,
    operation::Operation,
    opts::{canonicalize::canonicalize, constants::sccp::sccp, dce::dce},
    printable::Printable,
    result::ExpectOk,
};

use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::SimpleJIT};

use pliron_tensor::tensor::{interface_impls::MAX_FOLD_ELEMENTS, memory_management::MallocFreeTMM};

mod common;
use common::{
    assert_module_round_trips, lookup_fn, lower_to_llvm_ir, parse_module,
    tensor::{
        Operands, TestElement, arith_f32, arith_i32, arith_ui32, bufferize_module,
        check_binary_case, input_tensor, max_min_f32, max_min_i32, max_min_ui32, output_data,
        output_tensor, reduce_reference,
    },
};

/// Run canonicalization, SCCP and then DCE on the module, and return the optimized IR as text.
fn optimize_module(ctx: &mut Context, parsed_op: Ptr<Operation>) -> String {
    canonicalize(parsed_op, ctx).expect_ok(ctx);
    sccp(parsed_op, ctx).expect_ok(ctx);
    dce(parsed_op, ctx).expect_ok(ctx);
    verify_op(Operation::get_op_dyn(parsed_op, ctx).as_ref(), ctx).expect_ok(ctx);
    erase_locations(ctx, parsed_op);
    parsed_op.disp(ctx).to_string()
}

/// Get the names of all operations in `op`, in pre-order.
fn op_names(ctx: &Context, op: Ptr<Operation>) -> Vec<String> {
    let mut names = Vec::new();
    walk_op(
        ctx,
        &mut names,
        &WALKCONFIG_PREORDER_FORWARD,
        op,
        |ctx, names, node| {
            if let IRNode::Operation(op) = node {
                names.push(Operation::get_opid(op, ctx).to_string());
            }
        },
    );
    names
}

/// SCCP must fold tensor ops with constant operands, and DCE must then
/// remove the tensor ops whose results are not used.
#[test]
fn test_fold_and_dce() {
    let ctx = &mut Context::new();
    let input_ir = include_str!("resources/test_fold_and_dce.input.plir");
    let (parsed_op, module_op) = parse_module(ctx, input_ir);

    // - @test_fold_extract, @test_fold_int_arith, @test_fold_splats and
    //   @test_fold_int_casts must each return or store one constant.
    // - @test_no_fold_dynamic: the splat with a dynamic result shape must stay.
    // - @test_no_fold_undefined: the division by zero, the signed division overflow
    //   and the extract out of bounds must stay.
    // - @test_dce: the unused tensor ops must be removed.
    // - @test_fold_transpose must store only constants, for a dense and a splat source.
    // - @test_fold_max_min_*, @test_fold_arith_*, @test_fold_exp and @test_fold_reduce
    //   must store only constants, equal to the values that the lowering computes.
    let after_opts = optimize_module(ctx, parsed_op);
    expect_file!["resources/test_fold_and_dce.expect.plir"].assert_eq(&after_opts);
    assert_module_round_trips(&after_opts);

    bufferize_module(ctx, &mut MallocFreeTMM, parsed_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let extract = unsafe { lookup_fn::<extern "C" fn() -> f64>(&jit, "test_fold_extract") };
    assert_eq!(extract(), 6.0);

    let int_arith =
        unsafe { lookup_fn::<extern "C" fn(*mut u8) -> ()>(&jit, "test_fold_int_arith") };
    let mut int_arith_res = output_tensor::<i64>(&[2, 2]).build_ir_descriptor();
    int_arith(int_arith_res.as_mut_ptr());
    // ((c + 10) * 2 - c) / 3, for c = [1, 2, 3, 4].
    assert_eq!(
        unsafe { output_data::<i64>(&int_arith_res, 2) },
        [7, 7, 7, 8]
    );

    let splats = unsafe { lookup_fn::<extern "C" fn(*mut u8) -> ()>(&jit, "test_fold_splats") };
    let mut splats_res = output_tensor::<f64>(&[8]).build_ir_descriptor();
    splats(splats_res.as_mut_ptr());
    assert_eq!(unsafe { output_data::<f64>(&splats_res, 1) }, [5.0; 8]);

    let int_casts =
        unsafe { lookup_fn::<extern "C" fn(*mut u8) -> ()>(&jit, "test_fold_int_casts") };
    let mut int_casts_res = output_tensor::<f64>(&[4]).build_ir_descriptor();
    int_casts(int_casts_res.as_mut_ptr());
    // 300 truncated to 8 bits is 44.
    assert_eq!(
        unsafe { output_data::<f64>(&int_casts_res, 1) },
        [-1.0, 2.0, 44.0, -4.0]
    );

    let dynamic =
        unsafe { lookup_fn::<extern "C" fn(*mut u8) -> ()>(&jit, "test_no_fold_dynamic") };
    let mut dynamic_res = output_tensor::<f32>(&[3]).build_ir_descriptor();
    dynamic(dynamic_res.as_mut_ptr());
    assert_eq!(unsafe { output_data::<f32>(&dynamic_res, 1) }, [1.5; 3]);

    let arg_data = [1i64, 2, 3, 4];
    let arg = input_tensor(&[2, 2], &arg_data);
    let mut dce_res = output_tensor::<i64>(&[2, 2]).build_ir_descriptor();
    let dce_fn = unsafe { lookup_fn::<extern "C" fn(*const u8, *mut u8) -> ()>(&jit, "test_dce") };
    dce_fn(arg.build_ir_descriptor().as_ptr(), dce_res.as_mut_ptr());
    assert_eq!(unsafe { output_data::<i64>(&dce_res, 2) }, [1, 4, 9, 16]);

    let transpose =
        unsafe { lookup_fn::<extern "C" fn(*mut u8, *mut u8) -> ()>(&jit, "test_fold_transpose") };
    let mut transpose_res = output_tensor::<i64>(&[2, 2, 3]).build_ir_descriptor();
    let mut transpose_splat_res = output_tensor::<i64>(&[3, 2]).build_ir_descriptor();
    transpose(transpose_res.as_mut_ptr(), transpose_splat_res.as_mut_ptr());
    // Element [i, j, k] of the result must be element [j, k, i] of c = [0, 1, ..., 11].
    assert_eq!(
        unsafe { output_data::<i64>(&transpose_res, 3) },
        [0, 2, 4, 6, 8, 10, 1, 3, 5, 7, 9, 11]
    );
    assert_eq!(
        unsafe { output_data::<i64>(&transpose_splat_res, 2) },
        [1; 6]
    );

    let consts = Operands::Constants;
    check_binary_case(&jit, "test_fold_max_min_f32", consts, &max_min_f32());
    check_binary_case(&jit, "test_fold_max_min_i32", consts, &max_min_i32());
    check_binary_case(&jit, "test_fold_max_min_ui32", consts, &max_min_ui32());
    check_binary_case(&jit, "test_fold_arith_f32", consts, &arith_f32());
    check_binary_case(&jit, "test_fold_arith_i32", consts, &arith_i32());
    check_binary_case(&jit, "test_fold_arith_ui32", consts, &arith_ui32());

    // The folded exponentials must be the values that the lowering computes,
    // also for infinities, NaN and -0.0.
    let exp = unsafe { lookup_fn::<extern "C" fn(*mut u8, *mut u8) -> ()>(&jit, "test_fold_exp") };
    let mut exp_res = output_tensor::<f32>(&[8]).build_ir_descriptor();
    let mut splat_exp_res = output_tensor::<f64>(&[3]).build_ir_descriptor();
    exp(exp_res.as_mut_ptr(), splat_exp_res.as_mut_ptr());
    let (nan, inf) = (f32::NAN, f32::INFINITY);
    let exp_input = [0.0f32, 1.0, -1.0, 10.0, -inf, inf, nan, -0.0];
    for (x, actual) in exp_input
        .iter()
        .zip(unsafe { output_data::<f32>(&exp_res, 1) })
    {
        assert!(
            actual.same(x.exp()),
            "exp({x}): got {actual}, expected {}",
            x.exp()
        );
    }
    assert_eq!(
        unsafe { output_data::<f64>(&splat_exp_res, 1) },
        [0.5f64.exp(); 3]
    );

    // The folded reductions must be the values that the lowering computes:
    // - the float elements must be combined in the order of the lowering
    // - integer sums must wrap
    // - unsigned integers must use an unsigned comparison.
    type M = *mut u8;
    let reduce =
        unsafe { lookup_fn::<extern "C" fn(M, M, M, M, M) -> ()>(&jit, "test_fold_reduce") };
    let mut sum_res = output_tensor::<f32>(&[2, 1, 4]).build_ir_descriptor();
    let mut max_res = output_tensor::<f32>(&[1, 3, 1]).build_ir_descriptor();
    let mut splat_sum_res = output_tensor::<f32>(&[2, 1, 4]).build_ir_descriptor();
    let mut int_sum_res = output_tensor::<i32>(&[1, 2]).build_ir_descriptor();
    let mut umax_res = output_tensor::<u32>(&[2, 1]).build_ir_descriptor();
    reduce(
        sum_res.as_mut_ptr(),
        max_res.as_mut_ptr(),
        splat_sum_res.as_mut_ptr(),
        int_sum_res.as_mut_ptr(),
        umax_res.as_mut_ptr(),
    );
    let shape = [2, 3, 4];
    let input: Vec<f32> = (0..24).map(|i| i as f32 * 0.5 - 5.25).collect();
    let sum_init = [10.0f32, 20.0, 30.0, 40.0, -1.0, -2.0, -3.0, -4.0];
    assert_eq!(
        unsafe { output_data::<f32>(&sum_res, 3) },
        reduce_reference(&shape, &input, &[1], &sum_init, |a, b| a + b)
    );
    let max_init = [-inf, -inf, 100.0];
    assert_eq!(
        unsafe { output_data::<f32>(&max_res, 3) },
        reduce_reference(&shape, &input, &[0, 2], &max_init, f32::max)
    );
    let tenths = [0.1f32; 24];
    assert_eq!(
        unsafe { output_data::<f32>(&splat_sum_res, 3) },
        reduce_reference(&shape, &tenths, &[1], &[0.0; 8], |a, b| a + b)
    );
    assert_eq!(
        unsafe { output_data::<i32>(&int_sum_res, 2) },
        [i32::MIN, 4]
    );
    assert_eq!(
        unsafe { output_data::<u32>(&umax_res, 2) },
        [0x8000_0000, u32::MAX]
    );
}

/// SCCP must not fold a non-splat result of more than [MAX_FOLD_ELEMENTS] elements,
/// or a reduction of an input of more than [MAX_FOLD_ELEMENTS] elements.
#[test]
fn test_fold_size_limit() {
    let num_elements = MAX_FOLD_ELEMENTS + 1;
    let elements = (0..num_elements)
        .map(|element| element.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let ty = format!("tensor.ranked<{num_elements}:builtin.integer i64>");
    let matrix_ty = format!("tensor.ranked<1x{num_elements}:builtin.integer i64>");
    let total_ty = "tensor.ranked<1:builtin.integer i64>";
    let input_ir = format!(
        r#"builtin.module @test_module {{
        ^entry():
            llvm.func @test_fold_size_limit: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {{
            ^entry(sequence_sum_p: llvm.ptr(0), splat_sum_p: llvm.ptr(0), reshaped_p: llvm.ptr(0), total_p: llvm.ptr(0)):
                sequence = tensor.constant !outlined : {ty} !100;
                ones = tensor.constant !outlined : {ty} !101;
                sequence_sum = tensor.add sequence, ones : {ty};
                splat_sum = tensor.add ones, ones : {ty};
                reshaped = tensor.reshape sequence : {matrix_ty};
                zero = tensor.constant !outlined : {total_ty} !102;
                total = tensor.reduce <Sum> [0] sequence, zero : {total_ty};
                llvm.store *sequence_sum_p <- sequence_sum;
                llvm.store *splat_sum_p <- splat_sum;
                llvm.store *reshaped_p <- reshaped;
                llvm.store *total_p <- total;
                llvm.return
            }}
        }}

        outlined_attributes:
        !100 = [tensor_constant_value = memref.dense_elements <{ty} = [{elements}]>]
        !101 = [tensor_constant_value = memref.dense_elements <{ty} = splat 1>]
        !102 = [tensor_constant_value = memref.dense_elements <{total_ty} = splat 0>]
        "#
    );

    let ctx = &mut Context::new();
    let (parsed_op, module_op) = parse_module(ctx, &input_ir);
    optimize_module(ctx, parsed_op);

    // - `sequence_sum` must stay a `tensor.add`, because its result is not a splat
    //   and has too many elements.
    // - `reshaped` must stay a `tensor.reshape`, for the same reason.
    // - `splat_sum` must fold to a new `tensor.constant`, because its result is a splat.
    // - `total` must stay a `tensor.reduce`, because its input has too many elements,
    //   although its result has only one element.
    expect![[r#"
        [
            "builtin.module",
            "llvm.func",
            "tensor.constant",
            "tensor.constant",
            "tensor.add",
            "tensor.constant",
            "tensor.reshape",
            "tensor.constant",
            "tensor.reduce",
            "llvm.store",
            "llvm.store",
            "llvm.store",
            "llvm.store",
            "llvm.return",
        ]
    "#]]
    .assert_debug_eq(&op_names(ctx, parsed_op));

    bufferize_module(ctx, &mut MallocFreeTMM, parsed_op);
    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let f = unsafe {
        lookup_fn::<extern "C" fn(*mut u8, *mut u8, *mut u8, *mut u8) -> ()>(
            &jit,
            "test_fold_size_limit",
        )
    };
    let mut sequence_sum = output_tensor::<i64>(&[num_elements]).build_ir_descriptor();
    let mut splat_sum = output_tensor::<i64>(&[num_elements]).build_ir_descriptor();
    let mut reshaped = output_tensor::<i64>(&[1, num_elements]).build_ir_descriptor();
    let mut total = output_tensor::<i64>(&[1]).build_ir_descriptor();
    f(
        sequence_sum.as_mut_ptr(),
        splat_sum.as_mut_ptr(),
        reshaped.as_mut_ptr(),
        total.as_mut_ptr(),
    );
    let expected: Vec<i64> = (1..=num_elements as i64).collect();
    assert_eq!(unsafe { output_data::<i64>(&sequence_sum, 1) }, expected);
    assert_eq!(
        unsafe { output_data::<i64>(&splat_sum, 1) },
        vec![2; num_elements]
    );
    let sequence: Vec<i64> = (0..num_elements as i64).collect();
    assert_eq!(unsafe { output_data::<i64>(&reshaped, 2) }, sequence);
    assert_eq!(
        unsafe { output_data::<i64>(&total, 1) },
        [sequence.iter().sum::<i64>()]
    );
}

/// SCCP and canonicalization of `tensor.dim`.
#[test]
fn test_tensor_dim_opts() {
    let ctx = &mut Context::new();
    let input_ir = include_str!("resources/test_tensor_dim.input.plir");
    let (parsed_op, _) = parse_module(ctx, input_ir);

    // - `tensor.dim %input, %dim_idx` (dynamic index) and `tensor.dim %input, 0`
    //   (dynamic dimension) must not be transformed.
    // - `tensor.dim %input, 1` must fold to `4`.
    // - `tensor.dim %init, 0`, where `%init = tensor.splat %lowest, %rows`, must become `%rows`.
    // - In @test_tensor_dim_through_ops, `tensor.dim %generated, 0`, where
    //   `%generated = tensor.generate %rows` and
    //   `%rows = tensor.dim (tensor.add (tensor.elementwise_cast (tensor.exp %input))), 0`,
    //   must become `tensor.dim %input, 0`.
    let after_opts = optimize_module(ctx, parsed_op);
    expect_file!["resources/test_tensor_dim_opts.expect.plir"].assert_eq(&after_opts);
}

/// SCCP and canonicalization of `memref.dim`.
#[test]
fn test_memref_dim_opts() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_memref_dim.input.plir");

    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let after_opts = optimize_module(ctx, parsed_op);

    // - `memref.dim %memref, %dim_idx` (dynamic index) must not be transformed.
    // - With constant indices, @test_memref_dim_const_index must return the constant 16032.
    // - `memref.dim %memref, 0`, where `%memref = memref.alloc %rows`, must become `%rows`.
    // - DCE must then remove the ops whose results are not used.
    expect_file!["resources/test_memref_dim_opts.expect.plir"].assert_eq(&after_opts);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let dynamic_index =
        unsafe { lookup_fn::<fn(i64) -> i64>(&jit, "test_memref_dim_dynamic_index") };
    assert_eq!(dynamic_index(0), 16);
    assert_eq!(dynamic_index(1), 32);

    let const_index = unsafe { lookup_fn::<fn() -> i64>(&jit, "test_memref_dim_const_index") };
    assert_eq!(const_index(), 16032);

    let dynamic_alloc =
        unsafe { lookup_fn::<fn(i64) -> i64>(&jit, "test_memref_dim_dynamic_alloc") };
    assert_eq!(dynamic_alloc(7), 7);
}
