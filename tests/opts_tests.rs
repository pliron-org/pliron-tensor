// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Test pliron's optimizations (SCCP and DCE) on tensor and memref IR.
//! The optimized IR is then lowered and executed in a JIT.

use expect_test::{expect, expect_file};
use pliron::{
    context::{Context, Ptr},
    graph::walkers::{IRNode, WALKCONFIG_PREORDER_FORWARD, uninterruptible::immutable::walk_op},
    location::erase_locations,
    op::verify_op,
    operation::Operation,
    opts::{constants::sccp::sccp, dce::dce},
    printable::Printable,
    result::ExpectOk,
};

use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::SimpleJIT};

use pliron_tensor::tensor::{interface_impls::MAX_FOLD_ELEMENTS, memory_management::MallocFreeTMM};

mod common;
use common::{
    assert_module_round_trips, lookup_fn, lower_to_llvm_ir, parse_module,
    tensor::{bufferize_module, input_tensor, output_data, output_tensor},
};

/// Run SCCP and then DCE on the module, and return the optimized IR as text.
fn optimize_module(ctx: &mut Context, parsed_op: Ptr<Operation>) -> String {
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
    // - @test_dce: the unused `tensor.add` and `tensor.generate` must be removed.
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
}

/// SCCP must not fold a non-splat result of more than [MAX_FOLD_ELEMENTS] elements.
#[test]
fn test_fold_size_limit() {
    let num_elements = MAX_FOLD_ELEMENTS + 1;
    let elements = (0..num_elements)
        .map(|element| element.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let ty = format!("tensor.ranked<{num_elements}:builtin.integer i64>");
    let matrix_ty = format!("tensor.ranked<1x{num_elements}:builtin.integer i64>");
    let input_ir = format!(
        r#"builtin.module @test_module {{
        ^entry():
            llvm.func @test_fold_size_limit: llvm.func <llvm.void (llvm.ptr(0), llvm.ptr(0), llvm.ptr(0)) variadic = false> [] {{
            ^entry(sequence_sum_p: llvm.ptr(0), splat_sum_p: llvm.ptr(0), reshaped_p: llvm.ptr(0)):
                sequence = tensor.constant !outlined : {ty} !100;
                ones = tensor.constant !outlined : {ty} !101;
                sequence_sum = tensor.add sequence, ones : {ty};
                splat_sum = tensor.add ones, ones : {ty};
                reshaped = tensor.reshape sequence : {matrix_ty};
                llvm.store *sequence_sum_p <- sequence_sum;
                llvm.store *splat_sum_p <- splat_sum;
                llvm.store *reshaped_p <- reshaped;
                llvm.return
            }}
        }}

        outlined_attributes:
        !100 = [tensor_constant_value = memref.dense_elements <{ty} = [{elements}]>]
        !101 = [tensor_constant_value = memref.dense_elements <{ty} = splat 1>]
        "#
    );

    let ctx = &mut Context::new();
    let (parsed_op, module_op) = parse_module(ctx, &input_ir);
    optimize_module(ctx, parsed_op);

    // - `sequence_sum` must stay a `tensor.add`, because its result is not a splat
    //   and has too many elements.
    // - `reshaped` must stay a `tensor.reshape`, for the same reason.
    // - `splat_sum` must fold to a new `tensor.constant`, because its result is a splat.
    expect![[r#"
        [
            "builtin.module",
            "llvm.func",
            "tensor.constant",
            "tensor.constant",
            "tensor.add",
            "tensor.constant",
            "tensor.reshape",
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
        lookup_fn::<extern "C" fn(*mut u8, *mut u8, *mut u8) -> ()>(&jit, "test_fold_size_limit")
    };
    let mut sequence_sum = output_tensor::<i64>(&[num_elements]).build_ir_descriptor();
    let mut splat_sum = output_tensor::<i64>(&[num_elements]).build_ir_descriptor();
    let mut reshaped = output_tensor::<i64>(&[1, num_elements]).build_ir_descriptor();
    f(
        sequence_sum.as_mut_ptr(),
        splat_sum.as_mut_ptr(),
        reshaped.as_mut_ptr(),
    );
    let expected: Vec<i64> = (1..=num_elements as i64).collect();
    assert_eq!(unsafe { output_data::<i64>(&sequence_sum, 1) }, expected);
    assert_eq!(
        unsafe { output_data::<i64>(&splat_sum, 1) },
        vec![2; num_elements]
    );
    let sequence: Vec<i64> = (0..num_elements as i64).collect();
    assert_eq!(unsafe { output_data::<i64>(&reshaped, 2) }, sequence);
}

/// SCCP must fold `memref.dim` of a static dimension with a constant index,
/// and DCE must then remove the ops whose results are not used.
#[test]
fn test_memref_dim_folding() {
    let ctx = &mut Context::new();

    let input_ir = include_str!("resources/test_memref_dim.input.plir");

    let (parsed_op, module_op) = parse_module(ctx, input_ir);
    let after_opts = optimize_module(ctx, parsed_op);

    // A dynamic index must stay a `memref.dim`. With constant indices, the
    // function must return the constant 16032.
    expect_file!["resources/test_memref_dim_folding.expect.plir"].assert_eq(&after_opts);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = lower_to_llvm_ir(ctx, parsed_op, module_op, &llvm_ctx);
    let jit = SimpleJIT::new(llvm_ctx, llvm_ir).expect("Failed to create JIT");

    let dynamic_index =
        unsafe { lookup_fn::<fn(i64) -> i64>(&jit, "test_memref_dim_dynamic_index") };
    assert_eq!(dynamic_index(0), 16);
    assert_eq!(dynamic_index(1), 32);

    let const_index = unsafe { lookup_fn::<fn() -> i64>(&jit, "test_memref_dim_const_index") };
    assert_eq!(const_index(), 16032);
}
