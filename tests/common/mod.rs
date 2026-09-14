// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Helpers that the tensor and the memref conversion tests share.

use pliron::{
    builtin::{given_names::erase_given_names, ops::ModuleOp},
    combine::Parser,
    context::{Context, Ptr},
    init_env_logger_for_tests, input_error_noloc,
    irbuild::dialect_conversion::apply_dialect_conversion,
    irfmt::parsers::spaced,
    location::{self, erase_locations},
    op::verify_op,
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{
    core::{LLVMContext, LLVMModule},
    lljit::{JitSymbol, SimpleJIT},
};

use pliron_tensor::memref::conversions::MemrefToCF;

/// Parse `input_ir` into a module and verify it.
pub fn parse_module(ctx: &mut Context, input_ir: &str) -> (Ptr<Operation>, ModuleOp) {
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
    log::debug!("pliron module parsed:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);
    (parsed_op, module_op)
}

/// Print the parsed module, parse the text again, and print it a second time.
/// The two texts must be equal.
pub fn assert_module_round_trips(input_ir: &str) {
    let ctx = &mut Context::new();
    let (parsed_op, _) = parse_module(ctx, input_ir);
    let printed = print_without_names_or_locations(ctx, parsed_op);

    let ctx = &mut Context::new();
    let (reparsed_op, _) = parse_module(ctx, &printed);
    assert_eq!(printed, print_without_names_or_locations(ctx, reparsed_op));
}

/// Print `op` with no given name and no location.
fn print_without_names_or_locations(ctx: &Context, op: Ptr<Operation>) -> String {
    erase_given_names(ctx, op);
    erase_locations(ctx, op);
    op.disp(ctx).to_string()
}

/// Lower the module Memref -> CF -> LLVM dialect and emit its LLVM-IR.
pub fn lower_to_llvm_ir(
    ctx: &mut Context,
    parsed_op: Ptr<Operation>,
    module_op: ModuleOp,
    llvm_ctx: &LLVMContext,
) -> LLVMModule {
    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!(
        "pliron module after dialect conversion to LLVM:\n{}",
        module_op.disp(ctx)
    );
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, llvm_ctx, module_op).expect_ok(ctx);
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();
    llvm_ir
}

/// Look up `name` in `jit` and interpret it as a function of type `F`.
///
/// # Safety
/// `F` must be the type of the compiled function.
pub unsafe fn lookup_fn<'jit, F: Copy>(jit: &'jit SimpleJIT, name: &str) -> JitSymbol<'jit, F> {
    unsafe { jit.lookup_symbol::<F>(name) }.expect("Failed to lookup symbol")
}
