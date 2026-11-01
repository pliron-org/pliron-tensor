// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Helpers that the tests of tensor IR share.

use pliron::{
    context::{Context, Ptr},
    location::erase_locations,
    op::verify_op,
    operation::Operation,
    printable::Printable,
    result::ExpectOk,
};

use pliron_tensor::tensor::{
    bufferize::bufferize, memory_management::TensorMemoryManager, runtime_utils::TensorDesciptor,
};

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
