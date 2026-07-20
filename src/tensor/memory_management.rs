//! Tensor Memory Management

use pliron::{
    attribute::AttrObj,
    builtin::op_interfaces::{AtLeastNOpdsInterface, OneResultInterface, ResultNOfType},
    context::Context,
    derive::{op_interface, op_interface_impl},
    op::Op,
    result::Result,
    r#type::TypedHandle,
    value::Value,
};
use pliron_common_dialects::cf::ToCFDialect;

use crate::memref::{
    ops::{AllocOp, DeallocOp},
    types::RankedMemrefType,
};

/// Utility to create alloc and dealloc ops.
/// For a simple implementation that creates malloc-like allocs and free-like deallocs,
/// use [MallocFreeTMM].
pub trait TensorMemoryManager {
    /// Create a new memref allocation op for the given memref type and dynamic sizes.
    fn create_memref_alloc(
        &mut self,
        ctx: &mut Context,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Box<dyn MemrefAllocOpInterface>>;

    /// Create a new memref deallocation op for the given memref.
    fn create_memref_dealloc(
        &mut self,
        ctx: &mut Context,
        memref: Value,
    ) -> Result<Box<dyn MemrefDeallocOpInterface>>;

    /// Register runtime symbols for LLVM JIT.
    fn register_runtime_symbols(&self, jit: &pliron_llvm::llvm_sys::lljit::LLVMLLJIT)
    -> Result<()>;
}

/// An [Op] to Allocate a buffer (memref) for the given tensor type and dynamic sizes (if any).
/// For a simple malloc-like allocation, use [AllocOp](crate::memref::ops::AllocOp).
#[op_interface]
pub trait MemrefAllocOpInterface:
    OneResultInterface + ResultNOfType<0, RankedMemrefType> + ToCFDialect
{
    /// Create a new [Self] to allocate a buffer for `memref_ty` with given `dynamic_sizes`.
    /// Any IR static information that may be needed can be passed via `static_info`.
    fn try_new(
        ctx: &mut Context,
        static_info: Option<AttrObj>,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Self>
    where
        Self: Sized;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

/// An [Op] to Deallocate a buffer (memref). For a simple free-like deallocation,
/// use [DeallocOp](crate::memref::ops::DeallocOp).
#[op_interface]
pub trait MemrefDeallocOpInterface: AtLeastNOpdsInterface<1> + ToCFDialect {
    /// Create a new [Self] to deallocate the buffer in `memref`.
    /// Any IR static information that may be needed can be passed via `static_info`.
    fn try_new(ctx: &mut Context, static_info: Option<AttrObj>, memref: Value) -> Result<Self>
    where
        Self: Sized;

    fn verify(_op: &dyn Op, _ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        Ok(())
    }
}

#[op_interface_impl]
impl MemrefAllocOpInterface for AllocOp {
    fn try_new(
        ctx: &mut Context,
        _static_info: Option<AttrObj>,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Self> {
        Ok(Self::new(ctx, memref_ty, dynamic_sizes))
    }
}

#[op_interface_impl]
impl MemrefDeallocOpInterface for DeallocOp {
    fn try_new(ctx: &mut Context, _static_info: Option<AttrObj>, memref: Value) -> Result<Self> {
        Ok(Self::new(ctx, memref))
    }
}

/// A simple [TensorMemoryManager] implementation based on [AllocOp] and [DeallocOp].
/// Effectively calls `malloc` and `free` and does no other book-keeping.
pub struct MallocFreeTMM;

impl TensorMemoryManager for MallocFreeTMM {
    fn create_memref_alloc(
        &mut self,
        ctx: &mut Context,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Box<dyn MemrefAllocOpInterface>> {
        let alloc_op = crate::memref::ops::AllocOp::try_new(ctx, None, memref_ty, dynamic_sizes)?;
        Ok(Box::new(alloc_op))
    }

    fn create_memref_dealloc(
        &mut self,
        ctx: &mut Context,
        memref: Value,
    ) -> Result<Box<dyn MemrefDeallocOpInterface>> {
        let dealloc_op = crate::memref::ops::DeallocOp::try_new(ctx, None, memref)?;
        Ok(Box::new(dealloc_op))
    }

    fn register_runtime_symbols(
        &self,
        _jit: &pliron_llvm::llvm_sys::lljit::LLVMLLJIT,
    ) -> Result<()> {
        // No custom runtime symbols to register for malloc/free-based bufferization.
        Ok(())
    }
}
