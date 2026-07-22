// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Track tensor allocations and deallocations in the IR.
//! Provides tracked AllocOp and DeallocOp operations that expand, during lowering,
//! to allocations / deallocation and tracking operations.

use std::collections::HashSet;
use std::num::NonZero;

use pliron::{
    arg_error_noloc,
    attribute::AttrObj,
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::{
            AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface, CallOpCallable,
            NOpdsInterface, NResultsInterface, OneOpdInterface, OneResultInterface, OperandNOfType,
            ResultNOfType, SameResultsType, SymbolOpInterface, SymbolTableInterface,
        },
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    common_traits::Verify,
    context::Context,
    derive::{op_interface_impl, pliron_op},
    input_error,
    irbuild::{
        dialect_conversion::{DialectConversionRewriter, OperandsInfo},
        inserter::Inserter,
        rewriter::Rewriter,
    },
    op::Op,
    operation::Operation,
    result::Result,
    symbol_table::{SymbolTableCollection, nearest_symbol_table},
    r#type::{Typed, TypedHandle},
    utils::apint::APInt,
    value::Value,
    verify_err,
};
use pliron_common_dialects::{cf::ToCFDialect, index::ops::IndexConstantOp};
use pliron_llvm::{
    attributes::IntegerOverflowFlagsAttr,
    function_call_utils::{compute_type_size_in_bytes, get_size_type, lookup_or_insert_function},
    llvm_sys::lljit::JITSymbolGenericFlags,
    op_interfaces::{CastOpInterface, IntBinArithOpWithOverflowFlag},
    ops::{CallOp, FuncOp, IntToPtrOp, MulOp},
    types::{PointerType, VoidType},
};

use crate::{
    memref::{
        attributes::ConstPointerAttr,
        descriptor,
        type_interfaces::{MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::memory_management::{
        MemrefAllocOpInterface, MemrefDeallocOpInterface, TensorMemoryManager,
    },
};

unsafe extern "C" {
    fn malloc(size: usize) -> *mut ();
    fn free(ptr: *mut ());
}

#[derive(Default)]
/// A [TensorMemoryManager] that tracks all live allocations.
///
/// Pass a mutable reference to this struct to [crate::tensor::bufferize::bufferize].
/// After JIT execution, [TrackedTMM::tracked_allocations] returns any pointers
/// that were allocated but not yet freed. They can all be freed at once via [TrackedTMM::free_all].
/// Make sure to keep this struct alive for the entire duration of the JIT execution,
/// the [Drop] impl will free any still-tracked allocations when the struct goes out of scope.
///
/// The [tracked_malloc] and [tracked_dealloc] functions implement the runtime
/// side; they are called from JIT-compiled code via [TrackedAllocOp] and [TrackedDeallocOp].
pub struct TrackedTMM {
    tracked: HashSet<*const ()>,
}

unsafe impl Send for TrackedTMM {}
unsafe impl Sync for TrackedTMM {}

impl TrackedTMM {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the set of currently live (not yet freed) allocation pointers.
    pub fn tracked_allocations(&self) -> &HashSet<*const ()> {
        &self.tracked
    }

    /// Frees all currently tracked allocations and clears the tracked set.
    pub fn free_all(&mut self) {
        for &ptr in &self.tracked {
            unsafe { free(ptr as *mut ()) };
        }
        self.tracked.clear();
    }
}

impl Drop for TrackedTMM {
    fn drop(&mut self) {
        self.free_all();
    }
}

/// Allocates `size` bytes, records the pointer in `state`, and returns it.
/// Called from JIT-compiled code emitted by [TrackedAllocOp].
///
/// # Safety
/// `state` must be a valid pointer to a [TrackedTMM] for the lifetime of the call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tracked_malloc(state: *mut TrackedTMM, size: u64) -> *mut () {
    let ptr = unsafe { malloc(size as usize) };
    if let Some(state) = unsafe { state.as_mut() } {
        state.tracked.insert(ptr as *const ());
    }
    ptr
}

/// Removes `ptr` from `state`'s tracked set and frees it.
/// Called from JIT-compiled code emitted by [TrackedDeallocOp].
///
/// # Safety
/// `state` must be a valid pointer to a [TrackedTMM] for the lifetime of the call.
/// `ptr` must have been returned by [tracked_malloc] with the same `state`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tracked_dealloc(state: *mut TrackedTMM, ptr: *mut ()) {
    if let Some(state) = unsafe { state.as_mut() } {
        state.tracked.remove(&(ptr as *const ()));
    }
    unsafe { free(ptr) };
}

/// Get or create a declaration for the [tracked_malloc] function in the nearest symbol table.
fn lookup_or_create_tracked_malloc_fn(
    ctx: &mut Context,
    symbol_table_collection: &mut SymbolTableCollection,
    symbol_table_op: Box<dyn SymbolTableInterface>,
) -> Result<FuncOp> {
    let ptr_ty = PointerType::get(ctx, 0).into();
    let size_ty = get_size_type(ctx);
    lookup_or_insert_function(
        ctx,
        symbol_table_collection,
        symbol_table_op,
        "tracked_malloc".try_into().unwrap(),
        ptr_ty,
        vec![ptr_ty, size_ty],
        false,
    )
}

/// Get or create a declaration for the [tracked_dealloc] function in the nearest symbol table.
fn lookup_or_create_tracked_dealloc_fn(
    ctx: &mut Context,
    symbol_table_collection: &mut SymbolTableCollection,
    symbol_table_op: Box<dyn SymbolTableInterface>,
) -> Result<FuncOp> {
    let ptr_ty = PointerType::get(ctx, 0).into();
    lookup_or_insert_function(
        ctx,
        symbol_table_collection,
        symbol_table_op,
        "tracked_dealloc".try_into().unwrap(),
        VoidType::get(ctx).into(),
        vec![ptr_ty, ptr_ty],
        false,
    )
}

/// Emit IR ops that materialise the TMM self pointer attribute value as an LLVM `ptr`.
/// Appends a `ConstantOp` (i64) and an `IntToPtrOp` to `inserter`.
fn emit_tmm_state_ptr(ctx: &mut Context, inserter: &mut dyn Inserter, tmm_ptr: *const ()) -> Value {
    let i64_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let addr = tmm_ptr as u64;
    let addr_attr = IntegerAttr::new(i64_ty, APInt::from_u64(addr, NonZero::new(64).unwrap()));
    let const_op = ConstantOp::new(ctx, Box::new(addr_attr));
    inserter.append_op(ctx, &const_op);
    let ptr_ty = PointerType::get(ctx, 0).into();
    let inttoptr_op = IntToPtrOp::new(ctx, const_op.get_result(ctx), ptr_ty);
    inserter.append_op(ctx, &inttoptr_op);
    inttoptr_op.get_result(ctx)
}

#[derive(Debug, thiserror::Error)]
pub enum TrackedMemoryManagerErr {
    #[error(
        "TrackedAllocOp: number of dynamic dimension operands ({got}) does not match \
         the number of dynamic dimensions in the result type ({expected})"
    )]
    AllocNumDynamicDimsMismatch { expected: usize, got: usize },
    #[error("Missing self pointer attribute on Op")]
    MissingSelfPointerAttr,
    #[error("Nearest symbol table not found")]
    NearestSymbolTableNotFound,
    #[error("Expected static_info to be a ConstPointerAttr")]
    ExpectedConstPointerAttr,
    #[error("LLJIT symbol lookup or insertion failed: {0}")]
    LLJITSymbolError(String),
}

/// Op to allocate a tracked memref.
///
/// Similar to [`AllocOp`](crate::memref::ops::AllocOp) but lowers to a call to
/// [tracked_malloc] instead of `malloc`, recording the allocation in a [TrackedTMM].
///
/// ### Attributes
/// | attribute | description |
/// |-----|-------|
/// | `memref_tmm_ptr_alloc` | [ConstPointerAttr]: opaque pointer to the [TrackedTMM] state |
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One `Index` operand per dynamic dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The allocated memref |
#[pliron_op(
    name = "memref.tracked_alloc",
    format = "attr($memref_tmm_ptr_alloc, $ConstPointerAttr, label($tmm), delimiters(`[`, `]`))\
     ` dyn_dims ` `[` operands(CharSpace(`,`)) `]` ` : ` type($0)",
    interfaces = [
        NResultsInterface<1>,
        ResultNOfType<0, RankedMemrefType>,
        AtLeastNResultsInterface<1>,
        OneResultInterface,
        SameResultsType,
        AllResultsOfType<RankedMemrefType>,
    ],
    attributes = (memref_tmm_ptr_alloc: ConstPointerAttr),
)]
pub struct TrackedAllocOp;

impl TrackedAllocOp {
    /// Create a new [TrackedAllocOp] with `tmm` as the runtime state pointer.
    pub fn new_with_tmm(
        ctx: &mut Context,
        result_ty: TypedHandle<RankedMemrefType>,
        dynamic_dimensions: Vec<Value>,
        tmm: ConstPointerAttr,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            dynamic_dimensions,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_memref_tmm_ptr_alloc(ctx, tmm);
        op
    }

    /// Get the dynamic dimension operands.
    pub fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Verify for TrackedAllocOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_attr_memref_tmm_ptr_alloc(ctx).is_none() {
            return verify_err!(
                self.loc(ctx),
                TrackedMemoryManagerErr::MissingSelfPointerAttr
            );
        }
        let result_ty_ptr = self.get_result(ctx).get_type(ctx);
        let result_ty_ref = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<RankedMemrefType>()
            .expect("TrackedAllocOp result must be a RankedMemrefType");
        let expected = result_ty.num_dynamic_dimensions();
        let got = self.get_operation().deref(ctx).get_num_operands();
        if got != expected {
            return verify_err!(
                self.loc(ctx),
                TrackedMemoryManagerErr::AllocNumDynamicDimsMismatch { expected, got }
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl ToCFDialect for TrackedAllocOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let result_ty = OneResultInterface::result_type(self, ctx);
        let memref_ty = TypedHandle::<RankedMemrefType>::from_handle(result_ty, ctx)
            .expect("TrackedAllocOp result must be a RankedMemrefType");
        let dyn_dimensions = self.get_dynamic_dimensions(ctx);

        let (sizes, strides, num_elems) =
            descriptor::compute_sizes_strides(ctx, rewriter, memref_ty, dyn_dimensions);

        let element_ty = MultiDimensionalType::element_type(&*memref_ty.deref(ctx));
        let elem_size = compute_type_size_in_bytes(ctx, rewriter, element_ty);
        let alloc_size = MulOp::new_with_overflow_flag(
            ctx,
            elem_size,
            num_elems,
            IntegerOverflowFlagsAttr::default(),
        );
        rewriter.append_op(ctx, &alloc_size);

        let symbol_table_op = nearest_symbol_table(ctx, self.get_operation()).ok_or_else(|| {
            input_error!(
                self.loc(ctx),
                TrackedMemoryManagerErr::NearestSymbolTableNotFound
            )
        })?;
        let tracked_malloc_fn = lookup_or_create_tracked_malloc_fn(
            ctx,
            &mut SymbolTableCollection::default(),
            symbol_table_op,
        )?;

        let tmm_ptr = self.get_attr_memref_tmm_ptr_alloc(ctx).unwrap().0;
        let state_ptr_ir = emit_tmm_state_ptr(ctx, rewriter, tmm_ptr);

        let call_op = CallOp::new(
            ctx,
            CallOpCallable::Direct(SymbolOpInterface::get_symbol_name(&tracked_malloc_fn, ctx)),
            tracked_malloc_fn.get_type(ctx),
            vec![state_ptr_ir, alloc_size.get_result(ctx)],
        );
        rewriter.append_op(ctx, &call_op);
        let allocated_ptr = call_op.get_result(ctx);

        let offset = IndexConstantOp::new(ctx, 0);
        rewriter.append_op(ctx, &offset);

        let descriptor = descriptor::pack_descriptor(
            ctx,
            rewriter,
            memref_ty,
            descriptor::Descriptor {
                allocated_ptr,
                aligned_ptr: allocated_ptr,
                offset: offset.get_result(ctx),
                sizes,
                strides,
            },
        )?;

        Rewriter::replace_operation_with_values(
            rewriter,
            ctx,
            self.get_operation(),
            vec![descriptor],
        );
        Ok(())
    }
}

#[op_interface_impl]
impl MemrefAllocOpInterface for TrackedAllocOp {
    fn try_new(
        ctx: &mut Context,
        static_info: Option<AttrObj>,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Self> {
        let const_pointer = static_info
            .map(|attr| attr.downcast::<ConstPointerAttr>())
            .ok_or_else(|| arg_error_noloc!(TrackedMemoryManagerErr::ExpectedConstPointerAttr))?
            .map_err(|_| arg_error_noloc!(TrackedMemoryManagerErr::ExpectedConstPointerAttr))?;

        Ok(Self::new_with_tmm(
            ctx,
            memref_ty,
            dynamic_sizes,
            *const_pointer,
        ))
    }
}

/// Op to deallocate a tracked memref.
///
/// Similar to [`DeallocOp`](crate::memref::ops::DeallocOp) but lowers to a call to
/// [tracked_dealloc] instead of `free`, removing the pointer from the [TrackedTMM] tracker.
///
/// ### Attributes
/// | attribute | description |
/// |-----|-------|
/// | `memref_tmm_ptr_dealloc` | [ConstPointerAttr]: opaque pointer to the [TrackedTMM] state |
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | The memref to deallocate |
#[pliron_op(
    name = "memref.tracked_dealloc",
    format = "attr($memref_tmm_ptr_dealloc, $ConstPointerAttr, label($tmm), delimiters(`[`, `]`)) ` : ` $0",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<1>,
        AtLeastNOpdsInterface<1>,
        OneOpdInterface,
        OperandNOfType<0, RankedMemrefType>,
    ],
    attributes = (memref_tmm_ptr_dealloc: ConstPointerAttr),
)]
pub struct TrackedDeallocOp;

impl TrackedDeallocOp {
    /// Create a new [TrackedDeallocOp] with `tmm` as the runtime state pointer.
    pub fn new_with_tmm(ctx: &mut Context, memref: Value, tmm: ConstPointerAttr) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![memref],
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_memref_tmm_ptr_dealloc(ctx, tmm);
        op
    }

    /// Get the memref operand to deallocate.
    pub fn get_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

impl Verify for TrackedDeallocOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        if self.get_attr_memref_tmm_ptr_dealloc(ctx).is_none() {
            return verify_err!(
                self.loc(ctx),
                TrackedMemoryManagerErr::MissingSelfPointerAttr
            );
        }
        Ok(())
    }
}

#[op_interface_impl]
impl ToCFDialect for TrackedDeallocOp {
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let memref = self.get_memref(ctx);
        let allocated_ptr = descriptor::unpack_allocated_ptr(ctx, rewriter, memref);

        let symbol_table_op = nearest_symbol_table(ctx, self.get_operation()).ok_or_else(|| {
            input_error!(
                self.loc(ctx),
                TrackedMemoryManagerErr::NearestSymbolTableNotFound
            )
        })?;
        let tracked_dealloc_fn = lookup_or_create_tracked_dealloc_fn(
            ctx,
            &mut SymbolTableCollection::default(),
            symbol_table_op,
        )?;

        let tmm_ptr = self.get_attr_memref_tmm_ptr_dealloc(ctx).unwrap().0;
        let state_ptr_ir = emit_tmm_state_ptr(ctx, rewriter, tmm_ptr);

        let call_op = CallOp::new(
            ctx,
            CallOpCallable::Direct(SymbolOpInterface::get_symbol_name(&tracked_dealloc_fn, ctx)),
            tracked_dealloc_fn.get_type(ctx),
            vec![state_ptr_ir, allocated_ptr],
        );
        rewriter.append_op(ctx, &call_op);
        Rewriter::replace_operation(rewriter, ctx, self.get_operation(), call_op.get_operation());
        Ok(())
    }
}

#[op_interface_impl]
impl MemrefDeallocOpInterface for TrackedDeallocOp {
    fn try_new(ctx: &mut Context, static_info: Option<AttrObj>, memref: Value) -> Result<Self> {
        let const_pointer = static_info
            .map(|attr| attr.downcast::<ConstPointerAttr>())
            .ok_or_else(|| arg_error_noloc!(TrackedMemoryManagerErr::ExpectedConstPointerAttr))?
            .map_err(|_| arg_error_noloc!(TrackedMemoryManagerErr::ExpectedConstPointerAttr))?;
        Ok(Self::new_with_tmm(ctx, memref, *const_pointer))
    }
}

/// A [TensorMemoryManager] that creates [TrackedAllocOp] and [TrackedDeallocOp],
/// recording each allocation in `self` so that live allocations can be inspected
/// after JIT execution via [TrackedTMM::tracked_allocations].
impl TensorMemoryManager for TrackedTMM {
    fn create_memref_alloc(
        &mut self,
        ctx: &mut Context,
        memref_ty: TypedHandle<RankedMemrefType>,
        dynamic_sizes: Vec<Value>,
    ) -> Result<Box<dyn MemrefAllocOpInterface>> {
        let tmm_ptr = self as *mut TrackedTMM as *const ();
        let tmm_attr = ConstPointerAttr(tmm_ptr);
        let op = TrackedAllocOp::new_with_tmm(ctx, memref_ty, dynamic_sizes, tmm_attr);
        Ok(Box::new(op))
    }

    fn create_memref_dealloc(
        &mut self,
        ctx: &mut Context,
        memref: Value,
    ) -> Result<Box<dyn MemrefDeallocOpInterface>> {
        let tmm_ptr = self as *mut TrackedTMM as *const ();
        let tmm_attr = ConstPointerAttr(tmm_ptr);
        let op = TrackedDeallocOp::new_with_tmm(ctx, memref, tmm_attr);
        Ok(Box::new(op))
    }

    fn register_runtime_symbols(
        &self,
        jit: &pliron_llvm::llvm_sys::lljit::LLVMLLJIT,
    ) -> Result<()> {
        jit.add_symbol_mapping(
            "tracked_malloc",
            tracked_malloc as *const () as u64,
            JITSymbolGenericFlags::JITSymbolGenericFlagsNone,
        )
        .map_err(|e| {
            arg_error_noloc!(TrackedMemoryManagerErr::LLJITSymbolError(format!(
                "Failed to add symbol mapping for tracked_malloc: {e}"
            )))
        })?;
        jit.add_symbol_mapping(
            "tracked_dealloc",
            tracked_dealloc as *const () as u64,
            JITSymbolGenericFlags::JITSymbolGenericFlagsNone,
        )
        .map_err(|e| {
            arg_error_noloc!(TrackedMemoryManagerErr::LLJITSymbolError(format!(
                "Failed to add symbol mapping for tracked_dealloc: {e}"
            )))
        })?;
        Ok(())
    }
}
