// SPDX-License-Identifier: Apache-2.0
// Copyright (c) The pliron-tensor contributors

//! Tensor semantics -> memref semantics
//!
//! Tensors are values: every tensor-producing op conceptually yields a brand new tensor.
//! Buffers (memrefs) are storage: bufferization is the process of mapping tensor values
//! to buffers.
//!
//! The main entry point to the bufferizer is the function [bufferize].
//!
//! A naive approach is to give every tensor value its own buffer. The optimization part
//! is deciding when a result can instead reuse ("bufferize in place") the buffer of one
//! of its operands without affecting program semantics.
//!
//! Multiple live tensor values are allowed to share a buffer, as long as none of them
//! can observe a write performed through another. Concretely, the algorithm maintains
//! the invariant that at every in-place write:
//!
//! 1. No tensor value sharing that buffer, other than the values the writing op itself
//!    defines, is live after the writing op, and
//! 2. No other operand of the writing op shares that buffer (which would be read or
//!    written concurrently, through the same storage, during the op).
//!
//! Reads never need exclusive access: sharing a buffer between readers can't corrupt it,
//! so a read is always bufferized in-place. The burden is entirely on writes.
//!
//! A tensor may be backed by more than one buffer at a point: for example, a tensor block
//! arg may be backed by different buffers based on control-flow.
//!
//! Aliases may be created in the following ways:
//! 1. An operand of an op implementing [BufferizableOpInterface] may be specified to alias
//!    with a result of the same op. The algorithm will insert copies (if it deems necessary)
//!    and update the operand with the copy buffer, allowing the rewrite method to reuse the
//!    operand buffer for the result, safely. The rewrite method of the op, must, however
//!    ensure that it doesn't create new aliases that violate the invariant. For example,
//!    multiple results must not bufferize to the same memref.
//! 2. A tensor value passed as a successor operand to a successor block argument creates an
//!    implicit alias between the value and the successor block argument. If the value is live-in
//!    at the successor block or is passed to multiple argument positions of the same successor,
//!    the bufferizer will insert a copy to a new buffer and pass that to the successor instead.
//!
//! *Requirement*: if a bufferized op writes to memory, the write must either be through
//! an operand aliasing the written result, or into a buffer the op allocates itself and
//! that the result refers to. A write through an operand that declares no alias is
//! invisible to the copy insertion below, and would clobber that operand's buffer.
//!
//! ## Memref Layouts
//!
//! - Function entry tensor arguments and tensors loaded by `llvm.load` must have
//!   the [identity layout](MemrefLayout).
//! - After copy decisions (before any rewrites), layouts are inferred to a fixed point:
//!   - An op result gets [BufferizableOpInterface::result_layout], or a fully dynamic
//!     layout if its op does not implement the interface.
//!   - Block args, loop results and function results get the merged layout of the values
//!     that flow into them.
//! - `memref.cast` operations are inserted on these values when required.

use rustc_hash::{FxHashMap, FxHashSet};
use std::collections::hash_map::Entry;

use crate::tensor::types::RankedTensorType;
use pliron::{
    analyses::liveness::{Liveness, LivenessTq},
    builtin::op_interfaces::{BranchOpInterface, OneResultInterface, OperandSegmentInterface},
    common_traits::Verify,
    context::{Context, Ptr},
    derive::op_interface,
    graph::{
        dominance::DomInfo,
        walkers::{self, IRNode},
    },
    irbuild::{
        IRStatus,
        dialect_conversion::{
            DialectConversion, DialectConversionRewriter, OperandsInfo, apply_dialect_conversion,
        },
        inserter::{IRInserter, Inserter, OpInsertionPoint},
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    result::Result,
    symbol_table::SymbolTableCollection,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast, type_impls},
    utils::union_find::UnionFind,
    value::{DefiningEntity, Use, Value},
    verify_err_noloc,
};
use pliron_common_dialects::{
    cf::{op_interfaces::YieldingRegions, ops::ForOp},
    index::{ops::IndexConstantOp, types::IndexType},
};
use pliron_llvm::ops::{FuncOp, ReturnOp};
use thiserror::Error;

use crate::{
    memref::{
        ToMemrefType,
        layout::{MemrefLayout, fully_dynamic, is_cast_compatible, merge},
        ops::{CopyOp, DimOp, MemrefCastOp},
        type_interfaces::{Dimension, MultiDimensionalType, ShapedType},
        types::RankedMemrefType,
    },
    tensor::memory_management::TensorMemoryManager,
};

/// Is an alias May or Must
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AliasKind {
    May,
    Must,
}

/// Buffer relation
#[derive(Default, Debug, Clone, Copy, PartialEq, Eq)]
pub enum BufferRelation {
    /// Relationship b/w the operand and result buffers is unknown.
    #[default]
    Unknown,
    /// Operand buffer may contain the result buffer.
    Contains,
    /// Operand buffer and result buffer are equivalent.
    Equivalent,
}

/// Describes aliasing b/w operands and results.
pub struct Alias {
    /// The operand that may alias with [Self::result].
    pub operand: Use<Value>,
    /// The result that may alias with [Self::operand].
    pub result: Value,
    /// Alias kind (May or Must)
    pub kind: AliasKind,
    /// Buffer relation b/w the operand and result buffers.
    pub relation: BufferRelation,
}

impl Alias {
    /// Get all results aliasing with `opd`.
    pub fn get_aliases_for_operand(aliases: &[Alias], opd: Use<Value>) -> Vec<&Alias> {
        aliases
            .iter()
            .filter(|alias| alias.operand == opd)
            .collect::<Vec<_>>()
    }

    /// Get all operands aliasing with `res`.
    pub fn get_aliases_for_result(aliases: &[Alias], res: Value) -> Vec<&Alias> {
        aliases
            .iter()
            .filter(|alias| alias.result == res)
            .collect::<Vec<_>>()
    }

    /// Does the operand alias with multiple results?
    pub fn operand_aliases_with_multiple_results(aliases: &[Alias], opd: Use<Value>) -> bool {
        Self::get_aliases_for_operand(aliases, opd).len() > 1
    }

    /// Does this operand alias with other operands?
    pub fn operand_aliases_with_other_operands(aliases: &[Alias], opd: Use<Value>) -> bool {
        let aliasing_results: Vec<_> = Self::get_aliases_for_operand(aliases, opd)
            .iter()
            .map(|alias| alias.result)
            .collect();
        aliases
            .iter()
            .any(|alias| aliasing_results.contains(&alias.result) && alias.operand != opd)
    }
}

#[derive(Debug, Error)]

pub enum AliasErr {
    #[error("Invalid alias: the operand and result do not belong to the same op")]
    InvalidAlias,
    #[error(
        "Incorrect number of dynamic dimension operands: Must be equal to number of dynamic dimensions in the operand type"
    )]
    IncorrectNumDynamicDims,
    #[error("Invalid dynamic dimension operand type: Must be of type Index")]
    InvalidDynamicDimOpdType,
    #[error("Operand type is not a shaped type")]
    InvalidOperandType,
}

impl Verify for Alias {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        let DefiningEntity::Op(op) = self.result.defining_entity() else {
            return verify_err_noloc!(AliasErr::InvalidAlias);
        };
        if self.operand.user_op() != op {
            return verify_err_noloc!(AliasErr::InvalidAlias);
        }
        Ok(())
    }
}

/// [Op]s implementing this can participate in bufferization.
#[op_interface]
pub trait BufferizableOpInterface {
    /// Return true if this operation bufferizes to a memory read of operand `opd`.
    /// It will only be called on operands that have a tensor type.
    ///
    /// It is always safe to return `true`, but that may introduce unnecessary
    /// allocations and / or copies.
    fn operand_bufferizes_to_memory_read(&self, ctx: &Context, opd: Use<Value>) -> bool;

    /// Return true if this operation bufferizes to a memory write of operand `opd`.
    /// It will only be called on operands that have a tensor type.
    ///
    /// It is always safe to return `true`, but that may introduce unnecessary
    /// allocations and / or copies.
    fn operand_bufferizes_to_memory_write(&self, ctx: &Context, opd: Use<Value>) -> bool;

    /// Get post-bufferization aliasing info between this op's operands and results.
    /// If after bufferization, the buffer of an operand may alias with the buffer of a result,
    /// then, the returned vector should contain an [Alias] with the appropriate information.
    fn get_operand_result_aliases(&self, ctx: &Context) -> Vec<Alias>;

    /// Get the dynamic dimensions for the given operand.
    /// On `None`, `memref.dim` will be used (less efficient).
    /// It will only be called on aliasing operands that have a tensor type.
    fn get_operand_dynamic_dimensions(&self, ctx: &Context, opd: Use<Value>) -> Option<Vec<Value>>;

    /// Return true if `value` can be written to in place.
    ///
    /// The method is called only when
    /// 1. `value` is a result of this op, or an argument of a block
    ///    belonging to one of this op's regions.
    /// 2. `value`is of a tensor type.
    ///
    /// Default: results are writable, block arguments are not.
    /// An op with a region must explicitly opt a block argument into being writable.
    /// An op that wants a non-writable result (e.g. one backed by read-only memory)
    /// must override this to return `false`.
    fn is_writable(&self, _ctx: &Context, value: Value) -> bool {
        value.defining_block().is_none()
    }

    /// Compute the memref layout of `result` from the layouts of its aliased operands.
    fn result_layout(
        &self,
        ctx: &Context,
        result: Value,
        operand_layout: &dyn Fn(Use<Value>) -> MemrefLayout,
    ) -> MemrefLayout {
        let aliases = self.get_operand_result_aliases(ctx);
        let aliases = Alias::get_aliases_for_result(&aliases, result);
        match aliases.as_slice() {
            // No alias: the result gets a new buffer with identity layout.
            [] => None,
            // The result is the operand's buffer, thus it has the operand's layout.
            [alias] if alias.relation == BufferRelation::Equivalent => {
                operand_layout(alias.operand)
            }
            // The layout is not known statically.
            _ => {
                let ty = result.get_type(ctx);
                let ty = ty.deref(ctx);
                let rank = type_cast::<dyn ShapedType>(&*ty)
                    .expect("Result must be shaped")
                    .rank();
                fully_dynamic(rank)
            }
        }
    }

    /// Rewrite to use memref semantics.
    ///
    /// Operands will have already been bufferized (i.e., converted to memrefs).
    /// Non-aliasing results will need to be be bufferized (by allocating a new buffer).
    /// Aliasing results can assume that it's safe to reuse operand buffers.
    ///
    /// The rewrite must not create buffer sharing beyond what
    /// [Self::get_operand_result_aliases] declares: the analysis reasons only about the
    /// aliases reported there. This means that, for example, multiple results of the op
    /// must not bufferize to the same memref.
    ///
    /// `operands_info` semantics are as in [DialectConversion::rewrite], and can be used
    /// to get pre-conversion operand types.
    fn rewrite(
        &self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        bufferizer_state: &mut BufferizerState,
        operands_info: &OperandsInfo,
    ) -> Result<()>;

    fn verify(op: &dyn Op, ctx: &Context) -> Result<()>
    where
        Self: Sized,
    {
        let op = op
            .downcast_ref::<Self>()
            .expect("Failed to downcast op to Self");
        let aliases = op.get_operand_result_aliases(ctx);

        for alias in aliases {
            alias.verify(ctx)?;
            let opd_ty = alias.operand.get_type(ctx);
            let opd_ty = opd_ty.deref(ctx);
            let Some(shaped_ty) = type_cast::<dyn ShapedType>(&*opd_ty) else {
                return verify_err_noloc!(AliasErr::InvalidOperandType);
            };
            let dynamic_dims_opt = op.get_operand_dynamic_dimensions(ctx, alias.operand);
            let num_dynamic_dims = shaped_ty.num_dynamic_dimensions();
            if let Some(dynamic_dims) = dynamic_dims_opt {
                if dynamic_dims.len() != num_dynamic_dims {
                    return verify_err_noloc!(AliasErr::IncorrectNumDynamicDims);
                }
                if !dynamic_dims
                    .iter()
                    .all(|dim| dim.get_type(ctx).deref(ctx).is::<IndexType>())
                {
                    return verify_err_noloc!(AliasErr::InvalidDynamicDimOpdType);
                }
            }
        }
        Ok(())
    }
}

/// Return true if `value` can be written to in place, per
/// [BufferizableOpInterface::is_writable] of the op that owns it: the defining op,
/// if `value` is an op result, or the parent op of the defining block, if `value` is
/// a block argument.
///
/// If the owning op doesn't implement [BufferizableOpInterface] at all, the same
/// conservative default applies as the trait's default method: op results are
/// writable, block arguments are not.
fn is_value_writable(ctx: &Context, value: Value) -> bool {
    let owner_op = match value.defining_entity() {
        DefiningEntity::Op(op) => op,
        DefiningEntity::Block(block) => {
            let Some(parent_op) = block.deref(ctx).get_parent_op(ctx) else {
                return false;
            };
            parent_op
        }
    };
    let op_dyn = Operation::get_op_dyn(owner_op, ctx);
    match op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref()) {
        Some(op_iface) => op_iface.is_writable(ctx, value),
        None => value.defining_op().is_some(),
    }
}

/// A helper struct that implements [DialectConversion]
/// to bufferize from tensor semantics to memref semantics.
struct Bufferizer<'a> {
    state: BufferizerState<'a>,
    /// Aliasing operands that must be copied to a fresh buffer before the op runs.
    out_of_place_operands: FxHashSet<Use<Value>>,
    /// Set of successor operands that must be copied before being passed to a successor block,
    /// because the operand and the successor block argument would otherwise share a buffer while
    /// both are live.
    successor_operands_needing_copy: FxHashSet<Use<Value>>,
}

/// Common state required by Ops implementing [BufferizableOpInterface].
pub struct BufferizerState<'a> {
    /// Cached symbol tables
    pub symbol_tables: SymbolTableCollection,
    /// Tensor memory manager
    pub tmm: &'a mut dyn TensorMemoryManager,
    /// A counter for new names generated, helps with uniquing.
    pub name_counter: u64,
    /// Inferred layouts of tensor values.
    pub layouts: FxHashMap<Value, MemrefLayout>,
}

impl<'a> DialectConversion for Bufferizer<'a> {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn BufferizableOpInterface>(Operation::get_op_dyn(op, ctx).as_ref())
            || op
                .deref(ctx)
                .operands_as_uses()
                .any(|u| self.successor_operands_needing_copy.contains(&u))
    }

    fn rewrite(
        &mut self,
        ctx: &mut Context,
        rewriter: &mut DialectConversionRewriter,
        op: Ptr<Operation>,
        _operands_info: &OperandsInfo,
    ) -> Result<()> {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let op_iface_opt = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref());

        // Two kinds of operands need to be copied to a new buffer:
        // 1. Successor operands that were previously identified to need a copy.
        // 2. Aliasing operands the analysis decided to bufferize out-of-place.
        let opds_needing_copy: FxHashSet<_> = op
            .deref(ctx)
            .operands_as_uses()
            .filter(|opd| {
                self.successor_operands_needing_copy.contains(opd)
                    || self.out_of_place_operands.contains(opd)
            })
            .collect();

        for opd in opds_needing_copy {
            let dynamic_sizes =
                op_iface_opt.and_then(|iface| iface.get_operand_dynamic_dimensions(ctx, opd));
            let new_buffer = copy_to_identity_buffer(
                ctx,
                rewriter,
                self.state.tmm,
                opd.get_def(ctx),
                dynamic_sizes,
            )?;
            // Replace the operand with the new buffer.
            Operation::replace_operand(op, ctx, opd.find_index(ctx), new_buffer);
        }

        // Rewrite the op to use memref semantics.
        if let Some(op_iface) = op_iface_opt {
            op_iface.rewrite(ctx, rewriter, &mut self.state, _operands_info)?;
        }
        Ok(())
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_impls::<dyn ToMemrefType>(&*ty.deref(ctx))
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let converted = crate::memref::to_memref_type(ty, ctx)?;
        let ty = converted.deref(ctx);
        if let Some(ty) = ty.downcast_ref::<RankedMemrefType>() {
            Ok(RankedMemrefType::get(
                ctx,
                ty.element_type(),
                ty.shape().clone(),
                fully_dynamic(ty.rank()),
            )
            .into())
        } else {
            Ok(converted)
        }
    }
}

/// Bufferize `op` and its nested ops.
///
/// Bufferization steps:
///
/// 1. Operands passed as successor operands to a successor block create aliases.
///    If any such operand is live-in at the successor block or is passed to
///    multiple argument positions of the same successor, copy the operand to a
///    new buffer and pass that to the successor instead.
/// 2. Group values that share a buffer into buffer classes. This includes aliases
///    declared by [BufferizableOpInterface] ops and successor operands that share
///    a buffer with the block argument they are forwarded to. An alias is skipped
///    once its operand has been decided as out-of-place, since the copy severs it.
/// 3. For each operand of a [BufferizableOpInterface] op that aliases a result, decide
///    whether it can be bufferized in place:
///    (a) If the op only reads through the operand: always in-place.
///    (b) If the op writes through it: in-place only if no other operand of the op is in
///    the same buffer class, and no member of that class, other than the values this op
///    itself defines, is live after it. Otherwise a new buffer is allocated and the
///    operand copied into it.
///
///    This step walks the ops in program order and rebuilds the classes of step 2 as soon
///    as a decision invalidates them, so a copy decided for one op can let a later op
///    stay in place.
/// 4. Infer layouts to a fixed point and set tensor block argument types.
///    Dialect conversion cannot correctly set block argument types, because it converts
///    from the type only and cannot access the layouts of the incoming values.
/// 5. Rewrite the IR with [BufferizableOpInterface::rewrite].
/// 6. Cast the values passed to block arguments and [ReturnOp]s on type mis-match.
///
/// The algorithm is, at worst, O(n^2) in the number of ops.
pub fn bufferize(
    tmm: &mut dyn TensorMemoryManager,
    op: Ptr<Operation>,
    ctx: &mut Context,
) -> Result<IRStatus> {
    struct InPlaceBufferizationAnalysis {
        liveness: Liveness<LivenessTq>,
        dom_info: DomInfo,
        /// Values sharing a buffer, grouped into buffer classes.
        buffer_classes: UnionFind<Value>,
        /// Aliasing operands decided to need a copy. Everything not in here is
        /// bufferized in place. Only ever grows; recording one severs its alias edge
        /// the next time the buffer classes are rebuilt.
        out_of_place_operands: FxHashSet<Use<Value>>,
        successor_operands_needing_copy: FxHashSet<Use<Value>>,
    }

    impl InPlaceBufferizationAnalysis {
        /// Record that `opd` must be copied to a fresh buffer.
        fn record_out_of_place(&mut self, opd: Use<Value>) {
            self.out_of_place_operands.insert(opd);
        }
    }

    /// Pass 1: decide which successor operands must be copied before being
    /// forwarded to a successor block.
    fn analyze_successor_operands(
        ctx: &Context,
        state: &mut InPlaceBufferizationAnalysis,
        node: IRNode,
    ) {
        let IRNode::Operation(op) = node else {
            return;
        };

        let op_dyn = Operation::get_op_dyn(op, ctx);

        // Passing a tensor value as a successor operand creates an implicit alias between
        // the value and the successor block argument. We detect two cases where the value
        // and the block argument would both be live, and so must not share a buffer:
        //   (a) The value is live-in at the successor block (direct use there besides the block arg).
        //   (b) The same value is passed to multiple argument positions of the same successor.
        // *Note*: Two tensors T1 and T2 that may be passed from different predecessor blocks to the
        // same successor block argument may share a buffer. If they do, they acquired it through a
        // common alias, so they are in one buffer class and every write checks the whole class.
        if op.deref(ctx).get_num_successors() == 0 {
            return;
        }

        let Some(branch_iface) = op_cast::<dyn BranchOpInterface>(op_dyn.as_ref()) else {
            // Without BranchOpInterface we cannot identify which operands are successor
            // operands, so conservatively copy all tensor-typed operands.
            for opd_use in op.deref(ctx).operands_as_uses() {
                let val = opd_use.get_def(ctx);
                if type_impls::<dyn ToMemrefType>(&*val.get_type(ctx).deref(ctx)) {
                    state.successor_operands_needing_copy.insert(opd_use);
                }
            }
            return;
        };

        for opd_use in op.deref(ctx).operands_as_uses() {
            let val = opd_use.get_def(ctx);
            if !type_impls::<dyn ToMemrefType>(&*val.get_type(ctx).deref(ctx)) {
                continue;
            }
            let mut needs_copy = false;
            for succ_idx in 0..op.deref(ctx).get_num_successors() {
                let succ_opds = branch_iface.successor_operands(ctx, succ_idx);
                if !succ_opds.contains(&val) {
                    continue;
                }
                // (a) Liveness check.
                let succ_block = op.deref(ctx).get_successor(succ_idx);
                if state.liveness.is_live_at_point(
                    ctx,
                    &mut state.dom_info,
                    val,
                    OpInsertionPoint::AtBlockStart(succ_block),
                ) {
                    needs_copy = true;
                    break;
                }
                // (b) Duplicate check: value appears more than once in this successor's args.
                if succ_opds.iter().filter(|&&v| v == val).count() >= 2 {
                    needs_copy = true;
                    break;
                }
            }
            if needs_copy {
                state.successor_operands_needing_copy.insert(opd_use);
            }
        }
    }

    /// Pass 2: add `op`'s share of the buffer classes.
    ///
    /// An alias edge is skipped when the operand carrying it has already been decided
    /// to be bufferized out-of-place: the copy inserted for it severs that alias. Since
    /// the decision set only grows, classes only shrink, which only relaxes the checks
    /// in [analyze_in_place_operands] and so never invalidates a decision already made.
    fn build_buffer_classes(
        ctx: &Context,
        state: &mut InPlaceBufferizationAnalysis,
        op: Ptr<Operation>,
    ) {
        let op_dyn = Operation::get_op_dyn(op, ctx);

        // Aliases the op declares between its own operands and results.
        if let Some(op_iface) = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref()) {
            for alias in op_iface.get_operand_result_aliases(ctx) {
                if state.out_of_place_operands.contains(&alias.operand) {
                    // A copy will be inserted for this operand, severing the alias.
                    continue;
                }
                state
                    .buffer_classes
                    .union(alias.operand.get_def(ctx), alias.result);
            }
        }

        // A successor operand shares its buffer with the block argument it is forwarded to.
        if op.deref(ctx).get_num_successors() == 0 {
            // No successor operands to union with block arguments.
            return;
        }
        let Some(branch_iface) = op_cast::<dyn BranchOpInterface>(op_dyn.as_ref()) else {
            // Without BranchOpInterface there is no operand -> block argument mapping to
            // union over. `analyze_successor_operands` handles this by conservatively
            // copying every tensor-typed operand, and a copy severs the alias, so there
            // is no sharing left to record.
            return;
        };
        for succ_idx in 0..op.deref(ctx).get_num_successors() {
            let succ_block = op.deref(ctx).get_successor(succ_idx);
            for (arg_idx, val) in branch_iface
                .successor_operands(ctx, succ_idx)
                .into_iter()
                .enumerate()
            {
                if !type_impls::<dyn ToMemrefType>(&*val.get_type(ctx).deref(ctx)) {
                    continue;
                }
                let block_arg = succ_block.deref(ctx).get_argument(arg_idx);
                state.buffer_classes.union(val, block_arg);
            }
        }
    }

    /// Pass 3: decide which aliasing operands of `op` can be bufferized in place.
    fn analyze_in_place_operands(
        ctx: &Context,
        state: &mut InPlaceBufferizationAnalysis,
        op: Ptr<Operation>,
    ) {
        let op_dyn = Operation::get_op_dyn(op, ctx);
        let Some(op_iface) = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref()) else {
            return;
        };

        let aliases = op_iface.get_operand_result_aliases(ctx);
        if aliases.is_empty() {
            return;
        }

        for opd in op.deref(ctx).operands_as_uses() {
            let opd_ty = opd.get_type(ctx);
            if !type_impls::<dyn ToMemrefType>(&*opd_ty.deref(ctx)) {
                continue;
            }

            // Decisions are never revisited: re-granting one would restore its alias
            // edge and grow the classes, which could invalidate other decisions.
            if state.out_of_place_operands.contains(&opd) {
                continue;
            }

            let aliasing_results = Alias::get_aliases_for_operand(&aliases, opd);
            if aliasing_results.is_empty() {
                // Not an aliasing operand: nothing to decide, and nothing to copy.
                continue;
            }
            if Alias::operand_aliases_with_multiple_results(&aliases, opd)
                || Alias::operand_aliases_with_other_operands(&aliases, opd)
            {
                // Sharing here could leave several values on one buffer in ways the
                // checks below don't model, so give the operand its own buffer.
                state.record_out_of_place(opd);
                continue;
            }

            // Reading through the operand can't corrupt the buffer, so sharing it is safe
            // no matter who else holds it. Leaving it undecided leaves it in place.
            if !op_iface.operand_bufferizes_to_memory_write(ctx, opd) {
                continue;
            }

            let opd_class = state.buffer_classes.find(opd.get_def(ctx));

            // Another operand backed by the same buffer would be accessed through the
            // storage this operand is about to be written through, during the op itself.
            // A liveness query cannot catch that since the hazard is within the op.
            let mut conflicts_with_other_operand = false;
            for other in op.deref(ctx).operands_as_uses() {
                if other != opd && state.buffer_classes.find(other.get_def(ctx)) == opd_class {
                    conflicts_with_other_operand = true;
                    break;
                }
            }
            if conflicts_with_other_operand {
                state.record_out_of_place(opd);
                continue;
            }

            let class_members = state.buffer_classes.set_members(opd_class);

            // A write cannot happen in place if any value sharing this buffer is not
            // writable (see [BufferizableOpInterface::is_writable]).
            if class_members
                .iter()
                .any(|&member| !is_value_writable(ctx, member))
            {
                state.record_out_of_place(opd);
                continue;
            }

            // Writing in place is safe only if no other value sharing the buffer is live
            // after this op. Exclude this op's results: they contain the updated data.
            //
            // Values defined later can still be live here through a loop back edge.
            let class_live = class_members
                .iter()
                .filter(|member| {
                    // Exclude values this op defines
                    !matches!(member.defining_entity(), DefiningEntity::Op(def_op) if def_op == op)
                })
                .any(|&member| {
                    state.liveness.is_live_at_point(
                        ctx,
                        &mut state.dom_info,
                        member,
                        OpInsertionPoint::AfterOperation(op),
                    )
                });
            if class_live {
                state.record_out_of_place(opd);
            }
        }
    }

    /// Rebuild the buffer classes from scratch, honouring the decisions made so far.
    fn rebuild_buffer_classes(
        ctx: &Context,
        state: &mut InPlaceBufferizationAnalysis,
        ops: &[Ptr<Operation>],
    ) {
        state.buffer_classes = UnionFind::default();
        for &op in ops {
            build_buffer_classes(ctx, state, op);
        }
    }

    let mut analysis = InPlaceBufferizationAnalysis {
        liveness: Liveness::<LivenessTq>::default(),
        dom_info: DomInfo::default(),
        buffer_classes: UnionFind::default(),
        out_of_place_operands: FxHashSet::default(),
        successor_operands_needing_copy: FxHashSet::default(),
    };

    // Successor-operand copies don't depend on the buffer classes, so they're decided once.
    walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut analysis,
        &walkers::WALKCONFIG_PREORDER_FORWARD,
        op,
        analyze_successor_operands,
    );

    let ops = collect_ops(ctx, op);

    // Decide the in-place bufferization of every aliasing operand, in program order.
    //
    // Classes start maximal (nothing decided out-of-place yet), which is the conservative
    // end: every alias edge is present, so the checks are at their strictest. Each
    // out-of-place decision severs an edge and shrinks the classes, and is applied right
    // away so that the ops after it see the effect.
    rebuild_buffer_classes(ctx, &mut analysis, &ops);
    for &op in &ops {
        let decisions_before_op = analysis.out_of_place_operands.len();
        analyze_in_place_operands(ctx, &mut analysis, op);
        if analysis.out_of_place_operands.len() != decisions_before_op {
            // TODO: This rebuild (unlike the one outside the loop)
            // is for precision and not safety. A rebuild will not contain
            // alias edges that weren't there in a previous computation.
            // We can explore ways to avoid rebuilding. It is O(n).
            rebuild_buffer_classes(ctx, &mut analysis, &ops);
        }
    }

    let layouts = infer_layouts(
        ctx,
        op,
        &ops,
        &analysis.out_of_place_operands,
        &analysis.successor_operands_needing_copy,
    );
    set_block_argument_types(ctx, &layouts);
    let mut bufferizer = Bufferizer {
        state: BufferizerState {
            tmm,
            symbol_tables: SymbolTableCollection::new(),
            name_counter: 0,
            layouts,
        },
        out_of_place_operands: analysis.out_of_place_operands,
        successor_operands_needing_copy: analysis.successor_operands_needing_copy,
    };
    let status = apply_dialect_conversion(ctx, &mut bufferizer, op)?;
    cast_to_block_argument_types(ctx, op)?;
    cast_to_function_result_types(ctx, op)?;
    Ok(status)
}

/// Copy a source into a new buffer with identity layout.
pub(crate) fn copy_to_identity_buffer(
    ctx: &mut Context,
    rewriter: &mut DialectConversionRewriter,
    tmm: &mut dyn TensorMemoryManager,
    source: Value,
    dynamic_sizes: Option<Vec<Value>>,
) -> Result<Value> {
    let source_type = TypedHandle::<RankedMemrefType>::from_handle(source.get_type(ctx), ctx)?;
    let identity_type = source_type.deref(ctx).with_identity_layout(ctx);
    let dynamic_sizes = match dynamic_sizes {
        Some(sizes) => sizes,
        None => {
            let shape = source_type.deref(ctx).shape().clone();
            let mut sizes = Vec::new();
            for (i, dimension) in shape.iter().enumerate() {
                if matches!(dimension, Dimension::Dynamic) {
                    let index = IndexConstantOp::new(ctx, i);
                    rewriter.append_op(ctx, &index);
                    let dim = DimOp::new(ctx, source, index.get_result(ctx));
                    rewriter.append_op(ctx, &dim);
                    sizes.push(dim.get_result(ctx));
                }
            }
            sizes
        }
    };
    let allocation = tmm.create_memref_alloc(ctx, identity_type, dynamic_sizes)?;
    rewriter.append_operation(ctx, allocation.get_operation());
    let buffer = allocation.get_result(ctx);
    let copy = CopyOp::new(ctx, buffer, source);
    rewriter.append_op(ctx, &copy);
    Ok(buffer)
}

#[derive(Debug, Error)]
pub enum BufferizeErr {
    #[error("An incoming buffer cannot be cast to the inferred layout")]
    IncompatibleIncomingLayout,
}

/// Collect the operations nested in `root` in program order.
fn collect_ops(ctx: &Context, root: Ptr<Operation>) -> Vec<Ptr<Operation>> {
    let mut ops = Vec::new();
    walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut ops,
        &walkers::WALKCONFIG_PREORDER_FORWARD,
        root,
        |_, ops, node| {
            if let IRNode::Operation(op) = node {
                ops.push(op);
            }
        },
    );
    ops
}

/// Get the operands that flow into block arguments, each paired with its block argument:
///   - For a branch `op`, its successor operands.
///   - For a [ForOp], its init operands and the operands of its yield.
fn block_arg_operands(ctx: &Context, op: Ptr<Operation>) -> Vec<(Value, Use<Value>)> {
    let mut operands = Vec::new();
    let op_dyn = Operation::get_op_dyn(op, ctx);
    if let Some(branch) = op_cast::<dyn BranchOpInterface>(op_dyn.as_ref()) {
        for succ in 0..op.deref(ctx).get_num_successors() {
            let block = op.deref(ctx).get_successor(succ);
            for (arg, position) in block
                .deref(ctx)
                .arguments()
                .zip(branch.successor_operand_range(ctx, succ))
            {
                operands.push((arg, op.deref(ctx).get_operand_as_use(position)));
            }
        }
    }
    if let Some(for_op) = op_dyn.downcast_ref::<ForOp>() {
        let yield_op = for_op.get_yield(ctx, 0).get_operation();
        let start = for_op.segment_range(ctx, 1).start;
        for (i, arg) in for_op
            .get_loop_carried_variables(ctx)
            .into_iter()
            .enumerate()
        {
            operands.push((arg, op.deref(ctx).get_operand_as_use(start + i)));
            operands.push((arg, yield_op.deref(ctx).get_operand_as_use(i)));
        }
    }
    operands
}

/// Infer layouts to a fixed point.
fn infer_layouts(
    ctx: &Context,
    root: Ptr<Operation>,
    ops: &[Ptr<Operation>],
    out_of_place: &FxHashSet<Use<Value>>,
    successor_copies: &FxHashSet<Use<Value>>,
) -> FxHashMap<Value, MemrefLayout> {
    let mut arguments = Vec::new();
    walkers::uninterruptible::immutable::walk_op(
        ctx,
        &mut arguments,
        &walkers::WALKCONFIG_PREORDER_FORWARD,
        root,
        |ctx, arguments, node| {
            if let IRNode::BasicBlock(block) = node {
                arguments.extend(
                    block
                        .deref(ctx)
                        .arguments()
                        .filter(|arg| arg.get_type(ctx).deref(ctx).is::<RankedTensorType>()),
                );
            }
        },
    );
    let is_tensor = |value: Value| value.get_type(ctx).deref(ctx).is::<RankedTensorType>();
    let mut incoming: FxHashMap<Value, Vec<Use<Value>>> = FxHashMap::default();
    for &op in ops {
        for (arg, operand) in block_arg_operands(ctx, op) {
            if is_tensor(arg) {
                incoming.entry(arg).or_default().push(operand);
            }
        }
        // A loop result has the same incoming values as its block argument.
        if let Some(for_op) = Operation::get_op::<ForOp>(op, ctx) {
            for (result, arg) in op
                .deref(ctx)
                .results()
                .zip(for_op.get_loop_carried_variables(ctx))
            {
                if let Some(operands) = incoming.get(&arg).cloned() {
                    incoming.insert(result, operands);
                }
            }
        }
    }
    let shape = |value: Value| {
        let ty = value.get_type(ctx);
        let ty = ty.deref(ctx);
        type_cast::<dyn ShapedType>(&*ty)
            .expect("Tensor must be shaped")
            .shape()
            .clone()
    };
    // Compute the initial layouts for every argument.
    let mut layouts = FxHashMap::default();
    for &arg in &arguments {
        if incoming.contains_key(&arg) {
            continue;
        }
        let block = arg
            .defining_block()
            .expect("Block argument must have a block");
        let entry = block.deref(ctx).get_parent_op(ctx).is_some_and(|parent| {
            Operation::get_op::<FuncOp>(parent, ctx).is_some()
                && block
                    .deref(ctx)
                    .get_parent_region()
                    .is_some_and(|region| region.deref(ctx).get_entry_block() == Some(block))
        });
        // Tensors that come as function arguments are assumed to have an identity layout.
        layouts.insert(
            arg,
            if entry {
                None
            } else {
                fully_dynamic(shape(arg).len())
            },
        );
    }
    let copied =
        |use_: Use<Value>| out_of_place.contains(&use_) || successor_copies.contains(&use_);
    let get_layout = |use_: Use<Value>, layouts: &FxHashMap<Value, MemrefLayout>| {
        if copied(use_) {
            // Copies inserted by the bufferizer have identity layout.
            return Some(None);
        }
        let value = use_.get_def(ctx);
        if let Some(ty) = value
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<RankedMemrefType>()
        {
            return Some(ty.layout().clone());
        }
        layouts.get(&value).cloned()
    };
    loop {
        let mut changed = false;
        let mut update =
            |value: Value, layout: MemrefLayout, layouts: &mut FxHashMap<Value, MemrefLayout>| {
                match layouts.entry(value) {
                    Entry::Occupied(mut previous) => {
                        let merged = merge(previous.get(), &layout, &shape(value));
                        if *previous.get() != merged {
                            previous.insert(merged);
                            changed = true;
                        }
                    }
                    Entry::Vacant(slot) => {
                        slot.insert(layout);
                        changed = true;
                    }
                }
            };
        for (&value, operands) in &incoming {
            let layout = operands
                .iter()
                .filter_map(|operand| get_layout(*operand, &layouts))
                .reduce(|a, b| merge(&a, &b, &shape(value)));
            if let Some(layout) = layout {
                update(value, layout, &mut layouts);
            }
        }
        for &op in ops {
            let op_dyn = Operation::get_op_dyn(op, ctx);
            let iface = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref());
            if iface.is_some()
                && op
                    .deref(ctx)
                    .operands_as_uses()
                    .any(|u| is_tensor(u.get_def(ctx)) && get_layout(u, &layouts).is_none())
            {
                continue;
            }
            for result in op.deref(ctx).results() {
                // Loop results get their layouts from their incoming values.
                if !is_tensor(result) || incoming.contains_key(&result) {
                    continue;
                }
                let layout = match iface {
                    Some(iface) => iface.result_layout(ctx, result, &|u| {
                        get_layout(u, &layouts).expect("Operand layout must be known")
                    }),
                    None => fully_dynamic(shape(result).len()),
                };
                update(result, layout, &mut layouts);
            }
        }
        if !changed {
            break;
        }
    }
    // Values that are still unknown get a fully dynamic (safe) layout.
    let results = ops
        .iter()
        .flat_map(|op| op.deref(ctx).results().collect::<Vec<_>>())
        .filter(|&result| is_tensor(result));
    for value in arguments.into_iter().chain(results) {
        layouts
            .entry(value)
            .or_insert_with(|| fully_dynamic(shape(value).len()));
    }
    layouts
}

/// Convert tensor block arguments to memrefs with their inferred layouts.
fn set_block_argument_types(ctx: &Context, layouts: &FxHashMap<Value, MemrefLayout>) {
    for (&arg, layout) in layouts {
        if arg.defining_block().is_none() {
            continue;
        }
        let layout = layout.clone();
        let ty = arg.get_type(ctx);
        let ty = ty.deref(ctx);
        let ty = ty
            .downcast_ref::<RankedTensorType>()
            .expect("Argument must be ranked");
        arg.set_type(
            ctx,
            RankedMemrefType::get(ctx, ty.element_type(), ty.shape().clone(), layout).into(),
        );
    }
}

/// Cast `value` to `target`, with a `memref.cast` before `before`.
/// Return `value` unchanged if it already has the `target` type or is not a ranked memref.
fn cast_to_type(
    ctx: &mut Context,
    before: Ptr<Operation>,
    value: Value,
    target: TypeHandle,
) -> Result<Value> {
    let source = value.get_type(ctx);
    if source == target || !source.deref(ctx).is::<RankedMemrefType>() {
        return Ok(value);
    }
    let target = TypedHandle::<RankedMemrefType>::from_handle(target, ctx)?;
    let compatible = {
        let source = source.deref(ctx);
        let source = source
            .downcast_ref::<RankedMemrefType>()
            .expect("Source must be ranked");
        let target = target.deref(ctx);
        source.element_type() == target.element_type()
            && source.shape() == target.shape()
            && is_cast_compatible(source.layout(), target.layout(), source.shape())
    };
    if !compatible {
        return verify_err_noloc!(BufferizeErr::IncompatibleIncomingLayout);
    }
    let cast = MemrefCastOp::new(ctx, value, target);
    let mut inserter =
        IRInserter::<pliron::irbuild::listener::DummyListener>::new_before_operation(before);
    inserter.append_op(ctx, &cast);
    Ok(cast.get_result(ctx))
}

/// Cast each value passed to a block argument to the type of that argument.
fn cast_to_block_argument_types(ctx: &mut Context, root: Ptr<Operation>) -> Result<()> {
    for op in collect_ops(ctx, root) {
        for (arg, operand) in block_arg_operands(ctx, op) {
            let user = operand.user_op();
            let value = operand.get_def(ctx);
            let casted = cast_to_type(ctx, user, value, arg.get_type(ctx))?;
            if casted != value {
                Operation::replace_operand(user, ctx, operand.find_index(ctx), casted);
            }
        }
    }
    Ok(())
}

/// Cast each value returned from a function to the result type of that function.
fn cast_to_function_result_types(ctx: &mut Context, root: Ptr<Operation>) -> Result<()> {
    for op in collect_ops(ctx, root) {
        let Some(return_op) = Operation::get_op::<ReturnOp>(op, ctx) else {
            continue;
        };
        let Some(value) = return_op.retval(ctx) else {
            continue;
        };
        let Some(func) = op
            .deref(ctx)
            .get_parent_op(ctx)
            .and_then(|parent| Operation::get_op::<FuncOp>(parent, ctx))
        else {
            continue;
        };
        let result_type = func.get_type(ctx).deref(ctx).result_type();
        let casted = cast_to_type(ctx, op, value, result_type)?;
        if casted != value {
            Operation::replace_operand(op, ctx, 0, casted);
        }
    }
    Ok(())
}
