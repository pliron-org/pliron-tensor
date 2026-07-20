//! Tensor semantics -> memref semantics
//!
//! Tensors are values: every tensor-producing op conceptually yields a brand new tensor.
//! Buffers (memrefs) are storage: bufferization is the process of mapping tensor values
//! to buffers.
//!
//! A naive approach is to give every tensor value its own buffer. The optimization part
//! is deciding when a result can instead reuse ("bufferize in place") the buffer of one
//! of its operands without affecting program semantics.
//!
//! The function [bufferize], is the main entry point.
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

use rustc_hash::FxHashSet;

use pliron::{
    analyses::liveness::{Liveness, LivenessTq},
    builtin::op_interfaces::{BranchOpInterface, OneResultInterface},
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
        inserter::{Inserter, OpInsertionPoint},
    },
    op::{Op, op_cast, op_impls},
    operation::Operation,
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle, type_cast, type_impls},
    utils::union_find::UnionFind,
    value::{DefiningEntity, Use, Value},
    verify_err_noloc,
};
use pliron_common_dialects::index::{ops::IndexConstantOp, types::IndexType};
use pliron_llvm::ops::FuncOp;
use thiserror::Error;

use crate::{
    memref::{
        ToMemrefType,
        ops::{CopyOp, DimOp},
        type_interfaces::{Dimension, ShapedType},
        types::RankedMemrefType,
    },
    tensor::{conversions::lower_func_op_to_llvm, memory_management::TensorMemoryManager},
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
    fn get_dynamic_dimensions(&self, ctx: &Context, opd: Use<Value>) -> Option<Vec<Value>>;

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
        tmm: &mut dyn TensorMemoryManager,
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
            let dynamic_dims_opt = op.get_dynamic_dimensions(ctx, alias.operand);
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

/// A helper struct that implements [DialectConversion]
/// to bufferize from tensor semantics to memref semantics.
struct Bufferizer<'tmm, TMM: TensorMemoryManager> {
    tmm: &'tmm mut TMM,
    /// Set of operands that can be bufferized in-place (i.e., without copy).
    /// Whether a result (that alises to this operand) needs a new allocation
    /// or not is left to the op's rewrite method.
    in_place_bufferizable_operands: FxHashSet<Use<Value>>,
    /// Set of successor operands (identified by their operand index in the branch op)
    /// that must be copied before being passed to a successor block, because the
    /// operand and the successor block argument would otherwise share a buffer while
    /// both are live.
    successor_operands_needing_copy: FxHashSet<Use<Value>>,
}

impl<'tmm, TMM: TensorMemoryManager> DialectConversion for Bufferizer<'tmm, TMM> {
    fn can_convert_op(&self, ctx: &Context, op: Ptr<Operation>) -> bool {
        op_impls::<dyn BufferizableOpInterface>(Operation::get_op_dyn(op, ctx).as_ref())
            || Operation::get_op::<FuncOp>(op, ctx).is_some()
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
        // Handle FuncOp: update function type signature and entry block arg types.
        if let Some(func_op) = Operation::get_op::<FuncOp>(op, ctx) {
            return lower_func_op_to_llvm(&func_op, ctx);
        }

        let op_dyn = Operation::get_op_dyn(op, ctx);
        let op_iface_opt = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref());
        let aliases_opt = op_iface_opt.map(|iface| iface.get_operand_result_aliases(ctx));

        // Two kinds of operands need to be copied to a new buffer:
        let mut opds_needing_copy = FxHashSet::default();
        // 1. Successor operands that were previously identified to need a copy.
        // 2. Operands that alias with a result but weren't marked safe for in-place bufferization.
        for opd in op.deref(ctx).operands_as_uses() {
            if self.successor_operands_needing_copy.contains(&opd) {
                opds_needing_copy.insert(opd);
                continue;
            }
            let Some(aliases) = &aliases_opt else {
                continue;
            };
            let aliasing_results = Alias::get_aliases_for_operand(aliases, opd);
            if aliasing_results.is_empty() || self.in_place_bufferizable_operands.contains(&opd) {
                continue;
            }
            opds_needing_copy.insert(opd);
        }

        for opd in opds_needing_copy {
            // Create a new buffer for the operand.
            let opd_ty = opd.get_type(ctx);
            let ranked_memref_ty: TypedHandle<RankedMemrefType> =
                TypedHandle::from_handle(opd_ty, ctx)?;
            let dynamic_sizes = if let Some(dynamic_sizes) =
                op_iface_opt.and_then(|iface| iface.get_dynamic_dimensions(ctx, opd))
            {
                dynamic_sizes
            } else {
                // If dynamic dimensions are not provided by the op, use `memref.dim` to query them (less efficient).
                let mut dynamic_sizes = Vec::new();
                let shape = {
                    let ranked_memref_ty = ranked_memref_ty.deref(ctx);
                    ranked_memref_ty.shape().clone()
                };
                for (dim_idx, dim) in shape.iter().enumerate() {
                    if matches!(dim, Dimension::Static(_)) {
                        continue;
                    }
                    let dim_const = IndexConstantOp::new(ctx, dim_idx);
                    rewriter.append_op(ctx, &dim_const);
                    let dim_size = DimOp::new(ctx, opd.get_def(ctx), dim_const.get_result(ctx))
                        .get_result(ctx);
                    dynamic_sizes.push(dim_size);
                }
                dynamic_sizes
            };
            let alloc_op = self
                .tmm
                .create_memref_alloc(ctx, ranked_memref_ty, dynamic_sizes)?;
            let new_buffer = alloc_op.get_result(ctx);
            rewriter.append_operation(ctx, alloc_op.get_operation());

            // Copy the operand buffer to the new buffer.
            let copy_op = CopyOp::new(ctx, new_buffer, opd.get_def(ctx));
            rewriter.append_op(ctx, &copy_op);
            // Replace the operand with the new buffer.
            Operation::replace_operand(op, ctx, opd.find_index(ctx), new_buffer);
        }

        // Rewrite the op to use memref semantics.
        if let Some(op_iface) = op_iface_opt {
            op_iface.rewrite(ctx, rewriter, self.tmm, _operands_info)?;
        }
        Ok(())
    }

    fn can_convert_type(&self, ctx: &Context, ty: TypeHandle) -> bool {
        type_impls::<dyn ToMemrefType>(&*ty.deref(ctx))
    }

    fn convert_type(&mut self, ctx: &mut Context, ty: TypeHandle) -> Result<TypeHandle> {
        let to_memref_ty = type_cast::<dyn ToMemrefType>(&*ty.deref(ctx)).map(|t| t.converter());
        if let Some(to_memref_ty) = to_memref_ty {
            to_memref_ty(ty, ctx)
        } else {
            Ok(ty)
        }
    }
}

/// Bufferize `op` and its nested ops.
///
/// Bufferization happens in four steps:
///
/// 1. Operands passed as successor operands to a successor block create aliases.
///    If any such operand is live-in at the successor block or is passed to
///    multiple argument positions of the same successor, copy the operand to a
///    new buffer and pass that to the successor instead.
/// 2. Group values that share a buffer into buffer classes. This includes aliases
///    declared by [BufferizableOpInterface] ops and successor operands that share
///    a buffer with the block argument they are forwarded to.
/// 3. For each operand of a [BufferizableOpInterface] op that aliases a result, decide
///    whether it can be bufferized in place:
///    (a) If the op only reads through the operand: always in-place.
///    (b) If the op writes through it: in-place only if no other operand of the op is in
///    the same buffer class, and no member of that class, other than the values this op
///    itself defines, is live after it. Otherwise a new buffer is allocated and the
///    operand copied into it.
/// 4. Rewrite the IR using dialect conversion, which invokes [BufferizableOpInterface::rewrite].
pub fn bufferize<TMM: TensorMemoryManager>(
    tmm: &mut TMM,
    op: Ptr<Operation>,
    ctx: &mut Context,
) -> Result<IRStatus> {
    struct InPlaceBufferizationAnalysis {
        liveness: Liveness<LivenessTq>,
        dom_info: DomInfo,
        /// Values sharing a buffer, grouped into buffer classes.
        buffer_classes: UnionFind<Value>,
        in_place_bufferizable_operands: FxHashSet<Use<Value>>,
        successor_operands_needing_copy: FxHashSet<Use<Value>>,
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

    /// Pass 2: group values that share a buffer into buffer classes.
    ///
    /// Alias edges are recorded even where a copy will later sever them, which
    /// over-approximates sharing in the conservative direction.
    fn build_buffer_classes(ctx: &Context, state: &mut InPlaceBufferizationAnalysis, node: IRNode) {
        let IRNode::Operation(op) = node else {
            return;
        };

        let op_dyn = Operation::get_op_dyn(op, ctx);

        // Aliases the op declares between its own operands and results.
        if let Some(op_iface) = op_cast::<dyn BufferizableOpInterface>(op_dyn.as_ref()) {
            for alias in op_iface.get_operand_result_aliases(ctx) {
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

    /// Pass 3: decide which aliasing operands can be bufferized in place.
    fn analyze_in_place_operands(
        ctx: &Context,
        state: &mut InPlaceBufferizationAnalysis,
        node: IRNode,
    ) {
        let IRNode::Operation(op) = node else {
            return;
        };

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

            let aliasing_results = Alias::get_aliases_for_operand(&aliases, opd);
            if aliasing_results.is_empty()
                || Alias::operand_aliases_with_multiple_results(&aliases, opd)
                || Alias::operand_aliases_with_other_operands(&aliases, opd)
            {
                continue;
            }

            // Reading through the operand can't corrupt the buffer, so sharing it is safe
            // no matter who else holds it.
            if !op_iface.operand_bufferizes_to_memory_write(ctx, opd) {
                state.in_place_bufferizable_operands.insert(opd);
                continue;
            }

            let opd_class = state.buffer_classes.find(opd.get_def(ctx));

            // Another operand backed by the same buffer would be accessed through the
            // storage this operand is about to be written through, during the op itself.
            // No liveness query can catch that, since the hazard is within the op.
            let mut conflicts_with_other_operand = false;
            for other in op.deref(ctx).operands_as_uses() {
                if other != opd && state.buffer_classes.find(other.get_def(ctx)) == opd_class {
                    conflicts_with_other_operand = true;
                    break;
                }
            }
            if conflicts_with_other_operand {
                continue;
            }

            // Writing in place is safe only if nothing else that shares the buffer can
            // observe the write. Values this op defines are excluded: they are the results
            // of the write, not readers of the pre-write contents.
            //
            // This asks only whether a member is live just after the op; no program order
            // between the op and the member's definition is assumed. A member defined by a
            // textually later op still counts if it reaches this point via a back edge.
            let class_members = state.buffer_classes.set_members(opd_class);
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
            if !class_live {
                state.in_place_bufferizable_operands.insert(opd);
            }
        }
    }

    let mut analysis = InPlaceBufferizationAnalysis {
        liveness: Liveness::<LivenessTq>::default(),
        dom_info: DomInfo::default(),
        buffer_classes: UnionFind::default(),
        in_place_bufferizable_operands: FxHashSet::default(),
        successor_operands_needing_copy: FxHashSet::default(),
    };

    for pass in [
        analyze_successor_operands,
        build_buffer_classes,
        analyze_in_place_operands,
    ] {
        walkers::uninterruptible::immutable::walk_op(
            ctx,
            &mut analysis,
            &walkers::WALKCONFIG_PREORDER_FORWARD,
            op,
            pass,
        );
    }

    let mut bufferizer = Bufferizer::<TMM> {
        tmm,
        in_place_bufferizable_operands: analysis.in_place_bufferizable_operands,
        successor_operands_needing_copy: analysis.successor_operands_needing_copy,
    };
    apply_dialect_conversion(ctx, &mut bufferizer, op)
}
