//! Memref ops

use std::cell::Ref;

use pliron::{
    builtin::op_interfaces::{
        AllOperandsOfType, AllResultsOfType, AtLeastNOpdsInterface, AtLeastNResultsInterface,
        IsTerminatorInterface, NOpdsInterface, NRegionsInterface, NResultsInterface,
        OneOpdInterface, OneRegionInterface, OneResultInterface, OperandNOfType,
        OperandSegmentInterface, ResultNOfType, SameOperandsType, SameResultsType,
        SingleBlockRegionInterface,
    },
    combine::{
        Parser, attempt,
        parser::char::{self, spaces},
    },
    common_traits::Verify,
    context::{Context, Ptr},
    derive::pliron_op,
    identifier::Identifier,
    irbuild::{
        inserter::{BlockInsertionPoint, IRInserter, Inserter, OpInsertionPoint},
        listener::DummyListener,
    },
    irfmt::{
        parsers::{
            delimited_list_parser, int_parser, process_parsed_ssa_defs, spaced, ssa_opd_parser,
        },
        printers::{iter_with_sep, list_with_sep},
    },
    location::Location,
    op::{Op, OpObj},
    operation::Operation,
    parsable::{self, IntoParseResult, Parsable},
    printable::{self, ListSeparator, Printable},
    result::Result,
    r#type::{TypeObj, TypePtr, Typed, type_cast},
    value::Value,
    verify_err, verify_error,
};
use pliron_common_dialects::{
    cf::op_interfaces::{YieldingOp, YieldingRegion},
    index::types::IndexType,
};

use crate::memref::{
    attributes::{SliceParamAttr, SliceParamsAttr},
    op_interfaces::{CompatibleShapesOp, ElementWiseBinaryMemrefOpInterface, GenerateOpInterface},
    type_interfaces::{MultiDimensionalType, ShapedType},
    types::RankedMemrefType,
};

/// Op to allocate a memref.
/// See MLIR's [AllocOp](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefalloc-memrefallocop).
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension, to indicate the extent of that dimension |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The allocated memref of the specified type. |
#[pliron_op(
    name = "memref.alloc",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        NResultsInterface<1>,
        ResultNOfType<0, RankedMemrefType>,
        AtLeastNResultsInterface<1>,
        OneResultInterface,
        SameResultsType,
        AllResultsOfType<RankedMemrefType>,
    ],
)]
pub struct AllocOp;

#[derive(Debug, thiserror::Error)]
pub enum AllocOpVerifyError {
    #[error(
        "The number of dynamic dimension operands must match the number of dynamic dimensions in the result type (expected {expected}, got {got})"
    )]
    NumDynamicDimOperandsDoesNotMatchNumDynamicDims { expected: usize, got: usize },
}

impl AllocOp {
    /// Create a new `AllocOp` with the specified result type and dynamic dimension operands.
    pub fn new(
        ctx: &mut Context,
        result_ty: TypePtr<RankedMemrefType>,
        dynamic_dimensions: Vec<Value>,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_ty.into()],
            dynamic_dimensions,
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the dynamic dimension operands.
    pub fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().collect()
    }
}

impl Verify for AllocOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let result_ty = self.get_result(ctx).get_type(ctx).deref(ctx);
        let result_ty = result_ty
            .downcast_ref::<RankedMemrefType>()
            .expect("The result type of AllocOp must be a ranked memref type");

        let num_dynamic_dims = result_ty.num_dynamic_dimensions();
        let num_dynamic_dim_operands = self.get_operation().deref(ctx).get_num_operands();
        if num_dynamic_dim_operands != num_dynamic_dims {
            return verify_err!(
                self.loc(ctx),
                AllocOpVerifyError::NumDynamicDimOperandsDoesNotMatchNumDynamicDims {
                    expected: num_dynamic_dims,
                    got: num_dynamic_dim_operands
                }
            );
        }

        Ok(())
    }
}

/// Op to deallocate a memref.
/// See MLIR's [DeallocOp](https://mlir.llvm.org/docs/Dialects/MemRef/#memrefdealloc-memrefdeallocop).
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | The memref to deallocate. |
#[pliron_op(
    name = "memref.dealloc",
    format = "$0",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<1>,
        AtLeastNOpdsInterface<1>,
        OneOpdInterface,
        OperandNOfType<0, RankedMemrefType>,
    ],
)]
pub struct DeallocOp;

impl Verify for DeallocOp {
    fn verify(&self, _ctx: &Context) -> Result<()> {
        Ok(())
    }
}

impl DeallocOp {
    /// Create a new `DeallocOp` for the given memref.
    pub fn new(ctx: &mut Context, memref: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![memref],
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the memref operand to deallocate.
    pub fn get_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }
}

/// Op to generate a memref by applying a function to generate the value at each index.
///
/// ### Operands(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | A memref value (pointer) to where the values will be generated. |
///
/// ### Regions
///   - A single region containing the body that computes the values of the memref.
///   The region takes as many arguments as the rank of the memref type,
///   each representing an index along the corresponding dimension. The body should
///   yield a single value that matches the element type of the memref.
#[pliron_op(
    name = "memref.generate",
    format = "operands(CharSpace(`,`)) region($0)",
    interfaces = [
        SingleBlockRegionInterface,
        OneRegionInterface,
        NRegionsInterface<1>,
        NResultsInterface<0>,
        AtLeastNOpdsInterface<1>,
        NOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        YieldingRegion<YieldOp>,
    ],
    verifier = "succ"
)]
pub struct GenerateOp;

impl GenerateOp {
    /// Creates a new dynamically sized memref value.
    /// The `body_builder` function is called to populate the body of the region.
    /// It is provided with, as arguments, the current index values and an inserter
    /// (set to the end of the entry block). It must return the value yielded at that index.
    /// A [YieldOp] is automatically added at end of the body, taking this value as operand.
    pub fn new<State>(
        ctx: &mut Context,
        memref: Value,
        body_builder: fn(
            ctx: &mut Context,
            state: State,
            inserter: &mut IRInserter<DummyListener>,
            indices: Vec<Value>,
        ) -> Value,
        body_builder_state: State,
    ) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![memref],
            vec![],
            1,
        );
        let opop = Self { op };

        let rank = {
            let memref_type = memref.get_type(ctx).deref(ctx);
            let memref_type = memref_type
                .downcast_ref::<RankedMemrefType>()
                .expect("The memref operand must be of ranked memref type");

            memref_type.rank()
        };

        // Create the initializer region.
        let index_ty = IndexType::get(ctx);
        let region = opop.get_region(ctx);
        let op_inserter = &mut IRInserter::default();
        let entry_block = op_inserter.create_block(
            ctx,
            BlockInsertionPoint::AtRegionStart(region),
            Some("entry".try_into().unwrap()),
            vec![index_ty.into(); rank],
        );
        // Build the body.
        let indices = entry_block.deref(ctx).arguments().collect();
        let yield_value = body_builder(ctx, body_builder_state, op_inserter, indices);
        let yield_op = YieldOp::new(ctx, yield_value);
        op_inserter.set_insertion_point(OpInsertionPoint::AtBlockEnd(opop.get_exit(ctx)));
        op_inserter.append_op(ctx, yield_op);
        opop
    }

    /// Get the memref operand to which this op generates.
    pub fn get_destination_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the ranked memref type of the memref operand.
    pub fn get_destination_memref_type(&self, ctx: &Context) -> TypePtr<RankedMemrefType> {
        let memref_ty = self.get_destination_memref(ctx).get_type(ctx);
        TypePtr::from_ptr(memref_ty, ctx).expect("The memref operand must be of ranked memref type")
    }
}

impl GenerateOpInterface for GenerateOp {
    /// Get the shape of the destination memref.
    fn get_generated_shape<'a>(&'a self, ctx: &'a Context) -> Ref<'a, dyn ShapedType> {
        let memref_ty = self.get_destination_memref_type(ctx).deref(ctx);
        Ref::map(memref_ty, |memref_ty| {
            type_cast::<dyn ShapedType>(memref_ty)
                .expect("The memref operand type must implement ShapedType")
        })
    }
}

/// Yield a single value from within a region.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `value` | any type |
#[pliron_op(
    name = "memref.yield",
    format = "$0",
    interfaces = [
        NResultsInterface<0>,
        OneOpdInterface,
        NOpdsInterface<1>,
        YieldingOp,
        IsTerminatorInterface
    ],
    verifier = "succ"
)]
pub struct YieldOp;

impl YieldOp {
    /// Creates a new `YieldOp` with the specified operand.
    pub fn new(ctx: &mut Context, value: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![value],
            vec![],
            0,
        );
        Self { op }
    }
}

/// Op to store a value to a memref at specified indices.
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `value` | The value to store. Must be of the same element type as the memref. |
/// | `memref` | The memref to store to. |
/// | `indices` | One operand per dimension of the memref, indicating the index to store at along that dimension. Each index operand must be of type [Index](IndexType).
/// The number of index operands must match the rank of the memref.
#[pliron_op(
    name = "memref.store",
    interfaces = [
        NResultsInterface<0>,
        AtLeastNOpdsInterface<3>,
        OperandNOfType<1, RankedMemrefType>,
        OperandSegmentInterface,
    ]
)]
pub struct StoreOp;

impl Printable for StoreOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let value = self.get_value(ctx);
        let memref = self.get_destination_memref(ctx);
        let indices = self.get_indices(ctx);
        write!(
            f,
            "{} {} to {}[{}]",
            Self::get_opid_static(),
            value.disp(ctx),
            memref.disp(ctx),
            iter_with_sep(indices.iter(), printable::ListSeparator::CharSpace(',')).disp(ctx)
        )
    }
}

impl Parsable for StoreOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (value, memref, indices) = (
            ssa_opd_parser().skip(spaced(char::string("to"))),
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', ssa_opd_parser()),
        );

        let ((value, memref, indices), _) = (value, memref, indices)
            .parse_stream(state_stream)
            .into_result()?;
        let op = StoreOp::new(state_stream.state.ctx, value, memref, indices);
        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

impl StoreOp {
    /// Creates a new `StoreOp` with the specified operands.
    pub fn new(ctx: &mut Context, value: Value, memref: Value, indices: Vec<Value>) -> Self {
        let (operands, sizes) =
            Self::compute_segment_sizes(vec![vec![value], vec![memref], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Get the value operand to be stored.
    pub fn get_value(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the memref operand to which the value will be stored.
    pub fn get_destination_memref(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 1)[0]
    }

    /// Get the index operands indicating where the value will be stored.
    pub fn get_indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 2)
    }
}

#[derive(Debug, thiserror::Error)]
pub enum StoreOpVerifyError {
    #[error("The first operand must be of the same type as the memref's element type")]
    FirstOperandNotSameTypeAsMemrefElementType,
    #[error(
        "The number of index operands must match the rank of the memref (expected {expected}, got {got})"
    )]
    NumIndicesDoesNotMatchMemrefRank { expected: usize, got: usize },
    #[error("All index operands of StoreOp must be of IndexType")]
    IndexOperandNotOfIndexType,
}

impl Verify for StoreOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let value = self.get_value(ctx);
        let memref = self.get_destination_memref(ctx);
        let indices = self.get_indices(ctx);

        let memref_ty = memref.get_type(ctx).deref(ctx);
        let memref_ty = memref_ty
            .downcast_ref::<RankedMemrefType>()
            .expect("The second operand must be a ranked memref type");

        if value.get_type(ctx) != memref_ty.element_type() {
            return verify_err!(
                loc.clone(),
                StoreOpVerifyError::FirstOperandNotSameTypeAsMemrefElementType
            );
        }

        if indices.len() != memref_ty.rank() {
            return verify_err!(
                loc.clone(),
                StoreOpVerifyError::NumIndicesDoesNotMatchMemrefRank {
                    expected: memref_ty.rank(),
                    got: indices.len()
                }
            );
        }

        if !indices
            .iter()
            .all(|index| index.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc, StoreOpVerifyError::IndexOperandNotOfIndexType);
        }

        Ok(())
    }
}

/// Op to load a value from a memref at specified indices.
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `memref` | The memref to load from. |
/// | `indices` | One operand per dimension of the memref, indicating the index to load from along that dimension. Each index operand must be of type [Index](IndexType).
/// The number of index operands must match the rank of the memref.
///
/// ## Result(s)
/// | result | The loaded value. Must be of the same element type as the memref. |
#[pliron_op(
    name = "memref.load",
    interfaces = [
        NResultsInterface<1>,
        OneResultInterface,
        AtLeastNOpdsInterface<2>,
        OperandNOfType<0, RankedMemrefType>,
        OperandSegmentInterface,
    ],
)]
pub struct LoadOp;

impl Printable for LoadOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let memref = self.get_source_memref(ctx);
        let indices = self.get_indices(ctx);
        write!(
            f,
            "{} = {} {}[{}] : {}",
            self.get_result(ctx).disp(ctx),
            Self::get_opid_static(),
            memref.disp(ctx),
            iter_with_sep(indices.iter(), printable::ListSeparator::CharSpace(',')).disp(ctx),
            self.get_result(ctx).get_type(ctx).disp(ctx)
        )
    }
}

impl Parsable for LoadOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (memref, indices, res_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', ssa_opd_parser()),
            spaced(char::string(":")).with(Ptr::<TypeObj>::parser(())),
        );

        let ((memref, indices, res_ty), _) = (memref, indices, res_ty)
            .parse_stream(state_stream)
            .into_result()?;

        let op = LoadOp::new(state_stream.state.ctx, res_ty, memref, indices);

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum LoadOpVerifyErr {
    #[error(
        "The number of index operands must match the rank of the memref (expected {expected}, got {got})"
    )]
    NumIndicesDoesNotMatchMemrefRank { expected: usize, got: usize },
    #[error("All index operands of LoadOp must be of IndexType")]
    IndexOperandNotOfIndexType,
    #[error("The result type must be the same as the memref's element type")]
    ResultTypeNotSameAsMemrefElementType,
}

impl Verify for LoadOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);

        let memref = self.get_source_memref(ctx);
        let indices = self.get_indices(ctx);
        let result_ty = self.get_result(ctx).get_type(ctx);

        let memref_ty = memref.get_type(ctx).deref(ctx);
        let memref_ty = memref_ty
            .downcast_ref::<RankedMemrefType>()
            .expect("The first operand must be a ranked memref type");

        if result_ty != memref_ty.element_type() {
            return verify_err!(loc, LoadOpVerifyErr::ResultTypeNotSameAsMemrefElementType);
        }

        if indices.len() != memref_ty.rank() {
            return verify_err!(
                loc.clone(),
                LoadOpVerifyErr::NumIndicesDoesNotMatchMemrefRank {
                    expected: memref_ty.rank(),
                    got: indices.len()
                }
            );
        }

        if !indices
            .iter()
            .all(|index| index.get_type(ctx).deref(ctx).is::<IndexType>())
        {
            return verify_err!(loc, LoadOpVerifyErr::IndexOperandNotOfIndexType);
        }

        Ok(())
    }
}

impl LoadOp {
    /// Create a new `LoadOp` with the specified operands and result type.
    pub fn new(
        ctx: &mut Context,
        element_ty: Ptr<TypeObj>,
        memref: Value,
        indices: Vec<Value>,
    ) -> Self {
        let (operands, sizes) = Self::compute_segment_sizes(vec![vec![memref], indices]);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![element_ty],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_operand_segment_sizes(ctx, sizes);
        op
    }

    /// Get the memref operand to load from.
    pub fn get_source_memref(&self, ctx: &Context) -> Value {
        self.get_segment(ctx, 0)[0]
    }

    /// Get the index operands indicating where to load from.
    pub fn get_indices(&self, ctx: &Context) -> Vec<Value> {
        self.get_segment(ctx, 1)
    }
}

/// Op to query the runtime size of a memref dimension.
///
/// ## Operand(s)
/// | operand | description | type |
/// |-----|-------|----|
/// | `memref` | The memref to query | [RankedMemrefType] |
/// | `index` | The dimension index to query | [Index](IndexType). |
///
/// ## Result(s)
/// | result | The size of the queried dimension, as [Index](IndexType). |
#[pliron_op(
    name = "memref.dim",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        ResultNOfType<0, IndexType>,
        NOpdsInterface<2>,
        OperandNOfType<0, RankedMemrefType>,
        OperandNOfType<1, IndexType>,
    ],
    format = "$0 `, ` $1 ` : ` type($0)",
    verifier = "succ"
)]
pub struct DimOp;

impl DimOp {
    /// Create a new `DimOp`.
    pub fn new(ctx: &mut Context, memref: Value, dim_index: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![IndexType::get(ctx).into()],
            vec![memref, dim_index],
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the memref operand.
    pub fn get_source_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the dimension index operand.
    pub fn get_dimension_index(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Addition of two memrefs elementwise. The memrefs must have the same shape and element type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | The memref where the result will be stored. |
/// | `lhs` | The first memref value to add. |
/// | `rhs` | The second memref value to add. |
#[pliron_op(
    name = "memref.add",
    format = "$0 ` <- ` $1 ` + ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        SameOperandsType,
        AtLeastNOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        CompatibleShapesOp<RankedMemrefType>,
        AllResultsOfType<RankedMemrefType>,
        ElementWiseBinaryMemrefOpInterface,
    ],
    verifier = "succ"
)]
pub struct AddOp;

/// Subtraction of two memrefs elementwise. The memrefs must have the same shape and element type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | The memref where the result will be stored. |
/// | `lhs` | The first memref value to subtract from. |
/// | `rhs` | The second memref value to subtract. |
#[pliron_op(
    name = "memref.sub",
    format = "$0 ` <- ` $1 ` - ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        SameOperandsType,
        AtLeastNOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        CompatibleShapesOp<RankedMemrefType>,
        AllResultsOfType<RankedMemrefType>,
        ElementWiseBinaryMemrefOpInterface,
    ],
    verifier = "succ"
)]
pub struct SubOp;

/// Multiplication of two memrefs elementwise. The memrefs must have the same shape and element type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | The memref where the result will be stored. |
/// | `lhs` | The first memref value to multiply. |
/// | `rhs` | The second memref value to multiply. |
#[pliron_op(
    name = "memref.mul",
    format = "$0 ` <- ` $1 ` * ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        SameOperandsType,
        AtLeastNOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        CompatibleShapesOp<RankedMemrefType>,
        AllResultsOfType<RankedMemrefType>,
        ElementWiseBinaryMemrefOpInterface,
    ],
    verifier = "succ"
)]
pub struct MulOp;

/// Division of two memrefs elementwise. The memrefs must have the same shape and element type.
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | The memref where the result will be stored. |
/// | `lhs` | The dividend memref. |
/// | `rhs` | The divisor memref. |
#[pliron_op(
    name = "memref.div",
    format = "$0 ` <- ` $1 ` / ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        SameOperandsType,
        AtLeastNOpdsInterface<1>,
        AllOperandsOfType<RankedMemrefType>,
        CompatibleShapesOp<RankedMemrefType>,
        AllResultsOfType<RankedMemrefType>,
        ElementWiseBinaryMemrefOpInterface,
    ],
    verifier = "succ"
)]
pub struct DivOp;

/// Matrix multiplication op for memrefs.
/// Computes `res[i,j] = sum_k(lhs[i,k] * rhs[k,j])`.
/// `res` has shape [M, N], `lhs` has shape [M, K], `rhs` has shape [K, N].
///
/// ## Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `res` | Destination memref of shape [M, N]; result is written here. |
/// | `lhs` | Left-hand side memref of shape [M, K]. |
/// | `rhs` | Right-hand side memref of shape [K, N]. |
#[pliron_op(
    name = "memref.matmul",
    format = "$0 ` <- ` $1 ` X ` $2",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<3>,
        AllOperandsOfType<RankedMemrefType>,
    ],
)]
pub struct MatMulOp;

#[derive(Debug, thiserror::Error)]
pub enum MatMulOpVerifyErr {
    #[error("MatMulOp operands must be 2D ranked memrefs")]
    OperandNot2DMemref,
    #[error("MatMulOp operands must all have the same element type")]
    ElementTypeMismatch,
    #[error("MatMulOp lhs inner dim K ({lhs_k}) must match rhs outer dim K ({rhs_k})")]
    InnerDimMismatch { lhs_k: usize, rhs_k: usize },
    #[error("MatMulOp result dim {dim} (={result_d}) must match expected {expected}")]
    ResultDimMismatch {
        dim: usize,
        result_d: usize,
        expected: usize,
    },
}

impl Verify for MatMulOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;
        let loc = self.loc(ctx);
        let res = self.get_result_memref(ctx);
        let lhs = self.get_lhs_memref(ctx);
        let rhs = self.get_rhs_memref(ctx);

        let res_ty_ref = res.get_type(ctx);
        let res_binding = res_ty_ref.deref(ctx);
        let res_ty = res_binding
            .downcast_ref::<RankedMemrefType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DMemref))?;
        let lhs_ty_ref = lhs.get_type(ctx);
        let lhs_binding = lhs_ty_ref.deref(ctx);
        let lhs_ty = lhs_binding
            .downcast_ref::<RankedMemrefType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DMemref))?;
        let rhs_ty_ref = rhs.get_type(ctx);
        let rhs_binding = rhs_ty_ref.deref(ctx);
        let rhs_ty = rhs_binding
            .downcast_ref::<RankedMemrefType>()
            .ok_or_else(|| verify_error!(loc.clone(), MatMulOpVerifyErr::OperandNot2DMemref))?;

        if res_ty.rank() != 2 || lhs_ty.rank() != 2 || rhs_ty.rank() != 2 {
            return verify_err!(loc, MatMulOpVerifyErr::OperandNot2DMemref);
        }

        let elem_ty = lhs_ty.element_type();
        if rhs_ty.element_type() != elem_ty || res_ty.element_type() != elem_ty {
            return verify_err!(loc, MatMulOpVerifyErr::ElementTypeMismatch);
        }

        let res_shape = res_ty.shape();
        let lhs_shape = lhs_ty.shape();
        let rhs_shape = rhs_ty.shape();

        // K: lhs[1] must match rhs[0]
        if let (Dimension::Static(lhs_k), Dimension::Static(rhs_k)) = (&lhs_shape[1], &rhs_shape[0])
            && lhs_k != rhs_k
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::InnerDimMismatch {
                    lhs_k: *lhs_k,
                    rhs_k: *rhs_k
                }
            );
        }
        // M: lhs[0] must match res[0]
        if let (Dimension::Static(lhs_m), Dimension::Static(res_m)) = (&lhs_shape[0], &res_shape[0])
            && lhs_m != res_m
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 0,
                    result_d: *res_m,
                    expected: *lhs_m
                }
            );
        }
        // N: rhs[1] must match res[1]
        if let (Dimension::Static(rhs_n), Dimension::Static(res_n)) = (&rhs_shape[1], &res_shape[1])
            && rhs_n != res_n
        {
            return verify_err!(
                loc,
                MatMulOpVerifyErr::ResultDimMismatch {
                    dim: 1,
                    result_d: *res_n,
                    expected: *rhs_n
                }
            );
        }
        Ok(())
    }
}

impl MatMulOp {
    /// Create a new [MatMulOp].
    pub fn new(ctx: &mut Context, res: Value, lhs: Value, rhs: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![res, lhs, rhs],
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the destination memref operand.
    pub fn get_result_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the left-hand side memref operand.
    pub fn get_lhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }

    /// Get the right-hand side memref operand.
    pub fn get_rhs_memref(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(2)
    }
}

/// User-facing representation of a slice parameter (offset, size, or step).
/// Can be either a static constant or a dynamic value.
#[derive(Clone, Debug)]
pub enum SliceParam {
    /// A static usize constant value
    Static(usize),
    /// A dynamic Index value
    Dynamic(Value),
}

impl Printable for SliceParam {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        match self {
            SliceParam::Static(val) => write!(f, "{}", val),
            SliceParam::Dynamic(opd) => write!(f, "{}", opd.disp(ctx)),
        }
    }
}

impl Parsable for SliceParam {
    type Arg = ();
    type Parsed = SliceParam;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        _arg: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        attempt(ssa_opd_parser().map(SliceParam::Dynamic))
            .or(int_parser::<usize>().map(SliceParam::Static))
            .parse_stream(state_stream)
            .into()
    }
}

/// Copy one memref into another memref of the same logical shape.
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `destination` | The destination memref to write into. |
/// | `source` | The source memref to copy from. |
#[pliron_op(
    name = "memref.copy",
    format = "$0 ` <- ` $1",
    interfaces = [
        NResultsInterface<0>,
        NOpdsInterface<2>,
        AllOperandsOfType<RankedMemrefType>,
    ],
)]
pub struct CopyOp;

#[derive(Debug, thiserror::Error)]
pub enum CopyOpVerifyErr {
    #[error("CopyOp operands must have the same element type")]
    ElementTypeMismatch,
    #[error("CopyOp operands must have the same rank")]
    RankMismatch,
    #[error("CopyOp operands must have compatible shapes")]
    IncompatibleShapes,
}

impl Verify for CopyOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let destination = self.destination(ctx);
        let source = self.source(ctx);

        let destination_ty_ptr = destination.get_type(ctx);
        let destination_ty_binding = destination_ty_ptr.deref(ctx);
        let destination_ty = destination_ty_binding
            .downcast_ref::<RankedMemrefType>()
            .expect("CopyOp destination must be ranked memref type");
        let source_ty_ptr = source.get_type(ctx);
        let source_ty_binding = source_ty_ptr.deref(ctx);
        let source_ty = source_ty_binding
            .downcast_ref::<RankedMemrefType>()
            .expect("CopyOp source must be ranked memref type");

        if destination_ty.element_type() != source_ty.element_type() {
            return verify_err!(loc.clone(), CopyOpVerifyErr::ElementTypeMismatch);
        }

        if destination_ty.rank() != source_ty.rank() {
            return verify_err!(loc.clone(), CopyOpVerifyErr::RankMismatch);
        }

        for (dst_dim, src_dim) in destination_ty.shape().iter().zip(source_ty.shape().iter()) {
            if let (
                crate::memref::type_interfaces::Dimension::Static(dst),
                crate::memref::type_interfaces::Dimension::Static(src),
            ) = (dst_dim, src_dim)
                && dst != src
            {
                return verify_err!(loc, CopyOpVerifyErr::IncompatibleShapes);
            }
        }

        Ok(())
    }
}

impl CopyOp {
    pub fn new(ctx: &mut Context, destination: Value, source: Value) -> Self {
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![],
            vec![destination, source],
            vec![],
            0,
        );
        Self { op }
    }

    pub fn destination(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(1)
    }
}

/// Create a view into an existing memref by adjusting offset, sizes, and strides.
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `source` | The source memref to view. |
/// | `dynamic_offsets` | Zero or more [Index](IndexType) operands for dynamic offsets. |
/// | `dynamic_sizes` | Zero or more [Index](IndexType) operands for dynamic sizes. |
/// | `dynamic_steps` | Zero or more [Index](IndexType) operands for dynamic steps. |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | The computed subview memref. |
#[pliron_op(
    name = "memref.subview",
    interfaces = [
        NResultsInterface<1>,
        OneResultInterface,
        ResultNOfType<0, RankedMemrefType>,
        OperandNOfType<0, RankedMemrefType>,
    ],
    attributes = (memref_subview_slice_params: SliceParamsAttr)
)]
pub struct SubviewOp;

impl Printable for SubviewOp {
    fn fmt(
        &self,
        ctx: &Context,
        _state: &printable::State,
        f: &mut std::fmt::Formatter,
    ) -> std::fmt::Result {
        let source = self.source(ctx);
        write!(
            f,
            "${} = {} {}",
            self.get_result(ctx).disp(ctx),
            Self::get_opid_static(),
            source.disp(ctx)
        )?;

        let offsets = self.slice_offsets(ctx);
        let sizes = self.slice_sizes(ctx);
        let steps = self.slice_steps(ctx);

        write!(
            f,
            " [{}]",
            list_with_sep(&offsets, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&sizes, ListSeparator::CharSpace(',')).disp(ctx)
        )?;
        write!(
            f,
            " [{}]",
            list_with_sep(&steps, ListSeparator::CharSpace(',')).disp(ctx)
        )?;

        write!(f, " : {}", self.result_type(ctx).disp(ctx))
    }
}

impl Parsable for SubviewOp {
    type Arg = Vec<(Identifier, Location)>;
    type Parsed = OpObj;

    fn parse<'a>(
        state_stream: &mut parsable::StateStream<'a>,
        results: Self::Arg,
    ) -> parsable::ParseResult<'a, Self::Parsed> {
        let (source, offsets, sizes, steps, result_ty) = (
            ssa_opd_parser().skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())).skip(spaces()),
            delimited_list_parser('[', ']', ',', SliceParam::parser(())),
            spaced(char::string(":")).with(TypePtr::<RankedMemrefType>::parser(())),
        );

        let ((source, offsets, sizes, steps, result_ty), _) =
            (source, offsets, sizes, steps, result_ty)
                .parse_stream(state_stream)
                .into_result()?;

        let op = SubviewOp::new_with_result_type(
            state_stream.state.ctx,
            source,
            offsets,
            sizes,
            steps,
            result_ty,
        );

        process_parsed_ssa_defs(state_stream, &results, op.get_operation())?;
        Ok(OpObj::new(op)).into_parse_result()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SubviewOpVerifyErr {
    #[error("SubviewOp result and source ranks must match")]
    ResultSourceRankMismatch,
    #[error("SubviewOp: All dynamic operands must be of IndexType, but operand {index} is {ty}")]
    NonIndexOperand { index: usize, ty: String },
    #[error(
        "SubviewOp: Number of dynamic operands ({got}) does not match number of dynamic parameters ({expected})"
    )]
    NumDynamicOperandsMismatch { expected: usize, got: usize },
    #[error(
        "SubviewOp: Number of offsets ({got}) does not match rank of source memref ({expected})"
    )]
    NumOffsetsMismatch { expected: usize, got: usize },
    #[error("SubviewOp: Number of sizes ({got}) does not match rank of source memref ({expected})")]
    NumSizesMismatch { expected: usize, got: usize },
    #[error("SubviewOp: Number of steps ({got}) does not match rank of source memref ({expected})")]
    NumStepsMismatch { expected: usize, got: usize },
    #[error("SubviewOp: Missing memref_subview_slice_params attribute")]
    MissingSliceParamsAttr,
    #[error("SubviewOp: Static step values must be non-zero (got 0 at dimension {dim})")]
    InvalidStaticStep { dim: usize },
}

impl Verify for SubviewOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        let loc = self.loc(ctx);
        let op_ref = self.get_operation().deref(ctx);
        let mut operands = op_ref.operands();

        let source_operand = operands
            .next()
            .expect("SubviewOp must have a source operand");

        let source_ty_ptr = source_operand.get_type(ctx);
        let source_ty_binding = source_ty_ptr.deref(ctx);
        let source_ty = source_ty_binding
            .downcast_ref::<RankedMemrefType>()
            .expect("SubviewOp source must be ranked memref type");
        let result_ty_ptr = self.result_type(ctx);
        let result_ty_binding = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_binding
            .downcast_ref::<RankedMemrefType>()
            .expect("SubviewOp result must be ranked memref type");

        let rank = source_ty.rank();
        if result_ty.rank() != rank {
            return verify_err!(loc, SubviewOpVerifyErr::ResultSourceRankMismatch);
        }

        let slice_params = self
            .get_attr_memref_subview_slice_params(ctx)
            .ok_or_else(|| {
                verify_error!(loc.clone(), SubviewOpVerifyErr::MissingSliceParamsAttr)
            })?;

        if slice_params.offsets.len() != rank {
            return verify_err!(
                loc,
                SubviewOpVerifyErr::NumOffsetsMismatch {
                    expected: rank,
                    got: slice_params.offsets.len()
                }
            );
        }
        if slice_params.sizes.len() != rank {
            return verify_err!(
                loc,
                SubviewOpVerifyErr::NumSizesMismatch {
                    expected: rank,
                    got: slice_params.sizes.len()
                }
            );
        }
        if slice_params.steps.len() != rank {
            return verify_err!(
                loc,
                SubviewOpVerifyErr::NumStepsMismatch {
                    expected: rank,
                    got: slice_params.steps.len()
                }
            );
        }

        for (dim, step) in slice_params.steps.iter().enumerate() {
            if let SliceParamAttr::Static(0) = step {
                return verify_err!(loc, SubviewOpVerifyErr::InvalidStaticStep { dim });
            }
        }

        let num_dynamic_offsets = slice_params
            .offsets
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_sizes = slice_params
            .sizes
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();
        let num_dynamic_steps = slice_params
            .steps
            .iter()
            .filter(|p| matches!(p, SliceParamAttr::OperandIdx(_)))
            .count();

        let total_dynamic = num_dynamic_offsets + num_dynamic_sizes + num_dynamic_steps;
        let remaining_operands: Vec<_> = operands.collect();

        for (i, opd) in remaining_operands.iter().enumerate() {
            let opd_ty = opd.get_type(ctx);
            let opd_ty_ref = opd_ty.deref(ctx);
            if !opd_ty_ref.is::<IndexType>() {
                let ty_name = format!("{:?}", opd_ty_ref);
                return verify_err!(
                    loc,
                    SubviewOpVerifyErr::NonIndexOperand {
                        index: i + 1,
                        ty: ty_name
                    }
                );
            }
        }

        if remaining_operands.len() != total_dynamic {
            return verify_err!(
                loc,
                SubviewOpVerifyErr::NumDynamicOperandsMismatch {
                    expected: total_dynamic,
                    got: remaining_operands.len()
                }
            );
        }

        Ok(())
    }
}

impl SubviewOp {
    /// Create a new `SubviewOp` with the specified source memref, slice parameters.
    /// The result type is inferred from the source memref element type and the static sizes.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
    ) -> Self {
        let source_element_type = source
            .get_type(ctx)
            .deref(ctx)
            .downcast_ref::<RankedMemrefType>()
            .expect("SubviewOp source must be ranked memref type")
            .element_type();
        let result_shape = sizes
            .iter()
            .map(|size| match size {
                SliceParam::Static(size) => {
                    crate::memref::type_interfaces::Dimension::Static(*size)
                }
                SliceParam::Dynamic(_) => crate::memref::type_interfaces::Dimension::Dynamic,
            })
            .collect();
        let result_type = RankedMemrefType::get(ctx, source_element_type, result_shape);

        Self::new_with_result_type(ctx, source, offsets, sizes, steps, result_type)
    }

    /// Create a new `SubviewOp` with the specified source memref, slice parameters, and explicit result type.
    pub fn new_with_result_type(
        ctx: &mut Context,
        source: Value,
        offsets: Vec<SliceParam>,
        sizes: Vec<SliceParam>,
        steps: Vec<SliceParam>,
        result_type: TypePtr<RankedMemrefType>,
    ) -> Self {
        let mut operands = vec![source];
        let mut offset_attrs = Vec::new();
        let mut size_attrs = Vec::new();
        let mut step_attrs = Vec::new();

        for offset in offsets {
            match offset {
                SliceParam::Static(val) => offset_attrs.push(SliceParamAttr::Static(val)),
                SliceParam::Dynamic(val) => {
                    offset_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        for size in sizes {
            match size {
                SliceParam::Static(val) => size_attrs.push(SliceParamAttr::Static(val)),
                SliceParam::Dynamic(val) => {
                    size_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        for step in steps {
            match step {
                SliceParam::Static(val) => step_attrs.push(SliceParamAttr::Static(val)),
                SliceParam::Dynamic(val) => {
                    step_attrs.push(SliceParamAttr::OperandIdx(operands.len()));
                    operands.push(val);
                }
            }
        }

        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            operands,
            vec![],
            0,
        );
        let op = Self { op };
        op.set_attr_memref_subview_slice_params(
            ctx,
            SliceParamsAttr {
                offsets: offset_attrs,
                sizes: size_attrs,
                steps: step_attrs,
            },
        );
        op
    }

    /// Get the source memref operand.
    pub fn source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    // Helper function to convert slice parameter attributes into user-facing `SliceParam` values.
    fn slice_attr_to_params(&self, ctx: &Context, attrs: &[SliceParamAttr]) -> Vec<SliceParam> {
        let op_ref = self.get_operation().deref(ctx);
        attrs
            .iter()
            .map(|attr| match attr {
                SliceParamAttr::Static(val) => SliceParam::Static(*val),
                SliceParamAttr::OperandIdx(idx) => SliceParam::Dynamic(op_ref.get_operand(*idx)),
            })
            .collect()
    }

    /// Get the slice offsets as user-facing `SliceParam` values.
    pub fn slice_offsets(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_memref_subview_slice_params(ctx)
            .expect("SubviewOp must have memref_subview_slice_params attribute");
        self.slice_attr_to_params(ctx, &attrs.offsets)
    }

    /// Get the slice sizes as user-facing `SliceParam` values.
    pub fn slice_sizes(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_memref_subview_slice_params(ctx)
            .expect("SubviewOp must have memref_subview_slice_params attribute");
        self.slice_attr_to_params(ctx, &attrs.sizes)
    }

    /// Get the slice steps as user-facing `SliceParam` values.
    pub fn slice_steps(&self, ctx: &Context) -> Vec<SliceParam> {
        let attrs = self
            .get_attr_memref_subview_slice_params(ctx)
            .expect("SubviewOp must have memref_subview_slice_params attribute");
        self.slice_attr_to_params(ctx, &attrs.steps)
    }

    /// Get the dynamic operand values for the slice parameters, in the order they appear as operands.
    pub fn dynamic_operands(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}

/// Reshape a memref to a new shape by creating a new descriptor view over
/// the same underlying storage.
///
/// This operation does not copy elements. The source and result must have the
/// same element type and total element count.
///
/// ### Operand(s)
/// | operand | description |
/// |-----|-------|
/// | `source` | The source ranked memref to reshape from. |
/// | `dynamic_dimensions` | One [Index](IndexType) operand per dynamic dimension in the result type. |
///
/// ### Result(s)
/// | result | description |
/// |-----|-------|
/// | `result` | A reshaped ranked memref descriptor that aliases the source storage. |
#[pliron_op(
    name = "memref.reshape",
    format = "operands(CharSpace(`,`)) ` : ` type($0)",
    interfaces = [
        OneResultInterface,
        NResultsInterface<1>,
        AtLeastNOpdsInterface<1>,
        OperandNOfType<0, RankedMemrefType>,
        AllResultsOfType<RankedMemrefType>,
    ],
)]
pub struct ReshapeOp;

#[derive(Debug, thiserror::Error)]
pub enum ReshapeOpVerifyErr {
    #[error("ReshapeOp source must be a RankedMemrefType")]
    SourceNotRankedMemref,
    #[error("ReshapeOp source and result element types must match")]
    ElementTypeMismatch,
    #[error(
        "ReshapeOp: number of dynamic dimension operands ({got}) must match \
        number of dynamic dimensions in result type ({expected})"
    )]
    DynDimCountMismatch { expected: usize, got: usize },
    #[error("ReshapeOp: all dynamic dimension operands must be of IndexType")]
    DynDimNotIndex,
    #[error(
        "ReshapeOp: total element count of source ({src_count}) must match destination ({result_count})"
    )]
    ElementCountMismatch {
        src_count: usize,
        result_count: usize,
    },
}

impl Verify for ReshapeOp {
    fn verify(&self, ctx: &Context) -> Result<()> {
        use crate::memref::type_interfaces::Dimension;
        let loc = self.loc(ctx);

        let src_ty_ptr = self.get_source(ctx).get_type(ctx);
        let src_ty_ref = src_ty_ptr.deref(ctx);
        let src_ty = src_ty_ref
            .downcast_ref::<RankedMemrefType>()
            .ok_or_else(|| verify_error!(loc.clone(), ReshapeOpVerifyErr::SourceNotRankedMemref))?;

        let result_ty_ptr = self.result_type(ctx);
        let result_ty_ref = result_ty_ptr.deref(ctx);
        let result_ty = result_ty_ref
            .downcast_ref::<RankedMemrefType>()
            .expect("AllResultsOfType<RankedMemrefType> ensures result is RankedMemrefType");

        if src_ty.element_type() != result_ty.element_type() {
            return verify_err!(loc, ReshapeOpVerifyErr::ElementTypeMismatch);
        }

        let dyn_dims = self.get_dynamic_dimensions(ctx);
        let num_dynamic_in_result = result_ty
            .shape()
            .iter()
            .filter(|d| matches!(d, Dimension::Dynamic))
            .count();

        if dyn_dims.len() != num_dynamic_in_result {
            return verify_err!(
                loc.clone(),
                ReshapeOpVerifyErr::DynDimCountMismatch {
                    expected: num_dynamic_in_result,
                    got: dyn_dims.len()
                }
            );
        }

        for dyn_dim in &dyn_dims {
            if !dyn_dim.get_type(ctx).deref(ctx).is::<IndexType>() {
                return verify_err!(loc.clone(), ReshapeOpVerifyErr::DynDimNotIndex);
            }
        }

        // If both shapes are fully static, verify element count equality.
        let src_shape = src_ty.shape();
        let dst_shape = result_ty.shape();
        if src_shape.iter().all(|d| matches!(d, Dimension::Static(_)))
            && dst_shape.iter().all(|d| matches!(d, Dimension::Static(_)))
        {
            let src_count: usize = src_shape
                .iter()
                .map(|d| match d {
                    Dimension::Static(s) => *s,
                    _ => unreachable!(),
                })
                .product();
            let dst_count: usize = dst_shape
                .iter()
                .map(|d| match d {
                    Dimension::Static(s) => *s,
                    _ => unreachable!(),
                })
                .product();
            if src_count != dst_count {
                return verify_err!(
                    loc,
                    ReshapeOpVerifyErr::ElementCountMismatch {
                        src_count,
                        result_count: dst_count
                    }
                );
            }
        }

        Ok(())
    }
}

impl ReshapeOp {
    /// Create a new `ReshapeOp`.
    pub fn new(
        ctx: &mut Context,
        source: Value,
        dynamic_dimensions: Vec<Value>,
        result_type: TypePtr<RankedMemrefType>,
    ) -> Self {
        let mut operands = vec![source];
        operands.extend(dynamic_dimensions);
        let op = Operation::new(
            ctx,
            Self::get_concrete_op_info(),
            vec![result_type.into()],
            operands,
            vec![],
            0,
        );
        Self { op }
    }

    /// Get the source memref operand.
    pub fn get_source(&self, ctx: &Context) -> Value {
        self.get_operation().deref(ctx).get_operand(0)
    }

    /// Get the dynamic dimension operands.
    pub fn get_dynamic_dimensions(&self, ctx: &Context) -> Vec<Value> {
        self.get_operation().deref(ctx).operands().skip(1).collect()
    }
}
