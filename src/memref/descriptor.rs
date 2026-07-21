//! Memref Descriptor for runtime information about memref types.
//! This is analogous to the memref descriptor in MLIR.
//! A memref descriptor is a struct containing the following fields:
//! * Allocated pointer: A pointer to the allocated memory for the memref.
//!   This will the malloc'd pointer that is eventually free'd.
//! * Aligned pointer: A pointer to the aligned memory for the memref
//!   (the start of the memref).
//! * The offset of the memref (the number of elements between the start of the allocated
//!   memory and the start of the memref).
//! * The sizes (a rank length'd array) of each dimension of the memref.
//! * The strides (a rank length'd array) of each dimension of the memref.

use std::num::NonZero;

use pliron::{
    builtin::{
        attributes::IntegerAttr,
        op_interfaces::OneResultInterface,
        ops::ConstantOp,
        types::{IntegerType, Signedness},
    },
    context::Context,
    irbuild::inserter::Inserter,
    result::Result,
    r#type::{TypeHandle, Typed, TypedHandle},
    utils::apint::APInt,
    value::Value,
};
use pliron_common_dialects::index::ops::IndexConstantOp;
use pliron_llvm::{
    ToLLVMType,
    attributes::IntegerOverflowFlagsAttr,
    op_interfaces::IntBinArithOpWithOverflowFlag,
    ops::{
        AddOp, AllocaOp, ExtractValueOp, GepIndex, GetElementPtrOp, InsertValueOp, LoadOp, MulOp,
        StoreOp, UndefOp,
    },
    types::StructType,
};

use crate::memref::{
    type_interfaces::{Dimension, ShapedType},
    types::RankedMemrefType,
};

/// Given a [RankedMemrefType] and the [Value]s corresponding to its dynamic dimensions,
/// compute the sizes of each dimension as [Value]s, and the strides of each dimension as
/// [Value]s. Also return the total number of elements in the memref (the product of the sizes
/// of each dimension).
pub fn compute_sizes_strides(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    memref: TypedHandle<RankedMemrefType>,
    dynamic_sizes: Vec<Value>,
) -> (Vec<Value>, Vec<Value>, Value) {
    let mut dynamic_size_iter = dynamic_sizes.into_iter();

    // Compute sizes
    //   * using dynamic sizes for dynamic dimensions
    //   * using constant index ops for static dimensions
    let shape = memref.deref(ctx).shape().clone();
    let sizes: Vec<Value> = shape
        .iter()
        .map(|d| match d {
            Dimension::Static(size) => {
                let v = IndexConstantOp::new(ctx, *size);
                inserter.append_op(ctx, &v);
                v.get_result(ctx)
            }
            Dimension::Dynamic => dynamic_size_iter
                .next()
                .expect("Expected enough dynamic sizes for the memref type"),
        })
        .collect();
    assert!(
        dynamic_size_iter.next().is_none(),
        "Expected exactly enough dynamic sizes for the memref type"
    );

    enum Stride {
        Static(usize),
        Dynamic(Value),
    }

    let mut strides = Vec::new();
    let mut running_stride = Stride::Static(1);
    let last_stride = IndexConstantOp::new(ctx, 1);
    inserter.append_op(ctx, &last_stride);
    strides.push(last_stride.get_result(ctx));
    for (size_val, dim) in sizes.iter().zip(shape.iter()).rev() {
        match (running_stride, dim) {
            (Stride::Static(s), Dimension::Static(d)) => {
                let next_stride = s * d;
                let next_stride_val = IndexConstantOp::new(ctx, next_stride);
                inserter.append_op(ctx, &next_stride_val);
                running_stride = Stride::Static(next_stride);
                strides.push(next_stride_val.get_result(ctx));
            }
            (Stride::Dynamic(v), Dimension::Static(d)) => {
                let d_val = IndexConstantOp::new(ctx, *d);
                inserter.append_op(ctx, &d_val);
                let next_stride = MulOp::new_with_overflow_flag(
                    ctx,
                    v,
                    d_val.get_result(ctx),
                    IntegerOverflowFlagsAttr::default(),
                );
                inserter.append_op(ctx, &next_stride);
                running_stride = Stride::Dynamic(next_stride.get_result(ctx));
                strides.push(next_stride.get_result(ctx));
            }
            (Stride::Static(s), Dimension::Dynamic) => {
                let s_val = IndexConstantOp::new(ctx, s);
                inserter.append_op(ctx, &s_val);
                let next_stride = MulOp::new_with_overflow_flag(
                    ctx,
                    s_val.get_result(ctx),
                    *size_val,
                    IntegerOverflowFlagsAttr::default(),
                );
                inserter.append_op(ctx, &next_stride);
                running_stride = Stride::Dynamic(next_stride.get_result(ctx));
                strides.push(next_stride.get_result(ctx));
            }
            (Stride::Dynamic(v), Dimension::Dynamic) => {
                let next_stride = MulOp::new_with_overflow_flag(
                    ctx,
                    v,
                    *size_val,
                    IntegerOverflowFlagsAttr::default(),
                );
                inserter.append_op(ctx, &next_stride);
                running_stride = Stride::Dynamic(next_stride.get_result(ctx));
                strides.push(next_stride.get_result(ctx));
            }
        }
    }
    // The last stride computed in the loop is actually the total number of elements in the memref.
    let total_elements = strides.pop().unwrap();
    // The strides are computed in reverse order, so reverse them back.
    strides.reverse();

    (sizes, strides, total_elements)
}

/// Represents the fields of a memref descriptor.
pub struct Descriptor {
    pub allocated_ptr: Value,
    pub aligned_ptr: Value,
    pub offset: Value,
    pub sizes: Vec<Value>,
    pub strides: Vec<Value>,
}

/// Unpack / extract the allocated pointer, aligned pointer, offset, sizes, and strides from a memref descriptor.
pub fn unpack_descriptor(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
) -> Descriptor {
    let allocated_ptr = unpack_allocated_ptr(ctx, inserter, descriptor);
    let aligned_ptr = unpack_aligned_ptr(ctx, inserter, descriptor);
    let offset = unpack_offset(ctx, inserter, descriptor);
    let sizes = unpack_sizes(ctx, inserter, descriptor);
    let strides = unpack_strides(ctx, inserter, descriptor);
    Descriptor {
        allocated_ptr,
        aligned_ptr,
        offset,
        sizes,
        strides,
    }
}

/// Unpack / extract the sizes from a memref descriptor.
pub fn unpack_sizes(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
) -> Vec<Value> {
    let rank = {
        let ty = descriptor.get_type(ctx).deref(ctx);
        let struct_ty = ty
            .downcast_ref::<StructType>()
            .expect("Expected an LLVM struct type for the memref descriptor");
        let arr_ty = struct_ty.field_type(3).deref(ctx);
        let arr_ty = arr_ty
            .downcast_ref::<pliron_llvm::types::ArrayType>()
            .expect("Expected an array type for the sizes field in the memref descriptor");
        arr_ty.size() as usize
    };
    let sizes_arr = ExtractValueOp::new(ctx, descriptor, vec![3])
        .expect("Expected sizes array field in memref descriptor");
    inserter.append_op(ctx, &sizes_arr);
    let sizes_arr_val = sizes_arr.get_result(ctx);

    (0..rank)
        .map(|dim| {
            let size = ExtractValueOp::new(ctx, sizes_arr_val, vec![dim.try_into().unwrap()])
                .expect("Expected size field in sizes array");
            inserter.append_op(ctx, &size);
            size.get_result(ctx)
        })
        .collect()
}

/// Unpack / extract the strides from a memref descriptor.
pub fn unpack_strides(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
) -> Vec<Value> {
    let rank = {
        let ty = descriptor.get_type(ctx).deref(ctx);
        let struct_ty = ty
            .downcast_ref::<StructType>()
            .expect("Expected an LLVM struct type for the memref descriptor");
        let arr_ty = struct_ty.field_type(4).deref(ctx);
        let arr_ty = arr_ty
            .downcast_ref::<pliron_llvm::types::ArrayType>()
            .expect("Expected an array type for the strides field in the memref descriptor");
        arr_ty.size() as usize
    };
    let strides_arr = ExtractValueOp::new(ctx, descriptor, vec![4])
        .expect("Expected strides array field in memref descriptor");
    inserter.append_op(ctx, &strides_arr);
    let strides_array_val = strides_arr.get_result(ctx);
    (0..rank)
        .map(|dim| {
            let stride = ExtractValueOp::new(ctx, strides_array_val, vec![dim.try_into().unwrap()])
                .expect("Expected stride field in strides array");
            inserter.append_op(ctx, &stride);
            stride.get_result(ctx)
        })
        .collect()
}

/// Unpack / extract the allocated pointer from a memref descriptor.
pub fn unpack_allocated_ptr(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
) -> Value {
    let extracter = ExtractValueOp::new(ctx, descriptor, vec![0])
        .expect("Expected allocated pointer field in memref descriptor");
    inserter.append_op(ctx, &extracter);
    extracter.get_result(ctx)
}

/// Unpack / extract the aligned pointer from a memref descriptor.
pub fn unpack_aligned_ptr(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
) -> Value {
    let extracter = ExtractValueOp::new(ctx, descriptor, vec![1])
        .expect("Expected aligned pointer field in memref descriptor");
    inserter.append_op(ctx, &extracter);
    extracter.get_result(ctx)
}

/// Unpack / extract the offset from a memref descriptor.
pub fn unpack_offset(ctx: &mut Context, inserter: &mut dyn Inserter, descriptor: Value) -> Value {
    let extracter = ExtractValueOp::new(ctx, descriptor, vec![2])
        .expect("Expected offset field in memref descriptor");

    inserter.append_op(ctx, &extracter);
    extracter.get_result(ctx)
}

/// Unpack / extract the size of a specific dimension from a memref descriptor.
pub fn unpack_size(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
    dim: usize,
) -> Value {
    let sizes_arr = ExtractValueOp::new(ctx, descriptor, vec![3])
        .expect("Expected sizes array field in memref descriptor");
    inserter.append_op(ctx, &sizes_arr);
    let size = ExtractValueOp::new(
        ctx,
        sizes_arr.get_result(ctx),
        vec![dim.try_into().unwrap()],
    )
    .expect("Expected size field in sizes array");
    inserter.append_op(ctx, &size);
    size.get_result(ctx)
}

/// Unpack / extract the size of a specific dimension from a memref descriptor,
/// where the dimension index is provided as an SSA value.
///
/// This helper materializes the descriptor value into a stack slot and uses
/// GEP + Load to access `descriptor.sizes[dim_idx]`.
///
/// We can't just re-use [unpack_size] because `ExtractValueOp` requires constant indices.
pub fn unpack_size_dynamic_dim_idx(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
    dim_idx: Value,
) -> Value {
    let descriptor_ty = descriptor.get_type(ctx);
    let size_elem_ty = IntegerType::get(ctx, 64, Signedness::Signless);
    let one_attr = IntegerAttr::new(
        size_elem_ty,
        APInt::from_usize(
            1,
            NonZero::new(size_elem_ty.deref(ctx).width() as usize).unwrap(),
        ),
    );
    let one = ConstantOp::new(ctx, Box::new(one_attr));
    inserter.append_op(ctx, &one);

    let descriptor_slot = AllocaOp::new(ctx, descriptor_ty, one.get_result(ctx));
    inserter.append_op(ctx, &descriptor_slot);
    let descriptor_slot_ptr = descriptor_slot.get_result(ctx);

    let store_descriptor = StoreOp::new(ctx, descriptor, descriptor_slot_ptr);
    inserter.append_op(ctx, &store_descriptor);

    let size_ptr = GetElementPtrOp::new(
        ctx,
        descriptor_slot_ptr,
        vec![
            GepIndex::Constant(0),
            GepIndex::Constant(3),
            GepIndex::Value(dim_idx),
        ],
        descriptor_ty,
    );
    inserter.append_op(ctx, &size_ptr);

    let load_size = LoadOp::new(ctx, size_ptr.get_result(ctx), size_elem_ty.into());
    inserter.append_op(ctx, &load_size);
    load_size.get_result(ctx)
}

/// Unpack / extract the stride of a specific dimension from a memref descriptor.
pub fn unpack_stride(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    descriptor: Value,
    dim: usize,
) -> Value {
    let strides_arr = ExtractValueOp::new(ctx, descriptor, vec![4])
        .expect("Expected strides array field in memref descriptor");
    inserter.append_op(ctx, &strides_arr);
    let stride = ExtractValueOp::new(
        ctx,
        strides_arr.get_result(ctx),
        vec![dim.try_into().unwrap()],
    )
    .expect("Expected stride field in strides array");
    inserter.append_op(ctx, &stride);
    stride.get_result(ctx)
}

/// Pack the allocated pointer, aligned pointer, offset, sizes, and strides into a memref descriptor.
pub fn pack_descriptor(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    memref_type: TypedHandle<RankedMemrefType>,
    descriptor: Descriptor,
) -> Result<Value> {
    let llvm_struct_ty = memref_type.deref(ctx).convert(ctx)?;
    let llvm_struct_ty_detail = TypedHandle::<StructType>::from_handle(llvm_struct_ty, ctx)
        .expect("Expected LLVM struct type for the memref descriptor");

    // Begin with an undef value of the LLVM struct type, and insert the fields one by one using InsertValueOp.
    let undef_struct = UndefOp::new(ctx, llvm_struct_ty);
    inserter.append_op(ctx, &undef_struct);

    // Insert the allocated pointer
    let allocated_ptr_struct = InsertValueOp::new(
        ctx,
        undef_struct.get_result(ctx),
        descriptor.allocated_ptr,
        vec![0],
    );
    inserter.append_op(ctx, &allocated_ptr_struct);

    // Insert the aligned pointer
    let aligned_ptr_struct = InsertValueOp::new(
        ctx,
        allocated_ptr_struct.get_result(ctx),
        descriptor.aligned_ptr,
        vec![1],
    );
    inserter.append_op(ctx, &aligned_ptr_struct);

    // Insert the offset
    let offset_struct = InsertValueOp::new(
        ctx,
        aligned_ptr_struct.get_result(ctx),
        descriptor.offset,
        vec![2],
    );
    inserter.append_op(ctx, &offset_struct);

    // Insert the sizes array
    let sizes_array_ty = llvm_struct_ty_detail.deref(ctx).field_type(3);
    // Start with an undef value for the sizes array, and insert each size into it.
    // Then insert the sizes array into the struct.
    let sizes_array = UndefOp::new(ctx, sizes_array_ty);
    inserter.append_op(ctx, &sizes_array);
    let mut sizes_array = sizes_array.get_result(ctx);
    for (i, size) in descriptor.sizes.into_iter().enumerate() {
        let insert_op = InsertValueOp::new(ctx, sizes_array, size, vec![i.try_into().unwrap()]);
        sizes_array = insert_op.get_result(ctx);
        inserter.append_op(ctx, &insert_op);
    }
    let sizes_struct = InsertValueOp::new(ctx, offset_struct.get_result(ctx), sizes_array, vec![3]);
    inserter.append_op(ctx, &sizes_struct);

    // Insert the strides array
    let strides_array_ty = llvm_struct_ty_detail.deref(ctx).field_type(4);
    // Start with an undef value for the strides array, and insert each stride into it.
    // Then insert the strides array into the struct.
    let strides_array = UndefOp::new(ctx, strides_array_ty);
    inserter.append_op(ctx, &strides_array);
    let mut strides_array = strides_array.get_result(ctx);
    for (i, stride) in descriptor.strides.into_iter().enumerate() {
        let insert_op = InsertValueOp::new(ctx, strides_array, stride, vec![i.try_into().unwrap()]);
        strides_array = insert_op.get_result(ctx);
        inserter.append_op(ctx, &insert_op);
    }
    let strides_struct =
        InsertValueOp::new(ctx, sizes_struct.get_result(ctx), strides_array, vec![4]);
    inserter.append_op(ctx, &strides_struct);

    Ok(strides_struct.get_result(ctx))
}

/// Performs the index computation to get to the element at `indices` of the
/// memref descriptor pointed to by `memref`. The indices are linearized as:
///   `base_offset + index_0 * stride_0 + ... + index_n * stride_n`.
pub fn get_strided_element_ptr(
    ctx: &mut Context,
    inserter: &mut dyn Inserter,
    elem_ty: TypeHandle,
    memref: Value,
    indices: Vec<Value>,
) -> Value {
    let base_aligned_ptr = unpack_aligned_ptr(ctx, inserter, memref);
    let strides = unpack_strides(ctx, inserter, memref);
    let offset = unpack_offset(ctx, inserter, memref);

    let offsetted_ptr = GetElementPtrOp::new(
        ctx,
        base_aligned_ptr,
        vec![GepIndex::Value(offset)],
        elem_ty,
    );
    inserter.append_op(ctx, &offsetted_ptr);
    let offsetted_ptr = offsetted_ptr.get_result(ctx);
    let products: Vec<_> = indices
        .into_iter()
        .zip(strides)
        .map(|(index, stride)| {
            let index_offset = MulOp::new_with_overflow_flag(
                ctx,
                stride,
                index,
                IntegerOverflowFlagsAttr::default(),
            );
            inserter.append_op(ctx, &index_offset);
            index_offset.get_result(ctx)
        })
        .collect();
    let indexed_offset = products
        .into_iter()
        .reduce(|sum_of_products, product| {
            let sum = AddOp::new_with_overflow_flag(
                ctx,
                product,
                sum_of_products,
                IntegerOverflowFlagsAttr::default(),
            );
            inserter.append_op(ctx, &sum);
            sum.get_result(ctx)
        })
        .expect("Zero rank not handled");

    let final_gep = GetElementPtrOp::new(
        ctx,
        offsetted_ptr,
        vec![GepIndex::Value(indexed_offset)],
        elem_ty,
    );
    inserter.append_op(ctx, &final_gep);
    final_gep.get_result(ctx)
}

#[cfg(test)]
mod tests {
    use expect_test::expect;
    use pliron::{
        builtin::{
            op_interfaces::{OneResultInterface, SingleBlockRegionInterface},
            ops::ModuleOp,
            types::FP32Type,
        },
        context::Context,
        irbuild::{
            inserter::{IRInserter, Inserter},
            listener::DummyListener,
        },
        op::Op,
        printable::Printable,
    };
    use pliron_common_dialects::index::ops::IndexConstantOp;
    use pliron_llvm::{
        ops::{FuncOp, ReturnOp},
        types::{FuncType, VoidType},
    };

    use crate::memref::{type_interfaces::Dimension, types::RankedMemrefType};

    /// Test the `compute_sizes_strides` function by creating a memref type
    /// and checking that the correct size and stride computations are generated.
    #[test]
    fn test_sizes_strides_1() {
        let ctx = &mut Context::new();

        // Create a ranked memref type: memref<4x8xf32>
        let fp32 = FP32Type::get(ctx);
        let memref_type = RankedMemrefType::get(
            ctx,
            fp32.into(),
            vec![Dimension::Static(4), Dimension::Static(8)],
        );

        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let func_ty = FuncType::get(ctx, VoidType::get(ctx).into(), vec![], false);
        let test_fn = FuncOp::new(ctx, "test_fn".try_into().unwrap(), func_ty);
        test_fn
            .get_operation()
            .insert_at_front(module.get_body(ctx, 0), ctx);
        let entry = test_fn.get_or_create_entry_block(ctx);
        let mut inserter = IRInserter::<DummyListener>::default();
        inserter.set_insertion_point_to_block_end(entry);

        let (sizes, strides, total_elements) =
            super::compute_sizes_strides(ctx, &mut inserter, memref_type, vec![]);
        let ret_op = ReturnOp::new(ctx, None);
        inserter.append_op(ctx, &ret_op);
        expect![[r#"
            builtin.module @test_module 
            {
              ^block1v1():
                llvm.func @test_fn: llvm.func <llvm.void () variadic = false>
                  [] 
                {
                  ^entry_block2v1():
                    v0 = index.constant <index.constant 4> : index.index ;
                    v1 = index.constant <index.constant 8> : index.index ;
                    v2 = index.constant <index.constant 1> : index.index ;
                    v3 = index.constant <index.constant 8> : index.index ;
                    v4 = index.constant <index.constant 32> : index.index ;
                    llvm.return 
                }
            }"#]]
        .assert_eq(&module.disp(ctx).to_string());
        let sizes_printed = sizes
            .iter()
            .map(|v| v.disp(ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        expect!["v0, v1"].assert_eq(&sizes_printed);
        let strides_printed = strides
            .iter()
            .map(|v| v.disp(ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        expect!["v3, v2"].assert_eq(&strides_printed);
        expect!["v4"].assert_eq(&total_elements.disp(ctx).to_string());
    }

    /// Test the `compute_sizes_strides` function with a memref type that has dynamic dimensions.
    #[test]
    fn test_sizes_strides_2() {
        let ctx = &mut Context::new();

        // Create a ranked memref type: memref<?x?xf32>
        let fp32 = FP32Type::get(ctx);
        let memref_type = RankedMemrefType::get(
            ctx,
            fp32.into(),
            vec![Dimension::Dynamic, Dimension::Dynamic],
        );

        let module = ModuleOp::new(ctx, "test_module".try_into().unwrap());
        let func_ty = FuncType::get(ctx, VoidType::get(ctx).into(), vec![], false);
        let test_fn = FuncOp::new(ctx, "test_fn".try_into().unwrap(), func_ty);
        test_fn
            .get_operation()
            .insert_at_front(module.get_body(ctx, 0), ctx);
        let entry = test_fn.get_or_create_entry_block(ctx);
        let mut inserter = IRInserter::<DummyListener>::default();
        inserter.set_insertion_point_to_block_start(entry);

        let dyn_size1 = IndexConstantOp::new(ctx, 4);
        let dyn_size2 = IndexConstantOp::new(ctx, 8);
        inserter.append_op(ctx, &dyn_size1);
        inserter.append_op(ctx, &dyn_size2);
        inserter.set_insertion_point_to_block_end(entry);

        // Use dynamic sizes of 4 and 8 for the two dimensions.
        let (sizes, strides, total_elements) = super::compute_sizes_strides(
            ctx,
            &mut inserter,
            memref_type,
            vec![dyn_size1.get_result(ctx), dyn_size2.get_result(ctx)],
        );
        let ret_op = ReturnOp::new(ctx, None);
        inserter.append_op(ctx, &ret_op);
        expect![[r#"
            builtin.module @test_module 
            {
              ^block1v1():
                llvm.func @test_fn: llvm.func <llvm.void () variadic = false>
                  [] 
                {
                  ^entry_block2v1():
                    v0 = index.constant <index.constant 4> : index.index ;
                    v1 = index.constant <index.constant 8> : index.index ;
                    v2 = index.constant <index.constant 1> : index.index ;
                    v3 = index.constant <index.constant 1> : index.index ;
                    v4 = llvm.mul v3, v1 <{nsw=false,nuw=false}>: index.index ;
                    v5 = llvm.mul v4, v0 <{nsw=false,nuw=false}>: index.index ;
                    llvm.return 
                }
            }"#]]
        .assert_eq(&module.disp(ctx).to_string());
        let sizes_printed = sizes
            .iter()
            .map(|v| v.disp(ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        expect!["v0, v1"].assert_eq(&sizes_printed);
        let strides_printed = strides
            .iter()
            .map(|v| v.disp(ctx).to_string())
            .collect::<Vec<_>>()
            .join(", ");
        expect!["v4, v2"].assert_eq(&strides_printed);
        expect!["v5"].assert_eq(&total_elements.disp(ctx).to_string());
    }
}
