//! Test conversions of memref operations to CF / LLVM dialect.

use pliron::{
    builtin::ops::ModuleOp,
    combine::Parser,
    context::Context,
    init_env_logger_for_tests, input_error_noloc,
    irbuild::dialect_conversion::apply_dialect_conversion,
    irfmt::parsers::spaced,
    location,
    op::verify_op,
    operation::Operation,
    parsable::{self, state_stream_from_iterator},
    printable::Printable,
    result::ExpectOk,
};

use pliron_common_dialects::cf::to_llvm::CFToLLVM;
use pliron_llvm::llvm_sys::{core::LLVMContext, lljit::LLVMLLJIT, target::initialize_native};

use expect_test::expect;
use pliron_tensor::memref::conversions::MemrefToCF;
#[test]
fn test_alloc_generate() {
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_alloc_generate: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_res: builtin.integer i64, j_res: builtin.integer i64):
                    memref = memref.alloc : memref.ranked<16 x 16 : builtin.integer i64>;
                    memref.generate memref {
                      ^entry(i : index.index, j : index.index):
                        i_int = index.to_integer i to builtin.integer i64;
                        j_int = index.to_integer j to builtin.integer i64;
                        sum = llvm.add i_int, j_res <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield sum
                    };
                    i_index = index.from_integer i_res : index.index;
                    j_index = index.from_integer j_res : index.index;
                    result = memref.load memref[i_index, j_index]: builtin.integer i64;
                    llvm.return result
                }
            }
            "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    expect![[r#"
        builtin.module @test_module 
        {
          ^entry_block3v1() !0:
            llvm.func @test_alloc_generate: llvm.func <builtin.integer i64(builtin.integer i64, builtin.integer i64) variadic = false>
              [] 
            {
              ^entry_block2v1(i_res_v13: builtin.integer i64, j_res_v14: builtin.integer i64) !1:
                v66 = builtin.constant <builtin.integer <16: i64>> : builtin.integer i64;
                v67 = builtin.constant <builtin.integer <16: i64>> : builtin.integer i64;
                v68 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                v69 = builtin.constant <builtin.integer <16: i64>> : builtin.integer i64;
                v70 = builtin.constant <builtin.integer <256: i64>> : builtin.integer i64;
                v20 = llvm.zero : llvm.ptr (0);
                v21 = llvm.gep <builtin.integer i64> (v20)[Constant(1)] : llvm.ptr (0);
                v22 = llvm.ptrtoint v21 to builtin.integer i64;
                v23 = llvm.mul v22, v70 <{nsw=false,nuw=false}>: builtin.integer i64;
                v24 = llvm.call @malloc (v23) : llvm.func <llvm.ptr (0)(builtin.integer i64) variadic = false>;
                v71 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                v26 = llvm.undef : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v27 = llvm.insert_value v26[0], v24 : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v28 = llvm.insert_value v27[1], v24 : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v29 = llvm.insert_value v28[2], v71 : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v30 = llvm.undef : llvm.array [2 x builtin.integer i64];
                v31 = llvm.insert_value v30[0], v66 : llvm.array [2 x builtin.integer i64];
                v32 = llvm.insert_value v31[1], v67 : llvm.array [2 x builtin.integer i64];
                v33 = llvm.insert_value v29[3], v32 : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v34 = llvm.undef : llvm.array [2 x builtin.integer i64];
                v35 = llvm.insert_value v34[0], v69 : llvm.array [2 x builtin.integer i64];
                v36 = llvm.insert_value v35[1], v68 : llvm.array [2 x builtin.integer i64];
                v37 = llvm.insert_value v33[4], v36 : llvm.struct <{ llvm.ptr (0), llvm.ptr (0), builtin.integer i64, llvm.array [2 x builtin.integer i64], llvm.array [2 x builtin.integer i64] }>;
                v38 = llvm.extract_value v37[3] : llvm.array [2 x builtin.integer i64];
                v39 = llvm.extract_value v38[0] : builtin.integer i64;
                v40 = llvm.extract_value v38[1] : builtin.integer i64;
                v72 = builtin.constant <builtin.integer <0: i64>> : builtin.integer i64;
                v73 = builtin.constant <builtin.integer <1: i64>> : builtin.integer i64;
                llvm.br ^for_op_header_block10v1(v72)

              ^for_op_header_block10v1(v79: builtin.integer i64) !2:
                v80 = llvm.icmp v79 <ULT> v39 : builtin.integer i1;
                llvm.cond_br if v80 ^entry_block6v1(v79) else ^entry_split_block9v1()

              ^entry_block6v1(iv_v75: builtin.integer i64) !3:
                llvm.br ^for_op_header_block8v1(v72)

              ^for_op_header_block8v1(v76: builtin.integer i64):
                v77 = llvm.icmp v76 <ULT> v40 : builtin.integer i1;
                llvm.cond_br if v77 ^entry_block5v1(v76) else ^entry_split_block7v1()

              ^entry_block5v1(iv_v74: builtin.integer i64) !4:
                llvm.br ^entry_block4v1(iv_v75, iv_v74)

              ^entry_block4v1(v43: builtin.integer i64, v44: builtin.integer i64):
                llvm.br ^entry_block1v1(v43, v44)

              ^entry_block1v1(i_v7: builtin.integer i64, j_v8: builtin.integer i64) !5:
                sum_v6 = llvm.add i_v7, j_res_v14 <{nsw=false,nuw=false}>: builtin.integer i64 !6;
                v56 = llvm.extract_value v37[1] : llvm.ptr (0);
                v57 = llvm.extract_value v37[4] : llvm.array [2 x builtin.integer i64];
                v58 = llvm.extract_value v57[0] : builtin.integer i64;
                v59 = llvm.extract_value v57[1] : builtin.integer i64;
                v60 = llvm.extract_value v37[2] : builtin.integer i64;
                v61 = llvm.gep <builtin.integer i64> (v56, v60)[OperandIdx(1)] : llvm.ptr (0);
                v62 = llvm.mul v58, i_v7 <{nsw=false,nuw=false}>: builtin.integer i64;
                v63 = llvm.mul v59, j_v8 <{nsw=false,nuw=false}>: builtin.integer i64;
                v64 = llvm.add v63, v62 <{nsw=false,nuw=false}>: builtin.integer i64;
                v65 = llvm.gep <builtin.integer i64> (v61, v64)[OperandIdx(1)] : llvm.ptr (0);
                llvm.store *v65 <- sum_v6  !7;
                v78 = llvm.add iv_v74, v73 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block8v1(v78)

              ^entry_split_block7v1():
                v81 = llvm.add iv_v75, v73 <{nsw=false,nuw=false}>: builtin.integer i64;
                llvm.br ^for_op_header_block10v1(v81)

              ^entry_split_block9v1():
                v45 = llvm.extract_value v37[1] : llvm.ptr (0);
                v46 = llvm.extract_value v37[4] : llvm.array [2 x builtin.integer i64];
                v47 = llvm.extract_value v46[0] : builtin.integer i64;
                v48 = llvm.extract_value v46[1] : builtin.integer i64;
                v49 = llvm.extract_value v37[2] : builtin.integer i64;
                v50 = llvm.gep <builtin.integer i64> (v45, v49)[OperandIdx(1)] : llvm.ptr (0);
                v51 = llvm.mul v47, i_res_v13 <{nsw=false,nuw=false}>: builtin.integer i64;
                v52 = llvm.mul v48, j_res_v14 <{nsw=false,nuw=false}>: builtin.integer i64;
                v53 = llvm.add v52, v51 <{nsw=false,nuw=false}>: builtin.integer i64;
                v54 = llvm.gep <builtin.integer i64> (v50, v53)[OperandIdx(1)] : llvm.ptr (0);
                v55 = llvm.load v54  : builtin.integer i64 !8;
                llvm.return v55 !9
            } !10;
            llvm.func @malloc: llvm.func <llvm.ptr (0)(builtin.integer i64) variadic = false>
              []
        }"#]].assert_eq(&print_parsed);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    expect![[r#"
        ; ModuleID = 'test_module'
        source_filename = "test_module"

        define i64 @test_alloc_generate(i64 %0, i64 %1) {
        entry_block2v1:
          %v23 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 256
          %v24 = call ptr @malloc(i64 %v23)
          %v27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v24, 0
          %v28 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v27, ptr %v24, 1
          %v29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v28, i64 0, 2
          %v33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v29, [2 x i64] [i64 16, i64 16], 3
          %v37 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v33, [2 x i64] [i64 16, i64 1], 4
          %v38 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 3
          %v39 = extractvalue [2 x i64] %v38, 0
          %v40 = extractvalue [2 x i64] %v38, 1
          br label %for_op_header_block10v1

        for_op_header_block10v1:                          ; preds = %entry_split_block7v1, %entry_block2v1
          %v79 = phi i64 [ 0, %entry_block2v1 ], [ %v81, %entry_split_block7v1 ]
          %v80 = icmp ult i64 %v79, %v39
          br i1 %v80, label %entry_block6v1, label %entry_split_block9v1

        entry_block6v1:                                   ; preds = %for_op_header_block10v1
          %iv_v75 = phi i64 [ %v79, %for_op_header_block10v1 ]
          br label %for_op_header_block8v1

        for_op_header_block8v1:                           ; preds = %entry_block1v1, %entry_block6v1
          %v76 = phi i64 [ 0, %entry_block6v1 ], [ %v78, %entry_block1v1 ]
          %v77 = icmp ult i64 %v76, %v40
          br i1 %v77, label %entry_block5v1, label %entry_split_block7v1

        entry_block5v1:                                   ; preds = %for_op_header_block8v1
          %iv_v74 = phi i64 [ %v76, %for_op_header_block8v1 ]
          br label %entry_block4v1

        entry_block4v1:                                   ; preds = %entry_block5v1
          %v43 = phi i64 [ %iv_v75, %entry_block5v1 ]
          %v44 = phi i64 [ %iv_v74, %entry_block5v1 ]
          br label %entry_block1v1

        entry_block1v1:                                   ; preds = %entry_block4v1
          %i_v7 = phi i64 [ %v43, %entry_block4v1 ]
          %j_v8 = phi i64 [ %v44, %entry_block4v1 ]
          %sum_v6 = add i64 %i_v7, %1
          %v56 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 1
          %v57 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 4
          %v58 = extractvalue [2 x i64] %v57, 0
          %v59 = extractvalue [2 x i64] %v57, 1
          %v60 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 2
          %v61 = getelementptr i64, ptr %v56, i64 %v60
          %v62 = mul i64 %v58, %i_v7
          %v63 = mul i64 %v59, %j_v8
          %v64 = add i64 %v63, %v62
          %v65 = getelementptr i64, ptr %v61, i64 %v64
          store i64 %sum_v6, ptr %v65, align 4
          %v78 = add i64 %iv_v74, 1
          br label %for_op_header_block8v1

        entry_split_block7v1:                             ; preds = %for_op_header_block8v1
          %v81 = add i64 %iv_v75, 1
          br label %for_op_header_block10v1

        entry_split_block9v1:                             ; preds = %for_op_header_block10v1
          %v45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 1
          %v46 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 4
          %v47 = extractvalue [2 x i64] %v46, 0
          %v48 = extractvalue [2 x i64] %v46, 1
          %v49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v37, 2
          %v50 = getelementptr i64, ptr %v45, i64 %v49
          %v51 = mul i64 %v47, %0
          %v52 = mul i64 %v48, %1
          %v53 = add i64 %v52, %v51
          %v54 = getelementptr i64, ptr %v50, i64 %v53
          %v55 = load i64, ptr %v54, align 4
          ret i64 %v55
        }

        declare ptr @malloc(i64)
    "#]].assert_eq(&llvm_ir.to_string());

    // Let's try and execute this function
    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_alloc_generate")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..16 {
        for j in 0..16 {
            let result = f(i, j);
            assert_eq!(result, i + j);
        }
    }
}

#[test]
fn test_memref_dim_dynamic_index() {
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
          llvm.func @test_memref_dim: llvm.func <builtin.integer i64 (builtin.integer i64) variadic = false> [] {
            ^entry(dim_arg: builtin.integer i64):
            memref = memref.alloc : memref.ranked<16 x 32 : builtin.integer i64>;
            dim_idx = index.from_integer dim_arg : index.index;
            dim = memref.dim memref, dim_idx : index.index;
            dim_int = index.to_integer dim to builtin.integer i64;
            llvm.return dim_int
          }
        }
        "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    assert!(!print_parsed.contains("memref.dim"));
    assert!(print_parsed.contains("llvm.gep"));
    assert!(print_parsed.contains("llvm.load"));

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_memref_dim")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64) -> i64>(symbol_addr) };

    assert_eq!(f(0), 16);
    assert_eq!(f(1), 32);
}

#[test]
fn test_memref_dim_const_index() {
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
          llvm.func @test_memref_dim_const_index: llvm.func <builtin.integer i64 () variadic = false> [] {
            ^entry():
            memref = memref.alloc : memref.ranked<16 x 32 : builtin.integer i64>;
            idx0 = index.constant <index.constant 0> : index.index;
            idx1 = index.constant <index.constant 1> : index.index;
            dim0 = memref.dim memref, idx0 : index.index;
            dim1 = memref.dim memref, idx1 : index.index;
            dim0_i64 = index.to_integer dim0 to builtin.integer i64;
            dim1_i64 = index.to_integer dim1 to builtin.integer i64;
            thousand = builtin.constant <builtin.integer <1000: i64>> : builtin.integer i64;
            scaled = llvm.mul dim0_i64, thousand <{nsw = false, nuw = false}> : builtin.integer i64;
            encoded = llvm.add scaled, dim1_i64 <{nsw = false, nuw = false}> : builtin.integer i64;
            llvm.return encoded
          }
        }
        "#;

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
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    verify_op(&module_op, ctx).expect_ok(ctx);

    let print_parsed = format!("{}", module_op.disp(ctx));
    assert!(!print_parsed.contains("memref.dim"));
    assert!(!print_parsed.contains("llvm.alloca"));
    assert!(!print_parsed.contains("llvm.load"));

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| println!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_memref_dim_const_index")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn() -> i64>(symbol_addr) };

    // Encoded return value = dim0 * 1000 + dim1 = 16 * 1000 + 32
    assert_eq!(f(), 16032);
}

/// Test that `memref.subview` is correctly lowered to CF / LLVM.
/// The function allocates a 2×3 source memref, fills it with `src[i][j] = i*3 + j`,
/// then creates a 2×2 subview with offsets [0, 1] and steps [1, 1], and returns
/// `view[i_arg][j_arg]`.
///
/// Expected: `view[i][j] = src[i][1 + j] = i*3 + j + 1`.
#[test]
fn test_subview() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
            builtin.module @test_module {
              ^entry():
                llvm.func @test_subview: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
                  ^entry(i_arg: builtin.integer i64, j_arg: builtin.integer i64):
                    src = memref.alloc : memref.ranked<2 x 3 : builtin.integer i64>;
                    memref.generate src {
                      ^entry(i : index.index, j : index.index):
                        i_int = index.to_integer i to builtin.integer i64;
                        j_int = index.to_integer j to builtin.integer i64;
                        three = builtin.constant <builtin.integer <3: i64>> : builtin.integer i64;
                        row = llvm.mul i_int, three <{nsw = false, nuw = false}> : builtin.integer i64;
                        val = llvm.add row, j_int <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield val
                    };
                      view = memref.subview src [0, 1] [2, 2] [1, 1] : memref.ranked<2 x 2 : builtin.integer i64>;
                    i_idx = index.from_integer i_arg : index.index;
                    j_idx = index.from_integer j_arg : index.index;
                      result = memref.load view[i_idx, j_idx]: builtin.integer i64;
                    llvm.return result
                }
            }
            "#;

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
    log::debug!("parsed module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!("converted module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_subview")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    // dst[i][j] = src[i][1 + j] = i*3 + (1 + j) = i*3 + j + 1
    for i in 0..2_i64 {
        for j in 0..2_i64 {
            let result = f(i, j);
            assert_eq!(result, i * 3 + j + 1, "f({i}, {j}) = {result}");
        }
    }
}

/// Test that `memref.copy` is correctly lowered to CF / LLVM.
#[test]
fn test_copy() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
          llvm.func @test_copy: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
            ^entry(i_arg: builtin.integer i64, j_arg: builtin.integer i64):
            src = memref.alloc : memref.ranked<2 x 2 : builtin.integer i64>;
            memref.generate src {
              ^entry(i : index.index, j : index.index):
              i_int = index.to_integer i to builtin.integer i64;
              j_int = index.to_integer j to builtin.integer i64;
              ten = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64;
              row = llvm.mul i_int, ten <{nsw = false, nuw = false}> : builtin.integer i64;
              val = llvm.add row, j_int <{nsw = false, nuw = false}> : builtin.integer i64;
              memref.yield val
            };
            dst = memref.alloc : memref.ranked<2 x 2 : builtin.integer i64>;
            memref.copy dst <- src;
            i_idx = index.from_integer i_arg : index.index;
            j_idx = index.from_integer j_arg : index.index;
            result = memref.load dst[i_idx, j_idx]: builtin.integer i64;
            llvm.return result
          }
        }
        "#;

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
    log::debug!("parsed module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!("converted module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_copy")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..2_i64 {
        for j in 0..2_i64 {
            let result = f(i, j);
            assert_eq!(result, i * 10 + j, "f({i}, {j}) = {result}");
        }
    }
}

/// Test that a `memref.copy + memref.subview + memref.copy` insertion sequence
/// is correctly lowered to CF / LLVM.
///
/// The function initializes:
/// - `src[i][j] = i*10 + j` for a 2x2 source memref
/// - `dst[i][j] = 100 + i*4 + j` for a 3x4 destination memref
///
/// Then inserts `src` into `dst` at offsets [1, 1] by:
/// - copying `dst` into `res`
/// - taking `sub = memref.subview res [...]`
/// - copying `src` into `sub`
#[test]
fn test_insert_slice_sequence() {
    init_env_logger_for_tests!();
    let ctx = &mut Context::new();

    let input_ir = r#"
        builtin.module @test_module {
          ^entry():
          llvm.func @test_insert_slice: llvm.func <builtin.integer i64 (builtin.integer i64, builtin.integer i64) variadic = false> [] {
            ^entry(i_arg: builtin.integer i64, j_arg: builtin.integer i64):
            src = memref.alloc : memref.ranked<2 x 2 : builtin.integer i64>;
            memref.generate src {
              ^entry(i : index.index, j : index.index):
              i_int = index.to_integer i to builtin.integer i64;
              j_int = index.to_integer j to builtin.integer i64;
              ten = builtin.constant <builtin.integer <10: i64>> : builtin.integer i64;
              row = llvm.mul i_int, ten <{nsw = false, nuw = false}> : builtin.integer i64;
              val = llvm.add row, j_int <{nsw = false, nuw = false}> : builtin.integer i64;
              memref.yield val
            };
            dst = memref.alloc : memref.ranked<3 x 4 : builtin.integer i64>;
            memref.generate dst {
                      ^entry(di : index.index, dj : index.index):
                        di_int = index.to_integer di to builtin.integer i64;
                        dj_int = index.to_integer dj to builtin.integer i64;
              four = builtin.constant <builtin.integer <4: i64>> : builtin.integer i64;
              hundred = builtin.constant <builtin.integer <100: i64>> : builtin.integer i64;
                        drow = llvm.mul di_int, four <{nsw = false, nuw = false}> : builtin.integer i64;
                        dbase = llvm.add hundred, drow <{nsw = false, nuw = false}> : builtin.integer i64;
                        dval = llvm.add dbase, dj_int <{nsw = false, nuw = false}> : builtin.integer i64;
                        memref.yield dval
            };
            res = memref.alloc : memref.ranked<3 x 4 : builtin.integer i64>;
            one = index.constant <index.constant 1> : index.index;
            memref.copy res <- dst;
            sub = memref.subview res [one, one] [2, 2] [one, one] : memref.ranked<2 x 2 : builtin.integer i64>;
            memref.copy sub <- src;
            i_idx = index.from_integer i_arg : index.index;
            j_idx = index.from_integer j_arg : index.index;
            result = memref.load res[i_idx, j_idx]: builtin.integer i64;
            llvm.return result
          }
        }
        "#;

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
    log::debug!("parsed module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    apply_dialect_conversion(ctx, &mut MemrefToCF, parsed_op).expect_ok(ctx);
    apply_dialect_conversion(ctx, &mut CFToLLVM, parsed_op).expect_ok(ctx);
    log::debug!("converted module:\n{}", module_op.disp(ctx));
    verify_op(&module_op, ctx).expect_ok(ctx);

    let llvm_ctx = LLVMContext::default();
    let llvm_ir = pliron_llvm::to_llvm_ir::convert_module(ctx, &llvm_ctx, module_op).expect_ok(ctx);
    llvm_ir
        .verify()
        .inspect_err(|e| eprintln!("LLVM-IR verification failed: {}", e))
        .unwrap();
    log::debug!("LLVM-IR generated:\n{}", llvm_ir);

    initialize_native().expect("Failed to initialize native target for LLVM execution");
    let jit = LLVMLLJIT::new_with_default_builder().expect("Failed to create LLJIT");
    jit.add_module(llvm_ir)
        .expect("Failed to add module to JIT");
    let symbol_addr = jit
        .lookup_symbol("test_insert_slice")
        .expect("Failed to lookup symbol");
    assert!(symbol_addr != 0);
    let f = unsafe { std::mem::transmute::<u64, fn(i64, i64) -> i64>(symbol_addr) };

    for i in 0..3_i64 {
        for j in 0..4_i64 {
            let result = f(i, j);
            let expected = if (1..3).contains(&i) && (1..3).contains(&j) {
                (i - 1) * 10 + (j - 1)
            } else {
                100 + i * 4 + j
            };
            assert_eq!(result, expected, "f({i}, {j}) = {result}");
        }
    }
}
