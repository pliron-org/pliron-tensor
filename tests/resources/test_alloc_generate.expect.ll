; ModuleID = 'test_module'
source_filename = "test_module"

define i64 @test_alloc_generate(i64 %0, i64 %1) {
entry_block2v1:
  %v19 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 256
  %v20 = call ptr @malloc(i64 %v19)
  %v23 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v20, 0
  %v24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v23, ptr %v20, 1
  %v25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v24, i64 0, 2
  %v29 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v25, [2 x i64] [i64 16, i64 16], 3
  %memref_v33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v29, [2 x i64] [i64 16, i64 1], 4
  %v34 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 3
  %v35 = extractvalue [2 x i64] %v34, 0
  %v36 = extractvalue [2 x i64] %v34, 1
  br label %for_op_header_block9v1

for_op_header_block9v1:                           ; preds = %entry_split_block6v1, %entry_block2v1
  %v83 = phi i64 [ 0, %entry_block2v1 ], [ %v85, %entry_split_block6v1 ]
  %v84 = icmp ult i64 %v83, %v35
  br i1 %v84, label %entry_block5v1, label %entry_split_block8v1

entry_block5v1:                                   ; preds = %for_op_header_block9v1
  %iv_v73 = phi i64 [ %v83, %for_op_header_block9v1 ]
  br label %for_op_header_block7v1

for_op_header_block7v1:                           ; preds = %entry_block4v1, %entry_block5v1
  %v80 = phi i64 [ 0, %entry_block5v1 ], [ %v82, %entry_block4v1 ]
  %v81 = icmp ult i64 %v80, %v36
  br i1 %v81, label %entry_block3v3, label %entry_split_block6v1

entry_block3v3:                                   ; preds = %for_op_header_block7v1
  %iv_v72 = phi i64 [ %v80, %for_op_header_block7v1 ]
  br label %entry_block4v1

entry_block4v1:                                   ; preds = %entry_block3v3
  %i_v39 = phi i64 [ %iv_v73, %entry_block3v3 ]
  %j_v40 = phi i64 [ %iv_v72, %entry_block3v3 ]
  %sum_v7 = add i64 %i_v39, %1
  %v52 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 1
  %v53 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 4
  %v54 = extractvalue [2 x i64] %v53, 0
  %v55 = extractvalue [2 x i64] %v53, 1
  %v56 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 2
  %v57 = getelementptr i64, ptr %v52, i64 %v56
  %v58 = mul i64 %v54, %i_v39
  %v59 = mul i64 %v55, %j_v40
  %v60 = add i64 %v59, %v58
  %v61 = getelementptr i64, ptr %v57, i64 %v60
  store i64 %sum_v7, ptr %v61, align 4
  %v82 = add i64 %iv_v72, 1
  br label %for_op_header_block7v1

entry_split_block6v1:                             ; preds = %for_op_header_block7v1
  %v85 = add i64 %iv_v73, 1
  br label %for_op_header_block9v1

entry_split_block8v1:                             ; preds = %for_op_header_block9v1
  %v41 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 1
  %v42 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 4
  %v43 = extractvalue [2 x i64] %v42, 0
  %v44 = extractvalue [2 x i64] %v42, 1
  %v45 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v33, 2
  %v46 = getelementptr i64, ptr %v41, i64 %v45
  %v47 = mul i64 %v43, %0
  %v48 = mul i64 %v44, %1
  %v49 = add i64 %v48, %v47
  %v50 = getelementptr i64, ptr %v46, i64 %v49
  %result_v51 = load i64, ptr %v50, align 4
  ret i64 %result_v51
}

declare ptr @malloc(i64)
