; ModuleID = 'test_module'
source_filename = "test_module"

define i64 @test_alloc_generate(i64 %0, i64 %1) {
entry_block2v1:
  %v20 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 256
  %v21 = call ptr @malloc(i64 %v20)
  %v24 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v21, 0
  %v25 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v24, ptr %v21, 1
  %v27 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v25, i64 0, 2
  %v33 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v27, [2 x i64] [i64 16, i64 16], 3
  %memref_v39 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v33, [2 x i64] [i64 16, i64 1], 4
  %v40 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 3
  %v41 = extractvalue [2 x i64] %v40, 0
  %v43 = extractvalue [2 x i64] %v40, 1
  br label %for_op_header_block9v1

for_op_header_block9v1:                           ; preds = %entry_split_block6v1, %entry_block2v1
  %v107 = phi i64 [ 0, %entry_block2v1 ], [ %v109, %entry_split_block6v1 ]
  %v108 = icmp ult i64 %v107, %v41
  br i1 %v108, label %entry_block5v1, label %entry_split_block8v1

entry_block5v1:                                   ; preds = %for_op_header_block9v1
  %iv_v95 = phi i64 [ %v107, %for_op_header_block9v1 ]
  br label %for_op_header_block7v1

for_op_header_block7v1:                           ; preds = %entry_block4v1, %entry_block5v1
  %v104 = phi i64 [ 0, %entry_block5v1 ], [ %v106, %entry_block4v1 ]
  %v105 = icmp ult i64 %v104, %v43
  br i1 %v105, label %entry_block3v3, label %entry_split_block6v1

entry_block3v3:                                   ; preds = %for_op_header_block7v1
  %iv_v94 = phi i64 [ %v104, %for_op_header_block7v1 ]
  br label %entry_block4v1

entry_block4v1:                                   ; preds = %entry_block3v3
  %i_v47 = phi i64 [ %iv_v95, %entry_block3v3 ]
  %j_v48 = phi i64 [ %iv_v94, %entry_block3v3 ]
  %sum_v7 = add i64 %i_v47, %j_v48
  %v64 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 1
  %v65 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 4
  %v66 = extractvalue [2 x i64] %v65, 0
  %v68 = extractvalue [2 x i64] %v65, 1
  %v70 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 2
  %v96 = mul i64 %i_v47, %v66
  %v97 = add i64 %v70, %v96
  %v98 = mul i64 %j_v48, %v68
  %v99 = add i64 %v97, %v98
  %v77 = getelementptr i64, ptr %v64, i64 %v99
  store i64 %sum_v7, ptr %v77, align 4
  %v106 = add i64 %iv_v94, 1
  br label %for_op_header_block7v1

entry_split_block6v1:                             ; preds = %for_op_header_block7v1
  %v109 = add i64 %iv_v95, 1
  br label %for_op_header_block9v1

entry_split_block8v1:                             ; preds = %for_op_header_block9v1
  %v49 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 1
  %v50 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 4
  %v51 = extractvalue [2 x i64] %v50, 0
  %v53 = extractvalue [2 x i64] %v50, 1
  %v55 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %memref_v39, 2
  %v100 = mul i64 %0, %v51
  %v101 = add i64 %v55, %v100
  %v102 = mul i64 %1, %v53
  %v103 = add i64 %v101, %v102
  %v62 = getelementptr i64, ptr %v49, i64 %v103
  %result_v63 = load i64, ptr %v62, align 4
  ret i64 %result_v63
}

declare ptr @malloc(i64)
