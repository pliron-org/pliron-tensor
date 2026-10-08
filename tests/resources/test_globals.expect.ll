; ModuleID = 'test_module'
source_filename = "test_module"

@counter = private global [16 x i8] zeroinitializer, align 8
@defined_elsewhere = external global [16 x i8], align 8
@odd = private constant [3 x i24] [i24 11, i24 22, i24 33], align 4

define i64 @bump(i64 %0) {
entry_block2v1:
  %old_v37 = load i64, ptr @counter, align 4
  %new_v4 = add i64 %old_v37, %0
  store i64 %new_v4, ptr @counter, align 4
  ret i64 %new_v4
}

define i24 @read(i64 %0) {
entry_block3v1:
  %v97 = mul i64 %0, 1
  %v98 = add i64 0, %v97
  %v75 = getelementptr i24, ptr @odd, i64 %v98
  %res_v76 = load i24, ptr %v75, align 4
  ret i24 %res_v76
}
