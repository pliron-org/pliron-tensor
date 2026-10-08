; ModuleID = 'test_module'
source_filename = "test_module"

@__constant_2x3xfp64_8B_0 = private constant [48 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\00@\00\00\00\00\00\00\08@\00\00\00\00\00\00\10@\00\00\00\00\00\00\14@\00\00\00\00\00\00\18@", align 8
@__constant_2x3xfp64_8B_1 = private constant [48 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\00@\00\00\00\00\00\00\08@\00\00\00\00\00\00\10@\00\00\00\00\00\00\14@\00\00\00\00\00\00\18@", align 8
@__constant_2x2xinteger_8B_2 = private constant [32 x i8] c"\0A\00\00\00\00\00\00\00\14\00\00\00\00\00\00\00\1E\00\00\00\00\00\00\00(\00\00\00\00\00\00\00", align 8
@__constant_4x4xfp64_8B_3 = private constant [128 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?", align 8

define double @test_constant_extract(i64 %0, i64 %1) {
entry_block2v1:
  %v382 = mul i64 %0, 3
  %v383 = add i64 0, %v382
  %v384 = mul i64 %1, 1
  %v385 = add i64 %v383, %v384
  %v65 = getelementptr double, ptr @__constant_2x3xfp64_8B_0, i64 %v385
  %res_v66 = load double, ptr %v65, align 8
  ret double %res_v66
}

define void @test_constant_add(ptr %0, ptr %1) {
entry_block3v1:
  %arg_v8 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %v100 = mul i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), 6
  %v101 = call ptr @malloc(i64 %v100)
  %v104 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v101, 0
  %v105 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v104, ptr %v101, 1
  %v107 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v105, i64 0, 2
  %v113 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v107, [2 x i64] [i64 2, i64 3], 3
  %res_v119 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v113, [2 x i64] [i64 3, i64 1], 4
  %v120 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v119, 3
  %v121 = extractvalue [2 x i64] %v120, 0
  %v123 = extractvalue [2 x i64] %v120, 1
  br label %for_op_header_block19v1

for_op_header_block19v1:                          ; preds = %entry_split_block16v1, %entry_block3v1
  %v503 = phi i64 [ 0, %entry_block3v1 ], [ %v505, %entry_split_block16v1 ]
  %v504 = icmp ult i64 %v503, %v121
  br i1 %v504, label %entry_block10v1, label %entry_split_block18v1

entry_block10v1:                                  ; preds = %for_op_header_block19v1
  %iv_v414 = phi i64 [ %v503, %for_op_header_block19v1 ]
  br label %for_op_header_block17v1

for_op_header_block17v1:                          ; preds = %entry_block6v1, %entry_block10v1
  %v500 = phi i64 [ 0, %entry_block10v1 ], [ %v502, %entry_block6v1 ]
  %v501 = icmp ult i64 %v500, %v123
  br i1 %v501, label %entry_block9v1, label %entry_split_block16v1

entry_block9v1:                                   ; preds = %for_op_header_block17v1
  %iv_v413 = phi i64 [ %v500, %for_op_header_block17v1 ]
  br label %entry_block6v1

entry_block6v1:                                   ; preds = %entry_block9v1
  %v127 = phi i64 [ %iv_v414, %entry_block9v1 ]
  %v128 = phi i64 [ %iv_v413, %entry_block9v1 ]
  %v239 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 1
  %v240 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 4
  %v241 = extractvalue [2 x i64] %v240, 0
  %v243 = extractvalue [2 x i64] %v240, 1
  %v245 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 2
  %v415 = mul i64 %v127, %v241
  %v416 = add i64 %v245, %v415
  %v417 = mul i64 %v128, %v243
  %v418 = add i64 %v416, %v417
  %v252 = getelementptr double, ptr %v239, i64 %v418
  %v253 = load double, ptr %v252, align 8
  %v419 = mul i64 %v127, 3
  %v420 = add i64 0, %v419
  %v421 = mul i64 %v128, 1
  %v422 = add i64 %v420, %v421
  %v267 = getelementptr double, ptr @__constant_2x3xfp64_8B_1, i64 %v422
  %v268 = load double, ptr %v267, align 8
  %v131 = fadd double %v253, %v268
  %v269 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v119, 1
  %v270 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v119, 4
  %v271 = extractvalue [2 x i64] %v270, 0
  %v273 = extractvalue [2 x i64] %v270, 1
  %v275 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v119, 2
  %v423 = mul i64 %v127, %v271
  %v424 = add i64 %v275, %v423
  %v425 = mul i64 %v128, %v273
  %v426 = add i64 %v424, %v425
  %v282 = getelementptr double, ptr %v269, i64 %v426
  store double %v131, ptr %v282, align 8
  %v502 = add i64 %iv_v413, 1
  br label %for_op_header_block17v1

entry_split_block16v1:                            ; preds = %for_op_header_block17v1
  %v505 = add i64 %iv_v414, 1
  br label %for_op_header_block19v1

entry_split_block18v1:                            ; preds = %for_op_header_block19v1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v119, ptr %1, align 8
  ret void
}

define void @test_constant_accumulator(ptr %0, ptr %1, ptr %2) {
entry_block4v1:
  %lhs_v14 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %rhs_v15 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %v165 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 4
  %v166 = call ptr @malloc(i64 %v165)
  %v169 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v166, 0
  %v170 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v169, ptr %v166, 1
  %v172 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v170, i64 0, 2
  %v178 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v172, [2 x i64] [i64 2, i64 2], 3
  %res_v184 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v178, [2 x i64] [i64 2, i64 1], 4
  %v185 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 3
  %v186 = extractvalue [2 x i64] %v185, 0
  %v188 = extractvalue [2 x i64] %v185, 1
  br label %for_op_header_block23v1

for_op_header_block23v1:                          ; preds = %entry_split_block20v1, %entry_block4v1
  %v510 = phi i64 [ 0, %entry_block4v1 ], [ %v512, %entry_split_block20v1 ]
  %v511 = icmp ult i64 %v510, %v186
  br i1 %v511, label %entry_block12v1, label %entry_split_block22v1

entry_block12v1:                                  ; preds = %for_op_header_block23v1
  %iv_v455 = phi i64 [ %v510, %for_op_header_block23v1 ]
  br label %for_op_header_block21v1

for_op_header_block21v1:                          ; preds = %entry_block7v1, %entry_block12v1
  %v507 = phi i64 [ 0, %entry_block12v1 ], [ %v509, %entry_block7v1 ]
  %v508 = icmp ult i64 %v507, %v188
  br i1 %v508, label %entry_block11v1, label %entry_split_block20v1

entry_block11v1:                                  ; preds = %for_op_header_block21v1
  %iv_v454 = phi i64 [ %v507, %for_op_header_block21v1 ]
  br label %entry_block7v1

entry_block7v1:                                   ; preds = %entry_block11v1
  %v192 = phi i64 [ %iv_v455, %entry_block11v1 ]
  %v193 = phi i64 [ %iv_v454, %entry_block11v1 ]
  %v456 = mul i64 %v192, 2
  %v457 = add i64 0, %v456
  %v458 = mul i64 %v193, 1
  %v459 = add i64 %v457, %v458
  %v296 = getelementptr i64, ptr @__constant_2x2xinteger_8B_2, i64 %v459
  %v297 = load i64, ptr %v296, align 4
  %v298 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 1
  %v299 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 4
  %v300 = extractvalue [2 x i64] %v299, 0
  %v302 = extractvalue [2 x i64] %v299, 1
  %v304 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 2
  %v460 = mul i64 %v192, %v300
  %v461 = add i64 %v304, %v460
  %v462 = mul i64 %v193, %v302
  %v463 = add i64 %v461, %v462
  %v311 = getelementptr i64, ptr %v298, i64 %v463
  store i64 %v297, ptr %v311, align 4
  %v509 = add i64 %iv_v454, 1
  br label %for_op_header_block21v1

entry_split_block20v1:                            ; preds = %for_op_header_block21v1
  %v512 = add i64 %iv_v455, 1
  br label %for_op_header_block23v1

entry_split_block22v1:                            ; preds = %for_op_header_block23v1
  %v195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 3
  %v196 = extractvalue [2 x i64] %v195, 0
  %v198 = extractvalue [2 x i64] %v195, 1
  %v200 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 3
  %v201 = extractvalue [2 x i64] %v200, 0
  %v203 = extractvalue [2 x i64] %v200, 1
  br label %for_op_header_block29v1

for_op_header_block29v1:                          ; preds = %entry_split_block26v1, %entry_split_block22v1
  %v519 = phi i64 [ 0, %entry_split_block22v1 ], [ %v521, %entry_split_block26v1 ]
  %v520 = icmp ult i64 %v519, %v196
  br i1 %v520, label %entry_block15v1, label %entry_split_split_block28v1

entry_block15v1:                                  ; preds = %for_op_header_block29v1
  %iv_v470 = phi i64 [ %v519, %for_op_header_block29v1 ]
  br label %for_op_header_block27v1

for_op_header_block27v1:                          ; preds = %entry_split_block24v1, %entry_block15v1
  %v516 = phi i64 [ 0, %entry_block15v1 ], [ %v518, %entry_split_block24v1 ]
  %v517 = icmp ult i64 %v516, %v198
  br i1 %v517, label %entry_block14v1, label %entry_split_block26v1

entry_block14v1:                                  ; preds = %for_op_header_block27v1
  %iv_v469 = phi i64 [ %v516, %for_op_header_block27v1 ]
  br label %for_op_header_block25v1

for_op_header_block25v1:                          ; preds = %entry_block8v1, %entry_block14v1
  %v513 = phi i64 [ 0, %entry_block14v1 ], [ %v515, %entry_block8v1 ]
  %v514 = icmp ult i64 %v513, %v203
  br i1 %v514, label %entry_block13v1, label %entry_split_block24v1

entry_block13v1:                                  ; preds = %for_op_header_block25v1
  %iv_v468 = phi i64 [ %v513, %for_op_header_block25v1 ]
  br label %entry_block8v1

entry_block8v1:                                   ; preds = %entry_block13v1
  %v207 = phi i64 [ %iv_v470, %entry_block13v1 ]
  %v208 = phi i64 [ %iv_v469, %entry_block13v1 ]
  %v209 = phi i64 [ %iv_v468, %entry_block13v1 ]
  %v312 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 1
  %v313 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 4
  %v314 = extractvalue [2 x i64] %v313, 0
  %v316 = extractvalue [2 x i64] %v313, 1
  %v318 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 2
  %v471 = mul i64 %v207, %v314
  %v472 = add i64 %v318, %v471
  %v473 = mul i64 %v208, %v316
  %v474 = add i64 %v472, %v473
  %v325 = getelementptr i64, ptr %v312, i64 %v474
  %v326 = load i64, ptr %v325, align 4
  %v327 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 1
  %v328 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 4
  %v329 = extractvalue [2 x i64] %v328, 0
  %v331 = extractvalue [2 x i64] %v328, 1
  %v333 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 2
  %v475 = mul i64 %v207, %v329
  %v476 = add i64 %v333, %v475
  %v477 = mul i64 %v209, %v331
  %v478 = add i64 %v476, %v477
  %v340 = getelementptr i64, ptr %v327, i64 %v478
  %v341 = load i64, ptr %v340, align 4
  %v342 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 1
  %v343 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 4
  %v344 = extractvalue [2 x i64] %v343, 0
  %v346 = extractvalue [2 x i64] %v343, 1
  %v348 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 2
  %v479 = mul i64 %v209, %v344
  %v480 = add i64 %v348, %v479
  %v481 = mul i64 %v208, %v346
  %v482 = add i64 %v480, %v481
  %v355 = getelementptr i64, ptr %v342, i64 %v482
  %v356 = load i64, ptr %v355, align 4
  %v213 = mul i64 %v341, %v356
  %v214 = add i64 %v326, %v213
  %v357 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 1
  %v358 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 4
  %v359 = extractvalue [2 x i64] %v358, 0
  %v361 = extractvalue [2 x i64] %v358, 1
  %v363 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, 2
  %v483 = mul i64 %v207, %v359
  %v484 = add i64 %v363, %v483
  %v485 = mul i64 %v208, %v361
  %v486 = add i64 %v484, %v485
  %v370 = getelementptr i64, ptr %v357, i64 %v486
  store i64 %v214, ptr %v370, align 4
  %v515 = add i64 %iv_v468, 1
  br label %for_op_header_block25v1

entry_split_block24v1:                            ; preds = %for_op_header_block25v1
  %v518 = add i64 %iv_v469, 1
  br label %for_op_header_block27v1

entry_split_block26v1:                            ; preds = %for_op_header_block27v1
  %v521 = add i64 %iv_v470, 1
  br label %for_op_header_block29v1

entry_split_split_block28v1:                      ; preds = %for_op_header_block29v1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v184, ptr %2, align 8
  ret void
}

define void @test_constant_splat(ptr %0) {
entry_block5v1:
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr @__constant_4x4xfp64_8B_3, ptr @__constant_4x4xfp64_8B_3, i64 0, [2 x i64] [i64 4, i64 4], [2 x i64] [i64 4, i64 1] }, ptr %0, align 8
  ret void
}

declare ptr @malloc(i64)
