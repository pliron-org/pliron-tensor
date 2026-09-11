; ModuleID = 'test_module'
source_filename = "test_module"

@__constant_2x3xfp64_8B_0 = private constant [48 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\00@\00\00\00\00\00\00\08@\00\00\00\00\00\00\10@\00\00\00\00\00\00\14@\00\00\00\00\00\00\18@", align 8
@__constant_2x3xfp64_8B_1 = private constant [48 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\00@\00\00\00\00\00\00\08@\00\00\00\00\00\00\10@\00\00\00\00\00\00\14@\00\00\00\00\00\00\18@", align 8
@__constant_2x2xinteger_8B_2 = private constant [32 x i8] c"\0A\00\00\00\00\00\00\00\14\00\00\00\00\00\00\00\1E\00\00\00\00\00\00\00(\00\00\00\00\00\00\00", align 8
@__constant_4x4xfp64_8B_3 = private constant [128 x i8] c"\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?\00\00\00\00\00\00\F0?", align 8

define double @test_constant_extract(i64 %0, i64 %1) {
entry_block2v1:
  %v53 = mul i64 3, %0
  %v54 = mul i64 1, %1
  %v55 = add i64 %v54, %v53
  %v56 = getelementptr double, ptr @__constant_2x3xfp64_8B_0, i64 %v55
  %res_v57 = load double, ptr %v56, align 8
  ret double %res_v57
}

define void @test_constant_add(ptr %0, ptr %1) {
entry_block3v1:
  %arg_v8 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %v85 = mul i64 ptrtoint (ptr getelementptr (double, ptr null, i32 1) to i64), 6
  %v86 = call ptr @malloc(i64 %v85)
  %v89 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v86, 0
  %v90 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v89, ptr %v86, 1
  %v91 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v90, i64 0, 2
  %v95 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v91, [2 x i64] [i64 2, i64 3], 3
  %res_v99 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v95, [2 x i64] [i64 3, i64 1], 4
  %v100 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v99, 3
  %v101 = extractvalue [2 x i64] %v100, 0
  %v102 = extractvalue [2 x i64] %v100, 1
  br label %for_op_header_block19v1

for_op_header_block19v1:                          ; preds = %entry_split_block16v1, %entry_block3v1
  %v367 = phi i64 [ 0, %entry_block3v1 ], [ %v369, %entry_split_block16v1 ]
  %v368 = icmp ult i64 %v367, %v101
  br i1 %v368, label %entry_block10v1, label %entry_split_block18v1

entry_block10v1:                                  ; preds = %for_op_header_block19v1
  %iv_v314 = phi i64 [ %v367, %for_op_header_block19v1 ]
  br label %for_op_header_block17v1

for_op_header_block17v1:                          ; preds = %entry_block6v1, %entry_block10v1
  %v364 = phi i64 [ 0, %entry_block10v1 ], [ %v366, %entry_block6v1 ]
  %v365 = icmp ult i64 %v364, %v102
  br i1 %v365, label %entry_block9v1, label %entry_split_block16v1

entry_block9v1:                                   ; preds = %for_op_header_block17v1
  %iv_v313 = phi i64 [ %v364, %for_op_header_block17v1 ]
  br label %entry_block6v1

entry_block6v1:                                   ; preds = %entry_block9v1
  %v105 = phi i64 [ %iv_v314, %entry_block9v1 ]
  %v106 = phi i64 [ %iv_v313, %entry_block9v1 ]
  %v195 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 1
  %v196 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 4
  %v197 = extractvalue [2 x i64] %v196, 0
  %v198 = extractvalue [2 x i64] %v196, 1
  %v199 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %arg_v8, 2
  %v200 = getelementptr double, ptr %v195, i64 %v199
  %v201 = mul i64 %v197, %v105
  %v202 = mul i64 %v198, %v106
  %v203 = add i64 %v202, %v201
  %v204 = getelementptr double, ptr %v200, i64 %v203
  %v205 = load double, ptr %v204, align 8
  %v212 = mul i64 3, %v105
  %v213 = mul i64 1, %v106
  %v214 = add i64 %v213, %v212
  %v215 = getelementptr double, ptr @__constant_2x3xfp64_8B_1, i64 %v214
  %v216 = load double, ptr %v215, align 8
  %v109 = fadd double %v205, %v216
  %v217 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v99, 1
  %v218 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v99, 4
  %v219 = extractvalue [2 x i64] %v218, 0
  %v220 = extractvalue [2 x i64] %v218, 1
  %v221 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v99, 2
  %v222 = getelementptr double, ptr %v217, i64 %v221
  %v223 = mul i64 %v219, %v105
  %v224 = mul i64 %v220, %v106
  %v225 = add i64 %v224, %v223
  %v226 = getelementptr double, ptr %v222, i64 %v225
  store double %v109, ptr %v226, align 8
  %v366 = add i64 %iv_v313, 1
  br label %for_op_header_block17v1

entry_split_block16v1:                            ; preds = %for_op_header_block17v1
  %v369 = add i64 %iv_v314, 1
  br label %for_op_header_block19v1

entry_split_block18v1:                            ; preds = %for_op_header_block19v1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v99, ptr %1, align 8
  ret void
}

define void @test_constant_accumulator(ptr %0, ptr %1, ptr %2) {
entry_block4v1:
  %lhs_v14 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %0, align 8
  %rhs_v15 = load { ptr, ptr, i64, [2 x i64], [2 x i64] }, ptr %1, align 8
  %v137 = mul i64 ptrtoint (ptr getelementptr (i64, ptr null, i32 1) to i64), 4
  %v138 = call ptr @malloc(i64 %v137)
  %v141 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } undef, ptr %v138, 0
  %v142 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v141, ptr %v138, 1
  %v143 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v142, i64 0, 2
  %v147 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v143, [2 x i64] [i64 2, i64 2], 3
  %res_v151 = insertvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %v147, [2 x i64] [i64 2, i64 1], 4
  %v152 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 3
  %v153 = extractvalue [2 x i64] %v152, 0
  %v154 = extractvalue [2 x i64] %v152, 1
  br label %for_op_header_block23v1

for_op_header_block23v1:                          ; preds = %entry_split_block20v1, %entry_block4v1
  %v385 = phi i64 [ 0, %entry_block4v1 ], [ %v387, %entry_split_block20v1 ]
  %v386 = icmp ult i64 %v385, %v153
  br i1 %v386, label %entry_block12v1, label %entry_split_block22v1

entry_block12v1:                                  ; preds = %for_op_header_block23v1
  %iv_v332 = phi i64 [ %v385, %for_op_header_block23v1 ]
  br label %for_op_header_block21v1

for_op_header_block21v1:                          ; preds = %entry_block7v1, %entry_block12v1
  %v382 = phi i64 [ 0, %entry_block12v1 ], [ %v384, %entry_block7v1 ]
  %v383 = icmp ult i64 %v382, %v154
  br i1 %v383, label %entry_block11v1, label %entry_split_block20v1

entry_block11v1:                                  ; preds = %for_op_header_block21v1
  %iv_v331 = phi i64 [ %v382, %for_op_header_block21v1 ]
  br label %entry_block7v1

entry_block7v1:                                   ; preds = %entry_block11v1
  %v157 = phi i64 [ %iv_v332, %entry_block11v1 ]
  %v158 = phi i64 [ %iv_v331, %entry_block11v1 ]
  %v233 = mul i64 2, %v157
  %v234 = mul i64 1, %v158
  %v235 = add i64 %v234, %v233
  %v236 = getelementptr i64, ptr @__constant_2x2xinteger_8B_2, i64 %v235
  %v237 = load i64, ptr %v236, align 4
  %v238 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 1
  %v239 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 4
  %v240 = extractvalue [2 x i64] %v239, 0
  %v241 = extractvalue [2 x i64] %v239, 1
  %v242 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 2
  %v243 = getelementptr i64, ptr %v238, i64 %v242
  %v244 = mul i64 %v240, %v157
  %v245 = mul i64 %v241, %v158
  %v246 = add i64 %v245, %v244
  %v247 = getelementptr i64, ptr %v243, i64 %v246
  store i64 %v237, ptr %v247, align 4
  %v384 = add i64 %iv_v331, 1
  br label %for_op_header_block21v1

entry_split_block20v1:                            ; preds = %for_op_header_block21v1
  %v387 = add i64 %iv_v332, 1
  br label %for_op_header_block23v1

entry_split_block22v1:                            ; preds = %for_op_header_block23v1
  %v160 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 3
  %v161 = extractvalue [2 x i64] %v160, 0
  %v162 = extractvalue [2 x i64] %v160, 1
  %v163 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 3
  %v164 = extractvalue [2 x i64] %v163, 0
  %v165 = extractvalue [2 x i64] %v163, 1
  br label %for_op_header_block29v1

for_op_header_block29v1:                          ; preds = %entry_split_block26v1, %entry_split_block22v1
  %v394 = phi i64 [ 0, %entry_split_block22v1 ], [ %v396, %entry_split_block26v1 ]
  %v395 = icmp ult i64 %v394, %v161
  br i1 %v395, label %entry_block15v1, label %entry_split_split_block28v1

entry_block15v1:                                  ; preds = %for_op_header_block29v1
  %iv_v339 = phi i64 [ %v394, %for_op_header_block29v1 ]
  br label %for_op_header_block27v1

for_op_header_block27v1:                          ; preds = %entry_split_block24v1, %entry_block15v1
  %v391 = phi i64 [ 0, %entry_block15v1 ], [ %v393, %entry_split_block24v1 ]
  %v392 = icmp ult i64 %v391, %v162
  br i1 %v392, label %entry_block14v1, label %entry_split_block26v1

entry_block14v1:                                  ; preds = %for_op_header_block27v1
  %iv_v338 = phi i64 [ %v391, %for_op_header_block27v1 ]
  br label %for_op_header_block25v1

for_op_header_block25v1:                          ; preds = %entry_block8v1, %entry_block14v1
  %v388 = phi i64 [ 0, %entry_block14v1 ], [ %v390, %entry_block8v1 ]
  %v389 = icmp ult i64 %v388, %v165
  br i1 %v389, label %entry_block13v1, label %entry_split_block24v1

entry_block13v1:                                  ; preds = %for_op_header_block25v1
  %iv_v337 = phi i64 [ %v388, %for_op_header_block25v1 ]
  br label %entry_block8v1

entry_block8v1:                                   ; preds = %entry_block13v1
  %v168 = phi i64 [ %iv_v339, %entry_block13v1 ]
  %v169 = phi i64 [ %iv_v338, %entry_block13v1 ]
  %v170 = phi i64 [ %iv_v337, %entry_block13v1 ]
  %v248 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 1
  %v249 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 4
  %v250 = extractvalue [2 x i64] %v249, 0
  %v251 = extractvalue [2 x i64] %v249, 1
  %v252 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 2
  %v253 = getelementptr i64, ptr %v248, i64 %v252
  %v254 = mul i64 %v250, %v168
  %v255 = mul i64 %v251, %v169
  %v256 = add i64 %v255, %v254
  %v257 = getelementptr i64, ptr %v253, i64 %v256
  %v258 = load i64, ptr %v257, align 4
  %v259 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 1
  %v260 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 4
  %v261 = extractvalue [2 x i64] %v260, 0
  %v262 = extractvalue [2 x i64] %v260, 1
  %v263 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %lhs_v14, 2
  %v264 = getelementptr i64, ptr %v259, i64 %v263
  %v265 = mul i64 %v261, %v168
  %v266 = mul i64 %v262, %v170
  %v267 = add i64 %v266, %v265
  %v268 = getelementptr i64, ptr %v264, i64 %v267
  %v269 = load i64, ptr %v268, align 4
  %v270 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 1
  %v271 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 4
  %v272 = extractvalue [2 x i64] %v271, 0
  %v273 = extractvalue [2 x i64] %v271, 1
  %v274 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %rhs_v15, 2
  %v275 = getelementptr i64, ptr %v270, i64 %v274
  %v276 = mul i64 %v272, %v170
  %v277 = mul i64 %v273, %v169
  %v278 = add i64 %v277, %v276
  %v279 = getelementptr i64, ptr %v275, i64 %v278
  %v280 = load i64, ptr %v279, align 4
  %v174 = mul i64 %v269, %v280
  %v175 = add i64 %v258, %v174
  %v281 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 1
  %v282 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 4
  %v283 = extractvalue [2 x i64] %v282, 0
  %v284 = extractvalue [2 x i64] %v282, 1
  %v285 = extractvalue { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, 2
  %v286 = getelementptr i64, ptr %v281, i64 %v285
  %v287 = mul i64 %v283, %v168
  %v288 = mul i64 %v284, %v169
  %v289 = add i64 %v288, %v287
  %v290 = getelementptr i64, ptr %v286, i64 %v289
  store i64 %v175, ptr %v290, align 4
  %v390 = add i64 %iv_v337, 1
  br label %for_op_header_block25v1

entry_split_block24v1:                            ; preds = %for_op_header_block25v1
  %v393 = add i64 %iv_v338, 1
  br label %for_op_header_block27v1

entry_split_block26v1:                            ; preds = %for_op_header_block27v1
  %v396 = add i64 %iv_v339, 1
  br label %for_op_header_block29v1

entry_split_split_block28v1:                      ; preds = %for_op_header_block29v1
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } %res_v151, ptr %2, align 8
  ret void
}

define void @test_constant_splat(ptr %0) {
entry_block5v1:
  store { ptr, ptr, i64, [2 x i64], [2 x i64] } { ptr @__constant_4x4xfp64_8B_3, ptr @__constant_4x4xfp64_8B_3, i64 0, [2 x i64] [i64 4, i64 4], [2 x i64] [i64 4, i64 1] }, ptr %0, align 8
  ret void
}

declare ptr @malloc(i64)
