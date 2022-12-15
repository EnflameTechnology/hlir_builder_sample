/*=======================================================================
 * Copyright 2020-2021 Enflame. All Rights Reserved.
 *
 *Licensed under the Apache License, Version 2.0 (the "License");
 *you may not use this file except in compliance with the License.
 *You may obtain a copy of the License at
 *
 *http://www.apache.org/licenses/LICENSE-2.0
 *
 *Unless required by applicable law or agreed to in writing, software
 *distributed under the License is distributed on an "AS IS" BASIS,
 *WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *See the License for the specific language governing permissions and
 *limitations under the License.
 *=======================================================================
 */
#include "common/dtu_utils.h"
#include "dtu/hlir_builder/hlir_builder_client_ops.h"


// [[2.  1. 2.]   X     [[2  1]    =    [[11 14] 
//  [1.  3. 4.]]        [2 1]]           [15 27]

// inputs and outputs
float g_lhs[] = {2.0, 1.0, 2.0, 1.0, 3.0, 4.0};
float g_rhs[] = {2.0, 1.0, 2.0, 1.0};
float g_exps[] = {11, 14, 15, 27};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs),
                                    static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();//float32
  std::vector<int64_t> in_shape1{2, 3};
  std::vector<int64_t> in_shape2{2, 2};
  builder::Type InType1(in_shape1, dtype);
  builder::Type InType2(in_shape2, dtype);

  auto input1 = hlir_builder->CreateInput(InType1);
  auto input2 = hlir_builder->CreateInput(InType2);

  std::vector<builder::Op> inputs(2, input1);
  inputs.emplace_back(input2);

  auto res = builder::Gemm(/*std::vector<builder::Op> operands, */inputs,
                            /*float alpha = */1.0, 
                            /*float beta = */1.0, 
                            /*int64_t transA = */0, 
                            /*int64_t transB = */1);

  res.SetAttribute("op_name", builder::Attribute("Gemm"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);

  // compile(hlir_builder, exe_ptr);

  return hlir_builder;
}