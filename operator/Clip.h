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

// [[[[1, 2, 3],            [[[[3, 3, 3],
//    [4, 5, 6]],     --->     [4, 5, 6]],   
//   [[7, 8, 9]]              [[7, 8, 8]]
//    [10,11,12]]]]            [8, 8, 8]]]]

// inputs and outouts
float g_lhs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
float g_exps[] = {3, 3, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 2, 2, 3}; // NCHW
  builder::Type input_type(in_shape, dtype);
  builder::Type scalar_type(dtype);
  auto input = hlir_builder->CreateInput(input_type);
  // auto result_type = builder::Type(in_shape, dtype);
  float min_value = 3.0;
  float max_value = 8.0;
  void* data_ptr = static_cast<void*>(&min_value);
  auto min = builder::Const(hlir_builder, data_ptr, scalar_type);
  data_ptr = static_cast<void*>(&max_value);
  auto max = builder::Const(hlir_builder, data_ptr, scalar_type);
  // auto res = builder::Clamp(min, input, max, result_type);
  auto res = builder::Clamp(min, input, max);
  res.SetAttribute("op_name", builder::Attribute("clip"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);
  // compile(hlir_builder, exe_ptr);
  return hlir_builder;
}


