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
#include "common/fp16.hpp"

// tf.dtypes.cast(x, tf.int32)

// [[-11.  10. 11.]   ->   [[-11  10 11]
//  [-21.  21. -10.]]        [-21  21 -10]]

// builder::Op builder::Convert(builder::Op input, builder::Type resultType = builder::Type())

float g_lhs[] = {-11, 10, 11, -21, 21, -10};
int32_t g_exps[] = {-11, 10, 11, -21, 21, -10};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();//float32
  auto out_dtype = builder::PrimitiveType::S32();//int32_t
  std::vector<int64_t> in_shape{2, 3};
  builder::Type input_type(in_shape, dtype);
  builder::Type out_type(in_shape, out_dtype);
  auto input = hlir_builder->CreateInput(input_type);
  auto res = builder::Convert(input, out_type);
  res.SetAttribute("op_name", builder::Attribute("Convert"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);
  // compile(hlir_builder, exe_ptr);
  return hlir_builder;
}
