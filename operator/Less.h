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
#include "dtu/hlir_builder/hlir_builder.h"

// inputs and outputs
float g_lhs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
float g_rhs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51};
bool g_exps[] = {false, false, false, false, false, false,
                 false, false, false, false, false, false,
                 false, true, true, true, true, true,
                 true, true, true, true, true, true,
                 true, true, true};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto ptype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 3, 3, 3};
  builder::Type input_type(in_shape, ptype);
  builder::Type output_type(in_shape, builder::PrimitiveType::PRED());


  builder::Op arg0 = hlir_builder->CreateInput(input_type);
  builder::Op arg1 = hlir_builder->CreateInput(input_type);
  builder::Op res = builder::Less(/* builder::Op lhs = */arg0, 
                                  /* builder::Op rhs = */arg1, 
                                  /* std::vector<int64_t> broadcast_dimensions = */{}, 
                                  /* builder::Type resultType = */output_type);

  res.SetAttribute("op_name", builder::Attribute("Less"));
  hlir_builder->SetOutput({res});

  return hlir_builder;
}