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

/*
  expect result:{8640, -26061, -7980, -23406, 896, 3456};
*/

// inputs and outputs
float g_lhs[] = {-80, 219, -84, 166, -14, -54};
float g_rhs[] = {-108, -119, 95, -141, -64, -64};
float g_exps[] = {8640, -26061, -7980, -23406, 896, 3456};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs),
                                    static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  // shape
  std::vector<int64_t> lhs_shape = {2, 1, 3}; // left operator num
  std::vector<int64_t> rhs_shape = {2, 1, 3};
  // std::vector<int64_t> dimensions = {2, 1, 3}; //result dimens
  // type
  builder::Type lhs_type(lhs_shape, dtype);
  builder::Type rhs_type(rhs_shape, dtype);

  auto lhs_input = hlir_builder->CreateInput(lhs_type);
  auto rhs_input = hlir_builder->CreateInput(rhs_type);

  auto res = builder::Mul(lhs_input, rhs_input);
  hlir_builder->SetOutput({res});

  res.SetAttribute("op_name", builder::Attribute("Mul"));

  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);

  // compile(hlir_builder, exe_ptr);

  return hlir_builder;
}