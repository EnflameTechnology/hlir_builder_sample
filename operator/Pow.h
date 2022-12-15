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

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

/*
  expect result:{1, 1, 0, 4, 1, 3};
*/

// inputs and outputs
int32_t g_lhs[] = {0, 1, 0, 2, 1, 3};
int32_t g_rhs[] = {0, 1, 1, 2, 0, 1};
int32_t g_exps[] = {1, 1, 0, 4, 1, 3};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs),
                                    static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::S32();
  // shape
  std::vector<int64_t> shape = {2, 3};
  // type
  builder::Type In_type(shape, dtype);
  auto lhs_input = hlir_builder->CreateInput(In_type);
  auto rhs_input = hlir_builder->CreateInput(In_type);
  auto res = builder::Pow(lhs_input, rhs_input);
  hlir_builder->SetOutput({res});

  res.SetAttribute("op_name", builder::Attribute("Pow"));

  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);

  // compile(hlir_builder, exe_ptr);
  return hlir_builder;
}