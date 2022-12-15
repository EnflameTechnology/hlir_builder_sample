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

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// torch.reshape(input, shape)  shpae=[3,2]

// [[1, 2, 3]            [[1 2]
//               ->       [3 4]
//  [4, 5, 6]]            [5 6]]

// inputs
float g_lhs[] = {1, 2, 3, 4, 5, 6};
float g_exps[] = {1, 2, 3, 4, 5, 6};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 1, 2, 3}; // NCHW
  std::vector<int64_t> out_shape{1, 1, 3, 2}; // NCHW
  builder::Type input_type(in_shape, dtype);
  builder::Type output_type(out_shape, dtype);
  auto input = hlir_builder->CreateInput(input_type);
  auto res = builder::Reshape(input, output_type);
  res.SetAttribute("op_name", builder::Attribute("Reshape"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);

  return hlir_builder;
}