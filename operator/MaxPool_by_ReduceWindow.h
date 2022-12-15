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

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

// inputs and outouts
float g_lhs[] = {0, 1, 2, 3, 4,
                 5, 6, 7, 8, 9,
                 10, 11, 12, 13,
                 14, 15, 16, 17,
                 18, 19, 20, 21, 
                 22, 23, 24}; 
float g_exps[] = { 6, 8, 16, 18};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto builder = std::make_shared<builder::Builder>();
  builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 1, 5, 5}; // NCHW
  builder::Type input_type(in_shape, dtype);
  builder::Type scalar_type(dtype);
  auto input = builder->CreateInput(input_type);
  auto float_min = std::numeric_limits<float>::lowest();
  void* data_ptr = static_cast<void*>(&float_min);
  auto init_value = builder::Const(builder, data_ptr, scalar_type);
  topsExecutable_t exe_ptr;
  uint64_t output_count = 0;

  builder->AddFunc("body");
  auto arg0 = builder->CreateInput(scalar_type, "body");
  auto arg1 = builder->CreateInput(scalar_type, "body");
  auto maximum = builder::Max(arg0, arg1);
  builder->SetOutput({maximum}, "body");

  auto res = builder::ReduceWindow({input}, {init_value},
      /*window_dimensions=*/{1, 1, 2, 2}, {"body"}, /*window_strides=*/{1, 1, 2, 2},
      /*base_dilations=*/{}, /*window_dilations=*/{},
      /*padding=*/{{0, 0}, {0, 0}, {0, 0}, {0, 0}}, "NOTSET", false);
  builder->SetOutput({res});
  // compile(builder, &exe_ptr);
  return builder;
}
