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
#include "dtu/hlir_builder/hlir_builder.h"
#include "dtu/hlir_builder/hlir_builder_client_ops.h"

/*
[[[[ 0  1  2]             [[[[27 28 29]              [[[[27 29 31]
   [ 3  4  5]                [30 31 32]                 [33 35 37]
   [ 6  7  8]]               [33 34 35]]                [39 41 43]]

  [[ 9 10 11]               [[36 37 38]                [[45 47 49]
   [12 13 14]       +        [39 40 41]       =         [51 53 55]
   [15 16 17]]               [42 43 44]]                [57 59 61]]

  [[18 19 20]               [[45 46 47]                [[63 65 67]
   [21 22 23]                [48 49 50]                 [69 71 73]
   [24 25 26]]]]             [51 52 53]]]]              [75 77 79]]]]
*/

// inputs and outputs
float g_lhs[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
               14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26};
float g_rhs[] = {27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
               41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53};
float g_exps[] = {27, 29, 31, 33, 35, 37, 39, 41, 43, 45, 47, 49, 51, 53, 
                55, 57, 59, 61, 63, 65, 67, 69, 71, 73, 75, 77, 79};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto ptype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 3, 3, 3}; // NCHW
  builder::Type input_type(in_shape, ptype);

  builder::Op arg0 = hlir_builder->CreateInput(input_type);
  builder::Op arg1 = hlir_builder->CreateInput(input_type);
  builder::Op res = builder::Add(arg0, arg1);

  res.SetAttribute("op_name", builder::Attribute("Add"));
  hlir_builder->SetOutput({res});

  return hlir_builder;
}