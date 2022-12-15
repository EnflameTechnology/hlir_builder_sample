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
#include "common/fp16.hpp"

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

half g_lhs[] = {0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f, 8.f, 
                9.f, 10.f, 11.f, 12.f, 13.f, 14.f, 15.f, 16.f, 17.f, 
                18.f, 19.f, 20.f, 21.f, 22.f, 23.f, 24.f, 25.f, 26.f};
half g_rhs[] = {27.f, 28.f, 29.f, 30.f, 31.f, 32.f, 33.f, 34.f, 35.f, 
                36.f, 37.f, 38.f, 39.f, 40.f, 41.f, 42.f, 43.f, 44.f, 
                45.f, 46.f, 47.f, 48.f, 49.f, 50.f, 51.f, 52.f, 53.f};
half g_exps[] = {27.f, 29.f, 31.f, 33.f, 35.f, 37.f, 39.f, 41.f, 43.f, 45.f, 47.f, 49.f, 51.f, 53.f, 55.f, 57.f, 59.f, 61.f, 63.f, 65.f, 67.f, 69.f, 71.f, 73.f, 75.f, 77.f, 79.f};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
   auto hlir_builder = std::make_shared<builder::Builder>();
   hlir_builder->SetShapeInference(true);
   auto ptype = builder::PrimitiveType::F16();
   std::vector<int64_t> in_shape{1, 3, 3, 3}; // NCHW
   builder::Type input_type(in_shape, ptype);

   builder::Op arg0 = hlir_builder->CreateInput(input_type);
   builder::Op arg1 = hlir_builder->CreateInput(input_type);
   builder::Op res = builder::Add(arg0, arg1);

   res.SetAttribute("op_name", builder::Attribute("Add"));
   hlir_builder->SetOutput({res});

   return hlir_builder;
}