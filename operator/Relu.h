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

// m=nn.ReLU()
// m(tensor)

// [[[[ -1,   2,   3]       [[[[ 0,  2,  3]
//    [  4,  -5,  -6]          [ 4,  0,  0]
//    [ -7,   8,  -9]]         [ 0,  8,  0]]

//   [[ 10, -11, -12]         [[10,  0,  0]
//    [ 13, -14, -15]   -->    [13,  0,  0]
//    [ 16,  17, -18]]         [16, 17,  0]]             
                                  
//   [[-19, -20,  21]         [[ 0,  0, 21]
//    [-22,  23, -24]          [ 0, 23,  0]
//    [ 25, -26, -27]]]]       [25,  0,  0]]]]

// inputs
float g_lhs[] = {-1, 2, 3, 4, -5, -6, -7, 8, -9, 
                 10, -11, -12, 13, -14, -15, 16, 17, -18, 
                 -19, -20, 21, -22, 23, -24, 25, -26, -27};
float g_exps[] = {0, 2, 3, 4, 0, 0, 0, 8, 0, 
                 10, 0, 0, 13, 0, 0, 16, 17, 0, 
                 0, 0, 21, 0, 23, 0, 25, 0, 0};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder>  build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{1, 3, 3, 3}; // NCHW
  builder::Type input_type(in_shape, dtype);

  builder::Op input = hlir_builder->CreateInput(input_type);

  auto res = builder::Relu(input);

  res.SetAttribute("op_name", builder::Attribute("Relu"));
  hlir_builder->SetOutput({res});

  return hlir_builder;
}
