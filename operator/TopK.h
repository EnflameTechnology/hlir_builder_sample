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
  * This sample demonstrates how to use the Transpose operator.
  * The Transpose operator permutes the dimensions of the input according to the given permutation.
  * The input tensor is a 2D tensor of shape [2, 3].
  * The output tensor is a 2D tensor of shape [3, 2].
*/

float g_lhs[] = {0, 5, 3, 6,
                 3, 3, 0, 6,
                 3, 3, 0, 6,
                 3, 3, 0, 6,
                 4, 6, 6, 2,
                 9, 2, 1, 3,
                 3, 1, 5, 5,
                 3, 1, 5, 5,
                 8, 2, 6, 5,
                 5, 0, 5, 9,
                 7, 6, 7, 6,
                 5, 2, 9, 4};
float g_exps[] = { 0 };

std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> input_shape = {50};
    builder::Type input_type(input_shape, builder::PrimitiveType::F32());
    builder::Op operand = hlir_builder->CreateInput(input_type);

    int64_t k_value = 5;
    std::vector<int64_t> k_shape = {1};
    builder::Type k_type(k_shape, builder::PrimitiveType::S64());
    builder::Op k = builder::Const(hlir_builder, static_cast<void *>(&k_value), k_type);

    // output is a tuple with 2 elements, 
    // the first stands for value and the second index
    builder::Type output_type({{5},{5}}, {builder::PrimitiveType::F32(), builder::PrimitiveType::S32()});

    builder::Op res = builder::TopK(/* builder::Op input = */ operand, 
                                    /* builder::Op k = */ k, 
                                    /* int64_t axis = */ -1, 
                                    /* bool sorted = */ false, 
                                    /* bool largest =  */true, 
                                    /* builder::Type resultType = */ output_type);

    res.SetAttribute("op_name", builder::Attribute("Topk"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}