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
  * This sample shows how to use the HlirBuilder to generate a tile op.
  * The tile op is a simple op that repeats the input tensor along the
  * specified axis.
*/
float g_lhs[] = {7., 7., 7., 7., 3., 3., 7., 3., 3., 1., 5., 3.};

float g_exps[] = { 7, 7, 7, 7, 
                   3, 3, 7, 7, 
                   7, 7, 3, 3, 
                   7, 3, 3, 1, 
                   
                   5, 3, 7, 3, 
                   3, 1, 5, 3, 
                   7, 7, 7, 7, 
                   3, 3, 7, 7, 
                   
                   7, 7, 3, 3, 
                   7, 3, 3, 1, 
                   5, 3, 7, 3, 
                   3, 1, 5, 3};

std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};


std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> input_shape = {1, 2, 2, 3};
    builder::Type input_type(input_shape, builder::PrimitiveType::F32());
    builder::Op operand = hlir_builder->CreateInput(input_type);

    std::vector<int64_t> repeat_value = {1, 2, 2, 1};
    builder::Type repeat_type({4}, builder::PrimitiveType::S64());
    builder::Op repeat = builder::Const(hlir_builder,
                                        static_cast<void *>(repeat_value.data()),
                                        repeat_type);

    builder::Type output_type({1, 4, 4, 3}, builder::PrimitiveType::F32());

    builder::Op res = builder::Tile(/* builder::Op input = */ operand, 
                                    /* builder::Op repeats = */ repeat, 
                                    /* builder::Type resultType = */ output_type);

    res.SetAttribute("op_name", builder::Attribute("Topk"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}