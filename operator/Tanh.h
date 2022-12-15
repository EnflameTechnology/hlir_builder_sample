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
  * This sample shows how to use the HLIRBuilder to build a Tanh op.
  * The Tanh op is a unary op, which means it only has one input and one output.
  * The input and output are both float32 type.
*/

// inputs and outputs
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

float g_exps[] = { 0, 0.999909, 0.995055, 0.999988, 
                   0.995055, 0.995055, 0, 0.999988, 
                   0.995055, 0.995055, 0, 0.999988, 
                   0.995055, 0.995055, 0, 0.999988, 
                   
                   0.999329, 0.999988, 0.999988, 0.964028, 
                   1, 0.964028, 0.761594, 0.995055, 
                   0.995055, 0.761594, 0.999909, 0.999909, 
                   0.995055, 0.761594, 0.999909, 0.999909, 
                   
                   1, 0.964028, 0.999988, 0.999909, 
                   0.999909, 0, 0.999909, 1, 
                   0.999998, 0.999988, 0.999998, 0.999988, 
                   0.999909, 0.964028, 1, 0.999329};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> input_shape = {1, 4, 4, 3};
    builder::Type input_type(input_shape, builder::PrimitiveType::F32());
    builder::Op operand = hlir_builder->CreateInput(input_type);

    builder::Op res = builder::Tanh(/* builder::Op operand = */ operand, 
                                    /* builder::Type resultType = */ input_type);


    res.SetAttribute("op_name", builder::Attribute("Tanh"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}