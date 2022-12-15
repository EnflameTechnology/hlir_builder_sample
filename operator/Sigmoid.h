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

// inputs and outputs
float g_lhs[] = {0,  1,  2,  3,  4,  5,  6,  7,  
                8,  9,  10, 11, 12, 13, 14, 15, 
                16, 17, 18, 19, 20, 21, 22, 23, 
                24, 25, 26, 27, 28, 29, 30, 31, 
                32, 33, 34, 35, 36, 37, 38, 39, 
                40, 41, 42, 43, 44, 45, 46, 47, 
                48, 49, 50, 51, 52, 53, 54, 55, 
                56, 57, 58, 59, 60, 61, 62, 63};
float g_exps[] = {0.5, 0.731059, 0.880797, 0.952574, 
                  0.982014, 0.993307, 0.997527, 0.999089, 
                  0.999665, 0.999877, 0.999955, 0.999983, 
                  0.999994, 0.999998, 0.999999, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                  1, 1, 1, 1, 1, 1, 1, 1, 1};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> data_shape = {8,8};
    builder::Type data_type(data_shape, builder::PrimitiveType::F32());
    builder::Op operand = hlir_builder->CreateInput(data_type);

    builder::Type output_type({}, builder::PrimitiveType::NONE());

    builder::Op res = builder::Sigmoid(/* builder::Op input = */ operand, 
                                       /* builder::Type resultType = */ output_type);

    res.SetAttribute("op_name", builder::Attribute("Sigmoid"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}