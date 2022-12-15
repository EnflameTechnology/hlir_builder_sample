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

// inputs and outouts
float g_lhs[] = {1, 5, 3, 6,
                 3, 3, 1, 6,
                 3, 3, 1, 6,
                 3, 3, 1, 6,
                                     
                 4, 6, 6, 2,
                 9, 2, 1, 3,
                 3, 1, 5, 5,
                 3, 1, 5, 5,
                           
                 8, 2, 6, 5,
                 5, 1, 5, 9,
                 7, 6, 7, 6,
                 5, 2, 9, 4};

float g_exps[] = {1.      , 0.2     , 0.333333, 0.166667, 0.333333, 0.333333,
                  1.      , 0.166667, 0.333333, 0.333333, 1.      , 0.166667,
                  0.333333, 0.333333, 1.      , 0.166667, 0.25    , 0.166667,
                  0.166667, 0.5     , 0.111111, 0.5     , 1.      , 0.333333,
                  0.333333, 1.      , 0.2     , 0.2     , 0.333333, 1.      ,
                  0.2     , 0.2     , 0.125   , 0.5     , 0.166667, 0.2     ,
                  0.2     , 1.      , 0.2     , 0.111111, 0.142857, 0.166667,
                  0.142857, 0.166667, 0.2     , 0.5     , 0.111111, 0.25    };
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};


std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> data_shape = {1, 3, 4, 4};
    builder::Type data_type(data_shape, builder::PrimitiveType::F32());
    builder::Op operand = hlir_builder->CreateInput(data_type);

    builder::Type output_type({}, builder::PrimitiveType::NONE());

    builder::Op res = builder::Reciprocal(/* builder::Op operand = */ operand);


    res.SetAttribute("op_name", builder::Attribute("Reciprocal"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}