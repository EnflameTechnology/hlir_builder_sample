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
float g_rhs[] = {2};
float g_exps[] = { -2, 3, 1, 4, 
                   1, 1, -2, 4, 
                   1, 1, -2, 4, 
                   1, 1, -2, 4, 
                   
                   2, 4, 4, 0, 
                   7, 0, -1, 1, 
                   1, -1, 3, 3, 
                   1, -1, 3, 3, 
                   
                   6, 0, 4, 3, 
                   3, -2, 3, 7, 
                   5, 4, 5, 4, 
                   3, 0, 7, 2 };
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> lhs_shape = {1, 4, 4, 3};
    builder::Type lhs_type(lhs_shape, builder::PrimitiveType::F32());
    builder::Op lhs = hlir_builder->CreateInput(lhs_type);

    std::vector<int64_t> rhs_shape = {1};
    builder::Type rhs_type(rhs_shape, builder::PrimitiveType::F32());
    builder::Op rhs = hlir_builder->CreateInput(rhs_type);

    builder::Op res = builder::Sub(/* builder::Op lhs = */ lhs, 
                                   /* builder::Op rhs = */ rhs, 
                                   /* std::vector<int64_t> broadcast_dimensions = */ {}, 
                                   /* builder::Type resultType = */ lhs_type);


    res.SetAttribute("op_name", builder::Attribute("Sub"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}