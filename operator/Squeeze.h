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

float g_exps[] = {0, 5, 3, 6, 
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
                  5, 2, 9, 4 };
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> data_shape = {1, 4, 1, 4, 1, 3};
    builder::Type data_type(data_shape, builder::PrimitiveType::F32());
    builder::Op data = hlir_builder->CreateInput(data_type);

    builder::Type axes_type({2}, builder::PrimitiveType::S64());
    std::vector<int64_t> axes_data = {2, 4};
    builder::Op axes = builder::Const(hlir_builder, static_cast<void *>(axes_data.data()), axes_type);

    std::vector<int64_t> output_shape = {1, 4, 4, 3};
    builder::Type output_type(output_shape, builder::PrimitiveType::F32());

    builder::Op res = builder::Squeeze(/* builder::Op data = */ data, 
                                       /* builder::Op axes = */ axes, 
                                       /* builder::Type resultType = */ output_type);


    res.SetAttribute("op_name", builder::Attribute("Squeeze"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}