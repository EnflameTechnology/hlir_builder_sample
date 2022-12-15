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
  * This sample demonstrates how to use the Unsqueeze operator.
  * The Unsqueeze operator inserts a new axis of size 1 at the specified position.
  * The input tensor is a 2D tensor of shape [2, 3].
  * The output tensor is a 3D tensor of shape [1, 2, 3].
*/

// inputs and outputs
float g_lhs[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
float g_exps[] = {0, 1, 2, 3, 4, 5, 6, 7, 8};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> input_shape = {3, 3};
    builder::Type input_type(input_shape, builder::PrimitiveType::F32());
    builder::Op data = hlir_builder->CreateInput(input_type);

    builder::Type axes_type({2}, builder::PrimitiveType::S64());
    std::vector<int64_t> axes_data = {0, 1};
    builder::Op axes = builder::Const(hlir_builder, static_cast<void *>(axes_data.data()), axes_type);

    std::vector<int64_t> output_shape = {1, 1, 3, 3};
    builder::Type output_type(output_shape, builder::PrimitiveType::F32());

    builder::Op res = builder::Unsqueeze(/* builder::Op data = */ data, 
                                         /* builder::Op axes = */ axes, 
                                         /* builder::Type resultType = */ output_type);

    res.SetAttribute("op_name", builder::Attribute("Unsqueeze"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}