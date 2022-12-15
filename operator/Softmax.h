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
float g_lhs[] = {1,  2,  3,  4,  5,  6};
float g_exps[] = { 0.0900306, 0.244728, 0.665241, 0.0900306, 0.244728, 0.665241};                  
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    std::vector<int64_t> data_shape = {2,3};
    builder::Type data_type(data_shape, builder::PrimitiveType::F32());
    builder::Op logits = hlir_builder->CreateInput(data_type);

    builder::Op res = builder::Softmax(/* builder::Op logits = */ logits, 
                                       /* int64_t axis = */ -1, 
                                       /* bool accurate = */ true, 
                                       /* bool logarithmic = */ false, 
                                       /* float epsilon = */ 0, 
                                       /* builder::Type resultType = */ data_type);


    res.SetAttribute("op_name", builder::Attribute("Softmax"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}