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
float g_lhs[] = {8, 4, 2, 9, 8, 
                 6, 6, 3, 3, 4, 
                 9, 4, 8, 2, 6, 
                 6, 1, 5, 4, 2, 
                 2, 3, 2, 4, 4, 
                 
                 8, 1, 2, 2, 1, 
                 9, 2, 9, 4, 9, 
                 7, 8, 5, 4, 1, 
                 5, 7, 2, 6, 6, 
                 4, 7, 2, 5, 2, 
                 
                 3, 4, 8, 5, 1, 
                 9, 2, 2, 9, 6, 
                 6, 2, 1, 4, 4, 
                 7, 9, 5, 9, 9, 
                 6, 1, 9, 7, 3};

float g_exps[] = {8, 4, 2, 9, 8, 
                 6, 6, 3, 3, 4, 
                 9, 4, 8, 2, 6, 
                 6, 1, 5, 4, 2, 
                 2, 3, 2, 4, 4, 
                 
                 8, 1, 2, 2, 1, 
                 9, 2, 9, 4, 9, 
                 7, 8, 5, 4, 1, 
                 5, 7, 2, 6, 6, 
                 4, 7, 2, 5, 2, 
                 
                 3, 4, 8, 5, 1, 
                 9, 2, 2, 9, 6, 
                 6, 2, 1, 4, 4, 
                 7, 9, 5, 9, 9, 
                 6, 1, 9, 7, 3};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
   auto hlir_builder = std::make_shared<builder::Builder>();
   hlir_builder->SetShapeInference(true);
   auto ptype = builder::PrimitiveType::F32();
   std::vector<int64_t> in_shape{1, 5, 5, 3}; // NCHW
   builder::Type input_type(in_shape, ptype);
   builder::Op input = hlir_builder->CreateInput(input_type);

   builder::Type output_type({1*5*5*3}, ptype);


   // if the training mode is set to be true,
   // there's no element will be dropped out.
   builder::Op res = builder::Reshape(/* builder::Op data = */input, 
                                      /* builder::Type resultType = */output_type);

   res.SetAttribute("op_name", builder::Attribute("Dropout"));
   hlir_builder->SetOutput({res});

   return hlir_builder;
}