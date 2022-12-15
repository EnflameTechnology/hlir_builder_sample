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
float g_lhs[] = {1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1., 
              1., 1., 1., 1.};

float g_exps[] = { 3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712, 
                   3.41041, 4.29619, 3.01199, 5.06712 };

std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
   auto hlir_builder = std::make_shared<builder::Builder>();
   hlir_builder->SetShapeInference(true);
   auto ptype = builder::PrimitiveType::F32();

   std::vector<int64_t> in_shape{1, 3, 4, 4}; // NCHW
   builder::Type input_type(in_shape, ptype);
   builder::Op operand = hlir_builder->CreateInput(input_type);

   std::vector<int64_t> scale_shape{4};
   builder::Type scale_type(scale_shape, ptype);
   std::vector<float> scale_data = {1.0f, 1.0f, 1.0f, 1.0f};
   builder::Op scale = builder::Const(hlir_builder, static_cast<void *>(scale_data.data()), scale_type);

   builder::Type offset_type(scale_shape, ptype);
   std::vector<float> offset_data = {1.0f, 1.0f, 1.0f, 1.0f};
   builder::Op offset = builder::Const(hlir_builder, static_cast<void *>(offset_data.data()), offset_type);

   builder::Type mean_type(scale_shape, ptype);
   std::vector<float> mean_data = {0.34442666, 0.34195662, 0.31565964, 0.57439};
   builder::Op mean = builder::Const(hlir_builder, static_cast<void *>(mean_data.data()), mean_type);

   builder::Type variance_type(scale_shape, ptype);
   std::vector<float> variance_data = {0.07396074, 0.03984508, 0.11567877, 0.01094088};
   builder::Op variance = builder::Const(hlir_builder, static_cast<void *>(variance_data.data()), variance_type);

   float epsilon = 1e-5;

   std::vector<int64_t> output_shape{1, 3, 4, 4}; // NCHW
   builder::Type output_type(output_shape, ptype);

   builder::Op res = builder::BatchNormInference(/* builder::Op operand = */ operand, 
                                                 /* builder::Op scale = */scale, 
                                                 /* builder::Op offset = */ offset, 
                                                 /* builder::Op mean = */ mean, 
                                                 /* builder::Op variance = */ variance, 
                                                 /* float epsilon = */ epsilon, 
                                                 /* int64_t feature_index = */ 3 , 
                                                 /* builder::Type resultType = */ output_type);

   res.SetAttribute("op_name", builder::Attribute("BatchNorm"));
   hlir_builder->SetOutput({res});

   return hlir_builder;
}