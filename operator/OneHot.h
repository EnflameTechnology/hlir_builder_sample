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
float g_lhs[] = {0, 1, 2, 3, 0, 1, 2, 3};
float g_exps[] = {1, 0, 0, 0, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 0, 0, 0, 0, 0, 
                  0, 0, 0, 1, 0, 0, 0, 0, 
                  1, 0, 0, 0, 0, 0, 0, 0, 
                  0, 1, 0, 0, 0, 0, 0, 0, 
                  0, 0, 1, 0, 0, 0, 0, 0, 
                  0, 0, 0, 1, 0, 0, 0, 0};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};


static builder::Op OneHotx(builder::Op indices, int64_t depth, builder::Op on_value,
                   builder::Op off_value, int64_t axis) {
  auto builder = indices.GetBuilder();
  auto index_type = indices.GetType();
  auto index_shape = index_type.GetShape();
  auto index_dtype = index_type.GetPrimitiveType();
  auto dtype = on_value.GetType().GetPrimitiveType();
  int64_t dim = static_cast<int64_t>(index_shape.size());                                                     
  if (axis < 0) {
    axis += (dim + 1);                                                                                        
  }
  std::vector<int64_t> output_shape(index_shape);
  output_shape.insert(output_shape.begin() + axis, depth);
  auto output_index_type = builder::Type(output_shape, index_dtype);                                          
  auto output_type = builder::Type(output_shape, dtype);
  auto iota = builder::Iota(builder, axis, output_index_type);                                                
  std::vector<int64_t> broadcast_dims(output_shape.size());
  std::iota(broadcast_dims.begin(), broadcast_dims.end(), 0);                                                 
  broadcast_dims.erase(broadcast_dims.begin() + axis);
  auto broadcast_indices = builder::BroadcastInDim(indices, broadcast_dims,                                   
                                                   output_index_type);                                        
  auto pred = builder::Compare(broadcast_indices, iota, "EQ");
  auto ons = builder::BroadcastInDim(on_value, {}, output_type);
  auto offs = builder::BroadcastInDim(off_value, {}, output_type);                                            
  auto res = builder::Select(pred, ons, offs, output_type);                                                   
  return res;                                                                                                 
}

std::shared_ptr<builder::Builder> build_sample(){
   int depth = 8; //you can change this var if u want different 
   auto hlir_builder = std::make_shared<builder::Builder>();
   hlir_builder->SetShapeInference(true);

   auto indice_type_ = builder::PrimitiveType::F32();
   std::vector<int64_t> indice_shape{8}; 
   builder::Type indice_type(indice_shape, indice_type_);
   builder::Op indice = hlir_builder->CreateInput(indice_type);

   auto off_value_type_ = builder::PrimitiveType::S32();
   std::vector<int64_t> off_value_shape{1};
   builder::Type off_value_type(off_value_shape, off_value_type_);
   std::vector<float> off_value_ = {0};
   builder::Op off_value = builder::Const(hlir_builder, static_cast<void *>(off_value_.data()), off_value_type);
   
   builder::Type on_value_type(off_value_shape, off_value_type_);
   std::vector<float> on_value_ = {1};
   builder::Op on_value = builder::Const(hlir_builder, static_cast<void *>(on_value_.data()), off_value_type);


   builder::Op res = OneHotx(/* builder::Op indices = */indice, 
                              /* int64_t depth = */depth, 
                              /* builder::Op on_value = */ on_value, 
                              /* builder::Op off_value = */ off_value,
                              /* int64_t axis = */ -1);

   res.SetAttribute("op_name", builder::Attribute("OneHot"));
   hlir_builder->SetOutput({res});

   return hlir_builder;
}
