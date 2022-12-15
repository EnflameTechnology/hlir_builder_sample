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
std::vector<float> lhs(1*4*4*5, 1.0);
float*g_lhs = lhs.data();
float g_exps[] = {347.705, 347.705, 347.705, 347.705, 347.705, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  347.705, 347.705, 347.705, 347.705, 347.705, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  922.635, 922.635, 922.635, 922.635, 922.635, 
                  922.635, 922.635, 922.635, 922.635, 922.635, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  922.635, 922.635, 922.635, 922.635, 922.635, 
                  922.635, 922.635, 922.635, 922.635, 922.635, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  347.705, 347.705, 347.705, 347.705, 347.705, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  567.091, 567.091, 567.091, 567.091, 567.091, 
                  347.705, 347.705, 347.705, 347.705, 347.705};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};


builder::Op build_Relu(builder::Op input_op){
  auto res = builder::Relu(input_op);
  return res;
}

builder::Op build_BN(std::shared_ptr<builder::Builder> builder, builder::Op input_op){

  auto ptype = builder::PrimitiveType::F32();
  std::vector<int64_t> scale_shape{5};
  builder::Type scale_type(scale_shape, ptype);
  std::vector<float> scale_data(5, 1.);
  builder::Op scale = builder::Const(builder, static_cast<void *>(scale_data.data()), scale_type);

  builder::Type offset_type(scale_shape, ptype);
  std::vector<float> offset_data(5, 1.);
  builder::Op offset = builder::Const(builder, static_cast<void *>(offset_data.data()), offset_type);

  builder::Type mean_type(scale_shape, ptype);
  std::vector<float> mean_data = {0.34442666, 0.34195662, 0.31565964, 0.57439, 0.423424};
  builder::Op mean = builder::Const(builder, static_cast<void *>(mean_data.data()), mean_type);

  builder::Type variance_type(scale_shape, ptype);
  std::vector<float> variance_data = {0.07396074, 0.03984508, 0.11567877, 0.01094088, 0.018201};
  builder::Op variance = builder::Const(builder, static_cast<void *>(variance_data.data()), variance_type);

  float epsilon = 1e-5;

  builder::Op res = builder::BatchNormInference(/* builder::Op operand = */ input_op, 
                                                /* builder::Op scale = */scale, 
                                                /* builder::Op offset = */ offset, 
                                                /* builder::Op mean = */ mean, 
                                                /* builder::Op variance = */ variance, 
                                                /* float epsilon = */ epsilon, 
                                                /* int64_t feature_index = */ 3);

  return res;
}

builder::Op build_Conv2d(std::shared_ptr<builder::Builder> builder, builder::Op input_op){
  std::vector<int64_t> weight_shape{3, 3, 5, 5};
  builder::Type weight_type(weight_shape, builder::PrimitiveType::F32());
  std::vector<float> weight_data(3*3*5*5, 0.05f);
  builder::Op weight = builder::Const(builder, static_cast<void *>(weight_data.data()), weight_type);
  
  std::vector<int64_t> bias_shape{5};
  builder::Type bias_type(bias_shape, builder::PrimitiveType::F32());
  std::vector<float> bias_data(5, -2.f);
  builder::Op bias = builder::Const(builder, static_cast<void *>(bias_data.data()), bias_type);

  std::vector<builder::Op> input = {input_op, weight, bias};
  auto res = builder::Conv2D(/* std::vector<builder::Op> */ input, 
                             /* int64_t group = */ 1, 
                             /* std::string auto_pad = */ "NOTSET", 
                             /* std::string layout = */ "NHWC", 
                             /* std::vector<int64_t> stride = */ {1, 1}, 
                             /* std::vector<int64_t> padding = */ {1, 1, 1, 1});
  return res;
}

builder::Op build_Add(builder::Op arg0, builder::Op arg1){
  builder::Op res = builder::Add(arg0, arg1);
  return res;
}


builder::Op build_block(std::shared_ptr<builder::Builder> builder, builder::Op input_op){
    auto bn   = build_BN(builder, input_op);
    auto relu = build_Relu(bn);
    auto conv = build_Conv2d(builder, relu);
    auto add  = build_Add(input_op, conv);
    auto block_output = build_Relu(add);
    return block_output;
}


void build_graph(std::shared_ptr<builder::Builder> builder, builder::Type input_type){

  builder::Op input = builder->CreateInput(input_type);

  builder::Op block_output_0 = build_block(builder, input);
  builder::Op block_output_1 = build_block(builder, block_output_0);
  builder::Op block_output_2 = build_block(builder, block_output_1);

  builder->SetOutput({block_output_2});

}

std::shared_ptr<builder::Builder> build_sample(){
  auto builder = std::make_shared<builder::Builder>();
  builder->SetShapeInference(true);

  // set constant input
  auto ptype = builder::PrimitiveType::F32();
  std::vector<int64_t> input_shape{1, 4, 4, 5};
  builder::Type input_type(input_shape, ptype);

  // build and compile graph
  build_graph(builder, input_type);
  return builder;
}