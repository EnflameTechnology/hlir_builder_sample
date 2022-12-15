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

/*
input fearure map    |    convolution kernel map                   |    result map
shape: [1, 4, 4, 3]  |    kernel shape: [3, 3, 3, 4]               |    out shape: [1, 4, 4, 4]
                     |    which order as Size x Size x Cin x Cout  |    
                     |    padding = 1, stride = 1, bias = 0        |
*/

// inputs and outputs
// input feature map has shape 1x4x4x3
float g_lhs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
// kernel has shape 3x3x3x4, Size is 3x3, Cin is 3, Cout is 4
float g_rhs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
float g_bias[] = {0., 0., 0., 0.};
float g_exps[] = {12., 12., 12., 12., 18., 18., 18., 18., 18., 18., 18., 18., 12.,
                  12., 12., 12., 18., 18., 18., 18., 27., 27., 27., 27., 27., 27.,
                  27., 27., 18., 18., 18., 18., 18., 18., 18., 18., 27., 27., 27.,
                  27., 27., 27., 27., 27., 18., 18., 18., 18., 12., 12., 12., 12.,
                  18., 18., 18., 18., 18., 18., 18., 18., 12., 12., 12., 12.};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), 
                                    static_cast<void *>(g_rhs),
                                    static_cast<void *>(g_bias)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);
    auto ptype = builder::PrimitiveType::F32();

    // input feature map is described in the order of NHWC
    std::vector<int64_t> in_shape{1, 4, 4, 3};
    builder::Type input_type(in_shape, ptype);
    builder::Op input_feature_map = hlir_builder->CreateInput(input_type);

    // kernel was described in this order: Size, Size, Cin, Cout
    std::vector<int64_t> weight_shape{3, 3, 3, 4};
    builder::Type weight_type(weight_shape, ptype);
    builder::Op weight = hlir_builder->CreateInput(weight_type);

    std::vector<int64_t> bias_shape{4};
    builder::Type bias_type(bias_shape, ptype);
    builder::Op bias = hlir_builder->CreateInput(bias_type);

    std::vector<builder::Op> input_node = {input_feature_map, weight, bias};
    builder::Op res = builder::Conv2D(input_node, 1, "NOTSET", "NHWC", {1,1}, {1, 1, 1, 1});

    res.SetAttribute("op_name", builder::Attribute("Conv"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}