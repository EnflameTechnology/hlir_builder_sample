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
shape: [1, 3, 4, 4]  |    kernel shape: [4, 3, 3, 3]               |    out shape: [1, 4, 4, 4]
                     |    which order as Co x Ci H x W             |    
                     |    padding = 1, stride = 1, bias = 0        |
*/

// input feature map has shape 1x3x4x4 in NCHW
float g_lhs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
// kernel has shape 4x3x3x3 (CoCiHW)
float g_rhs[] = {1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
               1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.};
               
float g_exps[] = {12., 18., 18., 12., 18., 27., 27., 18., 18., 27., 27., 18., 12., 18.,
                  18., 12., 12., 18., 18., 12., 18., 27., 27., 18., 18., 27., 27., 18.,
                  12., 18., 18., 12., 12., 18., 18., 12., 18., 27., 27., 18., 18., 27.,
                  27., 18., 12., 18., 18., 12., 12., 18., 18., 12., 18., 27., 27., 18.,
                  18., 27., 27., 18., 12., 18., 18., 12.};

std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);
    auto ptype = builder::PrimitiveType::F32();


    std::vector<int64_t> in_shape{1, 3, 4, 4}; // NCHW
    builder::Type input_type(in_shape, ptype);
    builder::Op input_feature_map = hlir_builder->CreateInput(input_type);

    // kernel was described in this order: Cout, Cin, H, W
    std::vector<int64_t> weight_shape{4, 3, 3, 3};
    builder::Type weight_type(weight_shape, ptype);
    builder::Op weight = hlir_builder->CreateInput(weight_type);

    std::vector<int64_t> result_shape{1, 4, 4, 4}; // NCHW
    builder::Type result_type(result_shape, ptype);

    builder::ConvDimensionNumbers convDimNums(/* int64_t input_batch_dimension = */0, 
                                              /* int64_t input_feature_dimension = */1, 
                                              /* std::vector<int64_t> input_spatial_dimensions = */{4, 4}, 
                                              /*  int64_t kernel_input_feature_dimension = */3, 
                                              /* int64_t kernel_output_feature_dimension = */4, 
                                              /* std::vector<int64_t> kernel_spatial_dimensions = */{3, 3}, 
                                              /* int64_t output_batch_dimension = */0, 
                                              /* int64_t output_feature_dimension = */1, 
                                              /* std::vector<int64_t> output_spatial_dimensions = */{4, 4});

    builder::Op res = builder::Conv(/*builder::Op lhs= */input_feature_map, 
                                    /*builder::Op rhs= */weight,
                                    /*builder::ConvDimensionNumbers dimension_numbers= */convDimNums,
                                    /*std::vector<int64_t> window_strides =  */{1, 1},
                                    /*std::vector<int64_t> padding =  */{1, 1, 1, 1}, 
                                    /*std::vector<int64_t> lhs_dilation =  */{1, 1}, 
                                    /*std::vector<int64_t> rhs_dilation =  */{1, 1}, 
                                    /* std::vector<int64_t> window_reversal =  */{}, 
                                    /* std::string auto_pad =  */"", 
                                    /* int64_t feature_group_count =  */1, 
                                    /* int64_t batch_group_count =  */1, 
                                    /* std::vector<std::string> precision_config =  */{"DEFAULT", "DEFAULT"}, 
                                    /*  builder::Type resultType = */ result_type);


    res.SetAttribute("op_name", builder::Attribute("Conv"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}
