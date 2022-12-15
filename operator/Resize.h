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

[[[[1 2 3]                  [[[[1. 2. 3.]
   [4 5 6]         ->          [7. 8. 9.]]
   [7 8 9]]]]                 [[1. 2. 3.]
                               [7. 8. 9.]]]]
*/
float g_lhs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
float g_exps[] = {1, 2, 3, 4, 5, 6, 7, 8, 9};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);

    // specify the input tensor shape and data type
    // this is for the input feature map
    auto Iptype = builder::PrimitiveType::F32();
    std::vector<int64_t> in_shape{1, 1, 3, 3};
    builder::Type input_type(in_shape, Iptype);
    builder::Op input_feature = hlir_builder->CreateInput(input_type);

    // specify the output tensor shape and data type
    // this is for the output feature map
    auto Optype = builder::PrimitiveType::F32();
    std::vector<int64_t> out_shape{1, 2, 2, 3};
    builder::Type output_type(out_shape, Optype);

    // this is for the roi
    // 1-D tensor given as [start1, ..., startN, end1, ..., endN], 
    // where N is the rank of input tensor or the length of axes
    // it only effect if the coordinate_transformation_mode is 5 (aka "tf_crop_and_resize")
    auto Rptype = builder::PrimitiveType::S32();
    std::vector<int64_t> roi_shape{4*2};
    builder::Type roi_type(roi_shape, Rptype);
    std::vector<int64_t> roi_ = {0, 1, 0, 1, 0, 3, 0, 3};
    builder::Op roi = builder::Const(hlir_builder, static_cast<void *>(roi_.data()), roi_type);

    // this is for the size
    // the size it to describe the size of output tensor you want it to have.
    // it has 4 elements, each element stands for N, H, W, C
    builder::Type empty_type({}, builder::PrimitiveType::F16());
    auto size = builder::Const(hlir_builder, nullptr, empty_type);

    // this is for the scale
    // the scale is to describe the down/up sample scale of the resized output tensor
    // it has 4 elements, each element stands for N, H, W, C
    // and the first and second element should always be 1
    // ususally, size and scale are functionally similar, 
    // you can only specify one of them
    auto Scptype = builder::PrimitiveType::F32();
    std::vector<int64_t> scale_ = {1, 1, 1, 1};
    builder::Type scale_type(scale_, Scptype);
    builder::Op scale = builder::Const(hlir_builder, static_cast<void *>(scale_.data()), scale_type);

      
    builder::Op res = builder::Resize(/* input = */input_feature, 
                                      /* roi = */roi, 
                                      /* scales = */scale, 
                                      /* sizes = */size, 
                                      /* mode = */0, 
                                      /* coordinate_transformation_mode = */0, 
                                      /* exclude_outside = */false, 
                                      /* nearest_mode =  */1, 
                                      /* extrapolation_value =  */0.0, 
                                      /* cubic_coeff_a =  */-0.75, 
                                      /* resize_dimensions =  */{1,2},
                                      /* builder::Type() */output_type);

    /* 
        mode              |      coordinate_transformation_mode    |    nearest_mode
        -------------------------------------------------------------------------------
        {"nearest", 0}    |      {"half_pixel", 0}                 |    {"simple", 0}
        {"linear",  1}    |      {"asymmetric", 1}                 |    {"round_prefer_floor", 1}
        {"cubic",   2}    |      {"pytorch_half_pixel", 2}         |    {"round_prefer_ceil", 2}
                          |      {"tf_half_pixel_for_nn", 3}       |    {"floor", 3}
                          |      {"align_corners", 4}              |
                          |      {"tf_crop_and_resize", 5}         |
    */ 

    res.SetAttribute("op_name", builder::Attribute("resize"));
    hlir_builder->SetOutput({res});

    return hlir_builder;
}