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

// y = tf.concat([in1, in2], 1)

// [[-11, 10, 11], [-21, 21, -10]]   axis=1  ->   [[-11.  10.  11.  11.  10. -11.]
// [[11, 10, -11], [21, 21, 10]]                  [-21.  21. -10.  21.  21.  10.]]

// inputs and outouts
float g_lhs[] = {-11, 10, 11, -21, 21, -10};
float g_rhs[] = {11, 10, -11, 21, 21, 10};
float g_exps[] = {-11, 10, 11, -21, 21, -10, 
                   11, 10, -11, 21, 21, 10};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs), 
                                    static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample(){
    auto hlir_builder = std::make_shared<builder::Builder>();
    hlir_builder->SetShapeInference(true);
    auto in_dtype = builder::PrimitiveType::F32();
    std::vector<int64_t> in_shape{2, 3};
    std::vector<int64_t> out_shape{4, 3};
    builder::Type in_type(in_shape, in_dtype);
    builder::Type out_type(out_shape, in_dtype);
    auto lhs = hlir_builder->CreateInput(in_type);
    auto rhs = hlir_builder->CreateInput(in_type);
    int axis = 0;
    auto res = builder::Concatenate({lhs, rhs}, axis, out_type);
    res.SetAttribute("op_name", builder::Attribute("Concat"));
    hlir_builder->SetOutput({res});
    hlir_builder->Print(std::cout,
                        builder::PrintingFlags::ElideLargeElementsAttrs);
    return hlir_builder;
}
