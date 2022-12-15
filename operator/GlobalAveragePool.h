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

// l=[1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2]
// l=np.array(l, dtype=float()).reshape(2, 3, 2, 2)
// y1=tf.keras.layers.GlobalAveragePooling2D(data_format='channels_first')(l)
// print(y1) ---> [[2.5 6.5 3. ][2.5 6.5 3. ]]
// [[[[1 2]                 
//    [3 4]]                                        
//   [[5 6]                 
//    [7 8]]                 
//   [[9 0]               [[2.5 6.5 3.]
//    [1 2]]]     ->       [2.5 6.5 3.]]           
//  [[[1 2]                   
//    [3 4]]                  
//   [[5 6]
//    [7 8]]
//   [[9 0]
//    [1 2]]]]

// inputs and outputs
float g_lhs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2,
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2};
float g_exps[] = {2.5, 6.5, 3., 2.5, 6.5, 3.};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto inpType = builder::PrimitiveType::F32();
  std::vector<int64_t> in_shape{2, 3, 2, 2};
  std::vector<int64_t> out_shape{2, 3, 1, 1};
  builder::Type InType(in_shape, inpType);
  builder::Type OutType(out_shape, inpType);
  auto input = hlir_builder->CreateInput(InType);
  // dims (the dimensions of HW or D1..Dn)
  std::vector<int64_t> dims{2, 3};
  auto res = builder::GlobalAveragePool(input, dims, OutType);
  res.SetAttribute("op_name", builder::Attribute("GlobalAveragePool"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);
  return hlir_builder; 
}