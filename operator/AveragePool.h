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


// tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(2,2), padding='same', data_format='channels_last')
// [[[[ 1.  2.  3.  4.]
// [ 5.  6.  7.  8.]   ->     [[[[6    7.5]
// [ 9. 10. 11. 12.]   ->        [12  13.5]]]]
// [13. 14. 15. 16.]]]]


// inputs and outouts
float g_lhs[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
float g_exps[] = {6, 7.5, 12, 13.5};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();

  std::vector<int64_t> in_shape{1, 1, 4, 4};
  std::vector<int64_t> out_shape{1, 1, 2, 2};
  builder::Type input_type(in_shape, dtype);
  builder::Type output_type(out_shape, dtype);

  auto input = hlir_builder->CreateInput(input_type);

  auto res = builder::AveragePool(
      input, /*dim=*/{2, 3}, /*kernel_shape=*/{3, 3},
      /*ceil_mode=*/true, /*count_include_pad=*/false, 
      /*stride=*/{2, 2}, /*padding=*/{0, 1, 0, 1},
      /*auto_pad = */"", output_type);
  res.SetAttribute("op_name", builder::Attribute("AveragePool"));
  hlir_builder->SetOutput({res});
  hlir_builder->Print(std::cout,
                      builder::PrintingFlags::ElideLargeElementsAttrs);

  return hlir_builder;
}