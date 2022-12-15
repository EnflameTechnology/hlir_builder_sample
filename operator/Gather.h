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

in1 = tf.constant(data, dtype=tf.float32, shape=(4,3))
indices=tf.constant([3,1])
y = tf.gather(in1, indices, axis=0)

[[ 0.  1.  2.]
 [10. 11. 12.]      ->      [[30. 31. 32.]
 [20. 21. 22.]               [10. 11. 12.]]
 [30. 31. 32.]]

*/

float g_lhs[] = {8, 4, 2, 9, 
                 8, 6, 6, 3, 
                 3, 4, 9, 4, 
                 8, 2, 6, 6};
float g_exps[] = {4, 9, 4, 9, 8, 6};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};


static builder::Op Gather(builder::Op data, builder::Op indices, int64_t axis,
                   builder::Type OutType) {
  std::vector<int64_t> offset_dim;

  for (int64_t i = 0; i < axis; i++) {
    offset_dim.emplace_back(i);
  }
  auto data_shape = data.GetType().GetShape();
  auto indices_shape = indices.GetType().GetShape();
  for (int64_t i = axis + 1; i < data_shape.size(); i++) {
    offset_dim.emplace_back(i - 1 + indices_shape.size());
  }
  std::vector<int64_t> slice_size(data_shape);
  slice_size[axis] = 1;
  builder::GatherDimensionNumbers gnums(offset_dim, {axis}, {axis},
                                        indices_shape.size());
  auto op = builder::Gather(data, indices, gnums, slice_size, false, OutType);
  return op;
}

std::shared_ptr<builder::Builder> build_sample(){
  auto hlir_builder = std::make_shared<builder::Builder>();
  hlir_builder->SetShapeInference(true);
  auto dtype = builder::PrimitiveType::F32();
  auto indice_dtype = builder::PrimitiveType::S64();

  std::vector<int64_t> in_shape{4, 3};

  std::vector<int64_t> out_shape{2, 3};
  std::vector<int64_t> indices_shape{2};
  builder::Type input_type(in_shape, dtype);
  builder::Type output_type(out_shape, dtype);
  builder::Type indice_type(indices_shape, indice_dtype);
  auto data = hlir_builder->CreateInput(input_type);

  //create indice_data 
  std::vector<int64_t> indice_data{3, 1};
  auto indices = builder::Const(hlir_builder, static_cast<void *>(indice_data.data()), indice_type);

  auto res = Gather(data, indices, 0, output_type);
  hlir_builder->SetOutput({res});

  res.SetAttribute("op_name", builder::Attribute("Gather"));
   return hlir_builder;
}