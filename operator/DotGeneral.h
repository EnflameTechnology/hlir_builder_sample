/*=======================================================================
 *Copyright 2022 The Enflame Tech Company. All Rights Reserved.
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

#include <iostream>
#include <sstream>
#include <string>
#include <vector>


// | 0  1 |   | 1  2 |   | 3  4 |
// |      | * |      | = |      |
// | 2  3 |   | 3  4 |   |11 16 |

// inputs and outputs
float g_lhs[] = {0, 1, 2, 3};
float g_rhs[] = {1, 2, 3, 4};
float g_exps[] = {3, 4, 11, 16};
std::vector<void *> g_input_ptrs = {static_cast<void *>(g_lhs),
                                    static_cast<void *>(g_rhs)};
std::vector<void*> g_expects = {static_cast<void *>(g_exps)};

std::shared_ptr<builder::Builder> build_sample() {
  auto builder = std::make_shared<builder::Builder>();
  builder->SetShapeInference(true);
  auto ptype = builder::PrimitiveType::F32();
  std::vector<int64_t> shape = {2, 2};
  builder::Type type(shape, ptype);
  auto arg0 = builder->CreateInput(type);
  auto arg1 = builder->CreateInput(type);
  builder::DotDimensionNumbers dims_attr({}, {}, {1}, {0});
  auto res = builder::DotGeneral(arg0, arg1, dims_attr);
  res.SetAttribute("op_name", builder::Attribute("MatMul"));
  builder->SetOutput({res});
  // builder->Print(std::cout, builder::PrintingFlags::ElideLargeElementsAttrs);
  return builder;
}