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
#include "dtu/hlir_builder/hlir_builder.h"
#include "dtu/hlir_builder/hlir_builder_client_ops.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define STRINGIZE_HELPER(x) #x
#define INCLUDE_FILE(x) STRINGIZE_HELPER(x)
#include INCLUDE_FILE(FUNC_FILE)

int main() {
  topsExecutable_t exe_ptr;
  uint64_t output_count = 0;
  std::string op_name(INCLUDE_FILE(FUNC_FILE));
  op_name = getOPName(op_name);

  // stage 1: build the ir
  auto hlir_builder = build_sample();

  // stage 2: compile
  compile(hlir_builder, &exe_ptr);

  // stage 3: init device resource
  topsResource_t res_bundle;
  int ret = initDevice(exe_ptr, res_bundle);
  EXPECT_EQ(ret, 0);

  // stage 4: create input
  // float* exps = g_exps;
  std::vector<void *> input_ptrs(g_input_ptrs);

  std::vector<void *> output_ptrs;
  topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputCount,
                          &output_count);
  std::vector<uint64_t> output_size_list(output_count);
  topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputSizeList,
                          output_size_list.data());
  for (size_t i = 0; i < output_count; i++) {
    uint64_t output_size = output_size_list[i];
    void *host_output = malloc(output_size);
    output_ptrs.push_back(host_output);
  }

  DtuSampleResource dtu_resource;
  // stage 5.1: init resource
  initDtuSampleResource(exe_ptr, res_bundle, dtu_resource);

  // stage 5.2: run
  std::vector<uint64_t> &input_size = dtu_resource._input_size;
  std::vector<uint64_t> &output_size = dtu_resource._output_size;
  int input_count = input_size.size();
  std::vector<void*> &inputs = dtu_resource._inputs;
  std::vector<void*> &outputs = dtu_resource._outputs;
  topsStream_t& stream = dtu_resource._stream;

  // prepare inference data, H2D
  for (size_t i = 0; i < input_count; i++) {
    topsMemcpyAsync(inputs[i], input_ptrs[i], input_size[i], topsMemcpyHostToDevice, stream);
  }

  topsError_t run_ret = topsLaunchExecutableV2(exe_ptr, res_bundle, inputs.data(), 
                                            input_count, outputs.data(), output_count, stream);
  EXPECT_EQ(run_ret, topsSuccess);

  //copy output D2H
  uint64_t dim_index = 0;
  for (size_t i = 0; i < output_count; i++) {
    ret = topsMemcpyAsync(output_ptrs[i], outputs[i], output_size[i], topsMemcpyDeviceToHost, stream);
    EXPECT_EQ(ret, topsSuccess);
  }
  topsStreamSynchronize(stream);

  // stage 6: print output and verify
  std::vector<void*> expects(g_expects);
  printData(output_ptrs, output_size_list, g_exps, "Outputs");
  printData(expects, output_size_list, g_exps, "Expects");
  if(!checkOutputOK(output_ptrs, g_exps, expects, g_exps, output_size_list)){
    ret = 1;
    std::cout << op_name << " Output wrong, check your code!\n";
  } else {
    std::cout << op_name << " Output right!\n";
  }

  // stage 7: release dtu sample resource
  releaseDtuSampleResource(dtu_resource);
  topsDestroyExecutable(exe_ptr);
  topsDestroyResource(res_bundle);
  for (size_t i = 0; i < output_count; i++) {
    free(output_ptrs[i]);
  }
  return ret;
}
