#pragma once

#include "dtu/3_0/runtime/tops/tops_ext.h"
#include "dtu/3_0/runtime/tops/tops_runtime.h"
#include "dtu_compiler/tops_graph_compiler.h"
#include "hlir_builder/hlir_builder.h"

#include <memory>
#include <vector>
#include <string>
#include <iostream>

#define MAX_NUM 10

#define EXPECT_EQ(_src, _dst)                                                  \
  do {                                                                         \
    if ((_src) != (_dst)) {                                                    \
      printf("FAIL: %s:%d,%s() %s != %s\n", __FILE__, __LINE__, __func__,      \
             #_src, #_dst);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define EXPECT_NE(_src, _dst)                                                  \
  do {                                                                         \
    if ((_src) == (_dst)) {                                                    \
      printf("FAIL: %s:%d,%s() %s == %s\n", __FILE__, __LINE__, __func__,      \
             #_src, #_dst);                                                    \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define SAFE_DELETE(p) {        \
    if (NULL != (p)) {          \
        free((p));              \
        (p) = NULL;             \
    }                           \
} 

struct DtuSampleResource {
  topsStream_t _stream;
  std::vector<uint64_t> _input_size;
  std::vector<uint64_t> _output_size;
  std::vector<std::vector<uint64_t>> _output_shapes;
  std::vector<void*> _inputs;
  std::vector<void*> _outputs;
};


bool file_exists(const char *filename);
std::string getOPName(std::string src);
int get_compile_options(const char **options);
void compile(std::shared_ptr<builder::Builder> builder,
             topsExecutable_t *exe_ptr);
int initDevice(topsExecutable_t exe_ptr, topsResource_t& res_bundle,
              int device_id = 0);
int initDtuSampleResource(topsExecutable_t exe_ptr, topsResource_t& res_bundle,
                          DtuSampleResource& dtu_resource);
int run(topsExecutable_t exe_ptr, topsResource_t& res_bundle, DtuSampleResource& dtu_resource,
        std::vector<void *> &input_ptrs, std::vector<void *> &output_ptrs);
int releaseDtuSampleResource(DtuSampleResource& dtu_resource);
template <typename T1, typename T2> 
bool checkOutputOK(std::vector<void*> &outputs, T1* refs1, std::vector<void*> &expects,
                  T2* refs2, std::vector<uint64_t>& output_size_list){
  if(!refs2) {
    return true;
  }
  int outputs_size = outputs.size();
  int expects_size = expects.size();
  if(outputs_size != expects_size) {
    return false;
  }
  for (size_t i = 0; i < outputs_size; i++) {
    T1 *output_data = static_cast<T1 *>(outputs[i]);
    T2 *expect_data = static_cast<T2 *>(expects[i]);
    uint64_t output_size = output_size_list[i];
    for (int j = 0; j < output_size / sizeof(T1); ++j) {
      float diff  = float(output_data[j] - expect_data[j]);
      diff = diff>0?diff:-1*diff; 
      if(diff > 1e-3) {
        return false;
      }
    }
  }
  return true;
}
template <typename T> 
void printData(std::vector<void*> &outputs, std::vector<uint64_t>& output_size_list,
               T* refs, std::string tag){
  std::cout << tag << " :";
  int i = 0;
  for (auto output_ptr : outputs) {
    T *output_data = static_cast<T *>(output_ptr);
    if (!output_data){
      continue;
    }
    
    uint64_t output_size = output_size_list[i++];
    for (int j = 0; j < output_size / sizeof(T); ++j) {
      std::cout << output_data[j] << ", ";
    }
    std::cout << std::endl;
  }
}