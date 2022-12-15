#include "dtu_utils.h"

#include <iostream>
#include <unistd.h>

bool file_exists(const char *filename) { return (access(filename, 0) == 0); }

int get_compile_options(const char **options){
    std::string arch = builder::GetDeviceArch();
    //TODO, 0: static shape 1: dynamic shape
    bool is_dynamic_shape = 0;
    if (arch == "gcu200") {
        //pavo
        if (is_dynamic_shape) {
          const char *options_tmp[] = {"-arch=gcu200", "-resource=4c24s", "-hlir=hlir-pytorch-pipeline{dynamic-shape=true enable-fusion=false}"};
          memcpy(options, options_tmp, sizeof(options_tmp));
        } else {
          const char *options_tmp[] = {"-arch=gcu200", "-resource=4c24s", "-hlir=hlir-pytorch-pipeline{dynamic-shape=false enable-fusion=false}"};
          memcpy(options, options_tmp, sizeof(options_tmp));
        }
    } else if (arch == "gcu210") {
        // darado
        if (is_dynamic_shape) {
          const char *options_tmp[]  = {"-arch=gcu210", "-resource=1c4s", "-hlir=tops-hlir-pipeline{shape-inference=true dynamic-shape=true}"};
          memcpy(options, options_tmp, sizeof(options_tmp));
        } else {
          const char *options_tmp[]  = {"-arch=gcu210", "-resource=1c4s", "-hlir=tops-hlir-pipeline{shape-inference=true dynamic-shape=false}"};
          memcpy(options, options_tmp, sizeof(options_tmp));
        }
    } else {
        std::cout << "Unsupported arch: " << arch << std::endl;
        return -1;
    }
    std::cout << "arch: " << arch << ", is_dynamic_shape: " << is_dynamic_shape<< std::endl;
    //std::cout << "options: " << options[0] << ", " << options[1] << ", " << options[2] << std::endl;
    return 0;
}


void compile(std::shared_ptr<builder::Builder> builder,
             topsExecutable_t *exe_ptr) {
  topsgraphProgram program;
  // get the built IR from builder
  auto hlir_module = builder->GetModule();
  auto ret = topsgraphCreateProgramFromModule(&program, hlir_module.get());
  const char **options = new const char*[3];
  get_compile_options(options);

  topsgraphCompileProgram(program, 3, options);
  delete [] options;
  // get binary size and binary data
  size_t binary_size = 0;
  topsgraphGetBinSize(program, &binary_size);
  char *binary = new char[binary_size];
  topsgraphGetBin(program, binary);
  topsCreateExecutable(exe_ptr, binary, binary_size);
  delete [] binary;
  topsgraphDestroyProgram(&program);
  std::cout << "[INFO] compile done." << std::endl;
  return;
}


std::string getOPName(std::string src){
  int start=src.find_last_of('/');
  int end=src.find_last_of('.');
	std::string dst(src.substr(start+1, end-start-1));
  return dst;
}

int initDevice(topsExecutable_t exe_ptr, topsResource_t& res_bundle, int device_id){
  int count = 0;
  topsError_t ret = topsGetDevice(&device_id);
  EXPECT_EQ(ret, topsSuccess);
  topsGetDeviceCount(&count);
  if (device_id > count) {
    return -1;
  }
  ret = topsSetDevice(device_id);
  EXPECT_EQ(ret, topsSuccess);

  topsCreateResourceForExecutable(&res_bundle, exe_ptr);
  return 0;
}
int initDtuSampleResource(topsExecutable_t exe_ptr, topsResource_t& res_bundle,
                          DtuSampleResource& dtu_resource){
  void *dev_input = nullptr;
  void *dev_output = nullptr;

  topsStreamCreate(&dtu_resource._stream);
  topsCreateResourceForExecutable(&res_bundle, exe_ptr);

  // 1.1 query InputCount,output_count
  uint64_t input_count = 0, output_count = 0;
  EXPECT_EQ(topsExecutableQueryInfo(exe_ptr, topsExecutableInfoInputCount,
                                    &input_count),
            topsSuccess);
  EXPECT_EQ(topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputCount,
                                    &output_count),
            topsSuccess);
  std::cout << "input_count = " << input_count
            << ", output_count = " << output_count << std::endl;
  
  // 1.2 query InputSize,output_size
  auto &input_size = dtu_resource._input_size;
  auto &output_size = dtu_resource._output_size;
  input_size.resize((int)input_count);
  output_size.resize(output_count);
  EXPECT_EQ(topsExecutableQueryInfo(exe_ptr, topsExecutableInfoInputSizeList,
                                    input_size.data()),
            topsSuccess);
  EXPECT_EQ(topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputSizeList,
                                    output_size.data()),
            topsSuccess);

  // 1.3 query output shapes
  std::vector<uint64_t> output_rank_list(output_count, 0);
  topsError_t ret = topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputRank,
                                output_rank_list.data());
  EXPECT_EQ(ret, topsSuccess);

  uint64_t output_dims_size = 0;
  for (int i = 0; i < output_count; i++) {
    output_dims_size += output_rank_list[i];
  }

  std::vector<uint64_t> output_dim_list(output_dims_size, 0);
  ret = topsExecutableQueryInfo(exe_ptr, topsExecutableInfoOutputDimsList,
                                output_dim_list.data());
  EXPECT_EQ(ret, topsSuccess);

  uint64_t *outputs = output_dim_list.data();
  for (int i = 0; i < output_count; i++) {
    std::vector<uint64_t> output_shape;
    for (int j = 0; j < output_rank_list[i]; j++) {
      output_shape.push_back(*outputs++);
    }
    dtu_resource._output_shapes.push_back(output_shape);
  }

  // 2. prepare data addr, H2D
  for (size_t i = 0; i < input_count; i++) {
    topsMallocForResource(&dev_input, input_size[i], res_bundle);
    dtu_resource._inputs.push_back(dev_input);
  }
  for (size_t i = 0; i < output_count; i++) {
    topsMallocForResource(&dev_output, output_size[i], res_bundle);
    dtu_resource._outputs.push_back(dev_output);
  }
  
  return 0;
}

int run(topsExecutable_t exe_ptr, topsResource_t& res_bundle, DtuSampleResource& dtu_resource,
        std::vector<void *> &input_ptrs, std::vector<void *> &output_ptrs){
  std::vector<uint64_t> &input_size = dtu_resource._input_size;
  std::vector<uint64_t> &output_size = dtu_resource._output_size;
  int input_count = input_size.size();
  int output_count = output_size.size();
  std::vector<void*> &inputs = dtu_resource._inputs;
  std::vector<void*> &outputs = dtu_resource._outputs;
  topsStream_t& stream = dtu_resource._stream;

  // prepare data, H2D
  for (size_t i = 0; i < input_count; i++) {
    topsMemcpyAsync(inputs[i], input_ptrs[i], input_size[i], topsMemcpyHostToDevice, stream);
  }

  topsError_t ret = topsLaunchExecutableV2(exe_ptr, res_bundle, inputs.data(), 
                                            input_count, outputs.data(), output_count, stream);
  EXPECT_EQ(ret, topsSuccess);

  //copy output 
  uint64_t dim_index = 0;
  for (size_t i = 0; i < output_count; i++) {
    // D2H
    ret = topsMemcpyAsync(output_ptrs[i], outputs[i], output_size[i], topsMemcpyDeviceToHost, stream);
    EXPECT_EQ(ret, topsSuccess);
  }
  topsStreamSynchronize(stream);

  return 0;
}

int releaseDtuSampleResource(DtuSampleResource& dtu_resource){
  // release data
  int input_count = dtu_resource._input_size.size();
  int output_count = dtu_resource._output_size.size();
  for (size_t i = 0; i < input_count; i++) {
    topsFree(dtu_resource._inputs[i]);
  }
  for (size_t i = 0; i < output_count; i++) {
    topsFree(dtu_resource._outputs[i]);
  }
  topsStreamDestroy(dtu_resource._stream);
  return 0;
}
