#include "tensorrt_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

namespace auto_aim
{
TensorRTBackend::TensorRTBackend(const std::string & config_path) : BackendBase(config_path) {}

bool TensorRTBackend::init(const std::string & model_path, const BackendConfig & model_config)
{
  std::string ex_model_path = model_path;
   size_t dotPos = model_path.find_last_of(".");
   std::string model_path_raw = model_path.substr(0, dotPos);
   std::string ext = model_path.substr(dotPos + 1);
  if (ext != "engine") {
    if (ext == "onnx") {
      onnx_convert(model_path.c_str(), 4);
      ex_model_path = model_path_raw + ".engine";
    }
    else return false;
  }

  ex_context_ = create_ex_context(ex_model_path);

  cudaStreamCreate(&stream_);
  size_t input_size = model_config_.input_channels * model_config_.input_size.height *
                      model_config_.input_size.width * sizeof(float);
  cudaMalloc(&input_src_device_, input_size);
}

void TensorRTBackend::onnx_convert(const char * onnx_file, int memory_size)
{
  std::string path(onnx_file);
  std::string::size_type i_pos = (path.find_last_of('\\') + 1);
  if (i_pos == 0) i_pos = path.find_last_of('/') + 1;

  std::string model_dir = path.substr(0, i_pos);                       ///< 模型所在文件夹
  std::string model_name = path.substr(i_pos, path.length() - i_pos);  ///< 模型文件名

  std::string model_name_raw =
    model_name.substr(0, model_name.rfind("."));                          ///< 模型文件名（无后缀）
  std::string engine_file_path = model_dir + model_name_raw + ".engine";  ///< 目标转换文件

  nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(logger_);
  const auto explicit_batch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  nvinfer1::INetworkDefinition * network = builder->createNetworkV2(explicit_batch);
  auto parser = nvonnxparser::createParser(*network, logger_);

  parser->parseFromFile(onnx_file, 2);
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    tools::logger()->error("TensorRT 模型加载错误：{}", parser->getError(i)->desc());
  }

  auto config = builder->createBuilderConfig();
  size_t workspace_size = 1024 * 1024 * memory_size;
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
  auto serialized_network = builder->buildSerializedNetwork(*network, *config);

  std::ofstream engine_file(engine_file_path, std::ios::binary);
  engine_file.write(
    static_cast<const char *>(serialized_network->data()), serialized_network->size());
}

std::shared_ptr<nvinfer1::IExecutionContext> TensorRTBackend::create_ex_context(
  std::string model_path)
{
  std::ifstream file_ptr(model_path, std::ios::binary);
  if (!file_ptr.good()) {
    tools::logger()->error("模型文件无法打开");
    return std::shared_ptr<nvinfer1::IExecutionContext>();
  }
  size_t size = 0;
  file_ptr.seekg(0, file_ptr.end);
  size = file_ptr.tellg();
  file_ptr.seekg(0, file_ptr.beg);

  char * model_stream = new char[size];
  file_ptr.read(model_stream, size);
  file_ptr.close();

  nvinfer1::IRuntime * runtime = nvinfer1::createInferRuntime(logger_);
  nvinfer1::ICudaEngine * engine = runtime->deserializeCudaEngine(model_stream, size);
  return std::shared_ptr<nvinfer1::IExecutionContext>(engine->createExecutionContext());
}
}  // namespace auto_aim