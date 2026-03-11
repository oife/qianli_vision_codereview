#include "tensorrt_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

namespace auto_aim
{
TensorRTBackend::TensorRTBackend(const std::string & config_path) : BackendBase(config_path) {}

void TensorRTBackend::onnxToEngine(const char * onnx_file, int memory_size)
{
  std::string path(onnx_file);
  std::string::size_type i_pos = (path.find_last_of('\\') + 1);
  if (i_pos == 0) i_pos = path.find_last_of('/') + 1;

  std::string model_path = path.substr(0, i_pos);                      ///< 模型所在文件夹
  std::string model_name = path.substr(i_pos, path.length() - i_pos);  ///< 模型文件名

  std::string model_name_raw =
    model_name.substr(0, model_name.rfind("."));                           ///< 模型文件名（无后缀）
  std::string engine_file_path = model_path + model_name_raw + ".engine";  ///< 目标转换文件

  nvinfer1::IBuilder * builder = nvinfer1::createInferBuilder(logger_);
  const auto explicit_batch =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

  nvinfer1::INetworkDefinition * network = builder->createNetworkV2(explicit_batch);
  auto parser = nvonnxparser::createParser(*network, logger_);

  parser->parseFromFile(onnx_file, 2);
  for (int i = 0; i < parser->getNbErrors(); ++i) {
    tools::logger()->error("TensorRT model load error: {}", parser->getError(i)->desc());
  }
  tools::logger()->info("TensorRT model load successed.");

  auto config = builder->createBuilderConfig();
  size_t workspace_size = 1024 * 1024 * memory_size;
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size);
  auto serialized_network = builder->buildSerializedNetwork(*network, *config);

  std::ofstream engine_file(engine_file_path, std::ios::binary);
  engine_file.write(
    static_cast<const char *>(serialized_network->data()), serialized_network->size());
}
}  // namespace auto_aim