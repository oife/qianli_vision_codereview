#include "ort_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
ORTBackend::ORTBackend(const std::string & config_path) : BackendBase(config_path)
{
  device_ = yaml_["device"].as<std::string>();
}

bool ORTBackend::init(const std::string & model_path, const BackendConfig & model_config)
{
  try {
    model_config_ = model_config;

    Ort::SessionOptions session_options;
    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    input_names_ = {session_.GetInputNameAllocated(0, allocator).get()};
    output_names_ = {session_.GetOutputNameAllocated(0, allocator).get()};

    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("ONNX Runtime初始化失败：{}", e.what());
    return false;
  }
}

bool ORTBackend::infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx)
{
  if (input.empty()) return false;
  try {
    auto ort_ctx = static_cast<ORTCtx *>(ctx);

    cv::Mat processed_input;
    preprocess(input, processed_input);

    std::vector<int64_t> input_shape = {
      1, model_config_.input_size.height, model_config_.input_size.width, 3};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<uint8_t>(
      memory_info, const_cast<uint8_t *>(processed_input.data),
      processed_input.total() * processed_input.elemSize(), input_shape.data(), input_shape.size());

    ort_ctx->output_tensors_ = session_.Run(
      Ort::RunOptions{nullptr}, input_names_.data(), &input_tensor, 1, output_names_.data(), 1);

    float * output_data = ort_ctx->output_tensors_[0].GetTensorMutableData<float>();
    auto output_shape = ort_ctx->output_tensors_[0].GetTensorTypeAndShapeInfo().GetShape();

    output = cv::Mat(output_shape[1], output_shape[2], CV_32F, output_data);
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("ONNX Runtime推理失败：{}", e.what());
    return false;
  }
}

void ORTBackend::preprocess(const cv::Mat & src, cv::Mat & dist)
{
  cv::cvtColor(src, dist, cv::COLOR_BGR2RGB);
  dist.convertTo(dist, CV_32F);
  dist /= 255.0;
}

std::unique_ptr<BackendCtx> ORTBackend::create_ctx()
{
  auto ctx = std::make_unique<ORTCtx>();
  return ctx;
}

std::string ORTBackend::get_name() const { return "OnnxRuntime"; }

}  // namespace auto_aim