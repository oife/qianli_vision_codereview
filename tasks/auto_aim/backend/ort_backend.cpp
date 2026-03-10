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

    env_ = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ort_backend");

    Ort::SessionOptions session_options;
    session_ = Ort::Session(env_, model_path.c_str(), session_options);

    Ort::AllocatorWithDefaultOptions allocator;
    Ort::AllocatedStringPtr input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    Ort::AllocatedStringPtr output_name_ptr = session_.GetOutputNameAllocated(0, allocator);

    input_name_ = input_name_ptr.get();
    output_name_ = output_name_ptr.get();

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
      1, 3, model_config_.input_size.height, model_config_.input_size.width};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    auto input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, processed_input.ptr<float>(),
      processed_input.total() * processed_input.channels(), input_shape.data(), input_shape.size());

    const char * input_name_c = input_name_.c_str();
    const char * output_name_c = output_name_.c_str();

    ort_ctx->output_tensors_ =
      session_.Run(Ort::RunOptions{nullptr}, &input_name_c, &input_tensor, 1, &output_name_c, 1);

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
  dist =
    cv::dnn::blobFromImage(src, 1 / 255.0, model_config_.input_size, cv::Scalar(), true, false);
}

std::unique_ptr<BackendCtx> ORTBackend::create_ctx()
{
  auto ctx = std::make_unique<ORTCtx>();
  return ctx;
}

std::string ORTBackend::get_name() const { return "OnnxRuntime"; }

}  // namespace auto_aim