#include "backend.hpp"
#ifdef OPENVINO_AVAILABLE
#include "openvino_backend.hpp"
#endif

#ifdef TENSORRT_AVAILABLE
#include "tensorrt_backend.hpp"
#endif

#include <yaml-cpp/yaml.h>

#include <memory>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
BackendBase::BackendBase(const std::string & config_path) : config_path_(config_path) {}

void BackendBase::preprocess(const cv::Mat & img, double & scale) const
{
  if (img.empty()) return;
  auto x_scale = static_cast<double>(model_config_.input_size.width) / img.rows;
  auto y_scale = static_cast<double>(model_config_.input_size.height) / img.cols;
  scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  auto input = cv::Mat(model_config_.input_size, CV_8UC3, model_config_.padding_color);
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, input(roi), {w, h});
}

Backend::Backend(const std::string config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  std::string backend_type = yaml["backend"].as<std::string>();
#ifdef OPENVINO_AVAILABLE
  if (backend_type == "openvino") {
    backend_ = std::make_unique<OpenVINOBackend>(config_path);
  }
#endif
#ifdef TENSORRT_AVAILABLE
  if (backend_type == "tensorrt") {
    backend_ = std::make_unique<TensorRTBackend>(config_path);
  }
#endif
  tools::logger()->error("未知的后端类型：{}", backend_type);
  backend_ = nullptr;
}

bool Backend::init(const std::string & model_path, const ModelConfig & model_config)
{
  return backend_->init(model_path, model_config);
}

cv::Mat Backend::infer(const cv::Mat & input) { return backend_->infer(input); }

cv::Mat Backend::preprocess(const cv::Mat & img, double & scale)
{
  backend_->preprocess(img, scale);
}

const ModelConfig & Backend::get_model_config() const { return backend_->get_model_config(); }

std::string Backend::get_name() { return backend_->get_name(); }
}  // namespace auto_aim