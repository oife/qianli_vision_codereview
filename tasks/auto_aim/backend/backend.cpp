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

bool BackendBase::preprocess(const cv::Mat & img, cv::Mat & target, double & scale) const
{
  if (img.empty()) return false;
  auto x_scale = static_cast<double>(model_config_.input_size.width) / img.rows;
  auto y_scale = static_cast<double>(model_config_.input_size.height) / img.cols;
  scale = std::min(x_scale, y_scale);

  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, target(roi), {w, h});
  return (h != 0 && w != 0);
}

bool BackendBase::execute(const cv::Mat & img, cv::Mat & target, double & scale) const
{
  auto input = cv::Mat(model_config_.input_size, CV_8UC3, model_config_.padding_color);
  if (!preprocess(img, input, scale)) return false;
  if (!infer(input, target)) return false;
  return true;
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

bool Backend::infer(const cv::Mat & input, cv::Mat & output) const
{
  return backend_->infer(input, output);
}

bool Backend::preprocess(const cv::Mat & img, cv::Mat target, double & scale) const
{
  return backend_->preprocess(img, target, scale);
}

bool Backend::execute(const cv::Mat & img, cv::Mat & target, double & scale) const
{
  return backend_->execute(img, target, scale);
}

const ModelConfig & Backend::get_model_config() const { return backend_->get_model_config(); }

std::string Backend::get_name() { return backend_->get_name(); }
}  // namespace auto_aim