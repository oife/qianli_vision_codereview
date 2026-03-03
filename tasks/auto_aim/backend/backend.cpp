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

cv::Mat BackendBase::preprocess(
  const cv::Mat & img, double & scale, int & pad_top, int & pad_left) const
{
  if (img.empty()) return cv::Mat();
  cv::Mat resized;

  if (model_config_.keep_aspect_ratio) {
    double x_scale = static_cast<double>(model_config_.input_size.width) / img.cols;
    double y_scale = static_cast<double>(model_config_.input_size.height) / img.rows;
    scale = std::min(x_scale, y_scale);

    int w = static_cast<int>(img.cols * scale);
    int h = static_cast<int>(img.rows * scale);

    cv::resize(img, resized, cv::Size(w, h));

    pad_top = (model_config_.input_size.height - h) / 2;
    pad_left = (model_config_.input_size.width - w) / 2;

    cv::Mat padded(model_config_.input_size, img.type(), model_config_.padding_color);
    resized.copyTo(padded(cv::Rect(pad_left, pad_top, w, h)));
    resized = padded;
  } else {
    cv::resize(img, resized, model_config_.input_size);
    scale = 1.0;
    pad_top = pad_left = 0;
  }

  return resized;
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
#endif
    tools::logger()->error("未知的后端类型：{}", backend_type);
    backend_ = nullptr;
  }
}

bool Backend::init(const std::string & model_path, const ModelConfig & model_config)
{
  return backend_->init(model_path, model_config);
}

cv::Mat Backend::infer(const cv::Mat & input) { return backend_->infer(input); }

cv::Mat Backend::preprocess(const cv::Mat & img, double & scale, int & pad_top, int & pad_left)
{
  return backend_->preprocess(img, scale, pad_top, pad_left);
}

const ModelConfig & Backend::get_model_config() const { return backend_->get_model_config(); }

std::string Backend::get_name() { return backend_->get_name(); }
}  // namespace auto_aim