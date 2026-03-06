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

bool BackendBase::standarlize(const cv::Mat & img, cv::Mat & target, double & scale)
{
  if (img.empty()) return false;
  target = cv::Mat(model_config_.input_size, img.type(), model_config_.padding_color);
  auto x_scale = static_cast<double>(model_config_.input_size.width) / img.cols;
  auto y_scale = static_cast<double>(model_config_.input_size.height) / img.rows;
  scale = std::min(x_scale, y_scale);

  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(img, target(roi), {w, h});
  return (h != 0 && w != 0);
}

bool BackendBase::execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx)
{
  cv::Mat input;
  if (!standarlize(img, input, scale)) return false;
  if (!infer(input, target, ctx)) return false;
  return true;
}

Backend::Backend(const std::string config_path) { backend_allocate(config_path); }

bool Backend::init(const std::string & model_path, const BackendConfig & model_config)
{
  return backend_->init(model_path, model_config);
}

std::unique_ptr<BackendCtx> Backend::create_ctx() { return backend_->create_ctx(); }

bool Backend::infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx)
{
  return backend_->infer(input, output, ctx);
}

bool Backend::standarlize(const cv::Mat & img, cv::Mat target, double & scale)
{
  return backend_->standarlize(img, target, scale);
}

bool Backend::execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx)
{
  return backend_->execute(img, target, scale, ctx);
}

const BackendConfig & Backend::get_model_config() const { return backend_->get_model_config(); }

std::string Backend::get_name() const { return backend_->get_name(); }

void Backend::backend_allocate(const std::string config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  std::string backend_type = yaml["backend"].as<std::string>();
#ifdef OPENVINO_AVAILABLE
  if (backend_type == "openvino") {
    backend_ = std::make_unique<OpenVINOBackend>(config_path);
    return;
  }
#endif
#ifdef TENSORRT_AVAILABLE
  if (backend_type == "tensorrt") {
    backend_ = std::make_unique<TensorRTBackend>(config_path);
    return;
  }
#endif
  tools::logger()->error("未知的后端类型：{}", backend_type);
  backend_ = nullptr;
}

}  // namespace auto_aim