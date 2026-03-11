#include "backend.hpp"

#ifdef TENSORRT_AVAILABLE
#include "tensorrt_backend.hpp"
#endif

#include <chrono>
#include <memory>

#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace auto_aim
{
BackendBase::BackendBase(const std::string & config_path) : config_path_(config_path)
{
  yaml_ = YAML::LoadFile(config_path);
}

bool BackendBase::standarlize(const cv::Mat & img, cv::Mat & target, double & scale)
{
  if (img.empty()) return false;
  target = cv::Mat(model_config_.input_size, img.type(), model_config_.padding_color);
  auto x_scale = static_cast<double>(model_config_.input_size.width) / img.cols;
  auto y_scale = static_cast<double>(model_config_.input_size.height) / img.rows;
  scale = std::min(x_scale, y_scale);

  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);
  auto x = 0;
  auto y = 0;
  if (model_config_.center_pad) {
    x = (target.cols - w) / 2;
    y = (target.rows - h) / 2;
  }

  auto roi = cv::Rect(x, y, w, h);
  cv::resize(img, target(roi), {w, h});
  return (h != 0 && w != 0);
}

bool BackendBase::execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx)
{
  auto execute_start = std::chrono::steady_clock::now();
  cv::Mat input;
  auto preprocess_start = std::chrono::steady_clock::now();
  if (!standarlize(img, input, scale)) return false;
  auto preprocess_end = std::chrono::steady_clock::now();
  last_profile_.preprocess_ms =
    tools::delta_time(preprocess_end, preprocess_start) * 1000.0;
  if (!infer(input, target, ctx)) return false;
  last_profile_.total_ms =
    tools::delta_time(std::chrono::steady_clock::now(), execute_start) * 1000.0;
  return true;
}

Backend::Backend(const std::string config_path) : config_path_(config_path) {}

bool Backend::init(const std::string & model_path, const BackendConfig & model_config)
{
  if (!backend_) backend_allocate(config_path_, model_path);
  if (!backend_) return false;
  return backend_->init(model_path, model_config);
}

std::unique_ptr<BackendCtx> Backend::create_ctx() { return backend_->create_ctx(); }

bool Backend::infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx)
{
  return backend_->infer(input, output, ctx);
}

bool Backend::standarlize(const cv::Mat & img, cv::Mat & target, double & scale)
{
  return backend_->standarlize(img, target, scale);
}

bool Backend::execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx)
{
  return backend_->execute(img, target, scale, ctx);
}

const BackendConfig & Backend::get_model_config() const { return backend_->get_model_config(); }

std::string Backend::get_name() const { return backend_->get_name(); }

const BackendProfile & Backend::get_last_profile() const { return backend_->get_last_profile(); }

void Backend::backend_allocate(const std::string & config_path, const std::string & model_path)
{
  auto yaml = YAML::LoadFile(config_path);
  std::string backend_type = yaml["backend"].as<std::string>("tensorrt");
  auto suffix_pos = model_path.find_last_of('.');
  auto model_ext = suffix_pos == std::string::npos ? "" : model_path.substr(suffix_pos);
#ifdef TENSORRT_AVAILABLE
  if (backend_type == "tensorrt" || backend_type == "jetson" || backend_type == "openvino") {
    if (model_ext != ".onnx" && model_ext != ".engine") {
      tools::logger()->error(
        "TensorRT backend only supports .onnx/.engine models on Jetson, got: {}", model_path);
      backend_ = nullptr;
      return;
    }
    backend_ = std::make_unique<TensorRTBackend>(config_path);
    return;
  }
#endif
  tools::logger()->error("未知的后端类型：{}", backend_type);
  backend_ = nullptr;
}

}  // namespace auto_aim
