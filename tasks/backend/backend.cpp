#include "backend.hpp"
#ifdef OPENVINO_AVAILABLE
#include "openvino_backend.hpp"
#endif

#ifdef TENSORRT_AVAILABLE
#include "tensorrt_backend.hpp"
#endif

#ifdef ORT_AVAILABLE
#include "ort_backend.hpp"
#endif

#include <memory>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
BackendBase::BackendBase(const std::string & config_path) : config_path_(config_path)
{
  yaml_ = YAML::LoadFile(config_path);
  device_ = yaml_["device"].as<std::string>("CPU");
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

  int pad_x = 0;
  int pad_y = 0;
  if (model_config_.center_padding) {
    pad_x = (model_config_.input_size.width - w) / 2;
    pad_y = (model_config_.input_size.height - h) / 2;
  }

  auto roi = cv::Rect(pad_x, pad_y, w, h);
  cv::resize(img, target(roi), {w, h});
  return (h != 0 && w != 0);
}

bool BackendBase::execute(const cv::Mat & img, cv::Mat & target, BackendCtx * ctx)
{
  cv::Mat input;
  if (!standarlize(img, input, ctx->scale)) return false;

  // 记录 padding，供后处理使用（如需）
  if (ctx) {
    const int new_w = static_cast<int>(img.cols * ctx->scale);
    const int new_h = static_cast<int>(img.rows * ctx->scale);
    ctx->pad_x = model_config_.center_padding ? (model_config_.input_size.width - new_w) / 2.0 : 0.0;
    ctx->pad_y =
      model_config_.center_padding ? (model_config_.input_size.height - new_h) / 2.0 : 0.0;
  }
  if (!infer(input, target, ctx)) return false;
  return true;
}

void BackendBase::execute_async(const cv::Mat & img, BackendCtx * ctx)
{
  cv::Mat input;
  standarlize(img, input, ctx->scale);

  if (ctx) {
    const int new_w = static_cast<int>(img.cols * ctx->scale);
    const int new_h = static_cast<int>(img.rows * ctx->scale);
    ctx->pad_x = model_config_.center_padding ? (model_config_.input_size.width - new_w) / 2.0 : 0.0;
    ctx->pad_y =
      model_config_.center_padding ? (model_config_.input_size.height - new_h) / 2.0 : 0.0;
  }

  // 这里必须传入预处理后的 input，而不是原图 img
  infer_async(input, ctx);
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

void Backend::infer_async(const cv::Mat & input, BackendCtx * ctx)
{
  backend_->infer_async(input, ctx);
}

void Backend::wait_for_result(cv::Mat & output, BackendCtx * ctx)
{
  backend_->wait_for_result(output, ctx);
}

bool Backend::standarlize(const cv::Mat & img, cv::Mat & target, double & scale)
{
  return backend_->standarlize(img, target, scale);
}

bool Backend::execute(const cv::Mat & img, cv::Mat & target, BackendCtx * ctx)
{
  return backend_->execute(img, target, ctx);
}

void Backend::execute_async(const cv::Mat & img, BackendCtx * ctx)
{
  return backend_->execute_async(img, ctx);
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
#ifdef ORT_AVAILABLE
  if (backend_type == "onnxruntime" || backend_type == "ort") {
    backend_ = std::make_unique<ORTBackend>(config_path);
    return;
  }
#endif
  tools::logger()->error("未知的后端类型：{}", backend_type);
  backend_ = nullptr;
}

}  // namespace auto_aim