#include "openvino_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
OpenVINOBackend::OpenVINOBackend(const std::string & config_path) : BackendBase(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  device_ = yaml["device"].as<std::string>("CPU");
}

bool OpenVINOBackend::init(const std::string & model_path, const BackendConfig & model_config)
{
  try {
    model_config_ = model_config;

    auto model = core_.read_model(model_path);

    if (model_config_.preprocess) {
      ov::preprocess::PrePostProcessor ppp(model);
      auto & input = ppp.input();

      input.tensor()
        .set_element_type(ov::element::u8)
        .set_shape(
          {1, static_cast<size_t>(model_config.input_size.height),
           static_cast<size_t>(model_config.input_size.width), 3})
        .set_layout("NHWC")
        .set_color_format(ov::preprocess::ColorFormat::BGR);

      input.model().set_layout("NCHW");

      input.preprocess()
        .convert_element_type(ov::element::f32)
        .convert_color(ov::preprocess::ColorFormat::RGB)
        .scale(255.0);

      model = ppp.build();
    }

    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("OpenVINO初始化失败：{}", e.what());
    return false;
  }
}

bool OpenVINOBackend::infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx)
{
  if (input.empty()) return false;
  try {
    auto ov_ctx = static_cast<OpenVINOCtx *>(ctx);

    ov::Tensor input_tensor(
      ov::element::u8,
      {1, static_cast<size_t>(model_config_.input_size.height),
       static_cast<size_t>(model_config_.input_size.width), 3},
      const_cast<uchar *>(input.data));

    ov_ctx->infer_request_ = compiled_model_.create_infer_request();
    ov_ctx->infer_request_.set_input_tensor(input_tensor);
    ov_ctx->infer_request_.infer();

    auto output_tensor = ov_ctx->infer_request_.get_output_tensor();
    auto output_shape = output_tensor.get_shape();

    output = cv::Mat(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("OpenVINO推理失败：{}", e.what());
  }
}

std::unique_ptr<BackendCtx> OpenVINOBackend::create_ctx()
{
  auto ctx = std::make_unique<OpenVINOCtx>();
  ctx->infer_request_ = compiled_model_.create_infer_request();
  return ctx;
}

std::string OpenVINOBackend::get_name() const { return "OpenVINO"; }
}  // namespace auto_aim
