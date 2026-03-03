#include "openvino_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <opencv2/opencv.hpp>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
OpenVINOBackend::OpenVINOBackend(const std::string & config_path) : Backend(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  device_ = yaml["device"].as<std::string>("CPU");
}

bool OpenVINOBackend::init(const std::string & model_path)
{
  try {
    auto model = core_.read_model(model_path);

    ov::preprocess::PrePostProcessor ppp(model);
    auto & input = ppp.input();

    input.tensor()
      .set_element_type(ov::element::u8)
      .set_shape({1, input_size_.height, input_size_.width, 3})
      .set_layout("NHWC")
      .set_color_format(ov::preprocess::ColorFormat::BGR);

    input.model().set_layout("NCHW");

    input.preprocess()
      .convert_element_type(ov::element::f32)
      .convert_color(ov::preprocess::ColorFormat::RGB)
      .scale(255.0);

    model = ppp.build();
    compiled_model_ = core_.compile_model(
      model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    return true;
  } catch (const std::exception & e) {
    tools::logger()->error("OpenVINO初始化失败：{}", e.what());
    return false;
  }
}

cv::Mat OpenVINOBackend::infer(const cv::Mat & input)
{
  ov::Tensor input_tensor(
    ov::element::u8, {1, input_size_.height, input_size_.width, 3},
    const_cast<uchar *>(input.data));

  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();

  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  return output;
}

cv::Size OpenVINOBackend::get_input_size() const { return input_size_; }

}  // namespace auto_aim