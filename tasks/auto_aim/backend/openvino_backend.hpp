#ifndef AUTO_AIM__OPENVINO_BACKEND_HPP
#define AUTO_AIM__OPENVINO_BACKEND_HPP

#include <openvino/openvino.hpp>

#include "backend.hpp"

namespace auto_aim
{
class OpenVINOBackend : public BackendBase
{
public:
  explicit OpenVINOBackend(const std::string & config_path);
  ~OpenVINOBackend() override = default;

  bool init(const std::string & model_path, const ModelConfig & model_config) override;
  cv::Mat infer(const cv::Mat & input) override;
  std::string get_name() const override;

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  ov::InferRequest infer_request_;
  ov::Tensor output_tensor_;
  std::string device_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OPENVINO_BACKEND_HPP