#ifndef AUTO_AIM__OPENVINO_BACKEND_HPP
#define AUTO_AIM__OPENVINO_BACKEND_HPP

#include <openvino/openvino.hpp>

#include "backend.hpp"

namespace auto_aim
{
class OpenVINOBackend : public Backend
{
public:
  explicit OpenVINOBackend(const std::string & config_path);
  ~OpenVINOBackend() override = default;

  bool init(const std::string & model_path) override;
  cv::Mat infer(const cv::Mat & input) override;
  cv::Size get_input_size() const override;

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
  cv::Size input_size_{416, 416};
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OPENVINO_BACKEND_HPP