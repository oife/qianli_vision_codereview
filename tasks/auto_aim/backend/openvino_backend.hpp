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
  ~OpenVINOBackend() override;

  bool init(const std::string & model_path, const BackendConfig & model_config) override;
  bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx) override;
  std::unique_ptr<BackendCtx> create_ctx() override;
  std::string get_name() const override;

private:
  ov::Core core_;
  ov::CompiledModel compiled_model_;
  std::string device_;
};

class OpenVINOCtx : public BackendCtx
{
private:
  ov::InferRequest infer_request_;
  friend class OpenVINOBackend;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__OPENVINO_BACKEND_HPP