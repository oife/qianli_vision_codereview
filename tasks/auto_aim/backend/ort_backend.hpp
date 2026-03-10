#ifndef AUTO_AIM__ORT_BACKEND_HPP
#define AUTO_AIM__ORT_BACKEND_HPP

#include "onnxruntime_cxx_api.h"
#ifdef USE_CUDA
#include "provider_options.h"
#endif

#include "backend.hpp"

namespace auto_aim
{
class ORTBackend : public BackendBase
{
public:
  explicit ORTBackend(const std::string & config_path);

  bool init(const std::string & model_path, const BackendConfig & model_config) override;
  bool infer(const cv ::Mat & input, cv::Mat & output, BackendCtx * ctx) override;
  std::unique_ptr<BackendCtx> create_ctx() override;
  std::string get_name() const override;

private:
  void preprocess(const cv::Mat & src, cv::Mat & dist);

  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::string input_name_;
  std::string output_name_;
  std::string device_;
};

class ORTCtx : public BackendCtx
{
private:
  std::vector<Ort::Value> output_tensors_;
  friend class ORTBackend;
};
}  // namespace auto_aim

#endif