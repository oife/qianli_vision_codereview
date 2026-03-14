#ifndef AUTO_AIM__ORT_BACKEND_HPP
#define AUTO_AIM__ORT_BACKEND_HPP

#include <future>

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
  void infer_async(const cv::Mat & input, BackendCtx * ctx) override;
  void wait_for_result(cv::Mat & output, BackendCtx * ctx) override;
  std::unique_ptr<BackendCtx> create_ctx() override;
  std::string get_name() const override;

private:
  void preprocess(const cv::Mat & src, cv::Mat & dist);

  Ort::Env env_;
  Ort::Session session_{nullptr};
  std::string input_name_;
  std::string output_name_;

  class ORTCtx : public BackendCtx
  {
  public:
    ORTCtx() : input_tensor_(nullptr) {}

  private:
    Ort::Value input_tensor_;
    std::vector<Ort::Value> output_tensors_;
    std::future<std::vector<Ort::Value>> async_result_;
    friend class ORTBackend;
  };
};
}  // namespace auto_aim

#endif