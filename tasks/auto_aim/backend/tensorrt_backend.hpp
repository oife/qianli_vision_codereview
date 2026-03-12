#ifndef AUTO_AIM__TENSORRT_BACKEND_HPP
#define AUTO_AIM__TENSORRT_BACKEND_HPP

#include <NvInfer.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <vector>

#include "backend.hpp"

namespace auto_aim
{
class TensorRTCtx : public BackendCtx
{
public:
  TensorRTCtx();
  ~TensorRTCtx() override;

private:
  struct InferDeleter
  {
    template <typename T>
    void operator()(T * ptr) const
    {
      delete ptr;
    }
  };

  std::unique_ptr<nvinfer1::IExecutionContext, InferDeleter> context_;
  void * input_device_{nullptr};
  void * output_device_{nullptr};
  size_t input_bytes_{0};
  size_t output_bytes_{0};
  cudaStream_t stream_{nullptr};
  std::vector<float> host_output_;
  std::vector<__half> host_output_half_;

  friend class TensorRTBackend;
};

class TensorRTBackend : public BackendBase
{
public:
  explicit TensorRTBackend(const std::string & config_path);
  ~TensorRTBackend() override;

  bool init(const std::string & model_path, const BackendConfig & model_config) override;
  std::unique_ptr<BackendCtx> create_ctx() override;
  bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx) override;
  std::string get_name() const override;

private:
  struct InferDeleter
  {
    template <typename T>
    void operator()(T * ptr) const
    {
      delete ptr;
    }
  };

  class Logger : public nvinfer1::ILogger
  {
  public:
    void log(Severity severity, nvinfer1::AsciiChar const * msg) noexcept override;
  };

  bool load_engine(const std::string & engine_path);
  bool build_engine_from_onnx(const std::string & model_path, const std::string & engine_path);
  void setup_tensor_names();
  bool ensure_context_ready(TensorRTCtx & ctx);
  bool prepare_input(const cv::Mat & input, std::vector<float> & host_input) const;
  size_t tensor_bytes(const nvinfer1::Dims & dims, nvinfer1::DataType type) const;
  size_t element_size(nvinfer1::DataType type) const;

  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime, InferDeleter> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine, InferDeleter> engine_;
  std::string input_name_;
  std::string output_name_;
  bool use_fp16_{true};
  size_t workspace_size_{1ULL << 30};
};
}  // namespace auto_aim

#endif  // AUTO_AIM__TENSORRT_BACKEND_HPP
