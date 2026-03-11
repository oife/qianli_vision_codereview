#ifndef AUTO_AIM__TENSORRT_BACKEND_HPP
#define AUTO_AIM__TENSORRT_BACKEND_HPP

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda.h>

#include <fstream>

#include "backend.hpp"
#include "tools/logger/logger.hpp"

namespace auto_aim
{
class TensorRTBackend : public BackendBase
{
public:
  explicit TensorRTBackend(const std::string & config_path);
  ~TensorRTBackend() override;

  bool init(const std::string & model_path, const BackendConfig & model_config) override;
  bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx) override;
  std::unique_ptr<BackendCtx> create_ctx() override;
  std::string get_name() const override;

private:
  void onnx_convert(const char * onnx_file, int memory_size);

  std::shared_ptr<nvinfer1::IExecutionContext> create_ex_context(std::string model_path);

  nvinfer1::ICudaEngine * engine = nullptr;
  std::shared_ptr<nvinfer1::IExecutionContext> ex_context_ = nullptr;

  cudaStream_t stream_;
  void * input_src_device_;
  void * output_src_device_;

  class Logger : public nvinfer1::ILogger
  {
    void log(Severity severity, const char * msg) noexcept override
    {
      if (severity <= Severity::kWARNING) tools::logger()->info("{}", msg);
    }
  } logger_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__TENSORRT_BACKEND_HPP