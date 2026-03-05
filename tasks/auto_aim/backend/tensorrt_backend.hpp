#ifndef AUTO_AIM__TENSORRT_BACKEND_HPP
#define AUTO_AIM__TENSORRT_BACKEND_HPP

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include "backend.hpp"

namespace auto_aim
{
class TensorRTBackend : public BackendBase
{
public:
  explicit TensorRTBackend(const std::string & config_path);
  ~TensorRTBackend() override;

  bool init(const std::string & model_path, const BackendConfig & model_config) override;
  cv::Mat infer(const cv::Mat & input) override;
  // cv::Size get_input_size() const override;

private:
  nvinfer1::ICudaEngine * engine = nullptr;
  nvinfer1::IExecutionContext * context_ = nullptr;
  cv::Size input_size_{416, 416};

  void * device_buffers_[2];
  int input_index_;
  int output_index_;
  cudaStream_t stream_;
};
}  // namespace auto_aim

#endif  // AUTO_AIM__TENSORRT_BACKEND_HPP