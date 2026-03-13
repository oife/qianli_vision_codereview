#include "tensorrt_backend.hpp"

#include <NvOnnxParser.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <vector>

#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace auto_aim
{
namespace
{
std::string to_lower(std::string value)
{
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}
}  // namespace

TensorRTCtx::TensorRTCtx() { cudaStreamCreate(&stream_); }

TensorRTCtx::~TensorRTCtx()
{
  if (input_device_) cudaFree(input_device_);
  if (output_device_) cudaFree(output_device_);
  if (stream_) cudaStreamDestroy(stream_);
}

void TensorRTBackend::Logger::log(Severity severity, nvinfer1::AsciiChar const * msg) noexcept
{
  switch (severity) {
    case Severity::kINTERNAL_ERROR:
    case Severity::kERROR:
      tools::logger()->error("[TensorRT] {}", msg);
      break;
    case Severity::kWARNING:
      tools::logger()->warn("[TensorRT] {}", msg);
      break;
    case Severity::kINFO:
      tools::logger()->info("[TensorRT] {}", msg);
      break;
    default:
      break;
  }
}

TensorRTBackend::TensorRTBackend(const std::string & config_path) : BackendBase(config_path)
{
  auto precision = to_lower(yaml_["precision"].as<std::string>("fp16"));
  use_fp16_ = precision != "fp32";
  workspace_size_ = yaml_["trt_workspace_size"].as<size_t>(1ULL << 30);
}

TensorRTBackend::~TensorRTBackend() = default;

bool TensorRTBackend::init(const std::string & model_path, const BackendConfig & model_config)
{
  model_config_ = model_config;

  std::filesystem::path model(model_path);
  const auto engine_path = model.extension() == ".engine"
                             ? model.string()
                             : (model.string() + (use_fp16_ ? ".fp16.engine" : ".fp32.engine"));

  if (!load_engine(engine_path)) {
    if (model.extension() != ".onnx") {
      tools::logger()->error("TensorRT backend requires .onnx or .engine, got {}", model_path);
      return false;
    }
    if (!build_engine_from_onnx(model_path, engine_path)) return false;
    if (!load_engine(engine_path)) return false;
  }

  setup_tensor_names();
  return !input_name_.empty() && !output_name_.empty();
}

std::unique_ptr<BackendCtx> TensorRTBackend::create_ctx()
{
  auto ctx = std::make_unique<TensorRTCtx>();
  if (!engine_) return ctx;

  ctx->context_.reset(engine_->createExecutionContext());
  if (!ctx->context_) {
    tools::logger()->error("Failed to create TensorRT execution context");
    return ctx;
  }
  return ctx;
}

bool TensorRTBackend::infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx)
{
  if (input.empty() || ctx == nullptr) return false;

  auto * trt_ctx = static_cast<TensorRTCtx *>(ctx);
  if (!trt_ctx->context_) {
    tools::logger()->error("TensorRT context is not initialized");
    return false;
  }
  if (!ensure_context_ready(*trt_ctx)) return false;

  std::vector<float> host_input;
  if (!prepare_input(input, host_input)) return false;

  if (
    cudaMemcpyAsync(
      trt_ctx->input_device_, host_input.data(), trt_ctx->input_bytes_, cudaMemcpyHostToDevice,
      trt_ctx->stream_) != cudaSuccess) {
    tools::logger()->error("TensorRT input memcpy failed");
    return false;
  }

  if (!trt_ctx->context_->enqueueV3(trt_ctx->stream_)) {
    tools::logger()->error("TensorRT enqueueV3 failed");
    return false;
  }

  auto output_dims = trt_ctx->context_->getTensorShape(output_name_.c_str());
  auto output_type = engine_->getTensorDataType(output_name_.c_str());
  size_t count = tensor_bytes(output_dims, output_type) / element_size(output_type);

  if (output_type == nvinfer1::DataType::kFLOAT) {
    trt_ctx->host_output_.resize(count);
    if (
      cudaMemcpyAsync(
        trt_ctx->host_output_.data(), trt_ctx->output_device_, trt_ctx->output_bytes_,
        cudaMemcpyDeviceToHost, trt_ctx->stream_) != cudaSuccess) {
      tools::logger()->error("TensorRT output memcpy failed");
      return false;
    }
  } else if (output_type == nvinfer1::DataType::kHALF) {
    trt_ctx->host_output_half_.resize(count);
    if (
      cudaMemcpyAsync(
        trt_ctx->host_output_half_.data(), trt_ctx->output_device_, trt_ctx->output_bytes_,
        cudaMemcpyDeviceToHost, trt_ctx->stream_) != cudaSuccess) {
      tools::logger()->error("TensorRT output memcpy failed");
      return false;
    }
  } else {
    tools::logger()->error("Unsupported TensorRT output type");
    return false;
  }

  if (cudaStreamSynchronize(trt_ctx->stream_) != cudaSuccess) {
    tools::logger()->error("TensorRT stream synchronize failed");
    return false;
  }

  if (output_type == nvinfer1::DataType::kHALF) {
    trt_ctx->host_output_.resize(count);
    for (size_t i = 0; i < count; ++i) {
      trt_ctx->host_output_[i] = __half2float(trt_ctx->host_output_half_[i]);
    }
  }

  if (output_dims.nbDims == 1) {
    output = cv::Mat(1, output_dims.d[0], CV_32F, trt_ctx->host_output_.data());
  } else if (output_dims.nbDims == 2) {
    output = cv::Mat(output_dims.d[0], output_dims.d[1], CV_32F, trt_ctx->host_output_.data());
  } else if (output_dims.nbDims >= 3) {
    int rows = output_dims.d[output_dims.nbDims - 2];
    int cols = output_dims.d[output_dims.nbDims - 1];
    output = cv::Mat(rows, cols, CV_32F, trt_ctx->host_output_.data());
  } else {
    tools::logger()->error("Unexpected TensorRT output dimensions");
    return false;
  }
  return true;
}

void TensorRTBackend::infer_async(const cv::Mat & input, BackendCtx * ctx)
{
  auto trt_ctx = static_cast<TensorRTCtx *>(ctx);
  if (!ensure_context_ready(*trt_ctx)) return;

  std::vector<float> host_input;
  if (!prepare_input(input, host_input)) return;

  cudaMemcpyAsync(
    trt_ctx->input_device_, host_input.data(), trt_ctx->input_bytes_, cudaMemcpyHostToDevice,
    trt_ctx->stream_);
  trt_ctx->context_->enqueueV3(trt_ctx->stream_);

  cudaMemcpyAsync(
    trt_ctx->host_output_.data(), trt_ctx->output_device_, trt_ctx->output_bytes_,
    cudaMemcpyDeviceToHost, trt_ctx->stream_);
}

void TensorRTBackend::wait_for_result(cv::Mat & output, BackendCtx * ctx)
{
  auto * trt_ctx = static_cast<TensorRTCtx *>(ctx);

  cudaStreamSynchronize(trt_ctx->stream_);

  auto output_dims = trt_ctx->context_->getTensorShape(output_name_.c_str());
  int rows = output_dims.d[output_dims.nbDims - 2];
  int cols = output_dims.d[output_dims.nbDims - 1];
  output = cv::Mat(rows, cols, CV_32F, trt_ctx->host_output_.data());
}

std::string TensorRTBackend::get_name() const { return "TensorRT"; }

bool TensorRTBackend::load_engine(const std::string & engine_path)
{
  if (!std::filesystem::exists(engine_path)) return false;

  std::ifstream file(engine_path, std::ios::binary);
  if (!file.is_open()) return false;

  std::vector<char> engine_data(
    (std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  if (engine_data.empty()) return false;

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) {
    tools::logger()->error("Failed to create TensorRT runtime");
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
  if (!engine_) {
    tools::logger()->warn("Failed to deserialize cached TensorRT engine: {}", engine_path);
    return false;
  }

  tools::logger()->info("Loaded TensorRT engine: {}", engine_path);
  return true;
}

bool TensorRTBackend::build_engine_from_onnx(
  const std::string & model_path, const std::string & engine_path)
{
  std::unique_ptr<nvinfer1::IBuilder, InferDeleter> builder(nvinfer1::createInferBuilder(logger_));
  if (!builder) {
    tools::logger()->error("Failed to create TensorRT builder");
    return false;
  }

  const auto network_flags =
    1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
  std::unique_ptr<nvinfer1::INetworkDefinition, InferDeleter> network(
    builder->createNetworkV2(network_flags));
  std::unique_ptr<nvonnxparser::IParser, InferDeleter> parser(
    nvonnxparser::createParser(*network, logger_));
  std::unique_ptr<nvinfer1::IBuilderConfig, InferDeleter> config(builder->createBuilderConfig());
  if (!network || !parser || !config) {
    tools::logger()->error("Failed to create TensorRT build components");
    return false;
  }

  if (!parser->parseFromFile(
        model_path.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
    tools::logger()->error("Failed to parse ONNX model: {}", model_path);
    return false;
  }

  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_size_);
  if (use_fp16_ && builder->platformHasFastFp16()) {
    config->setFlag(nvinfer1::BuilderFlag::kFP16);
  }

  auto * input_tensor = network->getInput(0);
  auto input_dims = input_tensor->getDimensions();
  bool dynamic = false;
  for (int i = 0; i < input_dims.nbDims; ++i) {
    dynamic = dynamic || input_dims.d[i] == -1;
  }
  if (dynamic) {
    auto * profile = builder->createOptimizationProfile();
    if (!profile) {
      tools::logger()->error("Failed to create TensorRT optimization profile");
      return false;
    }
    auto opt_dims = input_dims;
    opt_dims.d[0] = 1;
    if (opt_dims.nbDims >= 4) {
      opt_dims.d[1] = model_config_.input_channels;
      opt_dims.d[2] = model_config_.input_size.height;
      opt_dims.d[3] = model_config_.input_size.width;
    }
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMIN, opt_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kOPT, opt_dims);
    profile->setDimensions(input_tensor->getName(), nvinfer1::OptProfileSelector::kMAX, opt_dims);
    config->addOptimizationProfile(profile);
  }

  std::unique_ptr<nvinfer1::IHostMemory, InferDeleter> serialized(
    builder->buildSerializedNetwork(*network, *config));
  if (!serialized) {
    tools::logger()->error("Failed to build TensorRT serialized engine");
    return false;
  }

  auto parent = std::filesystem::path(engine_path).parent_path();
  if (!parent.empty()) std::filesystem::create_directories(parent);
  std::ofstream out(engine_path, std::ios::binary);
  out.write(static_cast<const char *>(serialized->data()), serialized->size());
  out.close();
  tools::logger()->info("Built TensorRT engine: {}", engine_path);
  return true;
}

void TensorRTBackend::setup_tensor_names()
{
  input_name_.clear();
  output_name_.clear();
  if (!engine_) return;

  for (int i = 0; i < engine_->getNbIOTensors(); ++i) {
    auto name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
      input_name_ = name;
    } else if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kOUTPUT) {
      output_name_ = name;
    }
  }
}

bool TensorRTBackend::ensure_context_ready(TensorRTCtx & ctx)
{
  auto input_dims = engine_->getTensorShape(input_name_.c_str());
  if (input_dims.nbDims >= 4) {
    input_dims.d[0] = 1;
    input_dims.d[1] = model_config_.input_channels;
    input_dims.d[2] = model_config_.input_size.height;
    input_dims.d[3] = model_config_.input_size.width;
  }

  if (!ctx.context_->setInputShape(input_name_.c_str(), input_dims)) {
    tools::logger()->error("Failed to set TensorRT input shape");
    return false;
  }

  auto output_dims = ctx.context_->getTensorShape(output_name_.c_str());
  auto input_type = engine_->getTensorDataType(input_name_.c_str());
  auto output_type = engine_->getTensorDataType(output_name_.c_str());
  size_t input_bytes = tensor_bytes(input_dims, input_type);
  size_t output_bytes = tensor_bytes(output_dims, output_type);

  if (input_bytes == 0 || output_bytes == 0) {
    tools::logger()->error("TensorRT reported empty input or output tensor");
    return false;
  }

  if (ctx.input_bytes_ != input_bytes) {
    if (ctx.input_device_) cudaFree(ctx.input_device_);
    cudaMalloc(&ctx.input_device_, input_bytes);
    ctx.input_bytes_ = input_bytes;
  }
  if (ctx.output_bytes_ != output_bytes) {
    if (ctx.output_device_) cudaFree(ctx.output_device_);
    cudaMalloc(&ctx.output_device_, output_bytes);
    ctx.output_bytes_ = output_bytes;
  }

  if (
    !ctx.context_->setTensorAddress(input_name_.c_str(), ctx.input_device_) ||
    !ctx.context_->setTensorAddress(output_name_.c_str(), ctx.output_device_)) {
    tools::logger()->error("Failed to bind TensorRT tensors");
    return false;
  }
  return true;
}

bool TensorRTBackend::prepare_input(const cv::Mat & input, std::vector<float> & host_input) const
{
  cv::Mat float_input;
  double scale = model_config_.preprocess ? 1.0 / 255.0 : 1.0;
  input.convertTo(float_input, CV_32F, scale);

  if (model_config_.preprocess && float_input.channels() == 3) {
    cv::cvtColor(float_input, float_input, cv::COLOR_BGR2RGB);
  }

  const int channels = float_input.channels();
  const int plane = float_input.rows * float_input.cols;
  std::vector<cv::Mat> split_channels;
  cv::split(float_input, split_channels);

  host_input.resize(channels * plane);
  for (int c = 0; c < channels; ++c) {
    std::memcpy(
      host_input.data() + c * plane, split_channels[c].ptr<float>(), plane * sizeof(float));
  }
  return true;
}

size_t TensorRTBackend::tensor_bytes(const nvinfer1::Dims & dims, nvinfer1::DataType type) const
{
  size_t count = 1;
  for (int i = 0; i < dims.nbDims; ++i) {
    if (dims.d[i] < 0) return 0;
    count *= static_cast<size_t>(dims.d[i]);
  }
  return count * element_size(type);
}

size_t TensorRTBackend::element_size(nvinfer1::DataType type) const
{
  switch (type) {
    case nvinfer1::DataType::kFLOAT:
      return sizeof(float);
    case nvinfer1::DataType::kHALF:
      return sizeof(__half);
    case nvinfer1::DataType::kINT32:
      return sizeof(int32_t);
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kUINT8:
      return sizeof(uint8_t);
    case nvinfer1::DataType::kBOOL:
      return sizeof(bool);
    default:
      return 0;
  }
}

}  // namespace auto_aim
