#include "ort_backend.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "tools/logger/logger.hpp"

namespace auto_aim
{
ORTBackend::ORTBackend(const std::string & config_path) : BackendBase(config_path)
{
  device_ = yaml_["device"].as<std::string>("CPU");

  static std::shared_ptr<Ort::Env> global_env = nullptr;
  if (!global_env) {
    global_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "auto_aim");
  }
  env_ = global_env;
}

bool ORTBackend::init(const std::string & model_path, const BackendConfig & model_config)
{
  // try {
  //   model_config_ = model_config;

  //   Ort::SessionOptions session_options;
  //   session_options.SetIntraOpNumThreads(1);
  // }
}

}  // namespace auto_aim