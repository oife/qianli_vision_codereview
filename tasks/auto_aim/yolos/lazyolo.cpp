#include "lazyolo.hpp"

#include <fmt/chrono.h>

#include <filesystem>

#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"

namespace auto_aim
{
template <int Size, int ClassNum>
LazYOLO<Size, ClassNum>::LazYOLO(const std::string & config_path, bool debug)
: debug_(debug), detector_(config_path, false), backend_(config_path)
{
  yaml_ = YAML::LoadFile(config_path);

  model_path_ = yaml_[model_path_key()].as<std::string>();
  device_ = yaml_["device"].as<std::string>();
  binary_threshold_ = yaml_["threshold"].as<double>();
  min_confidance_ = yaml_["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml_["roi"]["x"].as<int>();
  y = yaml_["roi"]["y"].as<int>();
  width = yaml_["roi"]["width"].as<int>();
  height = yaml_["roi"]["height"].as<int>();
  use_roi_ = yaml_["use_roi"].as<bool>();
  use_traditional_ = yaml["use_traditional"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  BackendConfig config;
  config.input_size = cv::Size(SIZE, SIZE);

  if (!backend_.init(model_path_, config)) {
    throw std::runtime_error("Backend initializing failed");
  }
}

template <int Size, int ClassNum>
std::list<Armor> LazYOLO<Size, ClassNum>::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  double scale;
  cv::Mat output;
  auto ctx = backend_.create_ctx();
  backend_.execute(bgr_img, output, scale, ctx.get());

  output_process(output);

  return parse(scale, output, raw_img, frame_count);
}

template <int Size, int ClassNum>
std::list<Armor> LazYOLO<Size, ClassNum>::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
}
}  // namespace auto_aim