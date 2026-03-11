#ifndef AUTO_BUFF__YOLO11_BUFF_HPP
#define AUTO_BUFF__YOLO11_BUFF_HPP

#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/backend/backend.hpp"
#include "tools/logger/logger.hpp"

namespace auto_buff
{
const std::vector<std::string> class_names = {"buff", "r"};

class YOLO11_BUFF
{
public:
  struct Object
  {
    cv::Rect_<float> rect;
    int label{0};
    float prob{0.F};
    std::vector<cv::Point2f> kpt;
  };

  explicit YOLO11_BUFF(const std::string & config);

  std::vector<Object> get_multicandidateboxes(cv::Mat & image);
  std::vector<Object> get_onecandidatebox(cv::Mat & image);

private:
  std::vector<Object> infer(cv::Mat & image, bool use_nms);
  void draw_result(cv::Mat & image, const Object & obj) const;
  void save(const std::string & program_name, const cv::Mat & image);

  auto_aim::Backend backend_;
  std::string model_path_;
  bool enabled_{false};
  static constexpr int INPUT_SIZE = 640;
  static constexpr int NUM_POINTS = 6;
};
}  // namespace auto_buff

#endif
