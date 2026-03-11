#include "mt_detector.hpp"

namespace auto_aim
{
namespace multithread
{

MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug)
{
  tools::logger()->info("[MultiThreadDetector] initialized with synchronous TensorRT pipeline");
}

void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto armors = yolo_.detect(img, 0);
  queue_.push({std::move(img), std::move(armors), t});
}

std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  auto [img, armors, t] = queue_.pop();
  return {std::move(armors), t};
}

std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  auto [img, armors, t] = queue_.pop();
  return {std::move(img), std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
