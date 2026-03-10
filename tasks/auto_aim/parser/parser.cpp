#include "parser.hpp"

namespace auto_aim
{
template <typename T>
Parser<T>::Parser(int begin, int length) : begin_(begin), length_(length)
{
}

template <typename T>
const T & Parser<T>::back() const
{
  return datas_.back();
}

template <typename T>
const std::vector<T> & Parser<T>::data() const
{
  return datas_;
}

template <typename T>
const T & Parser<T>::operator[](int index) const
{
  return datas_[index];
}

template <typename T>
void Parser<T>::parse(const cv::Mat & row)
{
  if (row.rows != 1) {
    tools::logger()->error("Data to parse is not a single row!");
    return;
  }
  auto raw_data = row.colRange(begin_, begin_ + length_);
  datas_.emplace_back(process(raw_data));
}

KeyPointsParse::KeyPointsParse(int begin) : Parser<std::vector<cv::Point2f>>(begin, 8) {}

void KeyPointsParse::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  // 确保关键点数量为4
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  // 按y坐标排序，分为上下两组
  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  // 对上下两组分别按x坐标排序
  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  // 重新排列为：左上、右上、右下、左下
  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

std::vector<cv::Point2f> KeyPointsParse::process(const cv::Mat & raw_data) {}  // namespace auto_aim