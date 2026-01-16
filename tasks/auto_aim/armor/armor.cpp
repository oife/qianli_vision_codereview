#include "armor.hpp"

#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>

namespace auto_aim
{
Lightbar::Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id)
: id(id), rotated_rect(rotated_rect)
{
  // 获取旋转矩形的四个角点
  std::vector<cv::Point2f> corners(4);
  rotated_rect.points(&corners[0]);
  // 按y坐标排序，分为上下两组
  std::sort(corners.begin(), corners.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  // 计算中心点和上下端点
  center = rotated_rect.center;
  top = (corners[0] + corners[1]) / 2;      // 上边中点
  bottom = (corners[2] + corners[3]) / 2;   // 下边中点
  top2bottom = bottom - top;                // 从顶部到底部的向量

  // 保存关键点
  points.emplace_back(top);
  points.emplace_back(bottom);

  // 计算宽度、角度、长度和长宽比
  width = cv::norm(corners[0] - corners[1]);
  angle = std::atan2(top2bottom.y, top2bottom.x);
  angle_error = std::abs(angle - CV_PI / 2);  // 与垂直方向的偏差
  length = cv::norm(top2bottom);
  ratio = length / width;
}

Armor::Armor(const Lightbar & left, const Lightbar & right)
: left(left), right(right), duplicated(false)
{
  // 设置颜色和中心点
  color = left.color;
  center = (left.center + right.center) / 2;

  // 设置四个角点（左上、右上、右下、左下）
  points.emplace_back(left.top);
  points.emplace_back(right.top);
  points.emplace_back(right.bottom);
  points.emplace_back(left.bottom);

  // 计算两灯条中心点之间的距离
  auto left2right = right.center - left.center;
  auto width = cv::norm(left2right);
  auto max_lightbar_length = std::max(left.length, right.length);
  auto min_lightbar_length = std::min(left.length, right.length);
  
  // 计算比例
  ratio = width / max_lightbar_length;  // 两灯条中点连线与长灯条的长度之比
  side_ratio = max_lightbar_length / min_lightbar_length;  // 长灯条与短灯条的长度之比

  // 计算矩形误差（灯条角度与垂直方向的偏差）
  auto roll = std::atan2(left2right.y, left2right.x);
  auto left_rectangular_error = std::abs(left.angle - roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(right.angle - roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);
}

Armor::Armor(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints)
: class_id(class_id), confidence(confidence), box(box), points(armor_keypoints)
{
  // 计算中心点（四个角点的平均值）
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  
  // 计算左右宽度和上下长度
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  
  // 计算左右中心点
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  
  // 计算矩形误差
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  // 计算长宽比
  ratio = max_length / max_width;

  // 从属性映射表获取颜色、名称和类型
  if (class_id >= 0 && class_id < armor_properties.size()) {
    auto [color, name, type] = armor_properties[class_id];
    this->color = color;
    this->name = name;
    this->type = type;
  } else {
    // 默认值
    this->color = blue;
    this->name = not_armor;
    this->type = small;
  }
}

Armor::Armor(
  int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
  cv::Point2f offset)
: class_id(class_id), confidence(confidence), box(box), points(armor_keypoints)
{
  // 将ROI坐标系下的关键点转换到原图坐标系
  std::transform(
    armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  std::transform(
    points.begin(), points.end(), points.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  
  // 计算中心点（四个角点的平均值）
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  
  // 计算左右宽度和上下长度
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  
  // 计算左右中心点
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  
  // 计算矩形误差
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  // 计算长宽比
  ratio = max_length / max_width;

  // 从属性映射表获取颜色、名称和类型
  if (class_id >= 0 && class_id < armor_properties.size()) {
    auto [color, name, type] = armor_properties[class_id];
    this->color = color;
    this->name = name;
    this->type = type;
  } else {
    // 默认值
    this->color = blue;
    this->name = not_armor;
    this->type = small;
  }
}

Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints)
: confidence(confidence), box(box), points(armor_keypoints)
{
  // 计算中心点（四个角点的平均值）
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  
  // 计算左右宽度和上下长度
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  
  // 计算左右中心点
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  
  // 计算矩形误差
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  // 计算长宽比
  ratio = max_length / max_width;
  
  // 设置颜色（0:蓝色, 1:红色, 2:熄灭）
  color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
  
  // 设置名称（根据num_id映射）
  name = num_id == 0  ? ArmorName::sentry
         : num_id > 5 ? ArmorName(num_id)
                      : ArmorName(num_id - 1);  // TODO: 考虑Bb
  
  // 设置类型（num_id==1为大装甲板，其他为小装甲板）
  type = num_id == 1 ? ArmorType::big : ArmorType::small;
}

Armor::Armor(
  int color_id, int num_id, float confidence, const cv::Rect & box,
  std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset)
: confidence(confidence), box(box), points(armor_keypoints)
{
  // 将ROI坐标系下的关键点转换到原图坐标系
  std::transform(
    armor_keypoints.begin(), armor_keypoints.end(), armor_keypoints.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  std::transform(
    points.begin(), points.end(), points.begin(),
    [&offset](const cv::Point2f & point) { return point + offset; });
  
  // 计算中心点（四个角点的平均值）
  center = (armor_keypoints[0] + armor_keypoints[1] + armor_keypoints[2] + armor_keypoints[3]) / 4;
  
  // 计算左右宽度和上下长度
  auto left_width = cv::norm(armor_keypoints[0] - armor_keypoints[3]);
  auto right_width = cv::norm(armor_keypoints[1] - armor_keypoints[2]);
  auto max_width = std::max(left_width, right_width);
  auto top_length = cv::norm(armor_keypoints[0] - armor_keypoints[1]);
  auto bottom_length = cv::norm(armor_keypoints[3] - armor_keypoints[2]);
  auto max_length = std::max(top_length, bottom_length);
  
  // 计算左右中心点
  auto left_center = (armor_keypoints[0] + armor_keypoints[3]) / 2;
  auto right_center = (armor_keypoints[1] + armor_keypoints[2]) / 2;
  auto left2right = right_center - left_center;
  auto roll = std::atan2(left2right.y, left2right.x);
  
  // 计算矩形误差
  auto left_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[3] - armor_keypoints[0]).y, (armor_keypoints[3] - armor_keypoints[0]).x) -
    roll - CV_PI / 2);
  auto right_rectangular_error = std::abs(
    std::atan2(
      (armor_keypoints[2] - armor_keypoints[1]).y, (armor_keypoints[2] - armor_keypoints[1]).x) -
    roll - CV_PI / 2);
  rectangular_error = std::max(left_rectangular_error, right_rectangular_error);

  // 计算长宽比
  ratio = max_length / max_width;
  
  // 设置颜色（0:蓝色, 1:红色, 2:熄灭）
  color = color_id == 0 ? Color::blue : color_id == 1 ? Color::red : Color::extinguish;
  
  // 设置名称（根据num_id映射）
  name = num_id == 0 ? ArmorName::sentry : num_id > 5 ? ArmorName(num_id) : ArmorName(num_id - 1);
  
  // 设置类型（num_id==1为大装甲板，其他为小装甲板）
  type = num_id == 1 ? ArmorType::big : ArmorType::small;
}

}  // namespace auto_aim