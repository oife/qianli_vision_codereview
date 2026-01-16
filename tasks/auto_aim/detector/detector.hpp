#ifndef AUTO_AIM__DETECTOR_HPP
#define AUTO_AIM__DETECTOR_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/classifier/classifier.hpp"

namespace auto_aim
{

class Detector
{
public:
  /**
   * @brief 构造函数，初始化检测器
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式
   */
  Detector(const std::string & config_path, bool debug = true);

  /**
   * @brief 从图像中检测所有装甲板
   * @param bgr_img 输入的BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 检测到的装甲板列表
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count = -1);

  /**
   * @brief 在ROI区域内重新检测并更新装甲板的角点位置
   * @param armor 待更新的装甲板对象（输入输出参数）
   * @param bgr_img 输入的BGR彩色图像
   * @return 是否成功检测并更新装甲板角点
   */
  bool detect(Armor & armor, const cv::Mat & bgr_img);

  friend class YOLOV8;

private:
  Classifier classifier_;

  double threshold_;
  double max_angle_error_;
  double min_lightbar_ratio_, max_lightbar_ratio_;
  double min_lightbar_length_;
  double min_armor_ratio_, max_armor_ratio_;
  double max_side_ratio_;
  double min_confidence_;
  double max_rectangular_error_;

  bool debug_;
  std::string save_path_;

  // 利用PCA回归角点，参考自https://github.com/CSU-FYT-Vision/FYT2024_vision
  void lightbar_points_corrector(Lightbar & lightbar, const cv::Mat & gray_img) const;

  bool check_geometry(const Lightbar & lightbar) const;
  bool check_geometry(const Armor & armor) const;
  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;

  Color get_color(const cv::Mat & bgr_img, const std::vector<cv::Point> & contour) const;
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;
  ArmorType get_type(const Armor & armor);
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  void save(const Armor & armor) const;
  void show_result(
    const cv::Mat & binary_img, const cv::Mat & bgr_img, const std::list<Lightbar> & lightbars,
    const std::list<Armor> & armors, int frame_count) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__DETECTOR_HPP