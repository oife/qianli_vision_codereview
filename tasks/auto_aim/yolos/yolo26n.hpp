#ifndef AUTO_AIM__YOLO26N_HPP
#define AUTO_AIM__YOLO26N_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tasks/backend/backend.hpp"

namespace auto_aim
{
/**
 * @brief YOLO26n-Pose+Name v2 检测器
 *
 * 单输出模型 [1,27,8400]，列优先布局：
 * [0-3] cx,cy,w,h  [4-6] color(3)  [7-14] name(8)  [15-26] kpts(4x3)
 * 所有 scores 已 sigmoid，bbox/kpts 已解码到 640 尺度
 */
class YOLO26N : public YOLOBase
{
public:
  /**
   * @brief 构造函数，初始化YOLO26N检测器
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式
   */
  YOLO26N(const std::string & config_path, bool debug);

  /**
   * @brief 从图像中检测所有装甲板
   * @param bgr_img 输入的BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 检测到的装甲板列表
   */
  std::list<Armor> detect(const cv::Mat & bgr_img, int frame_count) override;

  /**
   * @brief 后处理YOLO模型输出，解析检测结果
   * @param scale 图像缩放比例
   * @param output YOLO模型输出的特征图
   * @param bgr_img 原始BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 解析后的装甲板列表
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) override;

private:
  static constexpr int INPUT_SIZE = 640;
  static constexpr int NUM_COLORS = 3;
  static constexpr int NUM_NAMES = 8;
  static constexpr int NUM_KPTS = 4;
  static constexpr float BIG_ARMOR_RATIO_THRESH = 2.5f;

  std::string model_path_;
  bool debug_;

  float score_threshold_ = 0.25f;
  float nms_threshold_ = 0.45f;
  double min_confidence_;

  Backend backend_;

  // letterbox 参数（在 detect 中计算，postprocess 中使用）
  float scale_ = 1.f;
  float pad_x_ = 0.f;
  float pad_y_ = 0.f;

  std::list<Armor> parse(
    const float * data, int num_anchors, const cv::Mat & bgr_img, int frame_count);

  ArmorType infer_armor_type(ArmorName name, const std::vector<cv::Point2f> & kpts) const;

  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  bool check_name(const Armor & armor) const;
  bool check_type(const Armor & armor) const;
  bool check_geometry(const Armor & armor, const cv::Mat & bgr_img) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO26N_HPP
