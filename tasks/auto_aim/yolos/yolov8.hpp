#ifndef AUTO_AIM__YOLOV8_HPP
#define AUTO_AIM__YOLOV8_HPP

#include <list>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/classifier/classifier.hpp"
#include "tasks/auto_aim/detector/detector.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"

namespace auto_aim
{

/**
 * @brief YOLOV8检测器实现类
 * 
 * 基于OpenVINO实现的YOLOV8目标检测器，用于检测图像中的装甲板
 * 支持ROI区域裁剪、分类器识别等功能
 */
class YOLOV8 : public YOLOBase
{
public:
  /**
   * @brief 构造函数，初始化YOLOV8检测器
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式
   */
  YOLOV8(const std::string & config_path, bool debug);

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
  Classifier classifier_;     ///< 分类器，用于识别装甲板图案
  Detector detector_;         ///< 传统检测器

  std::string device_;        ///< 推理设备（CPU/GPU等）
  std::string model_path_;    ///< 模型文件路径
  std::string save_path_;     ///< 保存路径
  std::string debug_path_;    ///< 调试路径
  bool debug_;                ///< 是否开启调试模式
  bool use_roi_;              ///< 是否使用ROI区域裁剪

  const int class_num_ = 2;              ///< 类别数量
  const float nms_threshold_ = 0.3;      ///< NMS阈值
  const float score_threshold_ = 0.7;     ///< 置信度阈值
  double min_confidence_;                 ///< 最小置信度
  double binary_threshold_;              ///< 二值化阈值

  ov::Core core_;                        ///< OpenVINO核心对象
  ov::CompiledModel compiled_model_;     ///< 编译后的模型

  cv::Rect roi_;                         ///< ROI区域
  cv::Point2f offset_;                   ///< ROI偏移量

  /**
   * @brief 检查装甲板的名称识别结果和置信度
   * @param armor 待检查的装甲板对象
   * @return 是否为有效装甲板且置信度满足要求
   */
  bool check_name(const Armor & armor) const;

  /**
   * @brief 检查装甲板的类型与名称是否匹配
   * @param armor 待检查的装甲板对象
   * @return 类型与名称是否匹配（小装甲板不能是1号或基地，大装甲板不能是2号、哨兵或前哨站）
   */
  bool check_type(const Armor & armor) const;

  /**
   * @brief 从图像中提取装甲板的图案ROI区域
   * @param bgr_img 输入的BGR彩色图像
   * @param armor 装甲板对象，包含角点信息
   * @return 提取的装甲板图案图像
   */
  cv::Mat get_pattern(const cv::Mat & bgr_img, const Armor & armor) const;

  /**
   * @brief 根据装甲板的名称判断装甲板类型（大/小）
   * @param armor 装甲板对象
   * @return 装甲板类型（ArmorType::big 或 ArmorType::small）
   */
  ArmorType get_type(const Armor & armor);

  /**
   * @brief 将像素坐标归一化到[0,1]范围
   * @param bgr_img 输入的BGR彩色图像
   * @param center 像素坐标中心点
   * @return 归一化后的坐标点（x/w, y/h）
   */
  cv::Point2f get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const;

  /**
   * @brief 解析YOLO模型输出，提取装甲板信息
   * @param scale 图像缩放比例
   * @param output YOLO模型输出的特征图（会被转置修改）
   * @param bgr_img 原始BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 解析后的装甲板列表
   */
  std::list<Armor> parse(double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

  /**
   * @brief 保存装甲板图案到文件，用于分类器训练数据收集
   * @param armor 待保存的装甲板对象
   */
  void save(const Armor & armor) const;

  /**
   * @brief 在调试模式下绘制检测结果
   * @param img 原始图像
   * @param armors 检测到的装甲板列表
   * @param frame_count 帧计数
   */
  void draw_detections(const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const;

  /**
   * @brief 对关键点进行排序，使其按顺序为：左上、右上、右下、左下
   * @param keypoints 待排序的关键点向量（输入输出参数，会被修改）
   */
  void sort_keypoints(std::vector<cv::Point2f> & keypoints);
};

}  // namespace auto_aim

#endif  // TOOLS__YOLOV8_HPP