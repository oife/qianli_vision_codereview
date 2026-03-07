#ifndef AUTO_AIM__LAZYOLO_HPP
#define AUTO_AIM__LAZYOLO_HPP

#include <yaml-cpp/yaml.h>

#include <list>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/backend/backend.hpp"
#include "tasks/auto_aim/detector/detector.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"

namespace auto_aim
{
/**
 * @brief LazYolo检测器基类
 * 
 * 通过继承此类可更低成本构建适配不同YOLO版本的检测器，用于检测图像中的装甲板
 * 支持ROI区域裁剪、分类器识别等功能
 */
template <int Size, int ClassNum>
class LazYOLO : public YOLOBase
{
public:
  /**
   * @brief 构造函数，初始化LazYOLO检测器
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式
   */
  LazYOLO(const std::string & config_path, bool debug);

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

protected:
  virtual void output_process(cv::Mat & output) {}

  virtual std::string model_path_key() const = 0;

  YAML::Node & get_yaml();

private:
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

  static constexpr int CLASS_NUM = ClassNum;
  static constexpr int SIZE = Size;
  static constexpr float MNS_THRESHOLD = 0.3;
  static constexpr float SCORE_THRESHOLD = 0.7;

  std::string device_;       ///< 推理设备（CPU/GPU等）
  std::string model_path_;   ///< 模型文件路径
  std::string save_path_;    ///< 保存路径
  std::string debug_path_;   ///< 调试路径
  bool debug_;               ///< 是否开启调试模式
  bool use_roi_;             ///< 是否使用ROI区域裁剪
  bool use_traditional_;     ///< 是否使用传统方法二次矫正角点
  double min_confidance_;    ///< 最小置信度
  double binary_threshold_;  ///< 二值化阈值

  Backend backend_;

  cv::Rect roi_;        ///< ROI区域
  cv::Point2f offset_;  ///< ROI偏移量

  Detector detector_;

  YAML::Node yaml_;
};
}  // namespace auto_aim

#endif