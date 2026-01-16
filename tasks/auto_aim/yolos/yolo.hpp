#ifndef AUTO_AIM__YOLO_HPP
#define AUTO_AIM__YOLO_HPP

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/armor/armor.hpp"

namespace auto_aim
{
/**
 * @brief YOLO检测器基类接口
 * 
 * 定义了YOLO检测器的基本接口，所有具体的YOLO实现类都应继承此类
 */
class YOLOBase
{
public:
  /**
   * @brief 从图像中检测装甲板
   * @param img 输入的BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 检测到的装甲板列表
   */
  virtual std::list<Armor> detect(const cv::Mat & img, int frame_count) = 0;

  /**
   * @brief 后处理YOLO模型输出，解析检测结果
   * @param scale 图像缩放比例
   * @param output YOLO模型输出的特征图
   * @param bgr_img 原始BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 解析后的装甲板列表
   */
  virtual std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count) = 0;
};

/**
 * @brief YOLO检测器工厂类
 * 
 * 根据配置文件中的yolo_name参数，自动创建对应的YOLO检测器实例
 * 支持的检测器类型：yolov5, yolov8, yolo11
 */
class YOLO
{
public:
  /**
   * @brief 构造函数，根据配置文件创建对应的YOLO检测器
   * @param config_path 配置文件路径
   * @param debug 是否开启调试模式，默认为true
   * @throws std::runtime_error 当yolo_name未知时抛出异常
   */
  YOLO(const std::string & config_path, bool debug = true);

  /**
   * @brief 从图像中检测装甲板
   * @param img 输入的BGR彩色图像
   * @param frame_count 帧计数，用于调试显示，默认为-1
   * @return 检测到的装甲板列表
   */
  std::list<Armor> detect(const cv::Mat & img, int frame_count = -1);

  /**
   * @brief 后处理YOLO模型输出，解析检测结果
   * @param scale 图像缩放比例
   * @param output YOLO模型输出的特征图
   * @param bgr_img 原始BGR彩色图像
   * @param frame_count 帧计数，用于调试显示
   * @return 解析后的装甲板列表
   */
  std::list<Armor> postprocess(
    double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count);

private:
  std::unique_ptr<YOLOBase> yolo_;  ///< YOLO检测器实例指针
};

}  // namespace auto_aim

#endif  // AUTO_AIM__YOLO_HPP