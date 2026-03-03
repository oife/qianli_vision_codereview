#ifndef AUTO_AIM__BACKEND_HPP
#define AUTO_AIM__BACKEND_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

namespace auto_aim
{

enum class BackendType
{
  OPENVINO,
  TENSORRT
};

/**
 * @brief 推理后端抽象基类
*/
class Backend
{
public:
  /**
     * @brief 构造函数
     * @param config_path 配置文件路径
     */
  explicit Backend(const std::string & config_path);

  virtual ~Backend() = default;

  /**
   * @brief 初始化模型
   * @param model_path 模型文件路径
   * @return 是否初始化成功
   */
  virtual bool init(const std::string & model_path) = 0;

  /**
   * @brief 模型推理
   * @param input 输入图像（已完成预处理）
   * @return 推理输出特征图
   */
  virtual cv::Mat infer(const cv::Mat & input) = 0;

  /**
   * @brief 获取输入尺寸
   * @return 模型输入尺寸
   */
  virtual cv::Size get_input_size() const = 0;

protected:
  std::string config_path_;
};

/**
 * @brief 后端工厂函数
 * @param backend_type 后端类型
 * @param config_path 配置文件路径
 * @return 后端指针实例
 */

std::shared_ptr<Backend> create_backend(
  const BackendType & backend_type, const std::string & config_path);

}  // namespace auto_aim

#endif  // AUTO_AIM__BACKEND_HPP