#ifndef AUTO_AIM__BACKEND_HPP
#define AUTO_AIM__BACKEND_HPP

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// TODO: Just for debug, remember to remove it.
// #define TENSORRT_AVAILABLE
#define OPENVINO_AVAILABLE

namespace auto_aim
{
struct BackendConfig
{
  cv::Size input_size{640, 640};
  cv::Scalar padding_color{0, 0, 0};
  bool preprocess{true};
};

class BackendCtx
{
public:
  BackendCtx() = default;
  virtual ~BackendCtx() = default;
};

/**
 * @brief 推理后端抽象基类
*/
class BackendBase
{
public:
  /**
   * @brief 构造函数
   * @param config_path 配置文件路径
   */
  explicit BackendBase(const std::string & config_path);

  virtual ~BackendBase() = default;

  /**
   * @brief 初始化模型
   * @param model_path 模型文件路径
   * @return 是否初始化成功
   */
  virtual bool init(const std::string & model_path, const BackendConfig & model_config) = 0;

  /**
   * @brief 创建推理上下文
   * @return 后端推理上下文
   */
  virtual std::unique_ptr<BackendCtx> create_ctx() = 0;

  /**
   * @brief 模型推理
   * @param input 输入图像（已完成预处理）
   * @param output 推理输出特征图
   */
  virtual bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx) = 0;

  /**
   * @brief 标准化图像
   * @param img 输入图像
   * @param target 输出图像
   * @param scale 返回的缩放比例
   * @return 标准化后的图像
   */
  bool standarlize(const cv::Mat & img, cv::Mat & target, double & scale);

  /**
   * @brief 完整执行预处理到推理的全流程
   * @param img 输入图像
   * @param target 输出图像
   * @param scale 返回的缩放比例
   */
  bool execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx);

  /**
   * @brief 获取模型配置
   */
  const BackendConfig & get_model_config() const { return model_config_; }

  /**
   * @brief 获取后端名称
   */
  virtual std::string get_name() const = 0;

protected:
  BackendConfig model_config_;
  std::string config_path_;
};

/**
 * @brief 后端工厂类
 * @param backend_type 后端类型
 * @param config_path 配置文件路径
 * @return 后端指针实例
 */
class Backend
{
public:
  /**
   * @brief 构造函数，根据配置文件创建对应的推理后端
   * @param config_path 配置文件路径
   */
  Backend(const std::string config_path);

  /**
   * @brief 初始化模型
   * @param model_path 模型文件路径
   * @return 是否初始化成功
   */
  bool init(const std::string & model_path, const BackendConfig & model_config);

  /**
   * @brief 创建推理上下文
   * @return 后端推理上下文
   */
  virtual std::unique_ptr<BackendCtx> create_ctx();

  /**
   * @brief 模型推理
   * @param input 输入图像（已完成预处理）
   * @return 推理输出特征图
   */
  bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx);

  /**
   * @brief 预处理图像
   * @param img 输入图像
   * @param scale 返回的缩放比例
   * @param pad_top 返回的上方填充
   * @param pad_left 返回的左方填充
   * @return 预处理后的图像
   */
  bool standarlize(const cv::Mat & img, cv::Mat target, double & scale);

  /**
   * @brief 完整执行预处理到推理的全流程
   * @param img 输入图像
   * @param target 输出图像
   * @param scale 返回的缩放比例
   */
  bool execute(const cv::Mat & img, cv::Mat & target, double & scale, BackendCtx * ctx);

  /**
   * @brief 获取模型配置
   * @return 模型配置
   */
  const BackendConfig & get_model_config() const;

  /**
   * @brief 获取后端名称
   */
  std::string get_name() const;

private:
  std::unique_ptr<BackendBase> backend_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__BACKEND_HPP