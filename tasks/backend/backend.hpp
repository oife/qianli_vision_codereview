#ifndef AUTO_AIM__BACKEND_HPP
#define AUTO_AIM__BACKEND_HPP

#include <yaml-cpp/yaml.h>

#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>

// TODO: Just for debug, remember to remove it.
// #define TENSORRT_AVAILABLE
// #define OPENVINO_AVAILABLE
// #define ORT_AVAILABLE

namespace auto_aim
{
struct BackendConfig
{
  cv::Size input_size{640, 640};
  int input_channels{3};
  cv::Scalar padding_color{0, 0, 0};
  // 是否采用居中 letterbox（否则 resize 后贴左上角）
  bool center_padding{false};
  bool preprocess{true};
  bool throughput_priority{false};
};

class BackendCtx
{
public:
  BackendCtx() = default;
  virtual ~BackendCtx() = default;
  double scale{1.0};
  // 对应 standarlize() 产生的 padding（像素）
  double pad_x{0.0};
  double pad_y{0.0};
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
   * @param ctx 推理上下文
   */
  virtual bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx) = 0;

  /**
   * @brief 异步模型推理
   * @param input 输入图像
   * @param ctx 推理上下文
   */
  virtual void infer_async(const cv::Mat & input, BackendCtx * ctx) = 0;

  /**
   * @brief 获取异步推理结果
   * @param output 推理输出特征图
   * @param ctx 推理上下文
   */
  virtual void wait_for_result(cv::Mat & output, BackendCtx * ctx) = 0;

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
   */
  bool execute(const cv::Mat & img, cv::Mat & target, BackendCtx * ctx);

  /**
   * @brief 执行预处理，并创建异步推理任务
   * @param img 输入图像
   * @param target 输出图像
   */
  void execute_async(const cv::Mat & img, BackendCtx * ctx);

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
  std::string device_;
  YAML::Node yaml_;
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
  std::unique_ptr<BackendCtx> create_ctx();

  /**
   * @brief 模型推理
   * @param input 输入图像（已完成预处理）
   * @return 推理输出特征图
   */
  bool infer(const cv::Mat & input, cv::Mat & output, BackendCtx * ctx);

  /**
   * @brief 异步模型推理
   * @param input 输入图像
   * @param ctx 推理上下文
   */
  void infer_async(const cv::Mat & input, BackendCtx * ctx);

  /**
   * @brief 获取异步推理结果
   * @param output 推理输出特征图
   * @param ctx 推理上下文
   */
  void wait_for_result(cv::Mat & output, BackendCtx * ctx);

  /**
   * @brief 预处理图像
   * @param img 输入图像
   * @param scale 返回的缩放比例
   * @return 预处理后的图像
   */
  bool standarlize(const cv::Mat & img, cv::Mat & target, double & scale);

  /**
   * @brief 完整执行预处理到推理的全流程
   * @param img 输入图像
   * @param target 输出图像
   */
  bool execute(const cv::Mat & img, cv::Mat & target, BackendCtx * ctx);

  /**
   * @brief 执行预处理，并创建异步推理任务
   * @param img 输入图像
   * @param target 输出图像
   */
  void execute_async(const cv::Mat & img, BackendCtx * ctx);

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
  void backend_allocate(const std::string config_path);

  std::unique_ptr<BackendBase> backend_;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__BACKEND_HPP