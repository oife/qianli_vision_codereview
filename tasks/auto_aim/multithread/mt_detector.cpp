/**
 * @file mt_detector.cpp
 * @brief 多线程检测器实现文件
 * @details 使用OpenVINO进行异步推理的多线程目标检测器
 */

#include "mt_detector.hpp"

#include <yaml-cpp/yaml.h>

namespace auto_aim
{
namespace multithread
{

/**
 * @brief 构造函数：初始化多线程检测器
 * @param config_path 配置文件路径
 * @param debug 是否开启调试模式
 * @details 从配置文件中加载模型路径和设备信息，配置OpenVINO预处理管道并编译模型
 */
MultiThreadDetector::MultiThreadDetector(const std::string & config_path, bool debug)
: yolo_(config_path, debug), backend_(config_path)
{
  // 加载YAML配置文件
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();

  BackendConfig config;
  config.input_size = cv::Size(640, 640);
  config.throughput_priority = true;

  if (!backend_.init(model_path, config)) {
    throw std::runtime_error("Backend initializing failed");
  }
}

/**
 * @brief 将图像推入检测队列进行异步推理
 * @param img 输入的OpenCV图像
 * @param t 图像的时间戳
 * @details 对输入图像进行预处理（缩放并填充到640x640），创建异步推理请求并推入队列
 */
void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  auto ctx = backend_.create_ctx();
  backend_.execute_async(img, ctx.get());
  queue_.push({img.clone(), t, std::move(ctx)});
}

/**
 * @brief 从队列中弹出并获取检测结果
 * @return 返回检测到的装甲板列表和对应的时间戳
 * @details 等待异步推理完成，获取输出张量并进行后处理，将检测框坐标映射回原始图像尺寸
 */
std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  // 从队列中弹出图像、时间戳和推理请求
  auto [img, t, ctx] = queue_.pop();

  cv::Mat output;
  backend_.wait_for_result(output, ctx.get());
  auto armors = yolo_.postprocess(ctx->scale, output, img, 0);

  return {std::move(armors), t};
}

/**
 * @brief 调试模式：从队列中弹出并获取检测结果（包含原始图像）
 * @return 返回原始图像、检测到的装甲板列表和对应的时间戳
 * @details 与pop()方法类似，但额外返回原始图像用于调试和可视化
 */
std::tuple<cv::Mat, std::list<Armor>, std::chrono::steady_clock::time_point>
MultiThreadDetector::debug_pop()
{
  // 从队列中弹出图像、时间戳和推理请求
  auto [img, t, ctx] = queue_.pop();

  cv::Mat output;
  backend_.wait_for_result(output, ctx.get());
  auto armors = yolo_.postprocess(ctx->scale, output, img, 0);

  // 返回原始图像、检测结果和时间戳（用于调试可视化）
  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
