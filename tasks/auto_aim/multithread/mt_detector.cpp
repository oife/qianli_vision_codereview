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
: yolo_(config_path, debug)
{
  // 加载YAML配置文件
  auto yaml = YAML::LoadFile(config_path);
  auto yolo_name = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  // 读取OpenVINO模型
  auto model = core_.read_model(model_path);
  // 创建预处理和后处理管道
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 配置输入张量的属性
  input.tensor()
    .set_element_type(ov::element::u8)  // 输入数据类型为uint8
    .set_shape({1, 640, 640, 3})  // 输入形状：批次大小1，高度640，宽度640，通道数3
    .set_layout("NHWC")  // 张量布局：N(批次) H(高度) W(宽度) C(通道)
    .set_color_format(ov::preprocess::ColorFormat::BGR);  // 输入颜色格式为BGR

  // 配置模型输入布局
  input.model().set_layout("NCHW");  // 模型期望的布局：N(批次) C(通道) H(高度) W(宽度)

  // 配置预处理步骤
  input.preprocess()
    .convert_element_type(ov::element::f32)  // 转换为float32类型
    .convert_color(ov::preprocess::ColorFormat::RGB)  // BGR转RGB
    // .resize(ov::preprocess::ResizeAlgorithm::RESIZE_LINEAR)  // 可选的线性缩放
    .scale(255.0);  // 归一化：除以255.0

  // 构建预处理管道
  model = ppp.build();
  // 编译模型，使用吞吐量优化模式
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::THROUGHPUT));

  tools::logger()->info("[MultiThreadDetector] initialized !");
}

/**
 * @brief 将图像推入检测队列进行异步推理
 * @param img 输入的OpenCV图像
 * @param t 图像的时间戳
 * @details 对输入图像进行预处理（缩放并填充到640x640），创建异步推理请求并推入队列
 */
void MultiThreadDetector::push(cv::Mat img, std::chrono::steady_clock::time_point t)
{
  // 计算缩放比例，保持宽高比
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);  // 选择较小的缩放比例以保持宽高比
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  // 图像预处理：创建640x640的黑色背景，将缩放后的图像放置在其中
  auto input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));  // 创建黑色背景
  auto roi = cv::Rect(0, 0, w, h);  // 定义感兴趣区域
  cv::resize(img, input(roi), {w, h});  // 将图像缩放到ROI大小

  // 创建推理请求
  auto input_port = compiled_model_.input();
  auto infer_request = compiled_model_.create_infer_request();
  // 创建输入张量，使用预处理后的图像数据
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  // 设置输入张量并启动异步推理
  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
  // 将原始图像、时间戳和推理请求推入队列（克隆图像以避免数据被覆盖）
  queue_.push({img.clone(), t, std::move(infer_request)});
}

/**
 * @brief 从队列中弹出并获取检测结果
 * @return 返回检测到的装甲板列表和对应的时间戳
 * @details 等待异步推理完成，获取输出张量并进行后处理，将检测框坐标映射回原始图像尺寸
 */
std::tuple<std::list<Armor>, std::chrono::steady_clock::time_point> MultiThreadDetector::pop()
{
  // 从队列中弹出图像、时间戳和推理请求
  auto [img, t, infer_request] = queue_.pop();
  // 等待异步推理完成
  infer_request.wait();

  // 后处理：获取推理输出
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  // 将输出张量转换为OpenCV Mat格式
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  // 计算缩放比例，用于将检测框坐标映射回原始图像尺寸
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  // 使用YOLO后处理函数解析检测结果（暂不支持ROI）
  auto armors = yolo_.postprocess(scale, output, img, 0);

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
  auto [img, t, infer_request] = queue_.pop();
  // 等待异步推理完成
  infer_request.wait();

  // 后处理：获取推理输出
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  // 将输出张量转换为OpenCV Mat格式
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());
  // 计算缩放比例，用于将检测框坐标映射回原始图像尺寸
  auto x_scale = static_cast<double>(640) / img.rows;
  auto y_scale = static_cast<double>(640) / img.cols;
  auto scale = std::min(x_scale, y_scale);
  // 使用YOLO后处理函数解析检测结果（暂不支持ROI）
  auto armors = yolo_.postprocess(scale, output, img, 0);

  // 返回原始图像、检测结果和时间戳（用于调试可视化）
  return {img, std::move(armors), t};
}

}  // namespace multithread

}  // namespace auto_aim
