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
  yolo_name_ = yaml["yolo_name"].as<std::string>();
  auto model_path = yaml[yolo_name_ + "_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();

  // 读取OpenVINO模型
  auto model = core_.read_model(model_path);
  // 创建预处理和后处理管道
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  // 配置输入张量的属性
  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 640, 640, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  // 配置模型输入布局
  input.model().set_layout("NCHW");

  // 配置预处理步骤
  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // YOLO26n FP16 模型需要额外的类型转换
  if (yolo_name_ == "yolo26n" &&
      model->input(0).get_element_type() == ov::element::f16) {
    input.preprocess().convert_element_type(ov::element::f16);
    for (size_t i = 0; i < model->outputs().size(); i++)
      ppp.output(i).postprocess().convert_element_type(ov::element::f32);
  }

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
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(img.rows * scale);
  auto w = static_cast<int>(img.cols * scale);

  cv::Mat input;
  if (yolo_name_ == "yolo26n") {
    // YOLO26n: 居中 letterbox + 灰色 114 填充
    input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));
    int pad_x = (640 - w) / 2;
    int pad_y = (640 - h) / 2;
    cv::resize(img, input(cv::Rect(pad_x, pad_y, w, h)), {w, h});
  } else {
    // 其他模型: 左上角对齐 + 黑色填充
    input = cv::Mat(640, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::resize(img, input(cv::Rect(0, 0, w, h)), {w, h});
  }

  // 创建推理请求
  auto infer_request = compiled_model_.create_infer_request();
  ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input.data);

  infer_request.set_input_tensor(input_tensor);
  infer_request.start_async();
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
