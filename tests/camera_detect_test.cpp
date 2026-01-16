/**
 * @file camera_detect_test.cpp
 * @brief 测试相机图像检测功能
 * 
 * 该测试程序用于验证目标检测功能，包括：
 * - 从相机读取图像
 * - 使用传统方法或YOLO进行装甲板检测
 * - 显示检测结果和性能统计
 */

#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "tasks/auto_aim/detector/detector.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明 }"
  "{@config-path   |configs/camera_detect.yaml| yaml配置文件的路径}"
  "{tradition t    |                     false| 是否使用传统方法识别}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto use_tradition = cli.get<bool>("tradition");

  tools::Exiter exiter;

  io::Camera camera(config_path);
  auto_aim::Detector detector(config_path, true);
  auto_aim::YOLO yolo(config_path, true);

  std::chrono::steady_clock::time_point timestamp;

  // FPS统计变量
  int frame_count = 0;
  auto fps_start_time = std::chrono::steady_clock::now();
  const auto fps_interval = std::chrono::seconds(1);  // 每秒更新一次FPS

  while (!exiter.exit()) {
    auto loop_start = std::chrono::steady_clock::now();
    
    cv::Mat img;
    std::list<auto_aim::Armor> armors;

    // 测量相机读取耗时
    auto read_start = std::chrono::steady_clock::now();
    camera.read(img, timestamp);
    auto read_end = std::chrono::steady_clock::now();
    double read_time = tools::delta_time(read_end, read_start) * 1000;

    if (img.empty()) break;

    // 测量检测耗时
    auto detect_start = std::chrono::steady_clock::now();
    if (use_tradition)
      armors = detector.detect(img);
    else
      armors = yolo.detect(img);
    auto detect_end = std::chrono::steady_clock::now();
    double detect_time = tools::delta_time(detect_end, detect_start) * 1000;
    
    // 统计帧数
    frame_count++;
    
    // 计算FPS：统计1秒内处理的帧数
    auto elapsed = std::chrono::steady_clock::now() - fps_start_time;
    if (elapsed >= fps_interval) {
      double fps = frame_count / tools::delta_time(std::chrono::steady_clock::now(), fps_start_time);
      auto loop_end = std::chrono::steady_clock::now();
      double loop_time = tools::delta_time(loop_end, loop_start) * 1000;
      tools::logger()->info("FPS: {:.2f} | 相机读取: {:.3f} ms | 检测耗时: {:.3f} ms | 循环总耗时: {:.3f} ms", 
                            fps, read_time, detect_time, loop_time);
      
      // 重置统计
      frame_count = 0;
      fps_start_time = std::chrono::steady_clock::now();
    }

    auto key = cv::waitKey(1);  // 改为1ms，减少等待时间对FPS的影响
    if (key == 'q') break;
  }

  return 0;
}