/**
 * @file camera_test.cpp
 * @brief 测试相机接口的基本读取功能
 * 
 * 该测试程序用于验证相机抽象接口的图像采集功能，包括：
 * - 从相机读取图像数据
 * - 计算并显示帧率（FPS）
 * - 可选显示视频流窗口
 */

#include "io/camera/camera.hpp"

#include <opencv2/opencv.hpp>

#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

// 定义命令行参数解析规则
// - help/h/usage/? : 显示帮助信息
// - config-path/c : 相机配置文件路径，默认为 configs/camera.yaml
// - display/d : 是否显示视频流窗口
const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/camera.yaml | yaml配置文件路径 }"
  "{d display      |                     | 显示视频流       }";

int main(int argc, char * argv[])
{
  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  // 初始化退出信号处理器，用于优雅地处理程序退出
  tools::Exiter exiter;

  // 获取配置文件路径和显示选项
  auto config_path = cli.get<std::string>("config-path");
  auto display = cli.has("display");
  
  // 根据配置文件初始化相机对象
  io::Camera camera(config_path);

  // 图像数据存储变量
  cv::Mat img;
  // 当前帧的时间戳
  std::chrono::steady_clock::time_point timestamp;
  // 上一帧的时间戳，用于计算帧率
  auto last_stamp = std::chrono::steady_clock::now();
  
  // 主循环：持续读取相机图像直到收到退出信号
  while (!exiter.exit()) {
    // 从相机读取图像和时间戳
    camera.read(img, timestamp);

    // 计算当前帧与上一帧的时间间隔（秒）
    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;

    // 计算并输出帧率（FPS = 1 / 时间间隔）
    tools::logger()->info("{:.2f} fps", 1 / dt);

    // 如果启用显示模式，显示图像
    if (display)
      cv::imshow("img", img);

    if (cv::waitKey(1) == 'q') break;
  }
}