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

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/camera.yaml | yaml配置文件路径 }"
  "{d display      |                     | 显示视频流       }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  auto config_path = cli.get<std::string>("config-path");
  auto display = cli.has("display");
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    camera.read(img, timestamp);

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;

    tools::logger()->info("{:.2f} fps", 1 / dt);

    if (!display) continue;
    cv::imshow("img", img);
    if (cv::waitKey(1) == 'q') break;
  }
}