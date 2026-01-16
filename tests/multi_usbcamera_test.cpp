/**
 * @file multi_usbcamera_test.cpp
 * @brief 测试多个USB相机同时读取
 * 
 * 该测试程序用于验证多相机同步采集功能，包括：
 * - 同时从多个USB相机读取图像
 * - 计算并显示帧率
 * - 可选显示所有相机的视频流
 */

#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera/camera.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                        | 输出命令行参数说明}"
  "{@config-path   | configs/sentry.yaml    | 位置参数，yaml配置文件路径 }"
  "{d display      |                        | 显示视频流       }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  tools::Exiter exiter;

  auto config_path = cli.get<std::string>(0);
  auto display = cli.has("display");

  io::USBCamera usbcam1("video0", config_path);
  io::USBCamera usbcam2("video2", config_path);
  // io::Camera camera("configs/camera.yaml");

  cv::Mat img1, img2, img3;
  std::chrono::steady_clock::time_point timestamp;
  auto last_stamp = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    usbcam1.read(img1, timestamp);
    usbcam2.read(img2, timestamp);
    // camera.read(img3, timestamp);

    auto dt = tools::delta_time(timestamp, last_stamp);
    last_stamp = timestamp;

    tools::logger()->info("{:.2f} fps", 1 / dt);

    if (!display) continue;
    cv::imshow("img1", img1);
    cv::imshow("img2", img2);
    // cv::imshow("img3", img3);

    if (cv::waitKey(1) == 'q') break;
  }
}