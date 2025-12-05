/**
 * @file cboard_test.cpp
 * @brief 测试控制板（CBoard）的IMU数据读取
 * 
 * 该测试程序用于验证控制板功能，包括：
 * - 读取IMU四元数数据
 * - 转换为欧拉角并显示
 * - 显示子弹速度信息
 */

#include "io/cboard/cboard.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                       | 输出命令行参数说明}"
  "{@config-path   | configs/standard.yaml | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;

  io::CBoard cboard(config_path);

  while (!exiter.exit()) {
    auto timestamp = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(1ms);

    Eigen::Quaterniond q = cboard.imu_at(timestamp);

    Eigen::Vector3d eulers = tools::eulers(q, 2, 1, 0) * 57.3;
    tools::logger()->info("z{:.2f} y{:.2f} x{:.2f} degree", eulers[0], eulers[1], eulers[2]);
    tools::logger()->info("bullet speed {:.2f} m/s", cboard.bullet_speed);
  }

  return 0;
}