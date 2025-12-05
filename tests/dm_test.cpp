/**
 * @file dm_test.cpp
 * @brief 测试DM IMU数据读取
 * 
 * 该测试程序用于验证DM IMU模块功能，包括：
 * - 读取IMU四元数数据
 * - 转换为欧拉角并显示
 */

#include <chrono>
#include <thread>

#include "io/dm_imu/dm_imu.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

using namespace std::chrono_literals;

int main()
{
  tools::logger()->set_level(spdlog::level::debug);
  
  tools::Exiter exiter;
  io::DM_IMU imu;

  while (!exiter.exit()) {
    auto timestamp = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(1ms);

    Eigen::Quaterniond q = imu.imu_at(timestamp);

    Eigen::Vector3d eulers = tools::eulers(q, 2, 1, 0) * 57.3;
    tools::logger()->info("z{:.2f} y{:.2f} x{:.2f} degree", eulers[0], eulers[1], eulers[2]);
  }

  return 0;
}