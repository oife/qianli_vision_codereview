/**
 * @file planner_test_offline.cpp
 * @brief 离线测试路径规划器
 * 
 * 该测试程序用于离线测试路径规划算法，包括：
 * - 创建模拟目标
 * - 路径规划计算
 * - 规划结果可视化
 * - 不涉及实际硬件控制
 */

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tasks/auto_aim/planner/planner.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |     | 输出命令行参数说明    }"
  "{d              | 3.0 | Target距离(m)       }"
  "{w              | 5.0 | Target角速度(rad/s) }"
  "{@config-path   |     | yaml配置文件路径     }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  auto d = cli.get<double>("d");
  auto w = cli.get<double>("w");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  auto_aim::Planner planner(config_path);
  auto_aim::Target target(d, w, 0.2, 0.1);

  auto t0 = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    target.predict(0.01);

    auto plan = planner.plan(target, 22);

    nlohmann::json data;
    data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);

    data["target_yaw"] = plan.target_yaw;
    data["target_pitch"] = plan.target_pitch;

    data["plan_yaw"] = plan.yaw;
    data["plan_yaw_vel"] = plan.yaw_vel;
    data["plan_yaw_acc"] = plan.yaw_acc;

    data["plan_pitch"] = plan.pitch;
    data["plan_pitch_vel"] = plan.pitch_vel;
    data["plan_pitch_acc"] = plan.pitch_acc;

    plotter.plot(data);

    std::this_thread::sleep_for(10ms);
  }

  return 0;
}