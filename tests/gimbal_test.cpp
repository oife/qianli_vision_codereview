/**
 * @file gimbal_test.cpp
 * @brief 测试云台控制和状态监控的综合小工具
 *
 * 该测试程序用于在「真实云台 + 下位机」环境下验证整条通信链路是否正常工作，
 * 主要包含以下几个方面：
 *
 * 1. 串口通信与状态读取
 *    - 按配置文件中的串口号创建 `io::Gimbal`，持续从云台读取 `GimbalToVision` 帧。
 *    - 实时解算云台姿态四元数，转换为 yaw / pitch 欧拉角，并读取角速度、弹速、累计发弹数等状态信息。
 *
 * 2. 云台控制命令发送
 *    - 以约 9 ms 一次（≈110 Hz）的频率调用 `gimbal.send(...)` 向云台发送 `VisionToGimbal` 控制帧。
 *    - 始终使能云台控制（control=true），yaw 目标在 0° 与 90° 之间循环切换，用于验证云台能否按期望转动。
 *
 * 3. 开火节奏与打弹延迟测试
 *    - 命令行参数 `-f`/`--f` 控制是否启用开火测试，仅在开启时才向云台下发开火请求。
    - 程序内部维护 0.2 s ON / 1.0 s OFF 的周期性开火节奏，通过计数器 `fire_count` 生成。
    - 通过对比 `bullet_count` 的变化与最近一次发出开火命令的时间戳 `fire_stamp`，
 *      统计「首次实际发弹」所用时间并打印日志，方便评估打弹链路延迟。
 *
 * 4. 数据可视化与监控
 *    - 将当前姿态角（由四元数解算）、云台回传的 yaw / pitch、角速度、弹速、发弹数以及开火标志等信息
 *      整理成一帧 JSON 数据，交给 `tools::Plotter` 做实时曲线绘制。
 *    - 可以直观观察云台跟随响应情况、发弹是否按预期节奏进行，以及通信是否丢包/异常。
 *
 * 使用方式（示例）：
 *   ./gimbal_test -f=1 path/to/config.yaml   // 启用周期性开火测试
 *   ./gimbal_test path/to/config.yaml       // 仅测试云台控制与状态回传，不开火
 */

#include "io/gimbal/gimbal.hpp"

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{f              | | 是否开火}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto test_fire = cli.get<bool>("f");
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  io::Gimbal gimbal(config_path);

  auto t0 = std::chrono::steady_clock::now();
  auto last_mode = gimbal.mode();
  uint16_t last_bullet_count = 0;

  auto fire = false;
  auto fire_count = 0;
  auto fire_stamp = std::chrono::steady_clock::now();
  auto first_fired = false;

  // yaw 在 0° 与 90° 之间循环切换
  constexpr double YAW_CYCLE_INTERVAL = 3.0;  // 每个目标停留 3 秒
  auto yaw_switch_time = std::chrono::steady_clock::now();
  double target_yaw = 0;  // 0 弧度

  while (!exiter.exit()) {
    auto mode = gimbal.mode();

    if (mode != last_mode) {
      tools::logger()->info("Gimbal mode changed: {}", gimbal.str(mode));
      last_mode = mode;
    }

    auto t = std::chrono::steady_clock::now();
    auto state = gimbal.state();
    auto q = gimbal.q(t);
    auto ypr = tools::eulers(q, 2, 1, 0);

    auto fired = state.bullet_count > last_bullet_count;
    last_bullet_count = state.bullet_count;

    if (!first_fired && fired) {
      first_fired = true;
      tools::logger()->info("Gimbal first fired after: {:.3f}s", tools::delta_time(t, fire_stamp));
    }

    if (fire && fire_count > 20) {
      // 0.2 s
      fire = false;
      fire_count = 0;
    } else if (!fire && fire_count > 100) {
      // 1s
      fire = true;
      fire_count = 0;
      fire_stamp = t;
      first_fired = false;
    }
    fire_count++;

    // 每隔 YAW_CYCLE_INTERVAL 秒切换 yaw 目标：0° <-> 90°
    if (tools::delta_time(t, yaw_switch_time) >= YAW_CYCLE_INTERVAL) {
      yaw_switch_time = t;
      target_yaw = (target_yaw == 0) ? (CV_PI / 2) : 0;
    }
    gimbal.send(true, test_fire && fire, target_yaw, 0.2, 0, 0, 0, 0);

    nlohmann::json data;
    data["q_yaw"] = ypr[0];
    data["q_pitch"] = ypr[1];
    data["q_roll"] = ypr[2];
    data["yaw"] = state.yaw;
    data["vyaw"] = state.yaw_vel;
    data["pitch"] = state.pitch;
    data["vpitch"] = state.pitch_vel;
    data["bullet_speed"] = state.bullet_speed;
    data["bullet_count"] = state.bullet_count;
    data["fired"] = fired ? 1 : 0;
    data["fire"] = test_fire && fire ? 1 : 0;
    data["t"] = tools::delta_time(t, t0);
    plotter.plot(data);

    std::this_thread::sleep_for(9ms);
  }

  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}