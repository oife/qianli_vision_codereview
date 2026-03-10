/**
 * @file gimbal_response_test.cpp
 * @brief 测试云台响应性能
 * 
 * 该测试程序用于测试云台对不同控制信号的响应，包括：
 * - 三角波信号测试
 * - 阶跃信号测试
 * - 圆形轨迹测试
 * - 响应延迟和精度分析
 */

#include <chrono>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{delta-angle a  |          8          | yaw轴delta角}"
  "{circle      c  |         0.2         | delta_angle的切片数}"
  "{signal-mode m  |     triangle_wave   | 发送信号的模式}"
  "{axis        x  |         yaw         | 发送信号的轴}"
  "{@config-path   | configs/sentry.yaml | 位置参数，yaml配置文件路径 }";

double yaw_cal(double t)
{
  double A = 7;
  double T = 4;  // s

  return A * std::sin(2 * M_PI * t / T);  // 31是云台yaw初始角度，单位为度
}

double pitch_cal(double t)
{
  double A = 7;
  double T = 4;  // s

  return A * std::sin(2 * M_PI * t / T + M_PI / 2) + 18;  // 18是云台pitch初始角度，单位为度
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>(0);
  auto delta_angle = cli.get<double>("delta-angle");
  auto circle = cli.get<double>("circle");
  auto signal_mode = cli.get<std::string>("signal-mode");
  auto axis = cli.get<std::string>("axis");
  if (cli.has("help") || config_path.empty()) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;

  io::Gimbal gimbal(config_path);

  auto init_angle = 0;
  double slice = circle * 100;  //切片数=周期*帧率
  auto dangle = delta_angle / slice;
  double cmd_angle = init_angle;

  int axis_index = axis == "yaw" ? 0 : 1;  // 0 for yaw, 1 for pitch

  double error = 0;
  int count = 0;

  // 发送初始零位指令，等待云台归零
  gimbal.send(true, false, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  std::this_thread::sleep_for(5s);  // 等待云台归零

  double cmd_yaw_rad = 0.0;
  double cmd_pitch_rad = 0.0;
  double last_cmd_yaw_rad = 0.0;
  double last_cmd_pitch_rad = 0.0;

  double t = 0;
  auto last_t = t;
  double dt = 0.005;  // 5ms, 模拟200fps

  auto t0 = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    nlohmann::json data;
    auto timestamp = std::chrono::steady_clock::now();

    std::this_thread::sleep_for(1ms);

    Eigen::Quaterniond q = gimbal.q(timestamp);

    Eigen::Vector3d eulers = tools::eulers(q, 2, 1, 0);

    if (signal_mode == "triangle_wave") {
      if (count == slice) {
        cmd_angle = init_angle;
        cmd_yaw_rad = 0.0;
        cmd_pitch_rad = 0.0;
        if (axis_index == 0)
          cmd_yaw_rad = cmd_angle / 57.3;
        else
          cmd_pitch_rad = cmd_angle / 57.3;
        count = 0;

      } else {
        cmd_angle += dangle;
        if (axis_index == 0)
          cmd_yaw_rad = cmd_angle / 57.3;
        else
          cmd_pitch_rad = cmd_angle / 57.3;
        count++;
      }

      gimbal.send(true, false, cmd_yaw_rad, 0.0f, 0.0f, cmd_pitch_rad, 0.0f, 0.0f);
      if (axis_index == 0) {
        data["cmd_yaw"] = cmd_yaw_rad * 57.3;
        data["last_cmd_yaw"] = last_cmd_yaw_rad * 57.3;
        data["gimbal_yaw"] = eulers[0] * 57.3;
      } else {
        data["cmd_pitch"] = cmd_pitch_rad * 57.3;
        data["last_cmd_pitch"] = last_cmd_pitch_rad * 57.3;
        data["gimbal_pitch"] = eulers[1] * 57.3;
      }
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      last_cmd_yaw_rad = cmd_yaw_rad;
      last_cmd_pitch_rad = cmd_pitch_rad;
      plotter.plot(data);
      std::this_thread::sleep_for(8ms);  //模拟自瞄100fps
    }

    else if (signal_mode == "step") {
      if (count == 300) {
        cmd_angle += delta_angle;
        count = 0;
      }
      cmd_yaw_rad = tools::limit_rad(cmd_angle / 57.3);
      count++;

      gimbal.send(true, false, cmd_yaw_rad, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
      data["cmd_yaw"] = cmd_yaw_rad * 57.3;
      data["last_cmd_yaw"] = last_cmd_yaw_rad * 57.3;
      data["gimbal_yaw"] = eulers[0] * 57.3;
      last_cmd_yaw_rad = cmd_yaw_rad;
      plotter.plot(data);
      std::this_thread::sleep_for(8ms);  //模拟自瞄100fps
    }

    else if (signal_mode == "circle") {
      std::cout << "t: " << t << std::endl;
      cmd_yaw_rad = yaw_cal(t) / 57.3;
      cmd_pitch_rad = pitch_cal(t) / 57.3;
      t += dt;
      if (t - last_t > 2) {
        t += 2.4;
        last_t = t;
      }
      gimbal.send(true, false, cmd_yaw_rad, 0.0f, 0.0f, cmd_pitch_rad, 0.0f, 0.0f);

      data["t"] = t;
      data["cmd_yaw"] = cmd_yaw_rad * 57.3;
      data["cmd_pitch"] = cmd_pitch_rad * 57.3;
      data["gimbal_yaw"] = eulers[0] * 57.3;
      data["gimbal_pitch"] = eulers[1] * 57.3;
      plotter.plot(data);
      std::this_thread::sleep_for(9ms);
    }
  }
  return 0;
}