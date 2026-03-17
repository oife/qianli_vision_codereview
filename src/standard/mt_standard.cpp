// 文件说明：多模式多线程主程序（Gimbal串口版），按Gimbal模式切换自瞄与打符流程，
// 使用多线程检测队列驱动瞄准与射击控制。适配战队电控串口通信。
#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter/shooter.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tasks/auto_buff/buff_aimer.hpp"
#include "tasks/auto_buff/buff_detector.hpp"
#include "tasks/auto_buff/buff_solver.hpp"
#include "tasks/auto_buff/buff_target.hpp"
#include "tasks/auto_buff/buff_type.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"
#include "tools/recorder/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  tools::Plotter plotter;
  tools::Recorder recorder;

  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);

  auto_aim::multithread::MultiThreadDetector detector(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);

  auto_buff::Buff_Detector buff_detector(config_path);
  auto_buff::Solver buff_solver(config_path);
  auto_buff::SmallTarget buff_small_target;
  auto_buff::BigTarget buff_big_target;
  auto_buff::Aimer buff_aimer(config_path);

  // CommandGener 绑定了 CBoard，这里改为内联命令生成逻辑
  // 用于命令生成线程的共享数据
  struct CommandInput
  {
    std::list<auto_aim::Target> targets;
    std::chrono::steady_clock::time_point t;
    double bullet_speed;
    Eigen::Vector3d gimbal_pos;
  };
  std::optional<CommandInput> latest_input;
  std::mutex cmd_mtx;
  std::atomic<bool> cmd_stop{false};

  // 命令生成线程：替代 CommandGener，通过 Gimbal 串口发送
  auto command_thread = std::thread([&]() {
    while (!cmd_stop) {
      std::optional<CommandInput> input;
      {
        std::lock_guard<std::mutex> lock(cmd_mtx);
        if (latest_input &&
            tools::delta_time(std::chrono::steady_clock::now(), latest_input->t) < 0.2) {
          input = latest_input;
        }
      }
      if (input) {
        auto command = aimer.aim(input->targets, input->t, input->bullet_speed);
        command.shoot = shooter.shoot(command, aimer, input->targets, input->gimbal_pos);
        gimbal.send(
          command.control, command.shoot, command.yaw, 0.0f, 0.0f, command.pitch, 0.0f, 0.0f);
      }
      std::this_thread::sleep_for(2ms);  // ~500Hz
    }
  });

  std::atomic<io::GimbalMode> mode{io::GimbalMode::IDLE};
  auto last_mode{io::GimbalMode::IDLE};

  auto detect_thread = std::thread([&]() {
    cv::Mat img;
    std::chrono::steady_clock::time_point t;

    while (!exiter.exit()) {
      if (mode.load() == io::GimbalMode::AUTO_AIM) {
        camera.read(img, t);
        detector.push(img, t);
      } else
        continue;
    }
  });

  while (!exiter.exit()) {
    mode = gimbal.mode();

    if (last_mode != mode) {
      tools::logger()->info("Switch to {}", gimbal.str(mode));
      last_mode = mode.load();
    }

    /// 自瞄
    if (mode.load() == io::GimbalMode::AUTO_AIM) {
      auto [img, armors, t] = detector.debug_pop();
      Eigen::Quaterniond q = gimbal.q(t);
      auto gs = gimbal.state();

      solver.set_R_gimbal2world(q);

      Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

      auto targets = tracker.track(armors, t);

      // 推送到命令生成线程
      {
        std::lock_guard<std::mutex> lock(cmd_mtx);
        latest_input = {targets, t, gs.bullet_speed, ypr};
      }
    }

    /// 打符
    else if (
      mode.load() == io::GimbalMode::SMALL_BUFF || mode.load() == io::GimbalMode::BIG_BUFF) {
      cv::Mat img;
      Eigen::Quaterniond q;
      std::chrono::steady_clock::time_point t;

      camera.read(img, t);
      q = gimbal.q(t);
      auto gs = gimbal.state();

      buff_solver.set_R_gimbal2world(q);

      auto power_runes = buff_detector.detect(img);

      buff_solver.solve(power_runes);

      io::Command buff_command;
      if (mode.load() == io::GimbalMode::SMALL_BUFF) {
        buff_small_target.get_target(power_runes, t);
        auto target_copy = buff_small_target;
        buff_command = buff_aimer.aim(target_copy, t, gs.bullet_speed, true);
      } else if (mode.load() == io::GimbalMode::BIG_BUFF) {
        buff_big_target.get_target(power_runes, t);
        auto target_copy = buff_big_target;
        buff_command = buff_aimer.aim(target_copy, t, gs.bullet_speed, true);
      }
      gimbal.send(
        buff_command.control, buff_command.shoot, buff_command.yaw, 0.0f, 0.0f,
        buff_command.pitch, 0.0f, 0.0f);

    } else
      gimbal.send(false, false, 0, 0, 0, 0, 0, 0);
  }

  cmd_stop = true;
  if (command_thread.joinable()) command_thread.join();
  if (detect_thread.joinable()) detect_thread.join();
  gimbal.send(false, false, 0, 0, 0, 0, 0, 0);

  return 0;
}
