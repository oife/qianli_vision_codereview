// 文件说明：Hero机器人使用MPC/五次多项式Planner进行轨迹规划与自瞄控制
#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>

#include "io/camera/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml | yaml配置文件路径 }";

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
  
  // 初始化hik摄像头
  io::Camera camera(config_path);
  
  // 初始化dm_imu
  // io::DM_IMU imu;
  
  // 初始化gimbal
  io::Gimbal gimbal(config_path);

  // 初始化自瞄模块（检测 + 解算 + 跟踪 + 轨迹规划）
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Planner planner(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  Eigen::Quaterniond gimbal_q;
  io::GimbalState gimbal_state;
  io::GimbalMode gimbal_mode;
  int frame_count = 0;
  auto t0 = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    // 读取hik摄像头图像和时间戳
    camera.read(img, t);
    if (img.empty()) {
      tools::logger()->warn("相机读取的图像为空，跳过此帧");
      continue;
    }

    // 读取gimbal数据
    gimbal_q = gimbal.q(t);  // 获取gimbal四元数（根据图像时间戳插值）
    gimbal_state = gimbal.state();  // 获取gimbal状态（yaw, pitch, bullet_speed等）
    gimbal_mode = gimbal.mode();  // 获取gimbal模式

    /// 自瞄核心逻辑（使用Planner进行轨迹规划）

    // 设置云台到世界的旋转（使用gimbal四元数）
    solver.set_R_gimbal2world(gimbal_q);

    // YOLO目标检测
    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    // 目标跟踪
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t);

    // 轨迹规划（使用gimbal状态中的子弹速度）
    auto aimer_start = std::chrono::steady_clock::now();
    std::optional<auto_aim::Target> target_opt;
    if (!targets.empty()) {
      target_opt = targets.front();
    } else {
      target_opt = std::nullopt;
    }
    auto plan = planner.plan(target_opt, gimbal_state.bullet_speed);

    // 发送plotter数据：时间、云台实际yaw/pitch、规划目标yaw/pitch
    {
      nlohmann::json data;
      data["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      data["gimbal_yaw"] = gimbal_state.yaw;
      data["gimbal_pitch"] = gimbal_state.pitch;
      data["target_yaw"] = plan.target_yaw;
      data["target_pitch"] = plan.target_pitch;
      plotter.plot(data);
    }

    // 发送控制命令到云台（根据规划结果设置角度、角速度和角加速度）
    gimbal.send(
      plan.control, plan.fire, plan.yaw, plan.yaw_vel, plan.yaw_acc, plan.pitch, plan.pitch_vel,
      plan.pitch_acc);

    // 调试输出

    // 打印command到日志
    tools::logger()->info(
      "[Plan] control: {}, fire: {}, yaw: {:.4f} rad ({:.2f} deg), pitch: {:.4f} rad ({:.2f} deg)",
      plan.control, plan.fire, plan.yaw, plan.yaw * 57.3, plan.pitch, plan.pitch * 57.3);

    auto finish = std::chrono::steady_clock::now();
    tools::logger()->info(
      "[{}] yolo: {:.1f}ms, tracker: {:.1f}ms, aimer: {:.1f}ms", frame_count,
      tools::delta_time(tracker_start, yolo_start) * 1e3,
      tools::delta_time(aimer_start, tracker_start) * 1e3,
      tools::delta_time(finish, aimer_start) * 1e3);

    // 绘制控制命令信息
    tools::draw_text(
      img,
      fmt::format(
        "plan is {},{:.2f},{:.2f},fire:{}", plan.control, plan.yaw * 57.3,
        plan.pitch * 57.3, plan.fire),
      {10, 60}, {154, 50, 205});

    // 绘制云台姿态信息
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
      {10, 90}, {255, 255, 255});

    // 绘制帧率信息
    static auto last_time = std::chrono::steady_clock::now();
    auto current_time = std::chrono::steady_clock::now();
    auto dt = tools::delta_time(current_time, last_time);
    last_time = current_time;
    if (dt > 0) {
      tools::draw_text(
        img, fmt::format("FPS: {:.1f}", 1.0 / dt), {10, 30}, {0, 255, 0});
    }

    // 绘制目标重投影（如果存在目标）
    if (!targets.empty()) {
      auto target = targets.front();

      // 绘制重投影装甲板位置（绿色）
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // 绘制planner瞄准位置（红色），使用planner的debug_xyza
      Eigen::Vector4d aim_xyza = planner.debug_xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      tools::draw_points(img, image_points, {0, 0, 255});
    }

    // 显示图像（可选缩小尺寸以提高显示性能）
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("hero_auto_aim", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;

    frame_count++;
  }

  return 0;
}
