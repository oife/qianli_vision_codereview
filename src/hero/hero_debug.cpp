// 文件说明：Hero机器人主程序，包含hik摄像头读取、dm_imu读取和gimbal数据读取功能，以及自瞄识别、跟踪和瞄准功能
#include <fmt/core.h>

#include <chrono>
#include <cmath>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer/aimer.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml | yaml配置文件路径 }";

using namespace std::chrono_literals;

namespace hero_shoot
{
// 基于 hero_test 的“更保守/更高把握度”门限
constexpr double SHOOT_YAW_THRESH = 0.6 / 57.3;    // rad
constexpr double SHOOT_PITCH_THRESH = 0.6 / 57.3;  // rad
constexpr int SHOOT_STABLE_FRAMES = 3;             // 连续稳定帧数
constexpr double SHOOT_COOLDOWN_SEC = 1;           // 冷却，避免过快连发
constexpr double SHOOT_ALIGN_TIME_THRESH_SEC = 0.03;
constexpr double SHOOT_CENTERLINE_THRESH = 2.0 / 57.3;
}  // namespace hero_shoot

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  
  // 初始化hik摄像头
  io::Camera camera(config_path);
  
  // 初始化dm_imu
  // io::DM_IMU imu;
  
  // 初始化gimbal
  io::Gimbal gimbal(config_path);

  // 初始化自瞄模块
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  Eigen::Quaterniond gimbal_q;
  io::GimbalState gimbal_state;
  io::GimbalMode gimbal_mode;
  io::Command last_command{};
  bool last_command_valid = false;
  int stable_count = 0;
  auto last_shoot_time = std::chrono::steady_clock::now() - std::chrono::seconds(5);
  int frame_count = 0;

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

    /// 自瞄核心逻辑

    // 设置云台到世界的旋转（使用gimbal四元数）
    solver.set_R_gimbal2world(gimbal_q);

    // YOLO目标检测
    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    // 目标跟踪
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t);

    // 瞄准控制（使用gimbal状态中的子弹速度）
    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, t, gimbal_state.bullet_speed);

    // 射击判断：稳定帧 + 更严格门限（与 hero 保持一致）
    command.shoot = false;
    if (!targets.empty() && aimer.debug_aim_point.valid && command.control) {
      if (last_command_valid) {
        double ye = std::abs(tools::limit_rad(command.yaw - last_command.yaw));
        double pe = std::abs(command.pitch - last_command.pitch);
        bool aligned = true;
        if (aimer.debug_aim_point.center_aim) {
          aligned =
            aimer.debug_aim_point.time_error < hero_shoot::SHOOT_ALIGN_TIME_THRESH_SEC &&
            aimer.debug_aim_point.centerline_error < hero_shoot::SHOOT_CENTERLINE_THRESH;
        }
        bool stable = (ye < hero_shoot::SHOOT_YAW_THRESH) && (pe < hero_shoot::SHOOT_PITCH_THRESH);
        stable_count = (stable && aligned) ? (stable_count + 1) : 0;

        auto now = std::chrono::steady_clock::now();
        double since = tools::delta_time(now, last_shoot_time);
        if (stable_count >= hero_shoot::SHOOT_STABLE_FRAMES &&
          since >= hero_shoot::SHOOT_COOLDOWN_SEC) {
          command.shoot = true;
          stable_count = 0;
          last_shoot_time = now;
        }
      }
    } else {
      stable_count = 0;
    }

    if (command.control) last_command = command;
    if (command.control) last_command_valid = true;

    // 发送控制命令到云台（角速度和角加速度设为0）
    gimbal.send(
      command.control, command.shoot, command.yaw, 0.0f, 0.0f, command.pitch, 0.0f, 0.0f);

    // 调试输出

    // 打印command到日志
    tools::logger()->info(
      "[Command] control: {}, shoot: {}, yaw: {:.4f} rad ({:.2f} deg), pitch: {:.4f} rad ({:.2f} deg), center_aim: {}, terr: {:.1f} ms, cerr: {:.2f} deg",
      command.control, command.shoot, command.yaw, command.yaw * 57.3, command.pitch,
      command.pitch * 57.3, aimer.debug_aim_point.center_aim,
      aimer.debug_aim_point.time_error * 1e3, aimer.debug_aim_point.centerline_error * 57.3);

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
        "command is {},{:.2f},{:.2f},shoot:{}", command.control, command.yaw * 57.3,
        command.pitch * 57.3, command.shoot),
      {10, 60}, {154, 50, 205});
    tools::draw_text(
      img,
      fmt::format(
        "aim:{} terr:{:.0f}ms cerr:{:.2f}d", aimer.debug_aim_point.center_aim ? "center" : "plate",
        aimer.debug_aim_point.time_error * 1e3, aimer.debug_aim_point.centerline_error * 57.3),
      {10, 120}, {200, 200, 255});

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

      // 绘制aimer瞄准位置（红色）
      auto aim_point = aimer.debug_aim_point;
      if (aim_point.valid) {
        Eigen::Vector4d aim_xyza = aim_point.xyza;
        auto image_points =
          solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 0, 255});

        if (aim_point.center_aim) {
          std::vector<cv::Point3f> center_world_points = {
            cv::Point3f(
              static_cast<float>(aim_point.center_xyz.x()), static_cast<float>(aim_point.center_xyz.y()),
              static_cast<float>(aim_point.center_xyz.z()))};
          auto center_pixels = solver.world2pixel(center_world_points);
          if (!center_pixels.empty())
            tools::draw_point(img, cv::Point(center_pixels.front()), {255, 0, 0}, 6);
        }
      }
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
