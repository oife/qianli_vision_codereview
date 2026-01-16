/**
 * @file camera_auto_aim_test.cpp
 * @brief 测试自动瞄准功能（相机实时测试）
 * 
 * 该测试程序用于实时测试自动瞄准系统，包括：
 * - 从相机读取图像
 * - 从文本文件读取固定IMU数据（四元数）
 * - YOLO目标检测
 * - 目标跟踪和预测
 * - 弹道解算和瞄准控制
 * - 可视化调试信息（重投影、状态显示等）
 */

#include <fmt/core.h>

#include <chrono>
#include <fstream>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "tasks/auto_aim/aimer/aimer.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明 }"
  "{config-path c  |configs/demo.yaml         | yaml配置文件的路径}"
  "{imu-path i     |                          | IMU数据文本文件路径（格式: w x y z）}";

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("config-path");
  auto imu_path = cli.get<std::string>("imu-path");

  tools::Plotter plotter;
  tools::Exiter exiter;

  // 初始化相机
  io::Camera camera(config_path);

  // 读取固定IMU数据
  Eigen::Quaterniond fixed_imu_q{1.0, 0.0, 0.0, 0.0};  // 默认单位四元数
  if (!imu_path.empty()) {
    std::ifstream imu_file(imu_path);
    if (imu_file.is_open()) {
      double w, x, y, z;
      if (imu_file >> w >> x >> y >> z) {
        fixed_imu_q = Eigen::Quaterniond{w, x, y, z};
        fixed_imu_q.normalize();
        tools::logger()->info(
          "已从文件读取固定IMU四元数: w={:.4f}, x={:.4f}, y={:.4f}, z={:.4f}", w, x, y, z);
      } else {
        tools::logger()->warn("无法从文件读取IMU数据，使用默认值（单位四元数）");
      }
      imu_file.close();
    } else {
      tools::logger()->warn("无法打开IMU文件: {}，使用默认值（单位四元数）", imu_path);
    }
  } else {
    tools::logger()->info("未指定IMU文件路径，使用默认值（单位四元数）");
  }

  // 初始化自瞄模块
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  auto_aim::Target last_target;
  io::Command last_command;
  int frame_count = 0;

  while (!exiter.exit()) {
    // 从相机读取图像
    camera.read(img, timestamp);
    if (img.empty()) {
      tools::logger()->warn("相机读取的图像为空，跳过此帧");
      continue;
    }

    /// 自瞄核心逻辑

    // 设置固定IMU数据（云台到世界的旋转）
    solver.set_R_gimbal2world(fixed_imu_q);

    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, timestamp);

    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, timestamp, 27, false);

    // 射击判断（当目标存在、瞄准点有效且控制角度变化小于2度时）
    if (
      !targets.empty() && aimer.debug_aim_point.valid &&
      std::abs(command.yaw - last_command.yaw) * 57.3 < 2)
      command.shoot = true;

    if (command.control) last_command = command;

    /// 调试输出

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

    // 绘制云台姿态信息
    tools::draw_text(
      img,
      fmt::format(
        "gimbal yaw{:.2f}", (tools::eulers(fixed_imu_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0]),
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

    nlohmann::json data;

    // 装甲板原始观测数据
    data["armor_num"] = armors.size();
    if (!armors.empty()) {
      const auto & armor = armors.front();
      data["armor_x"] = armor.xyz_in_world[0];
      data["armor_y"] = armor.xyz_in_world[1];
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;
      data["armor_center_x"] = armor.center_norm.x;
      data["armor_center_y"] = armor.center_norm.y;
    }

    // 云台和命令数据
    auto yaw = tools::eulers(fixed_imu_q, 2, 1, 0)[0];
    data["gimbal_yaw"] = yaw * 57.3;
    data["cmd_yaw"] = command.yaw * 57.3;
    data["shoot"] = command.shoot;

    if (!targets.empty()) {
      auto target = targets.front();

      std::vector<Eigen::Vector4d> armor_xyza_list;

      // 当前帧target更新后，绘制重投影装甲板位置（绿色）
      armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});
      }

      // 绘制aimer瞄准位置（红色）
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});

      // 观测器内部数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];
      data["vx"] = x[1];
      data["y"] = x[2];
      data["vy"] = x[3];
      data["z"] = x[4];
      data["vz"] = x[5];
      data["a"] = x[6] * 57.3;
      data["w"] = x[7];
      data["r"] = x[8];
      data["l"] = x[9];
      data["h"] = x[10];
      data["last_id"] = target.last_id;

      // 卡方检验数据
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");
      data["residual_distance"] = target.ekf().data.at("residual_distance");
      data["residual_angle"] = target.ekf().data.at("residual_angle");
      data["nis"] = target.ekf().data.at("nis");
      data["nees"] = target.ekf().data.at("nees");
      data["nis_fail"] = target.ekf().data.at("nis_fail");
      data["nees_fail"] = target.ekf().data.at("nees_fail");
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");
    }

    plotter.plot(data);

    // 显示图像（可选缩小尺寸以提高显示性能）
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("camera_auto_aim", img);
    auto key = cv::waitKey(1);
    if (key == 'q') break;

    frame_count++;
  }

  return 0;
}

