/**
 * @file full_pipeline_test.cpp
 * @brief 完整自瞄流水线测试（检测+跟踪+瞄准+可视化），无需下位机
 */

#include <fmt/core.h>

#include <chrono>
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

const std::string keys =
  "{help h usage ? |                                  | 输出命令行参数说明}"
  "{@config-path   | configs/camera_detect_yolo26n.yaml | yaml配置文件路径  }"
  "{no-gui         |                            false | 无GUI模式         }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto no_gui = cli.get<bool>("no-gui");

  tools::Exiter exiter;

  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();
  double total_detect = 0, total_track = 0, total_aim = 0;

  // 模拟 IMU 数据（单位四元数，假设云台水平朝前）
  Eigen::Quaterniond fake_imu(1, 0, 0, 0);

  tools::logger()->info("Full pipeline test started (fake IMU, no CBoard)");

  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) break;

    solver.set_R_gimbal2world(fake_imu);

    // 1. 检测
    auto t1 = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto t2 = std::chrono::steady_clock::now();

    // 2. 跟踪
    auto targets = tracker.track(armors, timestamp);
    auto t3 = std::chrono::steady_clock::now();

    // 3. 瞄准
    auto command = aimer.aim(targets, timestamp, 27, false);
    auto t4 = std::chrono::steady_clock::now();

    total_detect += tools::delta_time(t2, t1);
    total_track += tools::delta_time(t3, t2);
    total_aim += tools::delta_time(t4, t3);

    frame_count++;

    // 可视化
    if (!no_gui) {
      // 画检测到的装甲板
      for (const auto & armor : armors) {
        tools::draw_points(img, armor.points, {0, 255, 255});
      }

      // 画跟踪状态
      tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

      if (!targets.empty()) {
        auto target = targets.front();

        // 反投影所有预测装甲板位置（绿色）
        auto armor_xyza_list = target.armor_xyza_list();
        for (const auto & xyza : armor_xyza_list) {
          auto pts = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(img, pts, {0, 255, 0});
        }

        // 瞄准点（红色=有效，蓝色=无效）
        auto aim_point = aimer.debug_aim_point;
        auto aim_pts = solver.reproject_armor(
          aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
        if (aim_point.valid)
          tools::draw_points(img, aim_pts, {0, 0, 255});
        else
          tools::draw_points(img, aim_pts, {255, 0, 0});
      }

      cv::resize(img, img, {}, 0.5, 0.5);
      cv::imshow("full_pipeline", img);
      if (cv::waitKey(1) == 'q') break;
    }

    // FPS 统计
    if (frame_count % 100 == 0) {
      auto elapsed = tools::delta_time(std::chrono::steady_clock::now(), start_time);
      tools::logger()->info(
        "[{}] FPS: {:.1f} | detect: {:.2f}ms | track: {:.2f}ms | aim: {:.2f}ms | targets: {}",
        frame_count, frame_count / elapsed,
        total_detect / frame_count * 1000,
        total_track / frame_count * 1000,
        total_aim / frame_count * 1000,
        targets.size());
    }
  }

  auto total_time = tools::delta_time(std::chrono::steady_clock::now(), start_time);
  tools::logger()->info("=== Final Results ===");
  tools::logger()->info("Total frames: {}, Total time: {:.2f}s", frame_count, total_time);
  tools::logger()->info("Overall FPS: {:.1f}", frame_count / total_time);
  tools::logger()->info("Avg detect: {:.2f} ms", total_detect / frame_count * 1000);
  tools::logger()->info("Avg track:  {:.2f} ms", total_track / frame_count * 1000);
  tools::logger()->info("Avg aim:    {:.2f} ms", total_aim / frame_count * 1000);

  return 0;
}
