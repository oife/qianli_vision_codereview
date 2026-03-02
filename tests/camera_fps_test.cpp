/**
 * @file camera_fps_test.cpp
 * @brief 纯推理帧率测试（无显示）
 */

#include <fmt/core.h>
#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                            | 输出命令行参数说明}"
  "{config-path c  | configs/camera_detect.yaml | yaml配置文件路径  }"
  "{resize r       | 0                          | 是否resize到640   }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>("config-path");
  auto do_resize = cli.get<int>("resize");

  tools::Exiter exiter;

  io::Camera camera(config_path);
  auto_aim::YOLO yolo(config_path, false);  // debug=false，不显示

  cv::Mat img, resized;
  std::chrono::steady_clock::time_point timestamp;

  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();
  double total_read_time = 0, total_resize_time = 0, total_detect_time = 0;

  tools::logger()->info("Starting FPS test (no display), resize={}", do_resize);

  while (!exiter.exit() && frame_count < 500) {
    // 相机读取
    auto t1 = std::chrono::steady_clock::now();
    camera.read(img, timestamp);
    auto t2 = std::chrono::steady_clock::now();

    if (img.empty()) break;

    // 可选resize
    cv::Mat & input_img = img;
    if (do_resize) {
      cv::resize(img, resized, {640, 640});
      input_img = resized;
    }
    auto t3 = std::chrono::steady_clock::now();

    // 推理
    auto armors = yolo.detect(input_img, frame_count);
    auto t4 = std::chrono::steady_clock::now();

    total_read_time += tools::delta_time(t2, t1);
    total_resize_time += tools::delta_time(t3, t2);
    total_detect_time += tools::delta_time(t4, t3);

    frame_count++;

    if (frame_count % 100 == 0) {
      auto elapsed = tools::delta_time(std::chrono::steady_clock::now(), start_time);
      tools::logger()->info(
        "[{}] FPS: {:.1f} | read: {:.2f}ms | resize: {:.2f}ms | detect: {:.2f}ms",
        frame_count, frame_count / elapsed,
        total_read_time / frame_count * 1000,
        total_resize_time / frame_count * 1000,
        total_detect_time / frame_count * 1000);
    }
  }

  auto total_time = tools::delta_time(std::chrono::steady_clock::now(), start_time);
  tools::logger()->info("=== Final Results ===");
  tools::logger()->info("Total frames: {}, Total time: {:.2f}s", frame_count, total_time);
  tools::logger()->info("Overall FPS: {:.1f}", frame_count / total_time);
  tools::logger()->info("Avg read:   {:.2f} ms", total_read_time / frame_count * 1000);
  tools::logger()->info("Avg resize: {:.2f} ms", total_resize_time / frame_count * 1000);
  tools::logger()->info("Avg detect: {:.2f} ms", total_detect_time / frame_count * 1000);

  return 0;
}
