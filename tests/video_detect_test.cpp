/**
 * @file video_detect_test.cpp
 * @brief 视频文件检测测试，输出带检测结果的视频
 */

#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

const std::string keys =
  "{help h usage ? |                          | 输出命令行参数说明}"
  "{config-path c  | configs/camera_detect.yaml | yaml配置文件路径}"
  "{output o       | output.mp4               | 输出视频路径     }"
  "{@input         |                          | 输入视频路径     }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@input")) {
    cli.printMessage();
    return 0;
  }

  auto input_path = cli.get<std::string>("@input");
  auto config_path = cli.get<std::string>("config-path");
  auto output_path = cli.get<std::string>("output");

  tools::Exiter exiter;

  cv::VideoCapture video(input_path);
  if (!video.isOpened()) {
    tools::logger()->error("Failed to open video: {}", input_path);
    return -1;
  }

  int width = static_cast<int>(video.get(cv::CAP_PROP_FRAME_WIDTH));
  int height = static_cast<int>(video.get(cv::CAP_PROP_FRAME_HEIGHT));
  double fps = video.get(cv::CAP_PROP_FPS);
  int total_frames = static_cast<int>(video.get(cv::CAP_PROP_FRAME_COUNT));

  tools::logger()->info("Input video: {}x{} @ {:.2f}fps, {} frames", width, height, fps, total_frames);

  cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, {width, height});
  if (!writer.isOpened()) {
    tools::logger()->error("Failed to create output video: {}", output_path);
    return -1;
  }

  auto_aim::YOLO yolo(config_path, false);  // debug=false，不显示窗口

  cv::Mat img;
  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();

  while (!exiter.exit()) {
    video.read(img);
    if (img.empty()) break;

    auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto detect_end = std::chrono::steady_clock::now();
    double detect_time = tools::delta_time(detect_end, detect_start) * 1000;

    // 绘制检测结果
    for (const auto & armor : armors) {
      // 绘制四个角点
      for (int i = 0; i < 4; i++) {
        cv::line(img, armor.points[i], armor.points[(i + 1) % 4], {0, 255, 0}, 2);
        cv::circle(img, armor.points[i], 4, {0, 0, 255}, -1);
      }
      // 绘制信息
      auto info = fmt::format("{:.2f}", armor.confidence);
      cv::putText(img, info, armor.center, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
    }

    // 绘制帧信息
    auto info = fmt::format("[{}] {:.1f}ms {} armors", frame_count, detect_time, armors.size());
    cv::putText(img, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);

    writer.write(img);
    frame_count++;

    if (frame_count % 100 == 0) {
      auto elapsed = tools::delta_time(std::chrono::steady_clock::now(), start_time);
      double progress = 100.0 * frame_count / total_frames;
      double avg_fps = frame_count / elapsed;
      tools::logger()->info("Progress: {:.1f}% ({}/{}), avg {:.1f} fps", progress, frame_count, total_frames, avg_fps);
    }
  }

  auto total_time = tools::delta_time(std::chrono::steady_clock::now(), start_time);
  tools::logger()->info("Done! Processed {} frames in {:.1f}s, avg {:.1f} fps", frame_count, total_time, frame_count / total_time);
  tools::logger()->info("Output saved to: {}", output_path);

  return 0;
}
