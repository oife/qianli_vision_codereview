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
  "{infer-only     | false                    | 仅跑推理，不绘制不写视频}"
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
  bool infer_only = cli.get<bool>("infer-only");

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

  cv::VideoWriter writer;
  if (!infer_only) {
    writer.open(output_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, {width, height});
    if (!writer.isOpened()) {
      tools::logger()->error("Failed to create output video: {}", output_path);
      return -1;
    }
  }

  auto_aim::YOLO yolo(config_path, false);  // debug=false，不显示窗口

  cv::Mat img;
  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();
  double total_read_ms = 0.0;
  double total_detect_ms = 0.0;
  double total_backend_ms = 0.0;
  double total_preprocess_ms = 0.0;
  double total_infer_ms = 0.0;
  double total_prepare_input_ms = 0.0;
  double total_h2d_submit_ms = 0.0;
  double total_enqueue_submit_ms = 0.0;
  double total_d2h_submit_ms = 0.0;
  double total_sync_ms = 0.0;
  double total_output_convert_ms = 0.0;
  double total_postprocess_ms = 0.0;
  double total_draw_ms = 0.0;
  double total_write_ms = 0.0;
  double total_loop_ms = 0.0;

  while (!exiter.exit()) {
    auto loop_start = std::chrono::steady_clock::now();

    auto read_start = std::chrono::steady_clock::now();
    video.read(img);
    auto read_end = std::chrono::steady_clock::now();
    if (img.empty()) break;
    double read_time = tools::delta_time(read_end, read_start) * 1000;

    auto detect_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto detect_end = std::chrono::steady_clock::now();
    double detect_time = tools::delta_time(detect_end, detect_start) * 1000;
    auto detect_profile = yolo.get_last_profile();

    double draw_time = 0.0;
    double write_time = 0.0;
    auto write_end = detect_end;
    if (!infer_only) {
      auto draw_start = std::chrono::steady_clock::now();
      for (const auto & armor : armors) {
        for (int i = 0; i < 4; i++) {
          cv::line(img, armor.points[i], armor.points[(i + 1) % 4], {0, 255, 0}, 2);
          cv::circle(img, armor.points[i], 4, {0, 0, 255}, -1);
        }
        auto info = fmt::format("{:.2f}", armor.confidence);
        cv::putText(img, info, armor.center, cv::FONT_HERSHEY_SIMPLEX, 0.6, {0, 255, 0}, 2);
      }

      auto info = fmt::format("[{}] {:.1f}ms {} armors", frame_count, detect_time, armors.size());
      cv::putText(img, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {255, 255, 255}, 2);
      auto draw_end = std::chrono::steady_clock::now();
      draw_time = tools::delta_time(draw_end, draw_start) * 1000;

      auto write_start = std::chrono::steady_clock::now();
      writer.write(img);
      write_end = std::chrono::steady_clock::now();
      write_time = tools::delta_time(write_end, write_start) * 1000;
    }

    double loop_time = tools::delta_time(write_end, loop_start) * 1000;

    total_read_ms += read_time;
    total_detect_ms += detect_time;
    total_backend_ms += detect_profile.backend_execute_ms;
    total_preprocess_ms += detect_profile.backend.preprocess_ms;
    total_infer_ms += detect_profile.backend.infer_ms;
    total_prepare_input_ms += detect_profile.backend.prepare_input_ms;
    total_h2d_submit_ms += detect_profile.backend.h2d_submit_ms;
    total_enqueue_submit_ms += detect_profile.backend.enqueue_submit_ms;
    total_d2h_submit_ms += detect_profile.backend.d2h_submit_ms;
    total_sync_ms += detect_profile.backend.sync_ms;
    total_output_convert_ms += detect_profile.backend.output_convert_ms;
    total_postprocess_ms += detect_profile.postprocess_ms;
    total_draw_ms += draw_time;
    total_write_ms += write_time;
    total_loop_ms += loop_time;

    if (frame_count < 5) {
      tools::logger()->info(
        "Frame {} | read {:.2f} ms | detect {:.2f} ms [backend {:.2f} = pre {:.2f} + infer {:.2f} "
        "(prep {:.2f} + h2d {:.2f} + enqueue {:.2f} + d2h {:.2f} + sync {:.2f} + out {:.2f}) | "
        "post {:.2f}] | draw {:.2f} ms | write {:.2f} ms | loop {:.2f} ms | armors {}",
        frame_count, read_time, detect_time, detect_profile.backend_execute_ms,
        detect_profile.backend.preprocess_ms, detect_profile.backend.infer_ms,
        detect_profile.backend.prepare_input_ms, detect_profile.backend.h2d_submit_ms,
        detect_profile.backend.enqueue_submit_ms, detect_profile.backend.d2h_submit_ms,
        detect_profile.backend.sync_ms, detect_profile.backend.output_convert_ms,
        detect_profile.postprocess_ms, draw_time, write_time, loop_time, armors.size());
    }

    frame_count++;

    if (frame_count % 100 == 0) {
      auto elapsed = tools::delta_time(std::chrono::steady_clock::now(), start_time);
      double progress = 100.0 * frame_count / total_frames;
      double avg_fps = frame_count / elapsed;
      tools::logger()->info(
        "Progress: {:.1f}% ({}/{}), avg {:.1f} fps | avg read {:.2f} ms | avg detect {:.2f} ms "
        "[backend {:.2f} = pre {:.2f} + infer {:.2f} (prep {:.2f} + h2d {:.2f} + enqueue {:.2f} "
        "+ d2h {:.2f} + sync {:.2f} + out {:.2f}) | post {:.2f}] | avg draw {:.2f} ms | avg "
        "write {:.2f} ms | avg loop {:.2f} ms",
        progress, frame_count, total_frames, avg_fps, total_read_ms / frame_count,
        total_detect_ms / frame_count, total_backend_ms / frame_count,
        total_preprocess_ms / frame_count, total_infer_ms / frame_count,
        total_prepare_input_ms / frame_count, total_h2d_submit_ms / frame_count,
        total_enqueue_submit_ms / frame_count, total_d2h_submit_ms / frame_count,
        total_sync_ms / frame_count, total_output_convert_ms / frame_count,
        total_postprocess_ms / frame_count, total_draw_ms / frame_count,
        total_write_ms / frame_count, total_loop_ms / frame_count);
    }
  }

  auto total_time = tools::delta_time(std::chrono::steady_clock::now(), start_time);
  tools::logger()->info("Done! Processed {} frames in {:.1f}s, avg {:.1f} fps", frame_count, total_time, frame_count / total_time);
  if (frame_count > 0) {
    tools::logger()->info(
      "Final avg | read {:.2f} ms | detect {:.2f} ms [backend {:.2f} = pre {:.2f} + infer {:.2f} "
      "(prep {:.2f} + h2d {:.2f} + enqueue {:.2f} + d2h {:.2f} + sync {:.2f} + out {:.2f}) | "
      "post {:.2f}] | draw {:.2f} ms | write {:.2f} ms | loop {:.2f} ms",
      total_read_ms / frame_count, total_detect_ms / frame_count, total_backend_ms / frame_count,
      total_preprocess_ms / frame_count, total_infer_ms / frame_count,
      total_prepare_input_ms / frame_count, total_h2d_submit_ms / frame_count,
      total_enqueue_submit_ms / frame_count, total_d2h_submit_ms / frame_count,
      total_sync_ms / frame_count, total_output_convert_ms / frame_count,
      total_postprocess_ms / frame_count, total_draw_ms / frame_count,
      total_write_ms / frame_count, total_loop_ms / frame_count);
  }
  if (infer_only) {
    tools::logger()->info("Infer-only mode: no video was written");
  } else {
    tools::logger()->info("Output saved to: {}", output_path);
  }

  return 0;
}
