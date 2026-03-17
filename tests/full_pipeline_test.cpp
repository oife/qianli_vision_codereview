/**
 * @file full_pipeline_test.cpp
 * @brief 完整自瞄流水线测试（检测+跟踪+瞄准+可视化），无需下位机
 */

#include <fmt/core.h>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <fstream>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>
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
  "{video-path     |                                  | 离线视频路径      }"
  "{max-frames     | 0                                | 最大处理帧数      }"
  "{display-every  | 3                                | 每N帧刷新一次GUI  }"
  "{display-fps    | 12                               | GUI最大刷新FPS    }"
  "{mjpeg          |                             true | 开启MJPEG网页预览 }"
  "{mjpeg-port     | 9000                             | MJPEG服务端口     }"
  "{mjpeg-fps      | 10                               | MJPEG最大刷新FPS  }"
  "{no-gui         |                             true | 无GUI模式         }"
  "{q-path         |                                  | 四元数+state文件  }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto video_path = cli.get<std::string>("video-path");
  auto max_frames = cli.get<int>("max-frames");
  auto display_every = std::max(1, cli.get<int>("display-every"));
  auto display_fps = std::max(1, cli.get<int>("display-fps"));
  auto mjpeg = cli.get<bool>("mjpeg");
  auto mjpeg_port = cli.get<int>("mjpeg-port");
  auto mjpeg_fps = std::max(1, cli.get<int>("mjpeg-fps"));
  auto no_gui = cli.get<bool>("no-gui");
  auto q_path = cli.get<std::string>("q-path");

  tools::Exiter exiter;

  // 加载四元数+GimbalState文件
  struct QRecord {
    double t;
    Eigen::Quaterniond q;
    float bullet_speed;
  };
  std::vector<QRecord> q_records;
  bool use_q_file = !q_path.empty();
  if (use_q_file) {
    std::ifstream qf(q_path);
    if (!qf.is_open()) {
      tools::logger()->error("Failed to open q-path: {}", q_path);
      return -1;
    }
    std::string line;
    while (std::getline(qf, line)) {
      std::istringstream iss(line);
      double t_val, w, x, y, z;
      float yaw, yaw_vel, pitch, pitch_vel, bs;
      uint16_t bc;
      if (iss >> t_val >> w >> x >> y >> z) {
        QRecord rec;
        rec.t = t_val;
        rec.q = Eigen::Quaterniond(w, x, y, z);
        rec.bullet_speed = 27.0f;
        // 尝试读取扩展字段
        if (iss >> yaw >> yaw_vel >> pitch >> pitch_vel >> bs >> bc) {
          rec.bullet_speed = bs;
        }
        q_records.push_back(rec);
      }
    }
    tools::logger()->info("Loaded {} quaternion records from {}", q_records.size(), q_path);
  }

  std::optional<io::Camera> camera;
  cv::VideoCapture video;
  bool use_video = !video_path.empty();
  double video_fps = 0.0;
  auto video_t0 = std::chrono::steady_clock::now();
  std::mutex gui_mutex;
  cv::Mat gui_frame;
  std::atomic<bool> stop_gui{false};
  std::atomic<bool> gui_quit{false};
  auto gui_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / display_fps));
  std::mutex mjpeg_source_mutex;
  cv::Mat mjpeg_source_frame;
  std::mutex mjpeg_jpeg_mutex;
  std::vector<uchar> mjpeg_jpeg;
  std::atomic<uint64_t> mjpeg_jpeg_version{0};
  std::atomic<bool> stop_mjpeg{false};
  auto mjpeg_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / mjpeg_fps));

  if (use_video) {
    video.open(video_path);
    if (!video.isOpened()) {
      tools::logger()->error("Failed to open video: {}", video_path);
      return -1;
    }
    video_fps = video.get(cv::CAP_PROP_FPS);
    if (video_fps <= 0.0) video_fps = 30.0;
    tools::logger()->info("Using offline video: {} @ {:.2f} fps", video_path, video_fps);
  } else {
    camera.emplace(config_path);
  }

  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);

  std::thread gui_thread;
  std::thread mjpeg_encode_thread;
  std::thread mjpeg_server_thread;
  if (!no_gui) {
    gui_thread = std::thread([&] {
      cv::namedWindow("full_pipeline", cv::WINDOW_NORMAL);
      auto last_show_time = std::chrono::steady_clock::now() - gui_frame_interval;
      while (!stop_gui) {
        auto now = std::chrono::steady_clock::now();
        auto since_last = now - last_show_time;
        if (since_last < gui_frame_interval) {
          auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(gui_frame_interval - since_last);
          if (remain.count() > 0) std::this_thread::sleep_for(remain);
        }

        cv::Mat frame_to_show;
        {
          std::scoped_lock lock(gui_mutex);
          if (!gui_frame.empty()) {
            frame_to_show = std::move(gui_frame);
            gui_frame.release();
          }
        }
        if (!frame_to_show.empty()) {
          cv::imshow("full_pipeline", frame_to_show);
          last_show_time = std::chrono::steady_clock::now();
        }
        if (cv::waitKey(1) == 'q') {
          gui_quit = true;
          stop_gui = true;
          break;
        }
      }
      cv::destroyWindow("full_pipeline");
    });
  }

  if (mjpeg) {
    mjpeg_encode_thread = std::thread([&] {
      auto last_encode_time = std::chrono::steady_clock::now() - mjpeg_frame_interval;
      std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};
      while (!stop_mjpeg) {
        auto now = std::chrono::steady_clock::now();
        auto since_last = now - last_encode_time;
        if (since_last < mjpeg_frame_interval) {
          auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(mjpeg_frame_interval - since_last);
          if (remain.count() > 0) std::this_thread::sleep_for(remain);
        }

        cv::Mat frame_to_encode;
        {
          std::scoped_lock lock(mjpeg_source_mutex);
          if (!mjpeg_source_frame.empty()) {
            frame_to_encode = std::move(mjpeg_source_frame);
            mjpeg_source_frame.release();
          }
        }
        if (frame_to_encode.empty()) continue;

        std::vector<uchar> encoded;
        if (cv::imencode(".jpg", frame_to_encode, encoded, encode_params)) {
          std::scoped_lock lock(mjpeg_jpeg_mutex);
          mjpeg_jpeg = std::move(encoded);
          mjpeg_jpeg_version.fetch_add(1, std::memory_order_relaxed);
          last_encode_time = std::chrono::steady_clock::now();
        }
      }
    });

    mjpeg_server_thread = std::thread([&] {
      int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd < 0) {
        tools::logger()->error("MJPEG server socket create failed");
        stop_mjpeg = true;
        return;
      }

      int opt = 1;
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = INADDR_ANY;
      addr.sin_port = htons(static_cast<uint16_t>(mjpeg_port));
      if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        tools::logger()->error("MJPEG server bind failed on port {}", mjpeg_port);
        ::close(server_fd);
        stop_mjpeg = true;
        return;
      }
      if (listen(server_fd, 4) < 0) {
        tools::logger()->error("MJPEG server listen failed");
        ::close(server_fd);
        stop_mjpeg = true;
        return;
      }

      tools::logger()->info("MJPEG preview: http://0.0.0.0:{}/", mjpeg_port);

      auto send_all = [](int fd, const void * data, size_t size) {
        const char * p = static_cast<const char *>(data);
        while (size > 0) {
          ssize_t n = ::send(fd, p, size, MSG_NOSIGNAL);
          if (n <= 0) return false;
          p += n;
          size -= static_cast<size_t>(n);
        }
        return true;
      };

      while (!stop_mjpeg) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        timeval tv{};
        tv.tv_sec = 0;
        tv.tv_usec = 200000;
        int ready = ::select(server_fd + 1, &rfds, nullptr, nullptr, &tv);
        if (ready <= 0) continue;

        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        std::thread([&, client_fd] {
          char req_buf[1024];
          ssize_t req_len = ::recv(client_fd, req_buf, sizeof(req_buf) - 1, 0);
          if (req_len <= 0) {
            ::close(client_fd);
            return;
          }
          req_buf[req_len] = '\0';
          std::string req(req_buf);

          if (req.find("GET / ") != std::string::npos || req.find("GET /HTTP") != std::string::npos) {
            std::string html =
              "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
              "<html><body style='margin:0;background:#111'>"
              "<img src='/stream' style='max-width:100%;height:auto;display:block;margin:auto' />"
              "</body></html>";
            send_all(client_fd, html.data(), html.size());
            ::close(client_fd);
            return;
          }

          if (req.find("GET /stream") == std::string::npos) {
            const char * not_found =
              "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            send_all(client_fd, not_found, std::strlen(not_found));
            ::close(client_fd);
            return;
          }

          const char * header =
            "HTTP/1.1 200 OK\r\n"
            "Cache-Control: no-cache\r\n"
            "Connection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
          if (!send_all(client_fd, header, std::strlen(header))) {
            ::close(client_fd);
            return;
          }

          uint64_t last_version = 0;
          while (!stop_mjpeg) {
            std::vector<uchar> jpeg_copy;
            uint64_t version = mjpeg_jpeg_version.load(std::memory_order_relaxed);
            if (version == 0 || version == last_version) {
              std::this_thread::sleep_for(std::chrono::milliseconds(5));
              continue;
            }
            {
              std::scoped_lock lock(mjpeg_jpeg_mutex);
              jpeg_copy = mjpeg_jpeg;
              version = mjpeg_jpeg_version.load(std::memory_order_relaxed);
            }
            if (jpeg_copy.empty()) continue;

            auto part_header = fmt::format(
              "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
              jpeg_copy.size());
            if (!send_all(client_fd, part_header.data(), part_header.size()) ||
                !send_all(client_fd, jpeg_copy.data(), jpeg_copy.size()) ||
                !send_all(client_fd, "\r\n", 2)) {
              break;
            }
            last_version = version;
          }

          ::close(client_fd);
        }).detach();
      }

      ::close(server_fd);
    });
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();
  double total_read = 0, total_detect = 0, total_track = 0, total_aim = 0;

  // 模拟 IMU 数据（单位四元数，假设云台水平朝前）
  Eigen::Quaterniond fake_imu(1, 0, 0, 0);
  float current_bullet_speed = 27.0f;

  tools::logger()->info(
    "Full pipeline test started ({}, no CBoard)", use_q_file ? "recorded IMU" : "fake IMU");

  while (!exiter.exit() && !gui_quit) {
    if (max_frames > 0 && frame_count >= max_frames) break;

    auto t0 = std::chrono::steady_clock::now();
    if (use_video) {
      video.read(img);
      timestamp = video_t0 + std::chrono::microseconds(static_cast<int64_t>(frame_count * 1e6 / video_fps));
    } else {
      camera->read(img, timestamp);
    }
    auto t_read = std::chrono::steady_clock::now();
    if (img.empty()) break;

    // 使用录制的四元数或 fake_imu
    if (use_q_file && frame_count < static_cast<int>(q_records.size())) {
      auto & rec = q_records[frame_count];
      solver.set_R_gimbal2world(rec.q);
      current_bullet_speed = rec.bullet_speed;
    } else {
      solver.set_R_gimbal2world(fake_imu);
    }

    // 1. 检测
    auto t1 = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);
    auto t2 = std::chrono::steady_clock::now();

    // 2. 跟踪
    auto targets = tracker.track(armors, timestamp);
    auto t3 = std::chrono::steady_clock::now();

    // 3. 瞄准
    auto command = aimer.aim(targets, timestamp, current_bullet_speed, false);
    auto t4 = std::chrono::steady_clock::now();

    total_read += tools::delta_time(t_read, t0);
    total_detect += tools::delta_time(t2, t1);
    total_track += tools::delta_time(t3, t2);
    total_aim += tools::delta_time(t4, t3);

    frame_count++;

    // 可视化
    if (( !no_gui || mjpeg ) && frame_count % display_every == 0) {
      cv::Mat display = img.clone();

      // 画检测到的装甲板
      for (const auto & armor : armors) {
        tools::draw_points(display, armor.points, {0, 255, 255});
      }

      // 画跟踪状态
      tools::draw_text(display, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

      if (!targets.empty()) {
        auto target = targets.front();

        // 反投影所有预测装甲板位置（绿色）
        auto armor_xyza_list = target.armor_xyza_list();
        for (const auto & xyza : armor_xyza_list) {
          auto pts = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(display, pts, {0, 255, 0});
        }

        // 瞄准点（红色=有效，蓝色=无效）
        auto aim_point = aimer.debug_aim_point;
        auto aim_pts = solver.reproject_armor(
          aim_point.xyza.head(3), aim_point.xyza[3], target.armor_type, target.name);
        if (aim_point.valid)
          tools::draw_points(display, aim_pts, {0, 0, 255});
        else
          tools::draw_points(display, aim_pts, {255, 0, 0});
      }
      {
        if (!no_gui) {
          std::scoped_lock lock(gui_mutex);
          gui_frame = display.clone();
        }
        if (mjpeg) {
          std::scoped_lock lock(mjpeg_source_mutex);
          mjpeg_source_frame = std::move(display);
        }
      }
    }

    // FPS 统计
    if (frame_count % 100 == 0) {
      auto elapsed = tools::delta_time(std::chrono::steady_clock::now(), start_time);
      tools::logger()->info(
        "[{}] FPS: {:.1f} | read: {:.2f}ms | detect: {:.2f}ms | track: {:.2f}ms | aim: {:.2f}ms | targets: {}",
        frame_count, frame_count / elapsed,
        total_read / frame_count * 1000,
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
  tools::logger()->info("Avg read:   {:.2f} ms", total_read / frame_count * 1000);
  tools::logger()->info("Avg detect: {:.2f} ms", total_detect / frame_count * 1000);
  tools::logger()->info("Avg track:  {:.2f} ms", total_track / frame_count * 1000);
  tools::logger()->info("Avg aim:    {:.2f} ms", total_aim / frame_count * 1000);

  stop_gui = true;
  stop_mjpeg = true;
  if (gui_thread.joinable()) gui_thread.join();
  if (mjpeg_encode_thread.joinable()) mjpeg_encode_thread.join();
  if (mjpeg_server_thread.joinable()) mjpeg_server_thread.join();

  return 0;
}
