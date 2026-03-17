// 文件说明：Standard步兵自瞄入口（Gimbal串口版），适配战队电控系统。
// 使用io::Gimbal串口通信替代io::CBoard CAN通信，保留步兵Shooter开火逻辑。
#include <fmt/core.h>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <string>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "io/camera/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer/aimer.hpp"
#include "tasks/auto_aim/shooter/shooter.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"
#include "tools/recorder/recorder.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/standard3.yaml | yaml配置文件路径 }"
  "{gui            | false             | 开启GUI窗口      }"
  "{display-every  | 2                 | 每N帧更新显示    }"
  "{display-fps    | 12                | GUI最大刷新FPS   }"
  "{mjpeg          | false             | 开启MJPEG预览    }"
  "{mjpeg-port     | 9000              | MJPEG端口        }"
  "{mjpeg-fps      | 10                | MJPEG最大FPS     }"
  "{plotter        | false             | 开启Plotter数据  }"
  "{plotter-port   | 9870              | Plotter端口      }"
  "{plotter-tcp    | false             | Plotter用TCP模式 }"
  "{record         | false             | 开启录制         }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  auto gui = cli.get<bool>("gui");
  auto display_every = std::max(1, cli.get<int>("display-every"));
  auto display_fps = std::max(1, cli.get<int>("display-fps"));
  auto mjpeg = cli.get<bool>("mjpeg");
  auto mjpeg_port = cli.get<int>("mjpeg-port");
  auto mjpeg_fps = std::max(1, cli.get<int>("mjpeg-fps"));
  auto enable_plotter = cli.get<bool>("plotter");
  auto plotter_port = cli.get<int>("plotter-port");
  auto plotter_tcp = cli.get<bool>("plotter-tcp");
  auto enable_record = cli.get<bool>("record");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  std::mutex gui_mutex;
  cv::Mat gui_frame;
  std::atomic<bool> stop_gui{false}, gui_quit{false};
  auto gui_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / display_fps));
  std::mutex mjpeg_source_mutex, mjpeg_jpeg_mutex;
  cv::Mat mjpeg_source_frame;
  std::vector<uchar> mjpeg_jpeg;
  std::atomic<uint64_t> mjpeg_jpeg_version{0};
  std::atomic<bool> stop_mjpeg{false};
  auto mjpeg_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / mjpeg_fps));

  // 初始化hik摄像头
  io::Camera camera(config_path);

  // 初始化gimbal（串口通信，替代CBoard CAN）
  io::Gimbal gimbal(config_path);

  // 初始化自瞄模块
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Shooter shooter(config_path);  // 步兵特有：Shooter开火判断模块

  // 初始化Plotter
  std::unique_ptr<tools::Plotter> plotter;
  auto t0 = std::chrono::steady_clock::now();
  if (enable_plotter) {
    plotter = std::make_unique<tools::Plotter>(
      "127.0.0.1", static_cast<uint16_t>(plotter_port), plotter_tcp ? "tcp" : "udp");
  }

  // 初始化Recorder
  std::unique_ptr<tools::Recorder> recorder;
  if (enable_record) {
    recorder = std::make_unique<tools::Recorder>();
    tools::logger()->info("Recording enabled");
  }

  std::thread gui_thread;
  std::thread mjpeg_encode_thread;
  std::thread mjpeg_server_thread;
  if (gui) {
    gui_thread = std::thread([&] {
      cv::namedWindow("standard_auto_aim", cv::WINDOW_NORMAL);
      auto last_show_time = std::chrono::steady_clock::now() - gui_frame_interval;
      while (!stop_gui) {
        auto now = std::chrono::steady_clock::now();
        auto since_last = now - last_show_time;
        if (since_last < gui_frame_interval) {
          auto remain =
            std::chrono::duration_cast<std::chrono::milliseconds>(gui_frame_interval - since_last);
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
          cv::imshow("standard_auto_aim", frame_to_show);
          last_show_time = std::chrono::steady_clock::now();
        }
        if (cv::waitKey(1) == 'q') {
          gui_quit = true;
          stop_gui = true;
          break;
        }
      }
      cv::destroyWindow("standard_auto_aim");
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
          auto remain = std::chrono::duration_cast<std::chrono::milliseconds>(
            mjpeg_frame_interval - since_last);
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
      if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
          listen(server_fd, 4) < 0) {
        tools::logger()->error("MJPEG server start failed on port {}", mjpeg_port);
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
        if (::select(server_fd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
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
                !send_all(client_fd, "\r\n", 2))
              break;
            last_version = version;
          }
          ::close(client_fd);
        }).detach();
      }
      ::close(server_fd);
    });
  }

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond gimbal_q;
  io::GimbalState gimbal_state;
  io::GimbalMode gimbal_mode;
  io::GimbalMode last_gimbal_mode = io::GimbalMode::IDLE;
  int frame_count = 0;

  while (!exiter.exit() && !gui_quit) {
    // 读取hik摄像头图像和时间戳
    camera.read(img, t);
    if (img.empty()) {
      tools::logger()->warn("相机读取的图像为空，跳过此帧");
      continue;
    }

    // 读取gimbal数据（串口）
    gimbal_q = gimbal.q(t);
    gimbal_state = gimbal.state();
    gimbal_mode = gimbal.mode();

    // 模式切换日志（步兵特征：支持模式切换）
    if (last_gimbal_mode != gimbal_mode) {
      tools::logger()->info("Switch to {}", gimbal.str(gimbal_mode));
      last_gimbal_mode = gimbal_mode;
    }

    /// 自瞄核心逻辑

    // 设置云台到世界的旋转
    solver.set_R_gimbal2world(gimbal_q);

    Eigen::Vector3d ypr = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    // YOLO目标检测
    auto yolo_start = std::chrono::steady_clock::now();
    auto armors = yolo.detect(img, frame_count);

    // 目标跟踪
    auto tracker_start = std::chrono::steady_clock::now();
    auto targets = tracker.track(armors, t);

    // 瞄准控制（使用gimbal状态中的子弹速度）
    auto aimer_start = std::chrono::steady_clock::now();
    auto command = aimer.aim(targets, t, gimbal_state.bullet_speed);

    // 步兵特有：使用Shooter模块进行开火判断（基于距离分远近容差）
    Eigen::Vector3d gimbal_pos(ypr[0], ypr[1], ypr[2]);
    command.shoot = shooter.shoot(command, aimer, targets, gimbal_pos);

    // 发送控制命令到云台（通过串口）
    gimbal.send(
      command.control, command.shoot, command.yaw, 0.0f, 0.0f, command.pitch, 0.0f, 0.0f);

    // 录制
    if (recorder) recorder->record(img, gimbal_q, t, &gimbal_state);

    // Plotter 数据发送
    if (plotter) {
      nlohmann::json pdata;
      pdata["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      pdata["cmd_yaw"] = command.yaw * 57.3;
      pdata["cmd_pitch"] = command.pitch * 57.3;
      pdata["gimbal_yaw"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0];
      pdata["gimbal_pitch"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[1];
      pdata["control"] = command.control;
      pdata["shoot"] = command.shoot;
      pdata["bullet_speed"] = gimbal_state.bullet_speed;
      plotter->plot(pdata);
    }

    // 调试输出
    tools::logger()->info(
      "[Command] control: {}, shoot: {}, yaw: {:.4f} rad ({:.2f} deg), pitch: {:.4f} rad ({:.2f} deg)",
      command.control, command.shoot, command.yaw, command.yaw * 57.3, command.pitch,
      command.pitch * 57.3);

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

    // 绘制目标重投影
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
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid) tools::draw_points(img, image_points, {0, 0, 255});
    }

    if ((gui || mjpeg) && frame_count % display_every == 0) {
      cv::Mat display = img.clone();
      if (gui) {
        std::scoped_lock lock(gui_mutex);
        gui_frame = display.clone();
      }
      if (mjpeg) {
        std::scoped_lock lock(mjpeg_source_mutex);
        mjpeg_source_frame = std::move(display);
      }
    }

    frame_count++;
  }

  stop_gui = true;
  stop_mjpeg = true;
  if (gui_thread.joinable()) gui_thread.join();
  if (mjpeg_encode_thread.joinable()) mjpeg_encode_thread.join();
  if (mjpeg_server_thread.joinable()) mjpeg_server_thread.join();

  return 0;
}
