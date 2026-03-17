// 文件说明：标定数据采集程序，支持MJPEG预览和TCP远程控制。
// 保留原有键盘操作（s保存，q退出），同时支持工作台通过TCP 9880端口远程控制。
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "io/camera/camera.hpp"
#include "io/cboard/cboard.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

const std::string keys =
  "{help h usage ?  |                          | 输出命令行参数说明}"
  "{@config-path c  | configs/calibration.yaml | yaml配置文件路径 }"
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }"
  "{mjpeg           | false                    | 开启MJPEG预览    }"
  "{mjpeg-port      | 9000                     | MJPEG端口        }"
  "{mjpeg-fps       | 10                       | MJPEG最大FPS     }"
  "{tcp-control     | false                    | 开启TCP控制端口  }"
  "{tcp-port        | 9880                     | TCP控制端口      }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");
  auto mjpeg = cli.get<bool>("mjpeg");
  auto mjpeg_port = cli.get<int>("mjpeg-port");
  auto mjpeg_fps = std::max(1, cli.get<int>("mjpeg-fps"));
  auto tcp_control = cli.get<bool>("tcp-control");
  auto tcp_port = cli.get<int>("tcp-port");

  // 读取配置
  std::string imu_source = "can";
  int pattern_cols = 10;
  int pattern_rows = 7;
  try {
    auto yaml = YAML::LoadFile(config_path);
    if (yaml["imu_source"]) imu_source = yaml["imu_source"].as<std::string>();
    if (yaml["pattern_cols"] && yaml["pattern_rows"]) {
      pattern_cols = yaml["pattern_cols"].as<int>();
      pattern_rows = yaml["pattern_rows"].as<int>();
    }
  } catch (const std::exception & e) {
    tools::logger()->warn("[Capture] 读取配置文件失败: {}", e.what());
  }
  cv::Size pattern_size(pattern_cols, pattern_rows);

  std::filesystem::create_directory(output_folder);
  tools::logger()->info("[Capture] 标定板尺寸为{}列{}行", pattern_cols, pattern_rows);

  // 初始化相机和IMU
  io::Camera camera(config_path);
  std::unique_ptr<io::CBoard> cboard;
  std::unique_ptr<io::DM_IMU> dm_imu;
  std::shared_ptr<io::Gimbal> gimbal;
  std::function<Eigen::Quaterniond(std::chrono::steady_clock::time_point)> imu_getter;

  if (imu_source == "can" || imu_source == "CBoard") {
    cboard = std::make_unique<io::CBoard>(config_path);
    imu_getter = [&cboard](auto t) { return cboard->imu_at(t); };
    tools::logger()->info("[Capture] 使用CAN总线(CBoard)读取IMU数据");
  } else if (imu_source == "dm_imu" || imu_source == "serial") {
    dm_imu = std::make_unique<io::DM_IMU>();
    imu_getter = [&dm_imu](auto t) { return dm_imu->imu_at(t); };
    tools::logger()->info("[Capture] 使用串口(DM_IMU)读取IMU数据");
  } else if (imu_source == "gimbal") {
    gimbal = std::make_shared<io::Gimbal>(config_path);
    imu_getter = [gimbal](auto t) { return gimbal->q(t); };
    tools::logger()->info("[Capture] 使用云台(gimbal)串口读取IMU/姿态数据");
  } else {
    tools::logger()->error("[Capture] 未知的IMU数据源: {}", imu_source);
    return 1;
  }

  // TCP 控制命令：由控制线程写入，主循环读取
  std::atomic<bool> tcp_save{false};
  std::atomic<bool> tcp_quit{false};
  std::atomic<int> save_count{0};
  std::atomic<bool> stop_tcp{false};

  std::thread tcp_thread;
  if (tcp_control) {
    tcp_thread = std::thread([&] {
      int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd < 0) return;
      int opt = 1;
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = INADDR_ANY;
      addr.sin_port = htons(static_cast<uint16_t>(tcp_port));
      if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
          listen(server_fd, 2) < 0) {
        tools::logger()->error("[Capture] TCP control server failed on port {}", tcp_port);
        ::close(server_fd);
        return;
      }
      tools::logger()->info("[Capture] TCP control listening on port {}", tcp_port);

      while (!stop_tcp) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(server_fd, &rfds);
        timeval tv{0, 200000};
        if (::select(server_fd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_fd = ::accept(server_fd, reinterpret_cast<sockaddr *>(&client_addr), &client_len);
        if (client_fd < 0) continue;

        // 处理单个客户端连接（持久连接，逐行读命令）
        std::thread([&, client_fd] {
          char buf[256];
          std::string line_buf;
          while (!stop_tcp) {
            ssize_t n = ::recv(client_fd, buf, sizeof(buf) - 1, 0);
            if (n <= 0) break;
            buf[n] = '\0';
            line_buf += buf;
            size_t pos;
            while ((pos = line_buf.find('\n')) != std::string::npos) {
              auto cmd = line_buf.substr(0, pos);
              line_buf.erase(0, pos + 1);
              // 去除 \r
              if (!cmd.empty() && cmd.back() == '\r') cmd.pop_back();

              if (cmd == "save") {
                tcp_save = true;
                // 等待主循环处理完
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                auto resp = fmt::format("saved {}\n", save_count.load());
                ::send(client_fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
              } else if (cmd == "quit") {
                tcp_quit = true;
                auto resp = std::string("quitting\n");
                ::send(client_fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
              } else if (cmd == "count") {
                auto resp = fmt::format("{}\n", save_count.load());
                ::send(client_fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
              } else {
                auto resp = std::string("unknown command\n");
                ::send(client_fd, resp.c_str(), resp.size(), MSG_NOSIGNAL);
              }
            }
          }
          ::close(client_fd);
        }).detach();
      }
      ::close(server_fd);
    });
  }

  // MJPEG 服务
  std::mutex mjpeg_source_mutex, mjpeg_jpeg_mutex;
  cv::Mat mjpeg_source_frame;
  std::vector<uchar> mjpeg_jpeg;
  std::atomic<uint64_t> mjpeg_jpeg_version{0};
  std::atomic<bool> stop_mjpeg{false};
  auto mjpeg_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / mjpeg_fps));

  std::thread mjpeg_encode_thread, mjpeg_server_thread;
  if (mjpeg) {
    mjpeg_encode_thread = std::thread([&] {
      auto last_encode_time = std::chrono::steady_clock::now() - mjpeg_frame_interval;
      std::vector<int> encode_params = {cv::IMWRITE_JPEG_QUALITY, 80};
      while (!stop_mjpeg) {
        auto now = std::chrono::steady_clock::now();
        if (now - last_encode_time < mjpeg_frame_interval) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1));
          continue;
        }
        cv::Mat frame;
        {
          std::scoped_lock lock(mjpeg_source_mutex);
          if (!mjpeg_source_frame.empty()) {
            frame = std::move(mjpeg_source_frame);
            mjpeg_source_frame.release();
          }
        }
        if (frame.empty()) continue;
        std::vector<uchar> encoded;
        if (cv::imencode(".jpg", frame, encoded, encode_params)) {
          std::scoped_lock lock(mjpeg_jpeg_mutex);
          mjpeg_jpeg = std::move(encoded);
          mjpeg_jpeg_version.fetch_add(1, std::memory_order_relaxed);
          last_encode_time = std::chrono::steady_clock::now();
        }
      }
    });

    mjpeg_server_thread = std::thread([&] {
      int server_fd = ::socket(AF_INET, SOCK_STREAM, 0);
      if (server_fd < 0) { stop_mjpeg = true; return; }
      int opt = 1;
      setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
      sockaddr_in addr{};
      addr.sin_family = AF_INET;
      addr.sin_addr.s_addr = INADDR_ANY;
      addr.sin_port = htons(static_cast<uint16_t>(mjpeg_port));
      if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0 ||
          listen(server_fd, 4) < 0) {
        tools::logger()->error("[Capture] MJPEG server failed on port {}", mjpeg_port);
        ::close(server_fd); stop_mjpeg = true; return;
      }
      tools::logger()->info("[Capture] MJPEG preview: http://0.0.0.0:{}/", mjpeg_port);

      auto send_all = [](int fd, const void * data, size_t size) {
        const char * p = static_cast<const char *>(data);
        while (size > 0) {
          ssize_t n = ::send(fd, p, size, MSG_NOSIGNAL);
          if (n <= 0) return false;
          p += n; size -= static_cast<size_t>(n);
        }
        return true;
      };

      while (!stop_mjpeg) {
        fd_set rfds; FD_ZERO(&rfds); FD_SET(server_fd, &rfds);
        timeval tv{0, 200000};
        if (::select(server_fd + 1, &rfds, nullptr, nullptr, &tv) <= 0) continue;
        sockaddr_in ca{}; socklen_t cl = sizeof(ca);
        int cfd = ::accept(server_fd, reinterpret_cast<sockaddr *>(&ca), &cl);
        if (cfd < 0) continue;

        std::thread([&, cfd] {
          char req_buf[1024];
          ssize_t req_len = ::recv(cfd, req_buf, sizeof(req_buf) - 1, 0);
          if (req_len <= 0) { ::close(cfd); return; }
          req_buf[req_len] = '\0';
          std::string req(req_buf);
          if (req.find("GET / ") != std::string::npos || req.find("GET /HTTP") != std::string::npos) {
            std::string html =
              "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
              "<html><body style='margin:0;background:#111'>"
              "<img src='/stream' style='max-width:100%;height:auto;display:block;margin:auto' />"
              "</body></html>";
            send_all(cfd, html.data(), html.size()); ::close(cfd); return;
          }
          if (req.find("GET /stream") == std::string::npos) {
            const char * nf = "HTTP/1.1 404 Not Found\r\nConnection: close\r\nContent-Length: 0\r\n\r\n";
            send_all(cfd, nf, std::strlen(nf)); ::close(cfd); return;
          }
          const char * hdr =
            "HTTP/1.1 200 OK\r\nCache-Control: no-cache\r\nConnection: close\r\n"
            "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
          if (!send_all(cfd, hdr, std::strlen(hdr))) { ::close(cfd); return; }
          uint64_t last_ver = 0;
          while (!stop_mjpeg) {
            uint64_t ver = mjpeg_jpeg_version.load(std::memory_order_relaxed);
            if (ver == 0 || ver == last_ver) { std::this_thread::sleep_for(std::chrono::milliseconds(5)); continue; }
            std::vector<uchar> jpeg_copy;
            { std::scoped_lock lock(mjpeg_jpeg_mutex); jpeg_copy = mjpeg_jpeg; ver = mjpeg_jpeg_version.load(); }
            if (jpeg_copy.empty()) continue;
            auto ph = fmt::format("--frame\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n", jpeg_copy.size());
            if (!send_all(cfd, ph.data(), ph.size()) || !send_all(cfd, jpeg_copy.data(), jpeg_copy.size()) || !send_all(cfd, "\r\n", 2)) break;
            last_ver = ver;
          }
          ::close(cfd);
        }).detach();
      }
      ::close(server_fd);
    });
  }

  // 主采集循环
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int count = 0;

  while (!tcp_quit) {
    camera.read(img, timestamp);
    if (img.empty()) continue;
    Eigen::Quaterniond q = imu_getter(timestamp);

    // 绘制欧拉角和棋盘格检测
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("saved: {}", count), {40, 160}, {0, 255, 0});

    const double scale = 0.5;
    cv::Mat gray, gray_small;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray_small, {}, scale, scale);

    std::vector<cv::Point2f> centers_2d, centers_2d_small;
    auto success = cv::findChessboardCorners(
      gray_small, pattern_size, centers_2d_small,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE + cv::CALIB_CB_FAST_CHECK);
    if (success) {
      centers_2d.reserve(centers_2d_small.size());
      for (const auto & p : centers_2d_small)
        centers_2d.emplace_back(p.x / scale, p.y / scale);
      cv::cornerSubPix(
        gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    cv::drawChessboardCorners(img_with_ypr, pattern_size, centers_2d, success);

    // 推送到 MJPEG（全分辨率带标注）
    if (mjpeg) {
      std::scoped_lock lock(mjpeg_source_mutex);
      mjpeg_source_frame = img_with_ypr.clone();
    }

    // 本地 GUI 显示（缩小）
    cv::Mat display;
    cv::resize(img_with_ypr, display, {}, 0.5, 0.5);
    cv::imshow("Press s to save, q to quit", display);
    auto key = cv::waitKey(1);

    // 处理保存：键盘 's' 或 TCP 'save' 命令
    bool do_save = (key == 's') || tcp_save.exchange(false);
    if (key == 'q') break;

    if (do_save) {
      count++;
      auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
      auto q_path = fmt::format("{}/{}.txt", output_folder, count);
      cv::imwrite(img_path, img);
      write_q(q_path, q);
      save_count = count;
      tools::logger()->info("[{}] Saved in {}", count, output_folder);
    }
  }

  // 清理
  stop_mjpeg = true;
  stop_tcp = true;
  if (mjpeg_encode_thread.joinable()) mjpeg_encode_thread.join();
  if (mjpeg_server_thread.joinable()) mjpeg_server_thread.join();
  if (tcp_thread.joinable()) tcp_thread.join();

  tools::logger()->warn("注意四元数输出顺序为wxyz");
  return 0;
}
