// 文件说明：独立录制程序，录制相机画面+云台四元数+GimbalState，支持MJPEG预览。
#include <fmt/core.h>

#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <mutex>
#include <opencv2/opencv.hpp>
#include <sys/select.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "io/camera/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/recorder/recorder.hpp"

const std::string keys =
  "{help h usage ? |                       | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml     | yaml配置文件路径  }"
  "{mjpeg          | true                  | 开启MJPEG预览     }"
  "{mjpeg-port     | 9000                  | MJPEG端口         }"
  "{mjpeg-fps      | 10                    | MJPEG最大FPS      }"
  "{gui            | false                 | 开启GUI窗口       }"
  "{record-fps     | 30                    | 录制帧率          }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto mjpeg = cli.get<bool>("mjpeg");
  auto mjpeg_port = cli.get<int>("mjpeg-port");
  auto mjpeg_fps = std::max(1, cli.get<int>("mjpeg-fps"));
  auto gui = cli.get<bool>("gui");
  auto record_fps = std::max(1, cli.get<int>("record-fps"));

  tools::Exiter exiter;

  // 初始化
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  tools::Recorder recorder(record_fps);

  // MJPEG 相关变量
  std::mutex mjpeg_source_mutex, mjpeg_jpeg_mutex;
  cv::Mat mjpeg_source_frame;
  std::vector<uchar> mjpeg_jpeg;
  std::atomic<uint64_t> mjpeg_jpeg_version{0};
  std::atomic<bool> stop_mjpeg{false};
  auto mjpeg_frame_interval = std::chrono::microseconds(static_cast<int64_t>(1e6 / mjpeg_fps));

  // GUI 相关变量
  std::mutex gui_mutex;
  cv::Mat gui_frame;
  std::atomic<bool> stop_gui{false}, gui_quit{false};

  std::thread gui_thread;
  std::thread mjpeg_encode_thread, mjpeg_server_thread;

  if (gui) {
    gui_thread = std::thread([&] {
      cv::namedWindow("record", cv::WINDOW_NORMAL);
      while (!stop_gui) {
        cv::Mat frame_to_show;
        {
          std::scoped_lock lock(gui_mutex);
          if (!gui_frame.empty()) {
            frame_to_show = std::move(gui_frame);
            gui_frame.release();
          }
        }
        if (!frame_to_show.empty()) cv::imshow("record", frame_to_show);
        if (cv::waitKey(30) == 'q') {
          gui_quit = true;
          stop_gui = true;
          break;
        }
      }
      cv::destroyWindow("record");
    });
  }

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
        tools::logger()->error("[Record] MJPEG server failed on port {}", mjpeg_port);
        ::close(server_fd); stop_mjpeg = true; return;
      }
      tools::logger()->info("[Record] MJPEG preview: http://0.0.0.0:{}/", mjpeg_port);

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

  // 主录制循环
  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  int frame_count = 0;
  auto start_time = std::chrono::steady_clock::now();

  tools::logger()->info("[Record] Recording started (fps={})", record_fps);

  while (!exiter.exit() && !gui_quit) {
    camera.read(img, t);
    if (img.empty()) continue;

    auto gimbal_q = gimbal.q(t);
    auto gimbal_state = gimbal.state();

    // 录制
    recorder.record(img, gimbal_q, t, &gimbal_state);
    frame_count++;

    // 叠加信息用于预览
    if ((gui || mjpeg) && frame_count % 2 == 0) {
      cv::Mat display = img.clone();
      Eigen::Vector3d zyx = tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3;

      // FPS
      static auto last_time = std::chrono::steady_clock::now();
      auto now = std::chrono::steady_clock::now();
      auto dt = tools::delta_time(now, last_time);
      last_time = now;
      if (dt > 0)
        tools::draw_text(display, fmt::format("FPS: {:.1f}", 1.0 / dt), {10, 30}, {0, 255, 0});

      // 欧拉角
      tools::draw_text(display, fmt::format("Y:{:.1f} P:{:.1f} R:{:.1f}", zyx[0], zyx[1], zyx[2]), {10, 60}, {0, 0, 255});

      // 弹速 + 帧数
      tools::draw_text(display, fmt::format("bullet:{:.1f} frames:{}", gimbal_state.bullet_speed, frame_count), {10, 90}, {255, 255, 0});

      // REC 标记
      tools::draw_text(display, "REC", {10, 120}, {0, 0, 255});

      if (gui) {
        std::scoped_lock lock(gui_mutex);
        gui_frame = display.clone();
      }
      if (mjpeg) {
        std::scoped_lock lock(mjpeg_source_mutex);
        mjpeg_source_frame = std::move(display);
      }
    }
  }

  auto total_time = tools::delta_time(std::chrono::steady_clock::now(), start_time);
  tools::logger()->info("[Record] Done. {} frames in {:.1f}s ({:.1f} fps)", frame_count, total_time, frame_count / total_time);

  stop_gui = true;
  stop_mjpeg = true;
  if (gui_thread.joinable()) gui_thread.join();
  if (mjpeg_encode_thread.joinable()) mjpeg_encode_thread.join();
  if (mjpeg_server_thread.joinable()) mjpeg_server_thread.join();

  return 0;
}
