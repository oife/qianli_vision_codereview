#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <memory>
#include <opencv2/opencv.hpp>

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
  "{output-folder o |      assets/img_with_q   | 输出文件夹路径   }";

void write_q(const std::string q_path, const Eigen::Quaterniond & q)
{
  std::ofstream q_file(q_path);
  Eigen::Vector4d xyzw = q.coeffs();
  // 输出顺序为wxyz
  q_file << fmt::format("{} {} {} {}", xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  q_file.close();
}

void capture_loop(
  const std::string & config_path, const std::string & output_folder, const std::string & imu_source,
  const cv::Size & pattern_size)
{
  io::Camera camera(config_path);
  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;

  // 根据配置选择IMU数据源
  std::unique_ptr<io::CBoard> cboard;
  std::unique_ptr<io::DM_IMU> dm_imu;
  std::shared_ptr<io::Gimbal> gimbal;
  std::function<Eigen::Quaterniond(std::chrono::steady_clock::time_point)> imu_getter;

  if (imu_source == "can" || imu_source == "CBoard") {
    cboard = std::make_unique<io::CBoard>(config_path);
    imu_getter = [&cboard](std::chrono::steady_clock::time_point t) -> Eigen::Quaterniond {
      return cboard->imu_at(t);
    };
    tools::logger()->info("[Capture] 使用CAN总线(CBoard)读取IMU数据");
  } else if (imu_source == "dm_imu" || imu_source == "serial") {
    dm_imu = std::make_unique<io::DM_IMU>();
    imu_getter = [&dm_imu](std::chrono::steady_clock::time_point t) -> Eigen::Quaterniond {
      return dm_imu->imu_at(t);
    };
    tools::logger()->info("[Capture] 使用串口(DM_IMU)读取IMU数据");
  } else if (imu_source == "gimbal") {
    gimbal = std::make_shared<io::Gimbal>(config_path);
    imu_getter = [gimbal](std::chrono::steady_clock::time_point t) -> Eigen::Quaterniond {
      return gimbal->q(t);
    };
    tools::logger()->info("[Capture] 使用云台(gimbal)串口读取IMU/姿态数据");
  } else {
    tools::logger()->error(
      "[Capture] 未知的IMU数据源: {}，支持的值: can, dm_imu, gimbal", imu_source);
    return;
  }

  int count = 0;
  while (true) {
    camera.read(img, timestamp);
    Eigen::Quaterniond q = imu_getter(timestamp);

    // 在图像上显示欧拉角，用来判断imuabs系的xyz正方向，同时判断imu是否存在零漂
    auto img_with_ypr = img.clone();
    Eigen::Vector3d zyx = tools::eulers(q, 2, 1, 0) * 57.3;  // degree
    tools::draw_text(img_with_ypr, fmt::format("Z {:.2f}", zyx[0]), {40, 40}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("Y {:.2f}", zyx[1]), {40, 80}, {0, 0, 255});
    tools::draw_text(img_with_ypr, fmt::format("X {:.2f}", zyx[2]), {40, 120}, {0, 0, 255});

    // 使用缩小分辨率的灰度图进行棋盘格检测，加速“未识别到棋盘格”时的处理
    const double scale = 0.5;
    cv::Mat gray, gray_small;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, gray_small, {}, scale, scale);

    std::vector<cv::Point2f> centers_2d;
    std::vector<cv::Point2f> centers_2d_small;
    auto success = cv::findChessboardCorners(
      gray_small, pattern_size, centers_2d_small,
      cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE +
        cv::CALIB_CB_FAST_CHECK);  // 棋盘格角点
    if (success) {
      centers_2d.reserve(centers_2d_small.size());
      for (const auto & p : centers_2d_small) {
        centers_2d.emplace_back(p.x / scale, p.y / scale);
      }
      cv::cornerSubPix(
        gray, centers_2d, cv::Size(11, 11), cv::Size(-1, -1),
        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
    }
    cv::drawChessboardCorners(img_with_ypr, pattern_size, centers_2d, success);  // 显示识别结果
    cv::resize(img_with_ypr, img_with_ypr, {}, 0.5, 0.5);  // 显示时缩小图片尺寸

    // 按“s”保存图片和对应四元数，按“q”退出程序
    cv::imshow("Press s to save, q to quit", img_with_ypr);
    auto key = cv::waitKey(1);
    if (key == 'q')
      break;
    else if (key != 's')
      continue;

    // 保存图片和四元数
    count++;
    auto img_path = fmt::format("{}/{}.jpg", output_folder, count);
    auto q_path = fmt::format("{}/{}.txt", output_folder, count);
    cv::imwrite(img_path, img);
    write_q(q_path, q);
    tools::logger()->info("[{}] Saved in {}", count, output_folder);
  }

  // 离开该作用域时，camera、cboard和dm_imu会自动关闭
}

int main(int argc, char * argv[])
{
  // 读取命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);
  auto output_folder = cli.get<std::string>("output-folder");

  // 从配置文件读取IMU数据源选择和标定板参数
  std::string imu_source = "can";  // 默认使用CAN
  int pattern_cols = 10;
  int pattern_rows = 7;
  try {
    auto yaml = YAML::LoadFile(config_path);
    if (yaml["imu_source"]) {
      imu_source = yaml["imu_source"].as<std::string>();
    } else {
      tools::logger()->info("[Capture] 配置文件中未指定imu_source，默认使用CAN");
    }

    if (yaml["pattern_cols"] && yaml["pattern_rows"]) {
      pattern_cols = yaml["pattern_cols"].as<int>();
      pattern_rows = yaml["pattern_rows"].as<int>();
    } else {
      tools::logger()->info("[Capture] 配置文件中未指定pattern_cols/pattern_rows，默认使用10x7");
    }
  } catch (const std::exception & e) {
    tools::logger()->warn(
      "[Capture] 读取配置文件失败: {}，imu_source使用默认CAN，标定板尺寸使用默认10x7", e.what());
  }

  cv::Size pattern_size(pattern_cols, pattern_rows);

  // 新建输出文件夹
  std::filesystem::create_directory(output_folder);

  tools::logger()->info(
    "[Capture] 标定板尺寸为{}列{}行", static_cast<int>(pattern_size.width),
    static_cast<int>(pattern_size.height));
  // 主循环，保存图片和对应四元数
  capture_loop(config_path, output_folder, imu_source, pattern_size);

  tools::logger()->warn("注意四元数输出顺序为wxyz");

  return 0;
}
