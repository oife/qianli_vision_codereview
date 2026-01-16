// 文件说明：Hero机器人主程序，包含hik摄像头读取、dm_imu读取和gimbal数据读取功能
#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/hero.yaml | yaml配置文件路径 }";

using namespace std::chrono_literals;

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;
  
  // 初始化hik摄像头
  io::Camera camera(config_path);
  
  // 初始化dm_imu
  // io::DM_IMU imu;
  
  // 初始化gimbal
  // io::Gimbal gimbal(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond q;
  Eigen::Quaterniond gimbal_q;
  io::GimbalState gimbal_state;
  io::GimbalMode gimbal_mode;

  while (!exiter.exit()) {
    // 读取hik摄像头图像和时间戳
    camera.read(img, t);
    
    // 读取dm_imu数据（根据图像时间戳获取对应的IMU数据）
    // q = imu.imu_at(t);
    
    // 读取gimbal数据
    // gimbal_q = gimbal.q(t);  // 获取gimbal四元数（根据图像时间戳插值）
    // gimbal_state = gimbal.state();  // 获取gimbal状态（yaw, pitch, bullet_speed等）
    // gimbal_mode = gimbal.mode();  // 获取gimbal模式
  }

  return 0;
}
