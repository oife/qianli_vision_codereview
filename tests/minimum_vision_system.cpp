/**
 * @file minimum_vision_system.cpp
 * @brief 最小视觉系统测试 - 完整的自瞄系统集成测试
 * 
 * 该测试程序实现了完整的自动瞄准系统，包括：
 * - 多线程图像采集和检测
 * - IMU数据融合
 * - 目标跟踪和预测
 * - 弹道解算和瞄准控制
 * - 射击决策
 * - 可视化调试信息（重投影、状态显示等）
 */

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>

#include "io/camera/camera.hpp"
#include "io/dm_imu/dm_imu.hpp"
#include "tasks/auto_aim/aimer/aimer.hpp"
#include "tasks/auto_aim/multithread/mt_detector.hpp"
#include "tasks/auto_aim/shooter/shooter.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/tracker/tracker.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"

// 命令行参数定义
const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | | yaml配置文件路径 }";

int main(int argc, char * argv[])
{
  // 解析命令行参数
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help") || !cli.has("@config-path")) {
    cli.printMessage();
    return 0;
  }

  // 获取配置文件路径
  auto config_path = cli.get<std::string>("@config-path");

  // 初始化各个模块
  tools::Exiter exiter;              // 退出控制器
  tools::Plotter plotter;            // 数据可视化工具
  io::Camera camera(config_path);    // 相机模块，用于图像采集
  io::DM_IMU dm_imu;                 // IMU模块，用于获取姿态数据

  // 初始化自瞄系统各组件
  auto_aim::multithread::MultiThreadDetector detector(config_path);  // 多线程目标检测器
  auto_aim::Solver solver(config_path);                              // 弹道解算器
  auto_aim::Tracker tracker(config_path, solver);                    // 目标跟踪器
  auto_aim::Aimer aimer(config_path);                                // 瞄准控制器
  auto_aim::Shooter shooter(config_path);                            // 射击控制器

  // 启动检测线程：持续从相机读取图像并送入检测器
  auto detect_thread = std::thread([&]() {
    cv::Mat img;                                      // 存储采集的图像
    std::chrono::steady_clock::time_point t;         // 图像时间戳

    while (!exiter.exit()) {
      camera.read(img, t);                           // 从相机读取图像和时间戳
      detector.push(img, t);                         // 将图像推入检测队列
    }
  });

  // 初始化主循环所需变量
  auto last_t = std::chrono::steady_clock::now();    // 上一帧时间，用于计算帧率
  nlohmann::json data;                                // 存储调试数据，用于可视化

  // 主循环：处理检测结果并进行瞄准决策
  while (!exiter.exit()) {
    //1.
    // 从检测器获取检测结果（图像、装甲板列表、时间戳）
    auto [img, armors, t] = detector.debug_pop();

    // 获取IMU数据：根据时间戳获取对应的姿态四元数
    Eigen::Quaterniond q = dm_imu.imu_at(t);

    //2.
    // 设置云台到世界坐标系的旋转矩阵
    solver.set_R_gimbal2world(q);

    // 计算云台姿态角度（欧拉角：z-y-x顺序）
    Eigen::Vector3d gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    //3.
    // 目标跟踪：将检测到的装甲板进行关联和跟踪，得到目标列表
    auto targets = tracker.track(armors, t);

    //4.
    // 瞄准决策：根据目标列表计算瞄准指令（22可能是子弹速度或射速）
    auto command = aimer.aim(targets, t, 22);

    //5.
    // 射击控制：根据瞄准指令决定是否射击
    shooter.shoot(command, aimer, targets, gimbal_pos);

    // 计算帧间隔和帧率
    auto dt = tools::delta_time(t, last_t);
    last_t = t;

//==================可视化部分开始===================
    
    // 记录时间相关数据用于可视化
    data["dt"] = dt;                    // 帧间隔（秒）
    data["fps"] = 1 / dt;               // 帧率
    plotter.plot(data);                 // 绘制数据曲线

    // 记录装甲板原始观测数据
    data["armor_num"] = armors.size();  // 检测到的装甲板数量
    if (!armors.empty()) {
      // 选择最左侧的装甲板（x坐标最小）进行解算
      auto min_x = 1e10;
      auto & armor = armors.front();
      for (auto & a : armors) {
        if (a.center.x < min_x) {
          min_x = a.center.x;
          armor = a;
        }
      }  // 始终选择左侧装甲板
      
      // 对选中的装甲板进行坐标解算（像素坐标->世界坐标）
      solver.solve(armor);
      
      // 记录装甲板在世界坐标系中的位置和姿态
      data["armor_x"] = armor.xyz_in_world[0];              // 世界坐标系x坐标（米）
      data["armor_y"] = armor.xyz_in_world[1];              // 世界坐标系y坐标（米）
      data["armor_yaw"] = armor.ypr_in_world[0] * 57.3;     // 世界坐标系偏航角（度）
      data["armor_yaw_raw"] = armor.yaw_raw * 57.3;         // 原始偏航角（度）
    }

    // 如果有跟踪到的目标，进行可视化绘制和数据记录
    if (!targets.empty()) {
      auto target = targets.front();
      
      // 在图像上绘制跟踪器状态信息（白色文字）
      tools::draw_text(img, fmt::format("[{}]", tracker.state()), {10, 30}, {255, 255, 255});

      // 重投影目标的所有装甲板到图像上（绿色点）
      // 用于可视化目标在图像中的预期位置
      std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
      for (const Eigen::Vector4d & xyza : armor_xyza_list) {
        // xyza: [x, y, z, yaw] - 装甲板在世界坐标系中的位置和姿态
        auto image_points =
          solver.reproject_armor(xyza.head(3), xyza[3], target.armor_type, target.name);
        tools::draw_points(img, image_points, {0, 255, 0});  // 绿色：目标装甲板位置
      }

      // 重投影瞄准点（红色=有效，蓝色=无效）
      // 用于可视化当前瞄准位置
      auto aim_point = aimer.debug_aim_point;
      Eigen::Vector4d aim_xyza = aim_point.xyza;
      auto image_points =
        solver.reproject_armor(aim_xyza.head(3), aim_xyza[3], target.armor_type, target.name);
      if (aim_point.valid)
        tools::draw_points(img, image_points, {0, 0, 255});  // 红色：有效瞄准点
      else
        tools::draw_points(img, image_points, {255, 0, 0});  // 蓝色：无效瞄准点

      // 记录扩展卡尔曼滤波器（EKF）的状态估计数据
      Eigen::VectorXd x = target.ekf_x();
      data["x"] = x[0];                  // 目标x坐标（米）
      data["vx"] = x[1];                 // 目标x方向速度（米/秒）
      data["y"] = x[2];                  // 目标y坐标（米）
      data["vy"] = x[3];                 // 目标y方向速度（米/秒）
      data["z"] = x[4];                  // 目标z坐标（米）
      data["vz"] = x[5];                 // 目标z方向速度（米/秒）
      data["a"] = x[6] * 57.3;           // 目标偏航角（度）
      data["w"] = x[7];                  // 目标角速度（弧度/秒）
      data["r"] = x[8];                  // 目标半径（米）
      data["l"] = x[9];                  // 目标长度（米）
      data["h"] = x[10];                 // 目标高度（米）
      data["last_id"] = target.last_id;  // 最近匹配的装甲板ID
      data["distance"] = std::sqrt(x[0] * x[0] + x[2] * x[2] + x[4] * x[4]);  // 目标距离（米）

      // 记录卡方检验数据（用于评估滤波器性能）
      data["residual_yaw"] = target.ekf().data.at("residual_yaw");           // 偏航角残差
      data["residual_pitch"] = target.ekf().data.at("residual_pitch");       // 俯仰角残差
      data["residual_distance"] = target.ekf().data.at("residual_distance"); // 距离残差
      data["residual_angle"] = target.ekf().data.at("residual_angle");       // 角度残差
      data["nis"] = target.ekf().data.at("nis");                             // 归一化创新平方（NIS）
      data["nees"] = target.ekf().data.at("nees");                           // 归一化估计误差平方（NEES）
      data["nis_fail"] = target.ekf().data.at("nis_fail");                   // NIS检验失败标志
      data["nees_fail"] = target.ekf().data.at("nees_fail");                 // NEES检验失败标志
      data["recent_nis_failures"] = target.ekf().data.at("recent_nis_failures");  // 最近NIS失败次数
    }
    
    // 显示图像：缩小到50%尺寸以提高显示性能
    cv::resize(img, img, {}, 0.5, 0.5);
    cv::imshow("reprojection", img);
    
    // 检查键盘输入，按'q'键退出
    auto key = cv::waitKey(1);
    if (key == 'q') break;
  }

  // 等待检测线程结束
  detect_thread.join();

  return 0;
}