/**
 * @brief 诊断工具：遍历所有 24 种合法的 R_gimbal2imubody，实时显示云台欧拉角
 *
 * 使用方法：
 *   ./build/find_R_gimbal2imubody configs/xxx.yaml
 *
 * 操作步骤：
 *   1. 让云台水平朝前（初始位置），观察哪些候选值的 yaw≈0, pitch≈0, roll≈0
 *   2. 缓慢左转云台（yaw+），观察哪些候选值的 yaw 正方向增大
 *   3. 缓慢抬头（pitch+），观察哪些候选值的 pitch 正方向增大
 *   4. 符合以上三个条件的就是正确的 R_gimbal2imubody
 *
 * 按 'n'/'p' 翻页显示不同的候选值组，按 'q' 退出
 */
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <vector>

#include "io/gimbal/gimbal.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/math_tools/math_tools.hpp"

// 生成所有 24 种合法的旋转矩阵（行列式为 +1 的正交矩阵，元素只有 0, ±1）
// 这些矩阵对应坐标轴的所有可能排列和方向组合
std::vector<std::pair<Eigen::Matrix3d, std::string>> generate_all_24_rotations()
{
  std::vector<std::pair<Eigen::Matrix3d, std::string>> result;

  // 三个轴的排列 (0=x, 1=y, 2=z)
  int perms[6][3] = {{0, 1, 2}, {0, 2, 1}, {1, 0, 2}, {1, 2, 0}, {2, 0, 1}, {2, 1, 0}};
  // 每个轴的符号组合 (±1)
  int signs[8][3] = {{1, 1, 1},   {1, 1, -1},  {1, -1, 1},  {1, -1, -1},
                     {-1, 1, 1},  {-1, 1, -1}, {-1, -1, 1}, {-1, -1, -1}};

  for (auto & perm : perms) {
    for (auto & sign : signs) {
      Eigen::Matrix3d R = Eigen::Matrix3d::Zero();
      for (int row = 0; row < 3; row++) {
        R(row, perm[row]) = sign[row];
      }
      // 只保留行列式为 +1 的（proper rotation）
      if (std::abs(R.determinant() - 1.0) < 1e-6) {
        // 生成人类可读的描述
        std::string desc = "[";
        for (int i = 0; i < 3; i++) {
          for (int j = 0; j < 3; j++) {
            desc += fmt::format("{}", (int)R(i, j));
            if (i * 3 + j < 8) desc += ", ";
          }
        }
        desc += "]";

        // 标记是否是原来的 8 个对角矩阵之一
        bool is_diagonal = (R(0, 1) == 0 && R(0, 2) == 0 && R(1, 0) == 0 && R(1, 2) == 0 &&
                            R(2, 0) == 0 && R(2, 1) == 0);
        if (is_diagonal) desc += " (对角)";

        result.push_back({R, desc});
      }
    }
  }
  return result;
}

int main(int argc, char * argv[])
{
  const std::string keys =
    "{help h usage ? |                    | 输出命令行参数说明}"
    "{@config-path   | configs/hero.yaml  | yaml配置文件路径 }";

  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  auto config_path = cli.get<std::string>(0);

  tools::Exiter exiter;
  io::Gimbal gimbal(config_path);

  auto all_rotations = generate_all_24_rotations();
  fmt::print("共生成 {} 种合法的 R_gimbal2imubody 候选值\n", all_rotations.size());

  // 读取 q_inverted 配置（默认 false）
  auto yaml = YAML::LoadFile(config_path);
  bool q_inverted = yaml["q_inverted"] && yaml["q_inverted"].as<bool>();
  fmt::print("q_inverted: {}\n", q_inverted ? "true" : "false");

  int page = 0;
  const int per_page = 6;  // 每页显示 6 个候选值
  int total_pages = (all_rotations.size() + per_page - 1) / per_page;

  while (!exiter.exit()) {
    // 获取当前 IMU 四元数
    auto t = std::chrono::steady_clock::now();
    Eigen::Quaterniond q = gimbal.q(t);

    // 创建显示画布
    cv::Mat canvas(720, 1280, CV_8UC3, cv::Scalar(30, 30, 30));

    cv::putText(
      canvas,
      fmt::format(
        "Page {}/{} | 'n'=next, 'p'=prev, 'q'=quit", page + 1, total_pages),
      {20, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {200, 200, 200}, 1);

    cv::putText(
      canvas,
      "Move gimbal: check yaw/pitch/roll direction matches physical motion",
      {20, 60}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {150, 150, 255}, 1);

    int start = page * per_page;
    int end = std::min(start + per_page, (int)all_rotations.size());

    for (int idx = start; idx < end; idx++) {
      auto & [R_gimbal2imubody, desc] = all_rotations[idx];
      int row = idx - start;
      int y_base = 100 + row * 100;

      // 计算 R_gimbal2world
      Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
      if (q_inverted) R_imubody2imuabs.transposeInPlace();
      Eigen::Matrix3d R_gimbal2world =
        R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;

      // 提取欧拉角
      Eigen::Vector3d ypr = tools::eulers(R_gimbal2world, 2, 1, 0) * 57.3;

      // 判断是否"合理"（pitch 和 roll 在 ±30° 以内，说明可能正确）
      bool reasonable = std::abs(ypr[1]) < 30 && std::abs(ypr[2]) < 30;
      cv::Scalar color = reasonable ? cv::Scalar(0, 255, 0) : cv::Scalar(100, 100, 100);

      // 显示编号和矩阵描述
      cv::putText(
        canvas, fmt::format("#{:2d} {}", idx + 1, desc), {20, y_base},
        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

      // 显示欧拉角
      cv::putText(
        canvas,
        fmt::format(
          "yaw: {:+7.2f}°  pitch: {:+7.2f}°  roll: {:+7.2f}°", ypr[0], ypr[1], ypr[2]),
        {20, y_base + 30}, cv::FONT_HERSHEY_SIMPLEX, 0.6, color, 1);

      // 如果合理，额外标记
      if (reasonable) {
        cv::putText(
          canvas, "<-- CANDIDATE", {900, y_base + 15}, cv::FONT_HERSHEY_SIMPLEX, 0.7,
          {0, 255, 255}, 2);
      }
    }

    // 显示原始四元数
    cv::putText(
      canvas,
      fmt::format(
        "Raw q: w={:.4f} x={:.4f} y={:.4f} z={:.4f}", q.w(), q.x(), q.y(), q.z()),
      {20, 690}, cv::FONT_HERSHEY_SIMPLEX, 0.6, {200, 200, 200}, 1);

    cv::imshow("Find R_gimbal2imubody", canvas);
    auto key = cv::waitKey(30);
    if (key == 'q') break;
    if (key == 'n' && page < total_pages - 1) page++;
    if (key == 'p' && page > 0) page--;
  }

  return 0;
}
