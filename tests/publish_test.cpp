/**
 * @file publish_test.cpp
 * @brief 测试ROS2消息发布功能
 * 
 * 该测试程序用于验证ROS2消息发布功能，包括：
 * - 初始化ROS2节点
 * - 发布装甲板检测数据
 * - 测试消息发布频率
 */

#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "io/ros2/ros2.hpp"
#include "tasks/auto_aim/armor/armor.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"

int main(int argc, char ** argv)
{
  tools::Exiter exiter;
  io::ROS2 ros2;

  double i = 0;
  while (!exiter.exit()) {
    Eigen::Vector4d data{i, i + 1, 1, auto_aim::ArmorName::sentry + 1};
    ros2.publish(data);
    i++;

    std::this_thread::sleep_for(std::chrono::seconds(1));
    if (i > 1000) break;
  }
  return 0;
}
