/**
 * @file subscribe_test.cpp
 * @brief 测试ROS2消息订阅功能
 * 
 * 该测试程序用于验证ROS2消息订阅功能，包括：
 * - 初始化ROS2节点
 * - 订阅敌方状态消息
 * - 显示接收到的数据
 */

#include <rclcpp/rclcpp.hpp>
#include <thread>

#include "io/ros2/ros2.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/logger/logger.hpp"

int main(int argc, char ** argv)
{
  tools::Exiter exiter;
  io::ROS2 ros2;

  int i = 0;
  while (!exiter.exit()) {
    auto x = ros2.subscribe_enemy_status();
    // tools::logger()->info("invincible enemy ids size is{}", x.size());
    for (const auto & id : x) {
      tools::logger()->info("id:{}", id);
    }
    // i++;

    std::this_thread::sleep_for(std::chrono::microseconds(500));
    // if (i > 1000) break;
  }
  return 0;
}
