# ROS2 模块

ROS2通信模块，用于与导航系统进行ROS2消息通信。

## 功能

- 发布目标位置信息到导航系统
- 订阅敌方状态信息
- 订阅自瞄目标信息
- 提供通用的ROS2发布者创建接口
- 使用独立线程处理ROS2节点spin，确保消息正常收发

## 主要类

- `ROS2`: ROS2通信主类
  - `publish()`: 发布目标位置到导航系统
  - `subscribe_enemy_status()`: 订阅并获取敌方状态
  - `subscribe_autoaim_target()`: 订阅并获取自瞄目标
  - `create_publisher()`: 创建通用的ROS2发布者

- `Publish2Nav`: 发布到导航系统的节点类
- `Subscribe2Nav`: 从导航系统订阅的节点类

## 消息类型

- 目标位置：使用Eigen::Vector4d格式
- 敌方状态：sp_msgs::msg::EnemyStatusMsg
- 自瞄目标：sp_msgs::msg::AutoaimTargetMsg

