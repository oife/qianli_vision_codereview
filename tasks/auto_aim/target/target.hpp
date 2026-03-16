/**
 * @file target.hpp
 * @brief 自动瞄准系统目标跟踪类
 * @details 该文件定义了Target类，用于跟踪和预测装甲板目标的状态
 */

#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <chrono>
#include <limits>
#include <optional>
#include <queue>
#include <string>
#include <vector>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tools/extended_kalman_filter/extended_kalman_filter.hpp"

namespace auto_aim
{

/**
 * @brief 目标跟踪类
 * @details 使用扩展卡尔曼滤波器跟踪装甲板目标的状态，包括位置、速度、角度等
 */
class Target
{
public:
  ArmorName name;           /**< @brief 装甲板名称 */
  ArmorType armor_type;     /**< @brief 装甲板类型 */
  ArmorPriority priority;   /**< @brief 装甲板优先级 */
  bool jumped;              /**< @brief 是否发生跳跃 */
  int last_id;              /**< @brief 最后更新的装甲板ID，仅用于调试 */

  /**
   * @brief 默认构造函数
   */
  Target() = default;

  /**
   * @brief 基于装甲板信息的构造函数
   * @param armor 装甲板对象
   * @param t 时间戳
   * @param radius 装甲板半径
   * @param armor_num 装甲板数量
   * @param P0_dig 初始协方差对角线向量
   */
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig, double fixed_low_armor_distance = 0.0,
    double fixed_high_armor_distance = 0.0, double fixed_height_diff = 0.0);

  /**
   * @brief 简化的构造函数
   * @param x X坐标
   * @param vyaw Yaw角速度
   * @param radius 装甲板半径
   * @param h 高度
   */
  Target(double x, double vyaw, double radius, double h);

  void set_fixed_geometry(
    double fixed_low_armor_distance, double fixed_high_armor_distance, double fixed_height_diff);

  bool fixed_geometry_enabled() const;

  /**
   * @brief 根据时间戳进行状态预测
   * @param t 当前时间戳
   */
  void predict(std::chrono::steady_clock::time_point t);

  /**
   * @brief 根据时间间隔进行状态预测
   * @param dt 时间间隔（秒）
   */
  void predict(double dt);

  /**
   * @brief 使用新的装甲板观测数据更新状态
   * @param armor 新的装甲板观测数据
   */
  void update(const Armor & armor);

  /**
   * @brief 获取扩展卡尔曼滤波器的状态向量
   * @return 当前状态向量
   */
  Eigen::VectorXd ekf_x() const;

  /**
   * @brief 获取扩展卡尔曼滤波器对象的常量引用
   * @return 扩展卡尔曼滤波器对象的常量引用
   */
  const tools::ExtendedKalmanFilter & ekf() const;

  /**
   * @brief 获取所有装甲板的XYZA坐标列表
   * @return 装甲板坐标向量列表
   */
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  /**
   * @brief 检查滤波器是否发散
   * @return 如果发散返回true，否则返回false
   */
  bool diverged() const;

  /**
   * @brief 检查滤波器是否收敛
   * @return 如果收敛返回true，否则返回false
   */
  bool convergened();

  bool isinit = false;      /**< @brief 是否已初始化 */

  /**
   * @brief 检查初始化状态
   * @return 初始化状态
   */
  bool checkinit();

private:
  int armor_num_;           /**< @brief 装甲板数量 */
  int switch_count_;        /**< @brief 切换计数 */
  int update_count_;        /**< @brief 更新计数 */

  bool is_switch_;          /**< @brief 是否切换状态 */
  bool is_converged_;       /**< @brief 是否收敛 */

  tools::ExtendedKalmanFilter ekf_;     /**< @brief 扩展卡尔曼滤波器 */
  std::chrono::steady_clock::time_point t_;  /**< @brief 最后更新时间戳 */

  bool fixed_geometry_enabled_{false};
  double fixed_low_armor_distance_{0.0};
  double fixed_high_armor_distance_{0.0};
  double fixed_height_diff_{0.0};

  void apply_fixed_geometry();

  double observed_z_min_{std::numeric_limits<double>::infinity()};
  double observed_z_max_{-std::numeric_limits<double>::infinity()};

  void update_observed_z_extrema(double z);
  bool can_classify_high_low() const;
  bool classify_is_high(double z) const;

  /**
   * @brief 使用YPDA算法更新装甲板观测（yaw, pitch, distance, angle）
   * @param armor 装甲板观测数据
   * @param id 装甲板ID
   */
  void update_ypda(const Armor & armor, int id);

  /**
   * @brief 计算装甲板在世界坐标系下的XYZ坐标
   * @param x 状态向量
   * @param id 装甲板ID
   * @return 装甲板XYZ坐标
   */
  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;

  /**
   * @brief 计算观测函数的雅可比矩阵
   * @param x 状态向量
   * @param id 装甲板ID
   * @return 雅可比矩阵
   */
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP