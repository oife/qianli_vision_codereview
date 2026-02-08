#ifndef AUTO_AIM__PLANNER_HPP
#define AUTO_AIM__PLANNER_HPP

#include <Eigen/Dense>
#include <list>
#include <optional>
#include <vector>
#include <tuple>

#include "tasks/auto_aim/target/target.hpp"
#include "tasks/auto_aim/planner/quintic_polynomial/quintic_polynomial.hpp"
#include "tinympc/tiny_api.hpp"

namespace auto_aim
{
// MPC预测时域的时间步长（秒）
constexpr double DT = 0.01;
// 预测时域的一半（用于选择输出时间点）
constexpr int HALF_HORIZON = 50;
// MPC预测时域的总步数（HORIZON * DT = 1.0秒）
constexpr int HORIZON = HALF_HORIZON * 2;

// 轨迹矩阵：4行（yaw, yaw_vel, pitch, pitch_vel）x HORIZON列（时间步）
using Trajectory = Eigen::Matrix<double, 4, HORIZON>;

/**
 * @brief 求解器类型枚举
 */
enum class SolverType
{
  MPC,           // 模型预测控制求解器
  QUINTIC_POLY    // 五次多项式求解器
};

/**
 * @brief 规划结果结构体
 * 包含云台控制所需的所有信息：控制标志、开火决策、目标角度和MPC规划的状态
 */
struct Plan
{
  bool control;        // 是否启用控制（false表示无目标或求解失败）
  bool fire;           // 是否允许开火（轨迹误差小于阈值时）
  float target_yaw;    // 目标yaw角度（弧度）
  float target_pitch;  // 目标pitch角度（弧度）
  float yaw;           // MPC规划的yaw角度（弧度）
  float yaw_vel;       // MPC规划的yaw角速度（rad/s）
  float yaw_acc;       // MPC规划的yaw角加速度（rad/s²）
  float pitch;         // MPC规划的pitch角度（弧度）
  float pitch_vel;     // MPC规划的pitch角速度（rad/s）
  float pitch_acc;     // MPC规划的pitch角加速度（rad/s²）
};

/**
 * @brief 自动瞄准规划器
 * 
 * 支持两种求解器：
 * 1. MPC（模型预测控制）：优化控制输入，考虑约束条件
 * 2. 五次多项式：检测突变段并生成平滑过渡轨迹
 * 
 * 主要功能：
 * 1. 预测目标未来位置
 * 2. 计算弹道补偿后的瞄准点
 * 3. 生成参考轨迹
 * 4. 使用配置的求解器优化控制输入
 * 5. 判断是否满足开火条件
 */
class Planner
{
public:
  Eigen::Vector4d debug_xyza;  // 调试用：当前选择的装甲板位置和角度 [x, y, z, yaw]

  /**
   * @brief 构造函数
   * @param config_path YAML配置文件路径
   */
  Planner(const std::string & config_path);

  /**
   * @brief 执行规划（直接传入Target对象）
   * @param target 目标对象
   * @param bullet_speed 子弹速度（m/s，范围10-25，超出范围会使用默认值22）
   * @return Plan 规划结果
   */
  Plan plan(Target target, double bullet_speed);

  /**
   * @brief 执行规划（传入optional<Target>，支持无目标情况）
   * @param target 可选的目标对象（可能为std::nullopt）
   * @param bullet_speed 子弹速度（m/s）
   * @return Plan 规划结果（无目标时control=false）
   * 
   * 根据目标速度自动选择延迟时间，用于补偿通信和处理延迟
   */
  Plan plan(std::optional<Target> target, double bullet_speed);

private:
  double yaw_offset_;      // yaw角度偏移量（弧度）
  double pitch_offset_;   // pitch角度偏移量（弧度）
  double fire_thresh_;    // 开火阈值：轨迹误差小于此值才允许开火
  double low_speed_delay_time_;   // 低速目标延迟时间（秒）
  double high_speed_delay_time_;  // 高速目标延迟时间（秒）
  double decision_speed_;         // 判断高速/低速的阈值速度（m/s）

  SolverType solver_type_;        // 求解器类型（MPC或五次多项式）

  // MPC求解器相关
  TinySolver * yaw_solver_;       // yaw轴MPC求解器（仅在solver_type_=MPC时使用）
  TinySolver * pitch_solver_;     // pitch轴MPC求解器（仅在solver_type_=MPC时使用）

  // 五次多项式求解器参数（仅在solver_type_=QUINTIC_POLY时使用）
  double max_yaw_vel_;            // yaw轴最大角速度（rad/s）
  double max_yaw_acc_;            // yaw轴最大角加速度（rad/s²）
  double max_pitch_vel_;          // pitch轴最大角速度（rad/s）
  double max_pitch_acc_;          // pitch轴最大角加速度（rad/s²）
  double jump_threshold_;         // 突变检测阈值（角速度变化量，rad/s）

  /**
   * @brief 初始化yaw轴MPC求解器
   * @param config_path 配置文件路径
   */
  void setup_yaw_solver(const std::string & config_path);

  /**
   * @brief 初始化pitch轴MPC求解器
   * @param config_path 配置文件路径
   */
  void setup_pitch_solver(const std::string & config_path);

  /**
   * @brief 计算瞄准点（考虑弹道补偿）
   * @param target 目标对象
   * @param bullet_speed 子弹速度（m/s）
   * @return [yaw, pitch] 瞄准角度（弧度）
   * 
   * 选择距离最近的装甲板，计算弹道轨迹，返回补偿后的瞄准角度
   */
  Eigen::Matrix<double, 2, 1> aim(const Target & target, double bullet_speed);

  /**
   * @brief 生成参考轨迹
   * @param target 目标对象（会被修改，用于预测）
   * @param yaw0 初始yaw角度（用于相对角度计算）
   * @param bullet_speed 子弹速度（m/s）
   * @return Trajectory 参考轨迹矩阵 [yaw, yaw_vel, pitch, pitch_vel] x HORIZON
   * 
   * 生成未来HORIZON个时间步的参考轨迹，用于MPC跟踪
   */
  Trajectory get_trajectory(Target & target, double yaw0, double bullet_speed);

  /**
   * @brief 突变段结构体，记录突变段的起止时间
   */
  struct JumpSegment
  {
    int start_idx;  // 突变段开始索引
    int end_idx;    // 突变段结束索引
  };

  /**
   * @brief 检测参考轨迹中的突变段
   * 突变段特征：斜率（角速度）先突然变大然后快速变小，或先突然变小然后快速变大
   * @param traj 参考轨迹矩阵
   * @param axis 轴索引（0=yaw, 2=pitch）
   * @return 突变段列表，每个突变段包含起止索引
   */
  std::vector<JumpSegment> detect_jump_segments(const Trajectory & traj, int axis);

  /**
   * @brief 为突变段生成平滑过渡轨迹（使用五次多项式）
   * @param traj_original 原始参考轨迹
   * @param traj_smoothed 平滑后的轨迹
   * @param axis 轴索引（0=yaw, 2=pitch）
   * @param jump_points 突变点索引列表
   * @param v_max 最大速度约束
   * @param a_max 最大加速度约束
   * @return 是否成功生成过渡段
   */
  bool smooth_jumps(
    const Trajectory & traj_original, Trajectory & traj_smoothed, int axis,
    const std::vector<int> & jump_points, double v_max, double a_max);

  /**
   * @brief 为单个突变点搜索最优过渡段
   * @param traj 参考轨迹
   * @param axis 轴索引
   * @param jump_idx 突变点索引
   * @param v_max 最大速度约束
   * @param a_max 最大加速度约束
   * @return 成功时返回过渡段系数和起止索引，失败时返回空
   */
  std::optional<std::tuple<Vector6d, int, int>> search_transition(
    const Trajectory & traj, int axis, int jump_idx, double v_max, double a_max);

  /**
   * @brief 使用MPC求解器进行规划
   * @param traj 参考轨迹
   * @param yaw0 初始yaw角度
   * @return Plan 规划结果
   */
  Plan solve_with_mpc(const Trajectory & traj, double yaw0);

  /**
   * @brief 使用五次多项式求解器进行规划
   * @param traj 参考轨迹（不会被修改，内部会创建副本进行平滑处理）
   * @param yaw0 初始yaw角度
   * @return Plan 规划结果
   */
  Plan solve_with_quintic(const Trajectory & traj, double yaw0);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__PLANNER_HPP