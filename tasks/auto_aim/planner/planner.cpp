#include "planner.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "tools/math_tools/math_tools.hpp"
#include "tools/trajectory/trajectory.hpp"
#include "tools/yaml/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
Planner::Planner(const std::string & config_path)
{
  // 从配置文件加载参数
  auto yaml = tools::load(config_path);
  // 角度偏移量（配置文件中的度转换为弧度）
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  // 开火阈值：轨迹跟踪误差小于此值才允许开火
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  // 判断目标速度高低的阈值
  decision_speed_ = tools::read<double>(yaml, "decision_speed");
  // 高速目标延迟时间（用于补偿通信延迟）
  high_speed_delay_time_ = tools::read<double>(yaml, "high_speed_delay_time");
  // 低速目标延迟时间
  low_speed_delay_time_ = tools::read<double>(yaml, "low_speed_delay_time");

  // 读取求解器类型（默认使用MPC）
  std::string solver_type_str = yaml["solver_type"] ? yaml["solver_type"].as<std::string>() : "mpc";
  if (solver_type_str == "quintic_poly" || solver_type_str == "quintic") {
    solver_type_ = SolverType::QUINTIC_POLY;
  } else {
    solver_type_ = SolverType::MPC;
  }

  // 根据求解器类型初始化相应的求解器
  if (solver_type_ == SolverType::MPC) {
    // 初始化yaw和pitch轴的MPC求解器
    setup_yaw_solver(config_path);
    setup_pitch_solver(config_path);
    // MPC求解器不需要五次多项式参数，设置为默认值
    max_yaw_vel_ = 10.0;
    max_yaw_acc_ = 20.0;
    max_pitch_vel_ = 10.0;
    max_pitch_acc_ = 20.0;
    jump_threshold_ = 0.5;
  } else {
    // 初始化五次多项式求解器参数
    max_yaw_vel_ = yaml["max_yaw_vel"] ? yaml["max_yaw_vel"].as<double>() : 100.0;      // 默认10 rad/s
    max_yaw_acc_ = yaml["max_yaw_acc"] ? yaml["max_yaw_acc"].as<double>() : 500.0;     // 默认20 rad/s²
    max_pitch_vel_ = yaml["max_pitch_vel"] ? yaml["max_pitch_vel"].as<double>() : 100.0; // 默认10 rad/s
    max_pitch_acc_ = yaml["max_pitch_acc"] ? yaml["max_pitch_acc"].as<double>() : 500.0; // 默认20 rad/s²
    jump_threshold_ = yaml["jump_threshold"] ? yaml["jump_threshold"].as<double>() : 0.5; // 默认0.5 rad
    // 五次多项式求解器不需要MPC求解器
    yaw_solver_ = nullptr;
    pitch_solver_ = nullptr;
  }
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 步骤0：校验子弹速度，超出合理范围则使用默认值
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 步骤1：预测目标位置（考虑子弹飞行时间）
  // 选择距离最近的装甲板（在xy平面上的距离）
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();  // xy平面距离
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();  // 保存最近装甲板的xyz坐标
    }
  }
  // 计算弹道轨迹，获取子弹飞行时间
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  // 预测目标到子弹到达时刻的位置
  target.predict(bullet_traj.fly_time);

  // 步骤2：生成参考轨迹
  double yaw0;  // 初始yaw角度（用于相对角度计算）
  Trajectory traj;  // 参考轨迹矩阵
  try {
    // 计算初始瞄准点，获取初始yaw角度
    yaw0 = aim(target, bullet_speed)(0);
    // 生成完整的参考轨迹（HORIZON个时间步）
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    // 如果无法求解（如弹道无解），返回失败
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 步骤3：根据配置的求解器类型进行规划
  if (solver_type_ == SolverType::MPC) {
    return solve_with_mpc(traj, yaw0);
  } else {
    return solve_with_quintic(traj, yaw0);
  }
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  // 如果没有目标，返回控制关闭
  if (!target.has_value()) return {false};

  // 根据目标速度选择延迟时间
  // ekf_x()[7] 是目标在x方向的速度分量
  // 高速目标需要更长的延迟时间以补偿通信和处理延迟
  double delay_time =
    std::abs(target->ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  // 计算未来时刻（当前时间 + 延迟时间）
  auto future = std::chrono::steady_clock::now() + std::chrono::microseconds(int(delay_time * 1e6));

  // 预测目标到未来时刻的位置
  target->predict(future);

  // 调用主规划函数
  return plan(*target, bullet_speed);
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  // 从配置文件加载yaw轴MPC参数
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");  // 最大yaw角加速度
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");  // 状态权重矩阵
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");  // 控制权重矩阵

  // 定义状态空间模型：x = [角度, 角速度]^T, u = [角加速度]
  // x_{k+1} = A * x_k + B * u_k
  // A矩阵：位置积分模型 [1, DT; 0, 1]
  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  // B矩阵：加速度对速度的积分 [0; DT]
  Eigen::MatrixXd B{{0}, {DT}};
  // 扰动项（无扰动）
  Eigen::VectorXd f{{0, 0}};
  // 权重矩阵
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());  // 状态权重：对角矩阵
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());  // 控制权重：标量
  // 初始化TinyMPC求解器
  // 参数：求解器指针, A, B, f, Q, R, 缩放因子, 状态维度, 控制维度, 时域长度, 其他参数
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  // 设置约束条件
  // 状态约束：角度和角速度无限制（使用很大的数值）
  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  // 控制约束：角加速度限制在[-max_yaw_acc, max_yaw_acc]
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  // 设置求解器最大迭代次数
  yaw_solver_->settings->max_iter = 10;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  // 从配置文件加载pitch轴MPC参数
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");  // 最大pitch角加速度
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");  // 状态权重矩阵
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");  // 控制权重矩阵

  // 定义状态空间模型：与yaw轴相同的结构
  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  // 初始化TinyMPC求解器
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  // 设置约束条件
  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  // 设置求解器最大迭代次数
  pitch_solver_->settings->max_iter = 10;
}

Eigen::Matrix<double, 2, 1> Planner::aim(const Target & target, double bullet_speed)
{
  // 选择距离最近的装甲板（在xy平面上的距离）
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();  // xy平面距离
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();  // 保存最近装甲板的xyz坐标
      yaw = xyza[3];         // 保存装甲板的yaw角度
    }
  }
  // 保存调试信息
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  // 计算方位角（atan2返回[-π, π]范围）
  auto azim = std::atan2(xyz.y(), xyz.x());
  // 计算弹道轨迹（考虑重力和子弹速度）
  // 参数：子弹速度、水平距离、高度差
  auto bullet_traj = tools::Trajectory(bullet_speed, min_dist, xyz.z());
  // 如果弹道无解（如目标太高），抛出异常
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  // 返回瞄准角度：[yaw, pitch]
  // yaw: 方位角 + yaw偏移量（限制在[-π, π]）
  // pitch: 弹道俯仰角（取负号，因为pitch向下为正）+ pitch偏移量
  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}

Trajectory Planner::get_trajectory(Target & target, double yaw0, double bullet_speed)
{
  Trajectory traj;

  // 预测到过去时刻（HALF_HORIZON+1个时间步之前），用于计算初始速度
  // 这样第一个时间步的索引0对应-HALF_HORIZON*DT时刻
  target.predict(-DT * (HALF_HORIZON + 1));
  auto yaw_pitch_last = aim(target, bullet_speed);

  // 预测到-HALF_HORIZON*DT时刻（轨迹的起始点）
  target.predict(DT);  // 现在target处于-HALF_HORIZON*DT时刻
  auto yaw_pitch = aim(target, bullet_speed);

  // 生成HORIZON个时间步的参考轨迹
  // 轨迹从-HALF_HORIZON*DT开始，到+HALF_HORIZON*DT结束
  // 索引HALF_HORIZON对应t=0时刻（当前时刻）
  for (int i = 0; i < HORIZON; i++) {
    // 预测到下一个时间步
    target.predict(DT);
    auto yaw_pitch_next = aim(target, bullet_speed);

    // 使用中心差分法计算角速度
    // 速度 = (下一个时刻 - 上一个时刻) / (2 * DT)
    auto yaw_vel = tools::limit_rad(yaw_pitch_next(0) - yaw_pitch_last(0)) / (2 * DT);
    auto pitch_vel = (yaw_pitch_next(1) - yaw_pitch_last(1)) / (2 * DT);

    // 保存当前时间步的轨迹数据
    // yaw使用相对角度（减去初始yaw0），便于MPC跟踪
    traj.col(i) << tools::limit_rad(yaw_pitch(0) - yaw0), yaw_vel, yaw_pitch(1), pitch_vel;

    // 更新滑动窗口
    yaw_pitch_last = yaw_pitch;
    yaw_pitch = yaw_pitch_next;
  }

  return traj;
}

std::vector<int> Planner::detect_jumps(const Trajectory & traj, int axis)
{
  std::vector<int> jump_points;
  
  // 检查相邻时间步之间的位置变化
  for (int i = 1; i < HORIZON; ++i) {
    double pos_diff = std::abs(traj(axis, i) - traj(axis, i - 1));
    
    // 如果位置变化超过阈值，认为是突变点
    if (pos_diff > jump_threshold_) {
      jump_points.push_back(i);
    }
  }
  
  return jump_points;
}

std::optional<std::tuple<Vector6d, int, int>> Planner::search_transition(
  const Trajectory & traj, int axis, int jump_idx, double v_max, double a_max)
{
  // 从大到小搜索偏移量（0.5周期 → 0.01周期）
  const double max_offset = 0.5;  // 最大偏移量（周期数）
  const double min_offset = 0.01; // 最小偏移量（周期数）
  const double offset_step = 0.01; // 搜索步长（周期数）
  
  int vel_axis = axis + 1;  // 速度轴索引
  
  // 从最大偏移量开始搜索
  for (double offset_cycles = max_offset; offset_cycles >= min_offset; offset_cycles -= offset_step) {
    int offset = static_cast<int>(offset_cycles / DT);  // 转换为时间步数
    
    // 计算过渡段起止点
    int start_idx = std::max(0, jump_idx - offset);
    int end_idx = std::min(HORIZON - 1, jump_idx + offset);
    
    // 确保有足够的空间
    if (end_idx - start_idx < 2) {
      continue;
    }
    
    // 计算边界条件
    QuinticPolynomialSolver::BoundaryCondition start;
    start.pos = traj(axis, start_idx);
    start.vel = traj(vel_axis, start_idx);
    // 使用中心差分估算加速度
    if (start_idx > 0 && start_idx < HORIZON - 1) {
      start.acc = (traj(vel_axis, start_idx + 1) - traj(vel_axis, start_idx - 1)) / (2 * DT);
    } else {
      start.acc = 0.0;
    }
    
    QuinticPolynomialSolver::BoundaryCondition end;
    end.pos = traj(axis, end_idx);
    end.vel = traj(vel_axis, end_idx);
    // 使用中心差分估算加速度
    if (end_idx > 0 && end_idx < HORIZON - 1) {
      end.acc = (traj(vel_axis, end_idx + 1) - traj(vel_axis, end_idx - 1)) / (2 * DT);
    } else {
      end.acc = 0.0;
    }
    
    // 计算过渡时间
    double T = (end_idx - start_idx) * DT;
    
    // 求解五次多项式系数
    Vector6d coeffs = QuinticPolynomialSolver::solve(start, end, T);
    
    // 检查约束
    if (QuinticPolynomialSolver::check_constraints(coeffs, T, v_max, a_max, 200)) {
      // 找到满足约束的最小过渡段
      return std::make_tuple(coeffs, start_idx, end_idx);
    }
  }
  
  // 未找到满足约束的过渡段
  return std::nullopt;
}

bool Planner::smooth_jumps(
  Trajectory & traj, int axis, const std::vector<int> & jump_points, double v_max,
  double a_max)
{
  if (jump_points.empty()) {
    return true;
  }
  
  int vel_axis = axis + 1;
  bool success = true;
  
  // 从后往前处理突变点，避免索引变化影响
  for (auto it = jump_points.rbegin(); it != jump_points.rend(); ++it) {
    int jump_idx = *it;
    
    // 搜索最优过渡段
    auto result = search_transition(traj, axis, jump_idx, v_max, a_max);
    
    if (result.has_value()) {
      auto [coeffs, start_idx, end_idx] = result.value();
      
      // 计算过渡时间
      double T = (end_idx - start_idx) * DT;
      
      // 用五次多项式替换过渡段内的轨迹
      for (int i = start_idx; i <= end_idx; ++i) {
        double t = (i - start_idx) * DT;  // 相对于起点的时间
        
        // 更新位置和速度
        traj(axis, i) = QuinticPolynomialSolver::evaluate_pos(coeffs, t);
        traj(vel_axis, i) = QuinticPolynomialSolver::evaluate_vel(coeffs, t);
      }
    } else {
      // 无法找到满足约束的过渡段
      tools::logger()->warn(
        "Failed to find valid transition for jump at index {} on axis {}", jump_idx, axis);
      success = false;
    }
  }
  
  return success;
}

Plan Planner::solve_with_mpc(const Trajectory & traj, double yaw0)
{
  // 使用MPC求解yaw轴控制
  // 设置初始状态：参考轨迹的第一个时间步
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);  // [yaw, yaw_vel]
  tiny_set_x0(yaw_solver_, x0);
  // 设置参考轨迹：yaw和yaw_vel的参考值
  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  // 执行MPC求解
  tiny_solve(yaw_solver_);

  // 使用MPC求解pitch轴控制
  // 设置初始状态：pitch和pitch_vel
  x0 << traj(2, 0), traj(3, 0);  // [pitch, pitch_vel]
  tiny_set_x0(pitch_solver_, x0);
  // 设置参考轨迹：pitch和pitch_vel的参考值
  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  // 执行MPC求解
  tiny_solve(pitch_solver_);

  // 构建规划结果
  Plan plan;
  plan.control = true;  // 规划成功，启用控制

  // 目标角度：参考轨迹在HALF_HORIZON时刻的值（0.5秒后）
  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  // MPC规划的状态：在HALF_HORIZON时刻的预测状态
  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);  // 控制输入（加速度）

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  // 判断是否满足开火条件
  // 计算未来时刻（HALF_HORIZON + shoot_offset_）的轨迹跟踪误差
  // 如果误差小于阈值，说明云台能够准确跟踪目标，允许开火
  auto shoot_offset_ = 2;  // 开火判断的时间偏移（2个时间步 = 0.02秒）
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::solve_with_quintic(Trajectory & traj, double yaw0)
{
  // 检测并平滑yaw轴突变段
  auto yaw_jumps = detect_jumps(traj, 0);
  if (!yaw_jumps.empty()) {
    smooth_jumps(traj, 0, yaw_jumps, max_yaw_vel_, max_yaw_acc_);
  }

  // 检测并平滑pitch轴突变段
  auto pitch_jumps = detect_jumps(traj, 2);
  if (!pitch_jumps.empty()) {
    smooth_jumps(traj, 2, pitch_jumps, max_pitch_vel_, max_pitch_acc_);
  }

  // 构建规划结果（直接使用平滑后的参考轨迹）
  Plan plan;
  plan.control = true;  // 规划成功，启用控制

  // 目标角度：参考轨迹在HALF_HORIZON时刻的值（0.5秒后）
  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  // 五次多项式规划的状态：直接使用参考轨迹在HALF_HORIZON时刻的值
  plan.yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = traj(1, HALF_HORIZON);
  // 使用中心差分法计算加速度
  if (HALF_HORIZON > 0 && HALF_HORIZON < HORIZON - 1) {
    plan.yaw_acc = (traj(1, HALF_HORIZON + 1) - traj(1, HALF_HORIZON - 1)) / (2 * DT);
  } else {
    plan.yaw_acc = 0.0;
  }

  plan.pitch = traj(2, HALF_HORIZON);
  plan.pitch_vel = traj(3, HALF_HORIZON);
  // 使用中心差分法计算加速度
  if (HALF_HORIZON > 0 && HALF_HORIZON < HORIZON - 1) {
    plan.pitch_acc = (traj(3, HALF_HORIZON + 1) - traj(3, HALF_HORIZON - 1)) / (2 * DT);
  } else {
    plan.pitch_acc = 0.0;
  }

  // 判断是否满足开火条件
  // 对于五次多项式求解器，检查轨迹是否平滑（无突变）
  auto shoot_offset_ = 2;  // 开火判断的时间偏移（2个时间步 = 0.02秒）
  if (HALF_HORIZON + shoot_offset_ < HORIZON) {
    // 检查轨迹跟踪误差（相对于参考轨迹本身，应该很小）
    double yaw_error = std::abs(traj(0, HALF_HORIZON + shoot_offset_) - traj(0, HALF_HORIZON));
    double pitch_error = std::abs(traj(2, HALF_HORIZON + shoot_offset_) - traj(2, HALF_HORIZON));
    plan.fire = std::hypot(yaw_error, pitch_error) < fire_thresh_;
  } else {
    plan.fire = false;
  }

  return plan;
}

}  // namespace auto_aim