#include "aimer.hpp"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/trajectory/trajectory.hpp"

namespace auto_aim
{
Aimer::Aimer(const std::string & config_path)
: left_yaw_offset_(std::nullopt), right_yaw_offset_(std::nullopt)
{
  auto yaml = YAML::LoadFile(config_path);
  yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;        // degree to rad
  pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;    // degree to rad
  comming_angle_ = yaml["comming_angle"].as<double>() / 57.3;  // degree to rad
  leaving_angle_ = yaml["leaving_angle"].as<double>() / 57.3;  // degree to rad
  high_speed_delay_time_ = yaml["high_speed_delay_time"].as<double>();
  low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  decision_speed_ = yaml["decision_speed"].as<double>();
  center_aim_speed_thresh_ = yaml["center_aim_speed_thresh"].IsDefined()
                               ? yaml["center_aim_speed_thresh"].as<double>()
                               : std::numeric_limits<double>::infinity();
  default_bullet_speed_ = yaml["default_bullet_speed"].as<double>();
  if (yaml["left_yaw_offset"].IsDefined() && yaml["right_yaw_offset"].IsDefined()) {
    left_yaw_offset_ = yaml["left_yaw_offset"].as<double>() / 57.3;    // degree to rad
    right_yaw_offset_ = yaml["right_yaw_offset"].as<double>() / 57.3;  // degree to rad
    tools::logger()->info("[Aimer] successfully loading shootmode");
  }
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  bool to_now)
{
  debug_aim_point = {};
  if (targets.empty()) return {false, false, 0, 0};
  auto target = targets.front();

  double delay_time =
    std::abs(target.ekf_x()[7]) > decision_speed_ ? high_speed_delay_time_ : low_speed_delay_time_;

  if (bullet_speed < 9) bullet_speed = default_bullet_speed_;

  // 考虑detecor和tracker所消耗的时间，此外假设aimer的用时可忽略不计
  auto future = timestamp;
  if (to_now) {
    double dt;
    dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay_time;
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  else {
    auto dt = 0.005 + delay_time;  //detector-aimer耗时0.005+发弹延时0.1
    // tools::logger()->info("dt is {:.4f} second", dt);
    future += std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);
  }

  const bool use_center_aim =
    std::isfinite(center_aim_speed_thresh_) && std::abs(target.ekf_x()[7]) > center_aim_speed_thresh_;
  auto aim_point0 = use_center_aim ? choose_center_aim_point(target, bullet_speed) : choose_aim_point(target);
  debug_aim_point = aim_point0;
  if (!aim_point0.valid) {
    // tools::logger()->debug("Invalid aim_point0.");
    return {false, false, 0, 0};
  }

  Eigen::Vector3d xyz0 = aim_point0.xyza.head(3);
  auto d0 = std::sqrt(xyz0[0] * xyz0[0] + xyz0[1] * xyz0[1]);
  tools::Trajectory trajectory0(bullet_speed, d0, xyz0[2]);
  if (trajectory0.unsolvable) {
    tools::logger()->debug(
      "[Aimer] Unsolvable trajectory0: {:.2f} {:.2f} {:.2f}", bullet_speed, d0, xyz0[2]);
    debug_aim_point.valid = false;
    return {false, false, 0, 0};
  }

  if (use_center_aim) {
    debug_aim_point.fly_time = trajectory0.fly_time;
    if (std::isfinite(debug_aim_point.align_time)) {
      debug_aim_point.time_error = std::abs(debug_aim_point.fly_time - debug_aim_point.align_time);
    }
    double yaw = debug_aim_point.center_yaw + yaw_offset_;
    double pitch = trajectory0.pitch + pitch_offset_;
    return {true, false, yaw, pitch};
  }

  // 迭代求解飞行时间 (最多10次，收敛条件：相邻两次fly_time差 <0.001)
  double prev_fly_time = trajectory0.fly_time;
  tools::Trajectory current_traj = trajectory0;
  std::vector<Target> iteration_target(10, target);  // 创建10个目标副本用于迭代预测

  for (int iter = 0; iter < 10; ++iter) {
    // 预测目标在 future + prev_fly_time 时刻的位置
    auto predict_time = future + std::chrono::microseconds(static_cast<int>(prev_fly_time * 1e6));
    iteration_target[iter].predict(predict_time);

    // 计算瞄准点
    auto aim_point = choose_aim_point(iteration_target[iter]);
    debug_aim_point = aim_point;
    if (!aim_point.valid) {
      return {false, false, 0, 0};
    }

    // 计算新弹道
    Eigen::Vector3d xyz = aim_point.xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    current_traj = tools::Trajectory(bullet_speed, d, xyz.z());

    // 检查弹道是否可解
    if (current_traj.unsolvable) {
      tools::logger()->debug(
        "[Aimer] Unsolvable trajectory in iter {}: speed={:.2f}, d={:.2f}, z={:.2f}", iter + 1,
        bullet_speed, d, xyz.z());
      debug_aim_point.valid = false;
      return {false, false, 0, 0};
    }

    // 检查收敛条件
    if (std::abs(current_traj.fly_time - prev_fly_time) < 0.001) {
      break;
    }
    prev_fly_time = current_traj.fly_time;
  }

  // 计算最终角度
  Eigen::Vector3d final_xyz = debug_aim_point.xyza.head(3);
  double yaw = std::atan2(final_xyz.y(), final_xyz.x()) + yaw_offset_;
  double pitch = current_traj.pitch + pitch_offset_;  // 世界坐标系下 pitch 向上为正
  return {true, false, yaw, pitch};
}

io::Command Aimer::aim(
  std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
  io::ShootMode shoot_mode, bool to_now)
{
  double yaw_offset;
  if (shoot_mode == io::left_shoot && left_yaw_offset_.has_value()) {
    yaw_offset = left_yaw_offset_.value();
  } else if (shoot_mode == io::right_shoot && right_yaw_offset_.has_value()) {
    yaw_offset = right_yaw_offset_.value();
  } else {
    yaw_offset = yaw_offset_;
  }

  auto command = aim(targets, timestamp, bullet_speed, to_now);
  command.yaw = command.yaw - yaw_offset_ + yaw_offset;

  return command;
}

namespace
{
inline bool finite4(const Eigen::Vector4d & v)
{
  return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]) && std::isfinite(v[3]);
}

inline double time_to_centerline(double delta, double w)
{
  // delta: current angle error in (-pi, pi]
  // w: angular velocity (rad/s)
  // Return: smallest positive time to reach delta == 0, considering wrap-around.
  constexpr double TWO_PI = 2.0 * CV_PI;
  constexpr double EPS_W = 1e-3;

  if (std::abs(w) < EPS_W) return std::numeric_limits<double>::infinity();

  // Normalize delta into (-pi, pi]
  delta = tools::limit_rad(delta);

  if (w > 0) {
    // delta increases with time
    return (delta <= 0) ? (-delta / w) : ((TWO_PI - delta) / w);
  } else {
    // delta decreases with time
    return (delta >= 0) ? (delta / (-w)) : ((TWO_PI + delta) / (-w));
  }
}
}  // namespace

AimPoint Aimer::choose_aim_point(Target target)
{
  Eigen::VectorXd ekf_x = target.ekf_x();
  std::vector<Eigen::Vector4d> armor_xyza_list = target.armor_xyza_list();
  auto armor_num = armor_xyza_list.size();
  // 如果装甲板未发生过跳变，则只有当前装甲板的位置已知
  if (!target.jumped) return {true, armor_xyza_list[0]};

  // 整车旋转中心的球坐标yaw
  auto center_yaw = std::atan2(ekf_x[2], ekf_x[0]);

  // 如果delta_angle为0，则该装甲板中心和整车中心的连线在世界坐标系的xy平面过原点
  std::vector<double> delta_angle_list;
  for (int i = 0; i < armor_num; i++) {
    auto delta_angle = tools::limit_rad(armor_xyza_list[i][3] - center_yaw);
    delta_angle_list.emplace_back(delta_angle);
  }

  // 选择在可射击范围内的装甲板（先做可见性/射界筛选）
  std::vector<int> id_list;
  id_list.reserve(armor_num);
  for (int i = 0; i < armor_num; i++) {
    if (std::abs(delta_angle_list[i]) > 60 / 57.3) continue;
    id_list.push_back(i);
  }
  if (id_list.empty()) {
    tools::logger()->warn("[Aimer] Empty id list!");
    return {false, armor_xyza_list[0]};
  }

  // 当有明显角速度时：用“到达中心线(delta=0)的预测时间”做选板
  const double w = ekf_x[7];
  if (std::abs(w) > 1e-3) {
    int best_id = id_list.front();
    double best_t = std::numeric_limits<double>::infinity();

    for (int id : id_list) {
      const double t_center = time_to_centerline(delta_angle_list[id], w);
      if (t_center < best_t) {
        best_t = t_center;
        best_id = id;
      }
    }

    // 防止预测太远导致数值不稳定（通常 0~0.5s 内就能决定）
    best_t = std::clamp(best_t, 0.0, 0.5);

    target.predict(best_t);
    auto predicted_list = target.armor_xyza_list();
    if (best_id < 0 || best_id >= static_cast<int>(predicted_list.size())) {
      tools::logger()->warn("[Aimer] best_id out of range after predict: {}", best_id);
      return {false, armor_xyza_list[0]};
    }
    if (!finite4(predicted_list[best_id])) {
      tools::logger()->warn("[Aimer] Non-finite predicted aim point.");
      return {false, armor_xyza_list[0]};
    }

    lock_id_ = best_id;
    return {true, predicted_list[best_id]};
  }

  // 角速度不明显：退化为原来的锁定逻辑（防止 45°来回抖）
  if (id_list.size() > 1) {
    int id0 = id_list[0], id1 = id_list[1];
    if (lock_id_ != id0 && lock_id_ != id1)
      lock_id_ = (std::abs(delta_angle_list[id0]) < std::abs(delta_angle_list[id1])) ? id0 : id1;
    return {true, armor_xyza_list[static_cast<int>(lock_id_)]};
  }
  lock_id_ = -1;
  return {true, armor_xyza_list[id_list[0]]};

  // 不应到达这里
  return {false, armor_xyza_list[0]};
}

AimPoint Aimer::choose_center_aim_point(const Target & target, double bullet_speed)
{
  auto armor_xyza_list = target.armor_xyza_list();
  if (armor_xyza_list.empty()) return {};

  const auto ekf_x = target.ekf_x();
  const double w = ekf_x[7];
  const double center_yaw0 = std::atan2(ekf_x[2], ekf_x[0]);
  constexpr double MAX_ALIGN_TIME = 1.5;
  constexpr int MAX_TURNS = 2;

  bool found = false;
  double best_score = std::numeric_limits<double>::infinity();
  AimPoint best;

  auto consider = [&](const Target & predicted, int id, double t_align) {
    auto predicted_list = predicted.armor_xyza_list();
    if (id < 0 || id >= static_cast<int>(predicted_list.size())) return;

    const auto & xyza = predicted_list[id];
    if (!finite4(xyza)) return;

    Eigen::Vector3d xyz = xyza.head(3);
    double d = std::sqrt(xyz.x() * xyz.x() + xyz.y() * xyz.y());
    tools::Trajectory traj(bullet_speed, d, xyz.z());
    if (traj.unsolvable) return;

    const auto predicted_x = predicted.ekf_x();
    Eigen::Vector3d center_xyz(predicted_x[0], predicted_x[2], predicted_x[4]);
    double center_yaw = std::atan2(center_xyz.y(), center_xyz.x());
    double armor_yaw = std::atan2(xyz.y(), xyz.x());
    double centerline_error = std::abs(tools::limit_rad(armor_yaw - center_yaw));
    double time_error = std::abs(traj.fly_time - t_align);
    double w_abs = std::max(std::abs(w), 1e-3);
    double score = time_error + centerline_error / w_abs;

    if (!found || score < best_score) {
      found = true;
      best_score = score;
      best.valid = true;
      best.xyza = xyza;
      best.center_xyz = center_xyz;
      best.center_yaw = center_yaw;
      best.fly_time = traj.fly_time;
      best.align_time = t_align;
      best.time_error = time_error;
      best.centerline_error = centerline_error;
      best.locked_id = id;
      best.center_aim = true;
    }
  };

  if (std::abs(w) < 1e-3) return {};

  const double period = 2.0 * CV_PI / std::abs(w);
  for (int i = 0; i < static_cast<int>(armor_xyza_list.size()); ++i) {
    double delta = tools::limit_rad(armor_xyza_list[i][3] - center_yaw0);
    double first_t = time_to_centerline(delta, w);
    if (!std::isfinite(first_t)) continue;

    for (int turn = 0; turn <= MAX_TURNS; ++turn) {
      double t_align = first_t + turn * period;
      if (t_align < 0.0 || t_align > MAX_ALIGN_TIME) continue;
      auto predicted = target;
      predicted.predict(t_align);
      consider(predicted, i, t_align);
    }
  }

  return best;
}

}  // namespace auto_aim
