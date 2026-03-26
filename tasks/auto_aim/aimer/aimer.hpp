#ifndef AUTO_AIM__AIMER_HPP
#define AUTO_AIM__AIMER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <limits>

#include "io/cboard/cboard.hpp"
#include "io/cboard/command.hpp"
#include "tasks/auto_aim/target/target.hpp"

namespace auto_aim
{

struct AimPoint
{
  bool valid = false;
  Eigen::Vector4d xyza = Eigen::Vector4d::Zero();
  Eigen::Vector3d center_xyz = Eigen::Vector3d::Zero();
  double center_yaw = 0.0;
  double fly_time = std::numeric_limits<double>::infinity();
  double align_time = std::numeric_limits<double>::infinity();
  double time_error = std::numeric_limits<double>::infinity();
  double centerline_error = std::numeric_limits<double>::infinity();
  int locked_id = 0;
  bool center_aim = false;
};

class Aimer
{
public:
  AimPoint debug_aim_point;
  explicit Aimer(const std::string & config_path);
  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    bool to_now = true);

  io::Command aim(
    std::list<Target> targets, std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    io::ShootMode shoot_mode, bool to_now = true);

private:
  double yaw_offset_;
  std::optional<double> left_yaw_offset_, right_yaw_offset_;
  double pitch_offset_;
  double comming_angle_;
  double leaving_angle_;
  double lock_id_ = -1;
  double high_speed_delay_time_;
  double low_speed_delay_time_;
  double decision_speed_;
  double center_aim_speed_thresh_;
  double default_bullet_speed_;

  AimPoint choose_aim_point(Target target);
  AimPoint choose_center_aim_point(const Target & target, double bullet_speed);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__AIMER_HPP
