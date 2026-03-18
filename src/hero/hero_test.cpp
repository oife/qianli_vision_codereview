// 完整形态考核专用自瞄链路
// 目标：5米外平移+旋转车辆，小装甲板，四个装甲板
// 车辆旋转半径：短轴r=23cm，长轴r+dr=28cm（dr=5cm）
// 核心改动：
//   1. ExamTarget: r/dr定死(0.23/0.05)，v2极小约束直线平移
//   2. 射击逻辑：cooldown+角度门限，只在最有把握时发一帧shoot:true
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <list>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sys/select.h>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/cboard/command.hpp"
#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/extended_kalman_filter/extended_kalman_filter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/plotter/plotter.hpp"
#include "tools/recorder/recorder.hpp"
#include "tools/trajectory/trajectory.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/standard.yaml | yaml配置文件路径 }"
  "{record         | false                 | 开启录制         }"
  "{viz            | true                  | 开启可视化界面   }"
  "{headless       | false                 | 无GUI模式        }";

using namespace std::chrono_literals;

namespace exam
{
// 车辆旋转几何（定死）
constexpr double R_SHORT = 0.23;    // 短轴半径 23cm
constexpr double R_LONG = 0.28;     // 长轴半径 28cm
constexpr double DR = R_LONG - R_SHORT;  // 0.05m
constexpr double DH = -0.035;  // 长轴装甲板比短轴低35mm

// 车辆运动学先验
constexpr double TRANSLATE_SPEED = 1.0;   // 平移速度 ~1m/s（保留用于可读性）
constexpr double ROTATE_RPS = 0.45;       // 旋转速度 0.45转/秒
constexpr double ROTATE_W = ROTATE_RPS * 2 * M_PI;  // ≈2.83 rad/s

// 射击控制
constexpr double SHOOT_YAW_THRESH = 0.6 / 57.3;
constexpr double SHOOT_PITCH_THRESH = 0.6 / 57.3;
constexpr double SHOOT_COOLDOWN_SEC = 1.2;
constexpr int SHOOT_STABLE_FRAMES = 3;

// EKF参数
// 平移：v1适中，目标加速度不大（匀速平移为主）
// 旋转：v2适中，目标匀速旋转，角加速度小但不为零
constexpr double V1 = 20;     // 平移加速度方差（匀速~1m/s，加速度小）
constexpr double V2 = 1.0;    // 角加速度方差（匀速旋转，允许小幅变化）
constexpr double VZ = 0.01;   // z轴加速度方差（旋转中心高度不变，极小）
}  // namespace exam

// ============================================================
// ExamTarget: r/dr定死，直线平移约束
// ============================================================
class ExamTarget
{
public:
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  auto_aim::ArmorPriority priority;
  bool jumped;
  int last_id;

  ExamTarget() = default;

  ExamTarget(const auto_aim::Armor & armor, std::chrono::steady_clock::time_point t)
  : name(armor.name), armor_type(armor.type), jumped(false), last_id(0),
    update_count_(0), armor_num_(4), t_(t), is_converged_(false)
  {
    double r = exam::R_SHORT;
    priority = armor.priority;
    const Eigen::VectorXd & xyz = armor.xyz_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;

    auto center_x = xyz[0] + r * std::cos(ypr[0]);
    auto center_y = xyz[1] + r * std::sin(ypr[0]);
    auto center_z = xyz[2];

    // [x vx y vy z vz a w r dr dh]
    // w初始化为已知旋转角速度（符号未知，先设正值，EKF会自动修正）
    Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0],
                         exam::ROTATE_W, exam::R_SHORT, exam::DR, exam::DH}};

    // r/dr/dh协方差极小 → 锁死旋转几何
    // w协方差适中 → 允许EKF修正角速度符号和精确值
    // vx/vy协方差适中 → 匹配~1m/s平移
    // z/vz协方差恢复正常
    Eigen::VectorXd P0_dig{{1, 16, 1, 16, 0.1, 0.1, 0.4, 25, 1e-6, 1e-6, 1e-6}};
    Eigen::MatrixXd P0 = P0_dig.asDiagonal();

    auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a + b;
      c[6] = tools::limit_rad(c[6]);
      return c;
    };
    ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);
  }

  void predict(std::chrono::steady_clock::time_point t)
  {
    auto dt = tools::delta_time(t, t_);
    predict(dt);
    t_ = t;
  }

  void predict(double dt)
  {
    // clang-format off
    Eigen::MatrixXd F{
      {1, dt,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  1, dt,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  1, dt,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  1, dt,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0},
      {0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1}
    };
    // clang-format on
    double v1 = exam::V1, v2 = exam::V2, vz = exam::VZ;
    auto a = dt*dt*dt*dt/4, b = dt*dt*dt/2, c = dt*dt;
    // clang-format off
    Eigen::MatrixXd Q{
      {a*v1, b*v1,    0,    0,     0,     0,    0,    0, 0, 0, 0},
      {b*v1, c*v1,    0,    0,     0,     0,    0,    0, 0, 0, 0},
      {   0,    0, a*v1, b*v1,     0,     0,    0,    0, 0, 0, 0},
      {   0,    0, b*v1, c*v1,     0,     0,    0,    0, 0, 0, 0},
      {   0,    0,    0,    0, a*vz, b*vz,    0,    0, 0, 0, 0},
      {   0,    0,    0,    0, b*vz, c*vz,    0,    0, 0, 0, 0},
      {   0,    0,    0,    0,     0,     0, a*v2, b*v2, 0, 0, 0},
      {   0,    0,    0,    0,     0,     0, b*v2, c*v2, 0, 0, 0},
      {   0,    0,    0,    0,     0,     0,    0,    0, 0, 0, 0},
      {   0,    0,    0,    0,     0,     0,    0,    0, 0, 0, 0},
      {   0,    0,    0,    0,     0,     0,    0,    0, 0, 0, 0}
    };
    // clang-format on
    auto f = [&](const Eigen::VectorXd & x) -> Eigen::VectorXd {
      Eigen::VectorXd x_prior = F * x;
      x_prior[6] = tools::limit_rad(x_prior[6]);
      return x_prior;
    };
    ekf_.predict(F, Q, f);
  }

  // axis_hint: -1=未知, 0=短轴(id=0,2), 1=长轴(id=1,3)
  void update(const auto_aim::Armor & armor, int axis_hint = -1)
  {
    int id = 0;
    auto min_angle_error = 1e10;
    const auto & xyza_list = armor_xyza_list();
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      // 如果有axis_hint，只在对应的id集合里匹配
      if (axis_hint == 0 && (i == 1 || i == 3)) continue;  // 短轴只匹配id=0,2
      if (axis_hint == 1 && (i == 0 || i == 2)) continue;  // 长轴只匹配id=1,3
      xyza_i_list.push_back({xyza_list[i], i});
    }
    std::sort(xyza_i_list.begin(), xyza_i_list.end(),
      [](const auto & a, const auto & b) {
        return tools::xyz2ypd(a.first.head(3))[2] < tools::xyz2ypd(b.first.head(3))[2];
      });
    for (int i = 0; i < std::min(3, (int)xyza_i_list.size()); i++) {
      const auto & xyza = xyza_i_list[i].first;
      Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
      auto err = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                 std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
      if (err < min_angle_error) { id = xyza_i_list[i].second; min_angle_error = err; }
    }
    if (id != 0) jumped = true;
    last_id = id;
    update_count_++;

    Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
    auto cy = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
    auto da = tools::limit_rad(armor.ypr_in_world[0] - cy);
    Eigen::VectorXd R_dig{{4e-3, 4e-3, log(std::abs(da)+1)+1,
      log(std::abs(armor.ypd_in_world[2])+1)/200+9e-2}};
    Eigen::MatrixXd R = R_dig.asDiagonal();
    auto h = [&](const Eigen::VectorXd & x) -> Eigen::Vector4d {
      auto xyz = h_armor_xyz(x, id);
      auto ypd = tools::xyz2ypd(xyz);
      return {ypd[0], ypd[1], ypd[2], tools::limit_rad(x[6]+id*2*CV_PI/armor_num_)};
    };
    auto z_sub = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) -> Eigen::VectorXd {
      Eigen::VectorXd c = a - b;
      c[0] = tools::limit_rad(c[0]); c[1] = tools::limit_rad(c[1]); c[3] = tools::limit_rad(c[3]);
      return c;
    };
    Eigen::VectorXd z{{armor.ypd_in_world[0], armor.ypd_in_world[1],
                        armor.ypd_in_world[2], armor.ypr_in_world[0]}};
    ekf_.update(z, H, R, h, z_sub);
  }

  Eigen::VectorXd ekf_x() const { return ekf_.x; }
  const tools::ExtendedKalmanFilter & ekf() const { return ekf_; }

  std::vector<Eigen::Vector4d> armor_xyza_list() const
  {
    std::vector<Eigen::Vector4d> list;
    for (int i = 0; i < armor_num_; i++) {
      auto angle = tools::limit_rad(ekf_.x[6] + i * 2 * CV_PI / armor_num_);
      Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);
      list.push_back({xyz[0], xyz[1], xyz[2], angle});
    }
    return list;
  }

  bool diverged() const { return ekf_.x[8] < -0.5 || ekf_.x[8] > 1.0; }
  bool convergened() { if (update_count_ > 3 && !diverged()) is_converged_ = true; return is_converged_; }

private:
  int armor_num_, update_count_;
  bool is_converged_;
  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const
  {
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    bool use_l = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l ? x[8]+x[9] : x[8];
    return {x[0]-r*std::cos(angle), x[2]-r*std::sin(angle), use_l ? x[4]+x[10] : x[4]};
  }

  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const
  {
    auto angle = tools::limit_rad(x[6] + id * 2 * CV_PI / armor_num_);
    bool use_l = (armor_num_ == 4) && (id == 1 || id == 3);
    auto r = use_l ? x[8]+x[9] : x[8];
    // clang-format off
    Eigen::MatrixXd Ha{
      {1,0,0,0,0,0, r*std::sin(angle),0,-std::cos(angle), use_l?-std::cos(angle):0.0, 0},
      {0,0,1,0,0,0,-r*std::cos(angle),0,-std::sin(angle), use_l?-std::sin(angle):0.0, 0},
      {0,0,0,0,1,0,0,0,0,0, use_l?1.0:0.0},
      {0,0,0,0,0,0,1,0,0,0,0}
    };
    // clang-format on
    Eigen::MatrixXd Hypd = tools::xyz2ypd_jacobian(h_armor_xyz(x, id));
    // clang-format off
    Eigen::MatrixXd Hypda{
      {Hypd(0,0),Hypd(0,1),Hypd(0,2),0},
      {Hypd(1,0),Hypd(1,1),Hypd(1,2),0},
      {Hypd(2,0),Hypd(2,1),Hypd(2,2),0},
      {0,0,0,1}
    };
    // clang-format on
    return Hypda * Ha;
  }
};

// ============================================================
// ExamTracker: 用原始Solver + ExamTarget
// ============================================================
class ExamTracker
{
public:
  enum class HorizontalMove { Unknown = 0, Left = 1, Right = 2, Still = 3 };

  ExamTracker(const std::string & config_path, auto_aim::Solver & solver)
  : solver_(solver), detect_count_(0), temp_lost_count_(0),
    state_("lost"), last_timestamp_(std::chrono::steady_clock::now()),
    last_center_x_valid_(false), last_center_x_(0.0f), move_dir_(HorizontalMove::Unknown)
  {
    auto yaml = YAML::LoadFile(config_path);
    enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red")
      ? auto_aim::Color::red : auto_aim::Color::blue;
    min_detect_count_ = yaml["min_detect_count"].as<int>();
    max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  }

  std::string state() const { return state_; }
  HorizontalMove move_dir() const { return move_dir_; }

  std::list<ExamTarget> track(
    std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point t)
  {
    auto dt = tools::delta_time(t, last_timestamp_);
    last_timestamp_ = t;
    if (state_ != "lost" && dt > 0.1) { state_ = "lost"; }

    armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });
    armors.sort([](const auto_aim::Armor & a, const auto_aim::Armor & b) {
      cv::Point2f c(1440/2, 1080/2);
      return cv::norm(a.center-c) < cv::norm(b.center-c);
    });
    armors.sort([](const auto_aim::Armor & a, const auto_aim::Armor & b) {
      return a.priority < b.priority;
    });

    bool found = (state_ == "lost") ? set_target(armors, t) : update_target(armors, t);

    if (state_ == "lost") {
      if (found) { state_ = "detecting"; detect_count_ = 1; }
    } else if (state_ == "detecting") {
      if (found) { detect_count_++; if (detect_count_ >= min_detect_count_) state_ = "tracking"; }
      else { detect_count_ = 0; state_ = "lost"; }
    } else if (state_ == "tracking") {
      if (!found) { temp_lost_count_ = 1; state_ = "temp_lost"; }
    } else if (state_ == "temp_lost") {
      if (found) state_ = "tracking";
      else { temp_lost_count_++; if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost"; }
    }

    if (state_ != "lost" && target_.diverged()) { state_ = "lost"; return {}; }
    if (state_ != "lost") {
      auto & nis = target_.ekf().recent_nis_failures;
      if (std::accumulate(nis.begin(), nis.end(), 0) >= (int)(0.4 * target_.ekf().window_size))
        { state_ = "lost"; return {}; }
    }
    if (state_ == "lost") return {};
    return {target_};
  }

private:
  auto_aim::Solver & solver_;
  auto_aim::Color enemy_color_;
  int min_detect_count_, max_temp_lost_count_, detect_count_, temp_lost_count_;
  std::string state_;
  ExamTarget target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  bool last_center_x_valid_;
  float last_center_x_;
  HorizontalMove move_dir_;

  void update_move_dir(float cur_center_x)
  {
    if (!last_center_x_valid_) {
      move_dir_ = HorizontalMove::Unknown;
      last_center_x_ = cur_center_x;
      last_center_x_valid_ = true;
      return;
    }
    float dx = cur_center_x - last_center_x_;
    last_center_x_ = cur_center_x;
    constexpr float STILL_THRESH_PX = 1.5f;
    if (std::abs(dx) <= STILL_THRESH_PX) move_dir_ = HorizontalMove::Still;
    else move_dir_ = (dx < 0) ? HorizontalMove::Left : HorizontalMove::Right;
  }

  bool set_target(std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point t)
  {
    if (armors.empty()) return false;
    auto & armor = armors.front();
    solver_.solve(armor);
    target_ = ExamTarget(armor, t);
    update_move_dir(armor.center.x);
    return true;
  }

  bool update_target(std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point t)
  {
    target_.predict(t);

    // 收集同名同类型的装甲板
    std::vector<auto_aim::Armor *> matched;
    for (auto & armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type) continue;
      matched.push_back(&armor);
    }
    if (matched.empty()) return false;

    if (matched.size() == 1) {
      // 单板：无法判断长短轴，axis_hint=-1
      solver_.solve(*matched[0]);
      target_.update(*matched[0], -1);
      update_move_dir(matched[0]->center.x);
    } else {
      // 双板：图像中y更小（更高）的是短轴，y更大（更低）的是长轴
      for (auto * a : matched) solver_.solve(*a);
      auto * a0 = matched[0];
      auto * a1 = matched[1];
      // center.y 是像素坐标，y小=图像上方=更高=短轴
      bool a0_higher = (a0->center.y < a1->center.y);
      int hint0 = a0_higher ? 0 : 1;  // 0=短轴, 1=长轴
      int hint1 = a0_higher ? 1 : 0;
      target_.update(*a0, hint0);
      target_.update(*a1, hint1);
      // 用优先级更高/更靠近中心的那块的center.x来估计水平平移方向（matched已按优先级/中心排序）
      update_move_dir(matched[0]->center.x);
    }
    return true;
  }
};

// ============================================================
// ExamAimer: 选"弹丸到达时正对"的板瞄准
// ============================================================
struct ExamAimPoint { bool valid; Eigen::Vector4d xyza; double face_angle; int locked_id; };

class ExamAimer
{
public:
  ExamAimPoint debug_aim_point;
  double debug_face_angle = 999;

  explicit ExamAimer(const std::string & config_path)
  {
    auto yaml = YAML::LoadFile(config_path);
    yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
    pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
    low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
    low_speed_delay_time_left_ = yaml["low_speed_delay_time_left"].IsDefined()
      ? yaml["low_speed_delay_time_left"].as<double>() : low_speed_delay_time_;
    low_speed_delay_time_right_ = yaml["low_speed_delay_time_right"].IsDefined()
      ? yaml["low_speed_delay_time_right"].as<double>() : low_speed_delay_time_;
  }

  io::Command aim(std::list<ExamTarget> & targets,
    std::chrono::steady_clock::time_point timestamp, double bullet_speed,
    ExamTracker::HorizontalMove move_dir = ExamTracker::HorizontalMove::Unknown)
  {
    if (targets.empty()) return {false, false, 0, 0};
    auto target = targets.front();

    double delay = low_speed_delay_time_;
    if (move_dir == ExamTracker::HorizontalMove::Left) delay = low_speed_delay_time_left_;
    else if (move_dir == ExamTracker::HorizontalMove::Right) delay = low_speed_delay_time_right_;
    double dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + delay;
    auto future = timestamp + std::chrono::microseconds(int(dt * 1e6));
    target.predict(future);

    auto ap0 = choose_best_plate(target);
    debug_aim_point = ap0;
    if (!ap0.valid) return {false, false, 0, 0};

    // 锁定选中的板id，迭代中不再切换
    int locked_id = ap0.locked_id;

    Eigen::Vector3d xyz0 = ap0.xyza.head(3);
    double d0 = std::sqrt(xyz0[0]*xyz0[0] + xyz0[1]*xyz0[1]);
    tools::Trajectory traj(bullet_speed, d0, xyz0[2]);
    if (traj.unsolvable) { debug_aim_point.valid = false; return {false, false, 0, 0}; }

    double prev_ft = traj.fly_time;
    std::vector<ExamTarget> iters(10, target);
    for (int i = 0; i < 10; ++i) {
      iters[i].predict(future + std::chrono::microseconds((int)(prev_ft*1e6)));
      auto list = iters[i].armor_xyza_list();
      auto xyza = list[locked_id];
      Eigen::Vector3d xyz = xyza.head(3);
      double armor_angle = xyza[3];
      double view_angle = std::atan2(xyz.y(), xyz.x());
      double face = std::abs(tools::limit_rad(armor_angle - view_angle));
      debug_aim_point = {true, xyza, face, locked_id};

      double d = std::sqrt(xyz.x()*xyz.x() + xyz.y()*xyz.y());
      traj = tools::Trajectory(bullet_speed, d, xyz.z());
      if (traj.unsolvable) { debug_aim_point.valid = false; return {false, false, 0, 0}; }
      if (std::abs(traj.fly_time - prev_ft) < 0.001) break;
      prev_ft = traj.fly_time;
    }

    debug_face_angle = debug_aim_point.face_angle;

    Eigen::Vector3d fxyz = debug_aim_point.xyza.head(3);
    return {true, false, std::atan2(fxyz.y(), fxyz.x()) + yaw_offset_, traj.pitch + pitch_offset_};
  }

private:
  double yaw_offset_, pitch_offset_, low_speed_delay_time_;
  double low_speed_delay_time_left_, low_speed_delay_time_right_;

  ExamAimPoint choose_best_plate(const ExamTarget & target)
  {
    auto list = target.armor_xyza_list();
    int n = (int)list.size();

    int best = -1;
    double best_face = 1e10;
    for (int i = 0; i < n; i++) {
      Eigen::Vector3d xyz = list[i].head(3);
      double armor_angle = list[i][3];
      double view_angle = std::atan2(xyz.y(), xyz.x());
      double diff = tools::limit_rad(armor_angle - view_angle);
      double face = std::abs(diff);

      // 敌方始终逆时针旋转，只选即将从左边转过来的板(diff<0)
      // 或者已经非常正对的板(face<15°)
      if (diff > 0 && face > 15.0 / 57.3) continue;

      if (face < best_face) { best_face = face; best = i; }
    }

    if (best >= 0 && best_face < 60.0 / 57.3) return {true, list[best], best_face, best};
    if (best >= 0) return {true, list[best], best_face, best};
    return {false, list[0], 999, 0};
  }
};

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  auto enable_record = cli.get<bool>("record");
  auto viz = cli.get<bool>("viz");
  auto headless = cli.get<bool>("headless");
  if (cli.has("help") || !cli.has("@config-path")) { cli.printMessage(); return 0; }
  bool enable_viz = viz && !headless;

  tools::Exiter exiter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  ExamTracker tracker(config_path, solver);
  ExamAimer aimer(config_path);
  tools::Plotter plotter;
  auto yaml_cfg = YAML::LoadFile(config_path);
  double default_bullet_speed = yaml_cfg["default_bullet_speed"].IsDefined()
    ? yaml_cfg["default_bullet_speed"].as<double>() : 19.7;
  std::string fire_mode = yaml_cfg["fire_mode"].IsDefined()
    ? yaml_cfg["fire_mode"].as<std::string>() : "single";
  bool burst_mode = (fire_mode == "burst");
  bool no_fire = (fire_mode == "none");
  tools::logger()->info("fire_mode: {} ({})", fire_mode,
    no_fire ? "不射击" : (burst_mode ? "泼水连发" : "单发精准"));
  std::unique_ptr<tools::Recorder> recorder;
  if (enable_record) {
    recorder = std::make_unique<tools::Recorder>();
    tools::logger()->info("Recording enabled");
  }
  auto t0 = std::chrono::steady_clock::now();

  cv::Mat img;
  std::chrono::steady_clock::time_point t;
  Eigen::Quaterniond gimbal_q;
  io::GimbalState gimbal_state;
  io::Command last_command;
  int frame_count = 0;
  std::deque<float> bullet_speed_history;
  uint16_t last_bullet_count = 0;

  int shoot_count = 0, stable_count = 0;
  auto last_shoot_time = std::chrono::steady_clock::now() - std::chrono::seconds(5);
  bool burst_fire_enabled = true;  // burst模式下空格切换开火

  tools::logger()->info("===== 完整形态考核模式(无计时) =====");
  tools::logger()->info("r={:.3f}m, dr={:.3f}m (定死), w0={:.2f}rad/s, v1={:.1f}, v2={:.2f}",
    exam::R_SHORT, exam::DR, exam::ROTATE_W, exam::V1, exam::V2);
  tools::logger()->info("visualization: {}", enable_viz ? "ON" : "OFF");

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;

    // FPS：每秒打印一次（与可视化开关无关）
    {
      static auto fps_last_time = std::chrono::steady_clock::now();
      static int fps_frames = 0;
      fps_frames++;
      auto now = std::chrono::steady_clock::now();
      double fps_dt = tools::delta_time(now, fps_last_time);
      if (fps_dt >= 1.0) {
        tools::logger()->info("[FPS] {:.1f}", fps_frames / fps_dt);
        fps_frames = 0;
        fps_last_time = now;
      }
    }

    gimbal_q = gimbal.q(t);
    gimbal_state = gimbal.state();
    solver.set_R_gimbal2world(gimbal_q);

    // 录制原始图像（在任何绘制之前）
    if (recorder) recorder->record(img, gimbal_q, t);

    auto yolo_start = std::chrono::steady_clock::now();
    (void)yolo_start;
    auto armors = yolo.detect(img, frame_count);

    // 强制所有装甲板为红色三号小装甲板
    for (auto & a : armors) {
      a.color = auto_aim::Color::red;
      a.name = auto_aim::ArmorName::three;
      a.type = auto_aim::ArmorType::small;
    }

    auto tracker_start = std::chrono::steady_clock::now();
    (void)tracker_start;
    auto targets = tracker.track(armors, t);
    auto aimer_start = std::chrono::steady_clock::now();
    (void)aimer_start;

    // 弹速：使用下位机反馈，但过滤过高/过低的异常值
    {
      constexpr double BULLET_SPEED_MIN = 10.0;    // m/s，过低视为异常
      constexpr double BULLET_SPEED_MAX = 20.0;   // m/s，过高视为异常
      constexpr double BULLET_SPEED_JUMP_MAX = 5.0;  // m/s，相对历史均值的最大跳变

      if (gimbal_state.bullet_count != last_bullet_count) {
        double bs = gimbal_state.bullet_speed;
        bool in_range = std::isfinite(bs) && (bs >= BULLET_SPEED_MIN) && (bs <= BULLET_SPEED_MAX);

        bool ok = in_range;
        if (ok && !bullet_speed_history.empty()) {
          double avg = std::accumulate(bullet_speed_history.begin(), bullet_speed_history.end(), 0.0) /
                       bullet_speed_history.size();
          if (std::abs(bs - avg) > BULLET_SPEED_JUMP_MAX) ok = false;
        }

        if (ok) {
          bullet_speed_history.push_back((float)bs);
          if (bullet_speed_history.size() > 5) bullet_speed_history.pop_front();
        }
        last_bullet_count = gimbal_state.bullet_count;
      }
    }

    double bullet_speed = default_bullet_speed;
    if (!bullet_speed_history.empty())
      bullet_speed = std::accumulate(bullet_speed_history.begin(), bullet_speed_history.end(), 0.0f) /
                     bullet_speed_history.size();
    auto command = aimer.aim(targets, t, bullet_speed, tracker.move_dir());

    // 射击判断
    command.shoot = false;
    if (!no_fire && !targets.empty() && aimer.debug_aim_point.valid && command.control) {
      double ye = std::abs(command.yaw - last_command.yaw);
      double pe = std::abs(command.pitch - last_command.pitch);
      double fa = aimer.debug_face_angle;  // 板正对偏差(rad)

      bool facing = fa < 30.0 / 57.3 || fa > 100;
      auto now = std::chrono::steady_clock::now();
      double since = tools::delta_time(now, last_shoot_time);

      if (burst_mode) {
        // 泼水模式：板正对 + 空格切换开火
        if (facing && burst_fire_enabled) {
          command.shoot = true;
          shoot_count++;
          last_shoot_time = now;
          tools::logger()->info("BURST #{} | face:{:.1f}d", shoot_count, fa * 57.3);
        }
      } else {
        // 单发模式：需要稳定帧数 + cooldown
        bool stable = ye < exam::SHOOT_YAW_THRESH && pe < exam::SHOOT_PITCH_THRESH;
        if (stable && facing) stable_count++;
        else stable_count = 0;

        if (stable_count >= exam::SHOOT_STABLE_FRAMES && since >= exam::SHOOT_COOLDOWN_SEC) {
          command.shoot = true;
          shoot_count++;
          last_shoot_time = now;
          stable_count = 0;
          tools::logger()->info("SHOOT #{} | face:{:.1f}d ye:{:.3f}d pe:{:.3f}d",
            shoot_count, fa * 57.3, ye * 57.3, pe * 57.3);
        }
      }
    } else { stable_count = 0; }

    if (command.control) last_command = command;
    gimbal.send(command.control, command.shoot, command.yaw, 0.f, 0.f, command.pitch, 0.f, 0.f);

    if (enable_viz) {
      tools::logger()->info("[Cmd] c:{} s:{} y:{:.2f}d p:{:.2f}d | {}/{}",
        command.control, command.shoot, command.yaw*57.3, command.pitch*57.3,
        shoot_count, "∞");
    } else {
      static auto last_log_time = std::chrono::steady_clock::now();
      static int headless_frames = 0;
      headless_frames++;
      auto now = std::chrono::steady_clock::now();
      double log_dt = tools::delta_time(now, last_log_time);
      if (log_dt >= 1.0) {
        tools::logger()->info("[HL] FPS:{:.0f} bs:{:.1f} avg:{:.1f} c:{} shoot:{}/{} mode:{}",
          headless_frames / log_dt, gimbal_state.bullet_speed, bullet_speed,
          command.control, shoot_count, "∞", fire_mode);
        headless_frames = 0;
        last_log_time = now;
      }
    }

    // 可视化
    if (enable_viz) {
      tools::draw_text(img, fmt::format("EXAM shoot:{} count:{}",
        command.shoot, shoot_count), {10,60}, {0,255,255});

      static auto last_time = std::chrono::steady_clock::now();
      auto cur = std::chrono::steady_clock::now();
      auto ddt = tools::delta_time(cur, last_time); last_time = cur;
      if (ddt > 0) tools::draw_text(img, fmt::format("FPS:{:.1f}", 1.0/ddt), {10,30}, {0,255,0});
      tools::draw_text(img, fmt::format("bs:{:.1f} avg:{:.1f}", gimbal_state.bullet_speed, bullet_speed), {10,90}, {0,200,255});

      if (!targets.empty()) {
        auto tgt = targets.front();
        for (const auto & xyza : tgt.armor_xyza_list())
          tools::draw_points(img, solver.reproject_armor(xyza.head(3), xyza[3], tgt.armor_type, tgt.name), {0,255,0});
        if (aimer.debug_aim_point.valid) {
          auto ap = aimer.debug_aim_point.xyza;
          tools::draw_points(img, solver.reproject_armor(ap.head(3), ap[3], tgt.armor_type, tgt.name), {0,0,255});
        }
        auto ex = tgt.ekf_x();
        tools::draw_text(img, fmt::format("vx:{:.2f} vy:{:.2f} w:{:.3f} r:{:.3f} face:{:.1f}d",
          ex[1], ex[3], ex[7], ex[8], aimer.debug_face_angle*57.3), {10,120}, {200,200,255});
      }

      cv::imshow("hero_test", img);
      int key = cv::waitKey(1);
      if (key == ' ' && burst_mode) {
        burst_fire_enabled = !burst_fire_enabled;
        tools::logger()->info("burst fire: {}", burst_fire_enabled ? "ON" : "OFF");
      }
    } else if (burst_mode) {
      // 非可视化模式下非阻塞检测空格（用于burst模式切换开火）
      fd_set fds;
      FD_ZERO(&fds);
      FD_SET(STDIN_FILENO, &fds);
      struct timeval tv = {0, 0};
      if (select(STDIN_FILENO + 1, &fds, nullptr, nullptr, &tv) > 0) {
        char ch;
        if (read(STDIN_FILENO, &ch, 1) == 1 && ch == ' ') {
          burst_fire_enabled = !burst_fire_enabled;
          tools::logger()->info("burst fire: {}", burst_fire_enabled ? "ON" : "OFF");
        }
      }
    }

    // plotter
    {
      nlohmann::json pdata;
      pdata["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      pdata["cmd_yaw"] = command.yaw * 57.3;
      pdata["cmd_pitch"] = command.pitch * 57.3;
      pdata["gimbal_yaw"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0];
      pdata["gimbal_pitch"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[1];
      pdata["control"] = command.control;
      pdata["shoot"] = command.shoot;
      pdata["bullet_speed"] = gimbal_state.bullet_speed;
      plotter.plot(pdata);
    }

    frame_count++;
  }

  tools::logger()->info("===== 结束: count:{} =====", shoot_count);
  return 0;
}

