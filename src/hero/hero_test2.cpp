// 完整形态考核专用自瞄链路 v2
// yaw锁定旋转中心，预测装甲板位置，三点一线时开火
// 绿框=当前4板位置，红框=预测4板位置（弹丸到达时）
#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <chrono>
#include <cmath>
#include <deque>
#include <list>
#include <nlohmann/json.hpp>
#include <numeric>
#include <opencv2/opencv.hpp>

#include "io/camera/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/yolos/yolo.hpp"
#include "tools/exiter/exiter.hpp"
#include "tools/extended_kalman_filter/extended_kalman_filter.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"
#include "tools/trajectory/trajectory.hpp"
#include "tools/plotter/plotter.hpp"
#include "tools/recorder/recorder.hpp"
#include "io/cboard/command.hpp"

const std::string keys =
  "{help h usage ? | | 输出命令行参数说明}"
  "{@config-path   | configs/standard.yaml | yaml配置文件路径 }"
  "{record         | false                 | 开启录制         }";

using namespace std::chrono_literals;

namespace exam2
{
constexpr double R_SHORT = 0.23;
constexpr double R_LONG = 0.28;
constexpr double DR = R_LONG - R_SHORT;
constexpr double DH = -0.035;

constexpr double TRANSLATE_SPEED = 1.0;
constexpr double ROTATE_RPS = 0.45;
constexpr double ROTATE_W = ROTATE_RPS * 2 * M_PI;

// 射击控制
constexpr double SHOOT_COOLDOWN_SEC = 1.0;
constexpr double SHOOT_INLINE_THRESH = 3.0 / 57.3;  // 三点一线阈值(rad)
constexpr double SHOOT_FACE_THRESH = 30.0 / 57.3;    // 板正对阈值(rad)
constexpr int MAX_BULLETS = 50;

// EKF参数
constexpr double V1 = 20;
constexpr double V2 = 1.0;
constexpr double VZ = 0.01;
}  // namespace exam2

// ============================================================
// ExamTarget2: 与hero_test的ExamTarget完全相同
// ============================================================
class ExamTarget2
{
public:
  auto_aim::ArmorName name;
  auto_aim::ArmorType armor_type;
  auto_aim::ArmorPriority priority;
  bool jumped;
  int last_id;

  ExamTarget2() = default;

  ExamTarget2(const auto_aim::Armor & armor, std::chrono::steady_clock::time_point t)
  : name(armor.name), armor_type(armor.type), jumped(false), last_id(0),
    update_count_(0), armor_num_(4), t_(t), is_converged_(false)
  {
    double r = exam2::R_SHORT;
    priority = armor.priority;
    const Eigen::VectorXd & xyz = armor.xyz_in_world;
    const Eigen::VectorXd & ypr = armor.ypr_in_world;

    auto center_x = xyz[0] + r * std::cos(ypr[0]);
    auto center_y = xyz[1] + r * std::sin(ypr[0]);
    auto center_z = xyz[2];

    Eigen::VectorXd x0{{center_x, 0, center_y, 0, center_z, 0, ypr[0],
                         exam2::ROTATE_W, exam2::R_SHORT, exam2::DR, exam2::DH}};

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
    double v1 = exam2::V1, v2 = exam2::V2, vz = exam2::VZ;
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

  void update(const auto_aim::Armor & armor, int axis_hint = -1)
  {
    int id = 0;
    auto min_angle_error = 1e10;
    const auto & xyza_list = armor_xyza_list();
    std::vector<std::pair<Eigen::Vector4d, int>> xyza_i_list;
    for (int i = 0; i < armor_num_; i++) {
      if (axis_hint == 0 && (i == 1 || i == 3)) continue;
      if (axis_hint == 1 && (i == 0 || i == 2)) continue;
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
// ExamTracker2: 与hero_test的ExamTracker完全相同
// ============================================================
class ExamTracker2
{
public:
  ExamTracker2(const std::string & config_path, auto_aim::Solver & solver)
  : solver_(solver), detect_count_(0), temp_lost_count_(0),
    state_("lost"), last_timestamp_(std::chrono::steady_clock::now())
  {
    auto yaml = YAML::LoadFile(config_path);
    enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red")
      ? auto_aim::Color::red : auto_aim::Color::blue;
    min_detect_count_ = yaml["min_detect_count"].as<int>();
    max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  }

  std::string state() const { return state_; }

  std::list<ExamTarget2> track(
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
  ExamTarget2 target_;
  std::chrono::steady_clock::time_point last_timestamp_;

  bool set_target(std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point t)
  {
    if (armors.empty()) return false;
    auto & armor = armors.front();
    solver_.solve(armor);
    target_ = ExamTarget2(armor, t);
    return true;
  }

  bool update_target(std::list<auto_aim::Armor> & armors, std::chrono::steady_clock::time_point t)
  {
    target_.predict(t);
    std::vector<auto_aim::Armor *> matched;
    for (auto & armor : armors) {
      if (armor.name != target_.name || armor.type != target_.armor_type) continue;
      matched.push_back(&armor);
    }
    if (matched.empty()) return false;

    if (matched.size() == 1) {
      solver_.solve(*matched[0]);
      target_.update(*matched[0], -1);
    } else {
      for (auto * a : matched) solver_.solve(*a);
      auto * a0 = matched[0];
      auto * a1 = matched[1];
      bool a0_higher = (a0->center.y < a1->center.y);
      int hint0 = a0_higher ? 0 : 1;
      int hint1 = a0_higher ? 1 : 0;
      target_.update(*a0, hint0);
      target_.update(*a1, hint1);
    }
    return true;
  }
};

// ============================================================
// ExamAimer2: yaw锁定旋转中心，预测装甲板三点一线时开火
// ============================================================
struct AimResult {
  bool control;
  double yaw, pitch;
  // 预测的4板位置（弹丸到达时刻）
  std::vector<Eigen::Vector4d> predicted_plates;
  double fly_time;
  int best_plate;       // 最接近三点一线的板id
  double inline_angle;  // 该板与中心连线的yaw偏差(rad)
  double face_angle;    // 该板的正对角(rad)
  bool should_shoot;
};

class ExamAimer2
{
public:
  AimResult last_result;

  explicit ExamAimer2(const std::string & config_path)
  {
    auto yaml = YAML::LoadFile(config_path);
    yaw_offset_ = yaml["yaw_offset"].as<double>() / 57.3;
    pitch_offset_ = yaml["pitch_offset"].as<double>() / 57.3;
    low_speed_delay_time_ = yaml["low_speed_delay_time"].as<double>();
  }

  AimResult aim(std::list<ExamTarget2> & targets,
    std::chrono::steady_clock::time_point timestamp, double bullet_speed)
  {
    AimResult res{};
    if (targets.empty()) return res;
    auto target = targets.front();
    auto ex = target.ekf_x();

    // yaw锁定旋转中心
    double center_x = ex[0], center_y = ex[2], center_z = ex[4];
    double center_yaw = std::atan2(center_y, center_x);

    // 用旋转中心距离做初始弹道估算
    double center_d = std::sqrt(center_x*center_x + center_y*center_y);
    tools::Trajectory traj0(bullet_speed, center_d, center_z);
    if (traj0.unsolvable) return res;

    // 延迟补偿
    double dt = tools::delta_time(std::chrono::steady_clock::now(), timestamp) + low_speed_delay_time_;
    auto future = timestamp + std::chrono::microseconds(int(dt * 1e6));

    // 迭代求解：用最接近三点一线的板的距离来收敛fly_time
    double prev_ft = traj0.fly_time;
    std::vector<Eigen::Vector4d> pred_plates;
    int best_id = 0;
    double best_inline = 1e10, best_face = 1e10;
    tools::Trajectory traj = traj0;

    for (int iter = 0; iter < 10; ++iter) {
      ExamTarget2 pred = target;
      auto fut_t = future + std::chrono::microseconds((int)(prev_ft*1e6));
      pred.predict(fut_t);
      auto pex = pred.ekf_x();
      pred_plates = pred.armor_xyza_list();

      // 预测时刻的旋转中心
      double pc_x = pex[0], pc_y = pex[2];
      double pc_yaw = std::atan2(pc_y, pc_x);

      // 找最正对相机的板（face_angle最小 = 板朝向与视线方向偏差最小）
      best_id = 0; best_inline = 1e10; best_face = 1e10;
      for (int i = 0; i < (int)pred_plates.size(); i++) {
        Eigen::Vector3d xyz = pred_plates[i].head(3);
        double plate_yaw = std::atan2(xyz.y(), xyz.x());
        double inline_err = std::abs(tools::limit_rad(plate_yaw - pc_yaw));
        double armor_angle = pred_plates[i][3];
        // 板正对相机时，armor_angle 与 从板看向相机的方向(plate_yaw + PI) 差值为0
        double face = std::abs(tools::limit_rad(armor_angle - (plate_yaw + M_PI)));

        if (face < best_face) {
          best_face = face;
          best_inline = inline_err;
          best_id = i;
        }
      }

      // 用该板的实际距离和高度做弹道解算
      Eigen::Vector3d best_xyz = pred_plates[best_id].head(3);
      double d = std::sqrt(best_xyz.x()*best_xyz.x() + best_xyz.y()*best_xyz.y());
      traj = tools::Trajectory(bullet_speed, d, best_xyz.z());
      if (traj.unsolvable) return res;
      if (std::abs(traj.fly_time - prev_ft) < 0.001) break;
      prev_ft = traj.fly_time;
    }

    // 判断是否应该开火：板正对相机即可
    bool should_shoot = (best_face < exam2::SHOOT_FACE_THRESH);

    res.control = true;
    res.yaw = center_yaw + yaw_offset_;
    res.pitch = traj.pitch + pitch_offset_;
    res.predicted_plates = pred_plates;
    res.fly_time = traj.fly_time;
    res.best_plate = best_id;
    res.inline_angle = best_inline;
    res.face_angle = best_face;
    res.should_shoot = should_shoot;
    last_result = res;
    return res;
  }

private:
  double yaw_offset_, pitch_offset_, low_speed_delay_time_;
};

// ============================================================
// main
// ============================================================
int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  auto config_path = cli.get<std::string>("@config-path");
  auto enable_record = cli.get<bool>("record");
  if (cli.has("help") || !cli.has("@config-path")) { cli.printMessage(); return 0; }

  tools::Exiter exiter;
  io::Camera camera(config_path);
  io::Gimbal gimbal(config_path);
  auto_aim::YOLO yolo(config_path);
  auto_aim::Solver solver(config_path);
  ExamTracker2 tracker(config_path, solver);
  ExamAimer2 aimer(config_path);
  tools::Plotter plotter;
  auto yaml_cfg = YAML::LoadFile(config_path);
  double default_bullet_speed = yaml_cfg["default_bullet_speed"].IsDefined()
    ? yaml_cfg["default_bullet_speed"].as<double>() : 19.7;
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
  int frame_count = 0;
  std::deque<float> bullet_speed_history;
  uint16_t last_bullet_count = 0;

  int shoot_count = 0;
  auto last_shoot_time = std::chrono::steady_clock::now() - std::chrono::seconds(5);
  auto exam_start_time = std::chrono::steady_clock::now();
  bool exam_started = false;

  tools::logger()->info("===== 完整形态考核v2: yaw锁中心 =====");

  while (!exiter.exit()) {
    camera.read(img, t);
    if (img.empty()) continue;
    gimbal_q = gimbal.q(t);
    gimbal_state = gimbal.state();
    solver.set_R_gimbal2world(gimbal_q);

    if (recorder) recorder->record(img, gimbal_q, t, &gimbal_state);

    auto armors = yolo.detect(img, frame_count);
    for (auto & a : armors) {
      a.color = auto_aim::Color::red;
      a.name = auto_aim::ArmorName::three;
      a.type = auto_aim::ArmorType::small;
    }
    auto targets = tracker.track(armors, t);
    if (gimbal_state.bullet_speed >= 14 && gimbal_state.bullet_count != last_bullet_count) {
      bullet_speed_history.push_back(gimbal_state.bullet_speed);
      if (bullet_speed_history.size() > 5) bullet_speed_history.pop_front();
      last_bullet_count = gimbal_state.bullet_count;
    }
    double bullet_speed = default_bullet_speed;
    if (!bullet_speed_history.empty())
      bullet_speed = std::accumulate(bullet_speed_history.begin(), bullet_speed_history.end(), 0.0f) / bullet_speed_history.size();
    auto res = aimer.aim(targets, t, bullet_speed);

    // 射击判断
    bool shoot = false;
    if (res.control && res.should_shoot) {
      if (!exam_started) {
        exam_started = true;
        exam_start_time = std::chrono::steady_clock::now();
        tools::logger()->info("===== 考核计时开始 =====");
      }
      auto now = std::chrono::steady_clock::now();
      double since = tools::delta_time(now, last_shoot_time);
      double elapsed = tools::delta_time(now, exam_start_time);
      if (since >= exam2::SHOOT_COOLDOWN_SEC && elapsed < 60.0) {
        shoot = true;
        shoot_count++;
        last_shoot_time = now;
        tools::logger()->info("SHOOT #{} | {:.1f}s | inline:{:.1f}d face:{:.1f}d ft:{:.3f}s",
          shoot_count, elapsed, res.inline_angle*57.3, res.face_angle*57.3, res.fly_time);
        if (shoot_count >= exam2::MAX_BULLETS)
          tools::logger()->info("===== 50发打完 {:.1f}s =====", elapsed);
      }
    }

    gimbal.send(res.control, shoot, res.yaw, 0.f, 0.f, res.pitch, 0.f, 0.f);

    tools::logger()->info("[v2] c:{} s:{} y:{:.2f}d p:{:.2f}d il:{:.1f}d fa:{:.1f}d | {}/{}",
      res.control, shoot, res.yaw*57.3, res.pitch*57.3,
      res.inline_angle*57.3, res.face_angle*57.3, shoot_count, exam2::MAX_BULLETS);

    // 可视化
    static auto last_time = std::chrono::steady_clock::now();
    auto cur = std::chrono::steady_clock::now();
    auto ddt = tools::delta_time(cur, last_time); last_time = cur;
    if (ddt > 0) tools::draw_text(img, fmt::format("FPS:{:.1f}", 1.0/ddt), {10,30}, {0,255,0});
    tools::draw_text(img, fmt::format("bs:{:.1f} avg:{:.1f}", gimbal_state.bullet_speed, bullet_speed), {10,60}, {0,200,255});
    tools::draw_text(img, fmt::format("v2 shoot: {}/{}",
      shoot, shoot_count, exam2::MAX_BULLETS), {10,90}, {0,255,255});

    if (!targets.empty()) {
      auto tgt = targets.front();
      // 绿框：当前4板位置
      for (const auto & xyza : tgt.armor_xyza_list())
        tools::draw_points(img, solver.reproject_armor(xyza.head(3), xyza[3], tgt.armor_type, tgt.name), {0,255,0});
      // 红框：预测4板位置（弹丸到达时刻）
      for (const auto & xyza : res.predicted_plates)
        tools::draw_points(img, solver.reproject_armor(xyza.head(3), xyza[3], tgt.armor_type, tgt.name), {0,0,255});

      auto ex = tgt.ekf_x();
      tools::draw_text(img, fmt::format("vx:{:.2f} vy:{:.2f} w:{:.3f} r:{:.3f}",
        ex[1], ex[3], ex[7], ex[8]), {10,120}, {200,200,255});
      tools::draw_text(img, fmt::format("inline:{:.1f}d face:{:.1f}d ft:{:.3f}s best:{}",
        res.inline_angle*57.3, res.face_angle*57.3, res.fly_time, res.best_plate), {10,150}, {200,200,255});
    }

    // plotter
    {
      nlohmann::json pdata;
      pdata["t"] = tools::delta_time(std::chrono::steady_clock::now(), t0);
      pdata["cmd_yaw"] = res.yaw * 57.3;
      pdata["cmd_pitch"] = res.pitch * 57.3;
      pdata["gimbal_yaw"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[0];
      pdata["gimbal_pitch"] = (tools::eulers(gimbal_q.toRotationMatrix(), 2, 1, 0) * 57.3)[1];
      pdata["control"] = res.control;
      pdata["shoot"] = shoot;
      pdata["bullet_speed"] = gimbal_state.bullet_speed;
      pdata["inline_angle"] = res.inline_angle * 57.3;
      pdata["face_angle"] = res.face_angle * 57.3;
      plotter.plot(pdata);
    }

    cv::imshow("hero_test2", img);
    cv::waitKey(1);
    frame_count++;
  }

  if (exam_started) {
    tools::logger()->info("===== v2考核结束: {}/{}, {:.1f}s =====",
      shoot_count, exam2::MAX_BULLETS,
      tools::delta_time(std::chrono::steady_clock::now(), exam_start_time));
  }
  return 0;
}
