#include "tracker.hpp"

#include <yaml-cpp/yaml.h>

#include <tuple>

#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace auto_aim
{
/**
 * @brief 构造函数：加载配置并初始化追踪器参数
 * @param config_path YAML 配置文件路径
 * @param solver 装甲板位姿解算器，用于后续目标初始化与更新
 * @details 读取敌方颜色、状态机计数阈值等参数，并初始化内部状态与时间戳。
 */
Tracker::Tracker(const std::string & config_path, Solver & solver)
: solver_{solver},
  detect_count_(0),
  temp_lost_count_(0),
  state_{"lost"},
  pre_state_{"lost"},
  last_timestamp_(std::chrono::steady_clock::now()),
  omni_target_priority_{ArmorPriority::fifth}
{
  auto yaml = YAML::LoadFile(config_path);
  enemy_color_ = (yaml["enemy_color"].as<std::string>() == "red") ? Color::red : Color::blue;
  min_detect_count_ = yaml["min_detect_count"].as<int>();
  max_temp_lost_count_ = yaml["max_temp_lost_count"].as<int>();
  outpost_max_temp_lost_count_ = yaml["outpost_max_temp_lost_count"].as<int>();
  normal_temp_lost_count_ = max_temp_lost_count_;
}

/**
 * @brief 获取当前追踪器状态
 * @return 状态字符串（如 "lost"、"detecting"、"tracking" 等）
 */
std::string Tracker::state() const { return state_; }

/**
 * @brief 主跟踪接口（单相机版本）
 * @param armors 当前帧检测到的装甲板列表（传入时可被筛选或排序）
 * @param t 当前帧时间戳
 * @param use_enemy_color 是否使用敌方颜色过滤（暂未使用）
 * @return 当前有效目标列表（通常只返回一个 Target，若无有效目标则为空）
 * @details 完成时间步长计算、装甲板过滤与排序、目标选择/更新以及状态机驱动，
 *          同时进行发散检测与收敛性检测，状态为 lost 时返回空列表。
 */
std::list<Target> Tracker::track(
  std::list<Armor> & armors, std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  // 计算与上一帧的时间间隔，用于后续状态预测和异常检测
  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 若时间间隔超过 0.1 秒且当前不在 lost 状态，判定为相机离线，强制进入 lost 状态
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 过滤：仅保留敌方颜色的装甲板（根据配置的 enemy_color_）
  armors.remove_if([&](const auto_aim::Armor & a) { return a.color != enemy_color_; });

  // 通过重投影误差判断前哨站装甲板是否为顶部装甲板，过滤掉顶部装甲板
  // armors.remove_if([this](const auto_aim::Armor & a) {
  //   return a.name == ArmorName::outpost &&
  //          solver_.oupost_reprojection_error(a, 27.5 * CV_PI / 180.0) <
  //            solver_.oupost_reprojection_error(a, -15 * CV_PI / 180.0);
  // });

  // 排序1：按距离图像中心的距离升序排列（距离中心越近越靠前）
  // 优先选择靠近图像中心的装甲板，提高追踪稳定性
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO: 应从配置或相机参数获取
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 排序2：按优先级升序排列（优先级数值越小优先级越高，1 为最高优先级）
  // 确保优先级最高的装甲板位于列表首位，作为首选追踪目标
  armors.sort(
    [](const auto_aim::Armor & a, const auto_aim::Armor & b) { return a.priority < b.priority; });

  // ========== 目标选择/更新 ==========
  // 根据当前状态决定是创建新目标还是更新已有目标
  bool found;
  if (state_ == "lost") {
    // lost 状态：尝试从装甲板列表中创建新目标
    found = set_target(armors, t);
  } else {
    // 其他状态：尝试更新当前追踪目标
    found = update_target(armors, t);
  }

  // ========== 状态机更新 ==========
  // 根据 found 标志更新追踪器状态（lost / detecting / tracking / temp_lost）
  state_machine(found);

  // ========== 目标质量检测 ==========
  // 检测1：发散检测 - 检查 EKF 滤波器是否发散（协方差过大）
  // 若目标发散，说明状态估计不可靠，重置为 lost 状态
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {};
  }

  // 检测2：收敛性检测 - 检查 EKF 的 NIS（归一化创新平方）失败率
  // 若最近窗口内 NIS 失败次数超过 40%，说明滤波器收敛效果差，重置为 lost 状态
  if (
    std::accumulate(
      target_.ekf().recent_nis_failures.begin(), target_.ekf().recent_nis_failures.end(), 0) >=
    (0.4 * target_.ekf().window_size)) {
    tools::logger()->debug("[Target] Bad Converge Found!");
    state_ = "lost";
    return {};
  }

  // ========== 返回结果 ==========
  // 若最终状态为 lost，返回空列表（无有效目标）
  if (state_ == "lost") return {};

  // 否则返回包含当前目标的目标列表（通常只有一个目标）
  std::list<Target> targets = {target_};
  return targets;
}

/**
 * @brief 主跟踪接口（双相机/全向感知版本）
 * @param detection_queue 全向感知相机输出的检测结果队列（按时间顺序）
 * @param armors 主相机当前帧检测到的装甲板列表
 * @param t 当前帧时间戳
 * @param use_enemy_color 是否使用敌方颜色过滤（暂未使用）
 * @return 一个二元组：{可能的切换目标信息, 当前有效 Target 列表}
 * @details 在单相机逻辑基础上，额外考虑全向感知相机中更高优先级装甲板，
 *          支持从 tracking 切换到 switching 再回到 detecting 的状态迁移。
 */
std::tuple<omniperception::DetectionResult, std::list<Target>> Tracker::track(
  const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
  std::chrono::steady_clock::time_point t, bool use_enemy_color)
{
  omniperception::DetectionResult switch_target{std::list<Armor>(), t, 0, 0};
  omniperception::DetectionResult temp_target{std::list<Armor>(), t, 0, 0};
  if (!detection_queue.empty()) {
    temp_target = detection_queue.front();
  }

  auto dt = tools::delta_time(t, last_timestamp_);
  last_timestamp_ = t;

  // 时间间隔过长，说明可能发生了相机离线
  if (state_ != "lost" && dt > 0.1) {
    tools::logger()->warn("[Tracker] Large dt: {:.3f}s", dt);
    state_ = "lost";
  }

  // 优先选择靠近图像中心的装甲板
  armors.sort([](const Armor & a, const Armor & b) {
    cv::Point2f img_center(1440 / 2, 1080 / 2);  // TODO
    auto distance_1 = cv::norm(a.center - img_center);
    auto distance_2 = cv::norm(b.center - img_center);
    return distance_1 < distance_2;
  });

  // 按优先级排序，优先级最高在首位(优先级越高数字越小，1的优先级最高)
  armors.sort([](const Armor & a, const Armor & b) { return a.priority < b.priority; });

  bool found;
  if (state_ == "lost") {
    found = set_target(armors, t);
  }

  // 此时主相机画面中出现了优先级更高的装甲板，切换目标
  else if (state_ == "tracking" && !armors.empty() && armors.front().priority < target_.priority) {
    found = set_target(armors, t);
    tools::logger()->debug("auto_aim switch target to {}", ARMOR_NAMES[armors.front().name]);
  }

  // 此时全向感知相机画面中出现了优先级更高的装甲板，切换目标
  else if (
    state_ == "tracking" && !temp_target.armors.empty() &&
    temp_target.armors.front().priority < target_.priority && target_.convergened()) {
    state_ = "switching";
    switch_target = omniperception::DetectionResult{
      temp_target.armors, t, temp_target.delta_yaw, temp_target.delta_pitch};
    omni_target_priority_ = temp_target.armors.front().priority;
    found = false;
    tools::logger()->debug("omniperception find higher priority target");
  }

  else if (state_ == "switching") {
    found = !armors.empty() && armors.front().priority == omni_target_priority_;
  }

  else if (state_ == "detecting" && pre_state_ == "switching") {
    found = set_target(armors, t);
  }

  else {
    found = update_target(armors, t);
  }

  pre_state_ = state_;
  // 更新状态机
  state_machine(found);

  // 发散检测
  if (state_ != "lost" && target_.diverged()) {
    tools::logger()->debug("[Tracker] Target diverged!");
    state_ = "lost";
    return {switch_target, {}};  // 返回switch_target和空的targets
  }

  if (state_ == "lost") return {switch_target, {}};  // 返回switch_target和空的targets

  std::list<Target> targets = {target_};
  return {switch_target, targets};
}

/**
 * @brief 状态机迁移逻辑
 * @param found 本帧是否成功找到/更新目标
 * @details 在 lost / detecting / tracking / temp_lost / switching 之间进行状态转换，
 *          使用 detect_count_ 与 temp_lost_count_ 作为计数器，并对前哨站设置独立的丢失阈值。
 */
void Tracker::state_machine(bool found)
{
  if (state_ == "lost") {
    if (!found) return;

    state_ = "detecting";
    detect_count_ = 1;
  }

  else if (state_ == "detecting") {
    if (found) {
      detect_count_++;
      if (detect_count_ >= min_detect_count_) state_ = "tracking";
    } else {
      detect_count_ = 0;
      state_ = "lost";
    }
  }

  else if (state_ == "tracking") {
    if (found) return;

    temp_lost_count_ = 1;
    state_ = "temp_lost";
  }

  else if (state_ == "switching") {
    if (found) {
      state_ = "detecting";
    } else {
      temp_lost_count_++;
      if (temp_lost_count_ > 200) state_ = "lost";
    }
  }

  else if (state_ == "temp_lost") {
    if (found) {
      state_ = "tracking";
    } else {
      temp_lost_count_++;
      if (target_.name == ArmorName::outpost)
        //前哨站的temp_lost_count需要设置的大一些
        max_temp_lost_count_ = outpost_max_temp_lost_count_;
      else
        max_temp_lost_count_ = normal_temp_lost_count_;

      if (temp_lost_count_ > max_temp_lost_count_) state_ = "lost";
    }
  }
}

/**
 * @brief 目标创建：根据当前装甲板列表初始化追踪目标
 * @param armors 当前帧装甲板列表（已按优先级与位置排序）
 * @param t 当前时间戳
 * @return 是否成功创建目标（列表为空时返回 false）
 * @details 选取最高优先级装甲板，调用 solver 计算位姿，
 *          并根据兵种（平衡车、前哨站、基地、普通）设置不同的 EKF 初始参数。
 */
bool Tracker::set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  if (armors.empty()) return false;

  auto & armor = armors.front();
  solver_.solve(armor);

  // 根据兵种优化初始化参数
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);

  if (is_balance) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 2, P0_dig);
  }

  else if (armor.name == ArmorName::outpost) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 81, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.2765, 3, P0_dig);
  }

  else if (armor.name == ArmorName::base) {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1e-4, 0, 0}};
    target_ = Target(armor, t, 0.3205, 3, P0_dig);
  }

  else {
    Eigen::VectorXd P0_dig{{1, 64, 1, 64, 1, 64, 0.4, 100, 1, 1, 1}};
    target_ = Target(armor, t, 0.2, 4, P0_dig);
  }

  return true;
}

/**
 * @brief 目标更新：基于当前装甲板观测更新已有目标
 * @param armors 当前帧装甲板列表
 * @param t 当前时间戳
 * @return 是否成功找到并更新对应装甲板（未找到时返回 false）
 * @details 先将目标状态预测到当前时刻，再在装甲板列表中查找同名同类型装甲板，
 *          若存在则使用 solver 重新解算位姿并调用 EKF 进行状态更新。
 */
bool Tracker::update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t)
{
  target_.predict(t);

  int found_count = 0;
  double min_x = 1e10;  // 画面最左侧
  for (const auto & armor : armors) {
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
    min_x = armor.center.x < min_x ? armor.center.x : min_x;
  }

  if (found_count == 0) return false;

  for (auto & armor : armors) {
    if (
      armor.name != target_.name || armor.type != target_.armor_type
      //  || armor.center.x != min_x
    )
      continue;

    solver_.solve(armor);

    target_.update(armor);
  }

  return true;
}

}  // namespace auto_aim