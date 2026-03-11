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

  // 过滤：保留敌方颜色 + 被击中变灰（extinguish）的装甲板
  // extinguish 是目标被击中后灯条熄灭的状态，仍然是同一辆车，不能丢弃
  armors.remove_if([&](const auto_aim::Armor & a) {
    return a.color != enemy_color_ && a.color != Color::extinguish;
  });

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

  // ---- 第一步：找名称/类型完全匹配的完整装甲板（原有逻辑）----
  int found_count = 0;
  for (const auto & armor : armors) {
    if (armor.partial) continue;
    if (armor.name != target_.name || armor.type != target_.armor_type) continue;
    found_count++;
  }

  // ---- 第二步：若严格匹配为0，尝试扩展关联（not_armor / 灰色击中装甲板）----
  // 找最近的扩展候选用于 partial 判断的参考装甲板
  const Armor * ref_armor_ptr = nullptr;
  double ref_min_dist = 1e10;

  if (found_count == 0) {
    for (const auto & armor : armors) {
      if (armor.partial) continue;
      if (!is_same_vehicle_full(armor)) continue;
      found_count++;
      // 记录最近的完整候选作为 partial 判断的参考
      double d = cv::norm(armor.center - cv::Point2f(
        target_.armor_xyza_list().front()[0], target_.armor_xyza_list().front()[1]));
      if (d < ref_min_dist) { ref_min_dist = d; ref_armor_ptr = &armor; }
    }
  } else {
    // 严格匹配时，用严格匹配的第一个作为 partial 参考
    for (const auto & armor : armors) {
      if (armor.partial) continue;
      if (armor.name != target_.name || armor.type != target_.armor_type) continue;
      ref_armor_ptr = &armor;
      break;
    }
  }

  if (found_count == 0) return false;

  // ---- 第三步：完整装甲板 PnP + EKF update，保存解算结果用于桥接 ----
  struct SolvedFull {
    const Armor * ptr;
    int matched_id;
  };
  std::vector<SolvedFull> solved_full;

  for (auto & armor : armors) {
    if (armor.partial) continue;
    bool strict_match = (armor.name == target_.name && armor.type == target_.armor_type);
    bool extended_match = !strict_match && is_same_vehicle_full(armor);
    if (!strict_match && !extended_match) continue;

    solver_.solve(armor);

    // PnP 结果合理性检查：solver_.solve() 可能因异常或不足点提前返回
    double solve_dist = armor.ypd_in_world.norm();
    if (solve_dist < 0.3 || solve_dist > 15.0 || std::isnan(solve_dist)) {
      tools::logger()->debug("[Tracker] PnP result rejected: dist={:.2f}", solve_dist);
      continue;
    }

    // 3D 位置跳变检测：车不可能在几帧内瞬移几米
    // 将 PnP 位置与 EKF 预测的所有装甲板位置做 3D 距离对比
    auto xyza_list = target_.armor_xyza_list();
    double min_3d_dist = 1e10;
    for (const auto & xyza : xyza_list) {
      double d = (armor.xyz_in_world - xyza.head(3)).norm();
      min_3d_dist = std::min(min_3d_dist, d);
    }
    if (min_3d_dist > 2.0) {
      tools::logger()->debug(
        "[Tracker] PnP 3D jump rejected: {:.2f}m from nearest prediction", min_3d_dist);
      continue;
    }

    target_.update(armor);
    solved_full.push_back({&armor, target_.last_id});
  }

  // ---- 第四步：partial（2kpt）装甲板桥接 EKF update ----
  for (const auto & armor : armors) {
    if (!armor.partial) continue;

    // 判断是否属于同一辆车
    bool name_match = (armor.name == target_.name);
    bool geom_match = false;
    if (!name_match && armor.name == ArmorName::not_armor && ref_armor_ptr != nullptr) {
      geom_match = is_same_vehicle_partial(armor, *ref_armor_ptr);
    }
    if (!name_match && !geom_match) continue;

    // 找最近的已解算完整装甲板作为 PnP 锚点
    const SolvedFull * best_anchor = nullptr;
    double min_dist = 1e10;
    for (const auto & sf : solved_full) {
      double d = cv::norm(armor.center - sf.ptr->center);
      if (d < min_dist) {
        min_dist = d;
        best_anchor = &sf;
      }
    }

    if (best_anchor && min_dist < 400.0) {
      // 桥接更新：以邻接关系确定 ID，联合约束旋转中心和 r/l/h
      target_.update_bridge(*best_anchor->ptr, armor, best_anchor->matched_id, solver_);
    }
    // 无相邻完整装甲板时直接跳过：2kpt 数据无法独立解算，只会增加噪声
  }

  return true;
}

// ---- 辅助函数实现 ----

bool Tracker::is_same_vehicle_full(const Armor & armor) const
{
  // 情况①：类别不自信被归为 not_armor，但颜色和类型与目标一致
  bool is_not_armor_candidate = (armor.name == ArmorName::not_armor) &&
                                 (armor.color == enemy_color_) &&
                                 (armor.type == target_.armor_type);

  // 情况③：被击中，颜色变灰（extinguish），但类别和类型与目标一致
  bool is_hit_candidate = (armor.color == Color::extinguish) &&
                           (armor.name == target_.name) &&
                           (armor.type == target_.armor_type);

  if (!is_not_armor_candidate && !is_hit_candidate) return false;

  // 位置约束：候选装甲板中心必须落在 EKF 预测的某面装甲板的重投影附近
  // 对每面预测装甲板做重投影，取候选中心到最近重投影中心的像素距离
  auto xyza_list = target_.armor_xyza_list();
  double min_reproj_dist = 1e10;
  for (const auto & xyza : xyza_list) {
    auto reproj = solver_.reproject_armor(xyza.head(3), xyza[3], target_.armor_type, target_.name);
    if (reproj.empty()) continue;
    // 用重投影四点的中心估算重投影中心
    cv::Point2f reproj_center(0, 0);
    for (const auto & p : reproj) reproj_center += p;
    reproj_center /= (float)reproj.size();
    double d = cv::norm(armor.center - reproj_center);
    min_reproj_dist = std::min(min_reproj_dist, d);
  }

  // 像素距离阈值（辅助约束，主要防护靠 3D 跳变检测）
  return min_reproj_dist < 300.0;
}

bool Tracker::is_same_vehicle_partial(const Armor & armor, const Armor & ref_armor) const
{
  // armor: 2kpt partial（not_armor），只有两个可见关键点（一条灯条的上下端点）
  // ref_armor: 距离最近的完整装甲板（已严格或扩展匹配的参考）

  // --- 约束 1：像素距离（2kpt 装甲板中心到参考装甲板中心不能太远）---
  double pixel_dist = cv::norm(armor.center - ref_armor.center);
  if (pixel_dist > 400.0) return false;  // 超过 400px 肯定不是同一辆车

  // --- 约束 2：灯条延长线与装甲板短边延长线的交点在图像上方 ---
  // 取 2kpt partial 的两个可见点（一条灯条的上端和下端）
  cv::Point2f p_top, p_bot;
  bool found_top = false, found_bot = false;
  for (int k = 0; k < 4; k++) {
    if (k < (int)armor.kpt_visibility.size() && armor.kpt_visibility[k] > 0.5f) {
      if (!found_top) { p_top = armor.points[k]; found_top = true; }
      else { p_bot = armor.points[k]; found_bot = true; break; }
    }
  }
  if (!found_top || !found_bot) return false;

  // 取参考装甲板的"更近的那条短边"（左灯条或右灯条）
  // 左灯条：points[0](左上) 和 points[3](左下)
  // 右灯条：points[1](右上) 和 points[2](右下)
  double dist_left = cv::norm(armor.center - (ref_armor.points[0] + ref_armor.points[3]) / 2.f);
  double dist_right = cv::norm(armor.center - (ref_armor.points[1] + ref_armor.points[2]) / 2.f);
  cv::Point2f ref_top, ref_bot;
  if (dist_left < dist_right) {
    ref_top = ref_armor.points[0];  // 左上
    ref_bot = ref_armor.points[3];  // 左下
  } else {
    ref_top = ref_armor.points[1];  // 右上
    ref_bot = ref_armor.points[2];  // 右下
  }

  // 计算两条直线的交点
  // 直线1（灯条延长线）：p_top → p_bot → 延长
  // 直线2（参考短边延长线）：ref_top → ref_bot → 延长
  // 用参数方程求交点：P = A + t*(B-A), Q = C + s*(D-C)
  auto line_intersect = [](cv::Point2f a, cv::Point2f b, cv::Point2f c, cv::Point2f d,
                            cv::Point2f & intersection) -> bool {
    float denom = (a.x - b.x) * (c.y - d.y) - (a.y - b.y) * (c.x - d.x);
    if (std::abs(denom) < 1e-6f) return false;  // 平行
    float t = ((a.x - c.x) * (c.y - d.y) - (a.y - c.y) * (c.x - d.x)) / denom;
    intersection = a + t * (b - a);
    return true;
  };

  cv::Point2f intersect;
  if (!line_intersect(p_top, p_bot, ref_top, ref_bot, intersect)) {
    // 平行灯条（正对相机时两灯条平行），此时直接用距离约束即可
    return pixel_dist < 250.0;
  }

  // 交点在图像上方（y 坐标小于两个装甲板中心的平均 y）说明是同一辆车
  float avg_center_y = (armor.center.y + ref_armor.center.y) / 2.f;
  bool intersection_above = (intersect.y < avg_center_y);

  return intersection_above;
}

}  // namespace auto_aim