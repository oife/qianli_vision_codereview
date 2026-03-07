#ifndef AUTO_AIM__TRACKER_HPP
#define AUTO_AIM__TRACKER_HPP

#include <Eigen/Dense>
#include <chrono>
#include <list>
#include <string>

#include "tasks/auto_aim/armor/armor.hpp"
#include "tasks/auto_aim/solver/solver.hpp"
#include "tasks/auto_aim/target/target.hpp"
#include "tasks/omniperception/perceptron.hpp"
#include "tools/thread_safe_queue/thread_safe_queue.hpp"

namespace auto_aim
{
class Tracker
{
public:
  Tracker(const std::string & config_path, Solver & solver);

  std::string state() const;

  std::list<Target> track(
    std::list<Armor> & armors, std::chrono::steady_clock::time_point t,
    bool use_enemy_color = true);

  std::tuple<omniperception::DetectionResult, std::list<Target>> track(
    const std::vector<omniperception::DetectionResult> & detection_queue, std::list<Armor> & armors,
    std::chrono::steady_clock::time_point t, bool use_enemy_color = true);

private:
  Solver & solver_;
  Color enemy_color_;
  int min_detect_count_;
  int max_temp_lost_count_;
  int detect_count_;
  int temp_lost_count_;
  int outpost_max_temp_lost_count_;
  int normal_temp_lost_count_;
  std::string state_, pre_state_;
  Target target_;
  std::chrono::steady_clock::time_point last_timestamp_;
  ArmorPriority omni_target_priority_;

  void state_machine(bool found);

  bool set_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  bool update_target(std::list<Armor> & armors, std::chrono::steady_clock::time_point t);

  // ---- 扩展关联策略 ----

  /// 判断一个完整装甲板（≥3kpt）是否可视为和目标同一辆车
  /// 涵盖：①类别不自信被归为 not_armor；③装甲板被击中颜色变灰
  bool is_same_vehicle_full(const Armor & armor) const;

  /// 判断一个 2kpt partial 装甲板是否来自目标同一辆车
  /// 使用：①空间距离；②灯条延长线与装甲板短边延长线交点在图像上方的几何约束
  bool is_same_vehicle_partial(const Armor & armor, const Armor & ref_armor) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TRACKER_HPP