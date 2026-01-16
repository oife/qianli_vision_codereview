// 装甲板姿态解算与重投影工具：
// 1. 从标定文件读取相机与云台外参与相机内参；
// 2. 通过 solvePnP 计算装甲板在云台/世界坐标系下的位姿；
// 3. 提供 yaw 搜索优化与自定义代价函数计算；
// 4. 支持世界坐标点到像素坐标的投影工具。
#include "solver.hpp"

#include <yaml-cpp/yaml.h>

#include <vector>

#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace auto_aim
{
// 装甲板模型的物理尺寸（单位：米）
constexpr double LIGHTBAR_LENGTH = 56e-3;
constexpr double BIG_ARMOR_WIDTH = 230e-3;
constexpr double SMALL_ARMOR_WIDTH = 135e-3;

// 以装甲板中心为原点的 3D 角点坐标（相机右手系）
const std::vector<cv::Point3f> BIG_ARMOR_POINTS{
  {0, BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, BIG_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};
const std::vector<cv::Point3f> SMALL_ARMOR_POINTS{
  {0, SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, LIGHTBAR_LENGTH / 2},
  {0, -SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2},
  {0, SMALL_ARMOR_WIDTH / 2, -LIGHTBAR_LENGTH / 2}};

/**
 * @brief 构造函数：从配置文件读取相机与云台的外参与内参
 * @param config_path YAML 配置文件路径
 * @details 读取云台到 IMU、本体到相机的旋转和平移矩阵，以及相机内参和畸变参数，
 *          并完成 Eigen 与 OpenCV 之间的矩阵转换。
 */
Solver::Solver(const std::string & config_path) : R_gimbal2world_(Eigen::Matrix3d::Identity())
{
  auto yaml = YAML::LoadFile(config_path);

  auto R_gimbal2imubody_data = yaml["R_gimbal2imubody"].as<std::vector<double>>();
  auto R_camera2gimbal_data = yaml["R_camera2gimbal"].as<std::vector<double>>();
  auto t_camera2gimbal_data = yaml["t_camera2gimbal"].as<std::vector<double>>();
  R_gimbal2imubody_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_gimbal2imubody_data.data());
  R_camera2gimbal_ = Eigen::Matrix<double, 3, 3, Eigen::RowMajor>(R_camera2gimbal_data.data());
  t_camera2gimbal_ = Eigen::Matrix<double, 3, 1>(t_camera2gimbal_data.data());

  auto camera_matrix_data = yaml["camera_matrix"].as<std::vector<double>>();
  auto distort_coeffs_data = yaml["distort_coeffs"].as<std::vector<double>>();
  Eigen::Matrix<double, 3, 3, Eigen::RowMajor> camera_matrix(camera_matrix_data.data());
  Eigen::Matrix<double, 1, 5> distort_coeffs(distort_coeffs_data.data());
  cv::eigen2cv(camera_matrix, camera_matrix_);
  cv::eigen2cv(distort_coeffs, distort_coeffs_);
}

/**
 * @brief 获取云台到世界坐标系的旋转矩阵
 * @return 当前云台到世界系的 3x3 旋转矩阵
 */
Eigen::Matrix3d Solver::R_gimbal2world() const { return R_gimbal2world_; }

/**
 * @brief 使用 IMU 姿态更新云台到世界系的旋转矩阵
 * @param q IMU 姿态四元数（机体坐标到世界/惯性坐标的旋转）
 * @details 通过 R_gimbal2imubody 将 IMU 姿态转换到云台坐标系，再得到云台到世界的旋转。
 */
void Solver::set_R_gimbal2world(const Eigen::Quaterniond & q)
{
  Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
  R_gimbal2world_ = R_gimbal2imubody_.transpose() * R_imubody2imuabs * R_gimbal2imubody_;
}

/**
 * @brief 使用 solvePnP 求解装甲板位姿并补齐多坐标系下的姿态信息
 * @param armor 待更新的装甲板对象（输入为像素角点，输出为多系下位姿）
 * @details 通过 PnP 计算装甲板在相机系中的平移与旋转，再依次转换到云台系和世界系，
 *          同时计算装甲板在云台系与世界系下的 yaw-pitch-roll 以及 ypd 表示。
 *          对非平衡装甲板，进一步调用 optimize_yaw 进行 yaw 搜索优化。
 */
void Solver::solve(Armor & armor) const
{
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  // 相机系 -> 云台系 -> 世界系
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  // 平衡车装甲板 yaw 直接采用 PnP 结果，不做搜索优化（pitch 假设不成立）
  auto is_balance = (armor.type == ArmorType::big) &&
                    (armor.name == ArmorName::three || armor.name == ArmorName::four ||
                     armor.name == ArmorName::five);
  if (is_balance) return;

  optimize_yaw(armor);
}

/**
 * @brief 根据假设位姿重投影装甲板角点
 * @param xyz_in_world 装甲板中心在世界坐标系下的位置
 * @param yaw 世界系下的装甲板 yaw 角（弧度）
 * @param type 装甲板类型（大装甲板 / 小装甲板）
 * @param name 装甲板名称（前哨站、英雄、步兵等，用于区分固定 pitch）
 * @return 重新投影到图像上的四个角点像素坐标
 * @details 根据给定 yaw 和固定 pitch 构造装甲板到世界系的旋转矩阵，
 *          再通过外参转换到相机系，最后利用相机内参进行透视投影。
 */
std::vector<cv::Point2f> Solver::reproject_armor(
  const Eigen::Vector3d & xyz_in_world, double yaw, ArmorType type, ArmorName name) const
{
  // 根据假设的 yaw / pitch 构造装甲板到世界系的旋转矩阵
  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto pitch = (name == ArmorName::outpost) ? -15.0 * CV_PI / 180.0 : 15.0 * CV_PI / 180.0;
  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, rvec);
  cv::Vec3d tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  const auto & object_points = (type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;
  cv::projectPoints(object_points, rvec, tvec, camera_matrix_, distort_coeffs_, image_points);
  return image_points;
}

/**
 * @brief 计算前哨站装甲板在给定 pitch 下的重投影误差
 * @param armor 目标装甲板（输入为像素角点）
 * @param pitch 世界系下的装甲板 pitch 角（弧度）
 * @return 对应 pitch 假设下的像素重投影误差（四个角点误差之和）
 * @details 先通过 PnP 解算原始 yaw 与平移，再替换为指定 pitch 重新构造旋转，
 *          将装甲板重投影回图像平面并与真实角点比较。
 */
double Solver::oupost_reprojection_error(Armor armor, const double & pitch)
{
  // 先用 PnP 解算 yaw、平移，再替换 pitch 计算重投影误差
  const auto & object_points =
    (armor.type == ArmorType::big) ? BIG_ARMOR_POINTS : SMALL_ARMOR_POINTS;

  cv::Vec3d rvec, tvec;
  cv::solvePnP(
    object_points, armor.points, camera_matrix_, distort_coeffs_, rvec, tvec, false,
    cv::SOLVEPNP_IPPE);

  Eigen::Vector3d xyz_in_camera;
  cv::cv2eigen(tvec, xyz_in_camera);
  armor.xyz_in_gimbal = R_camera2gimbal_ * xyz_in_camera + t_camera2gimbal_;
  armor.xyz_in_world = R_gimbal2world_ * armor.xyz_in_gimbal;

  cv::Mat rmat;
  cv::Rodrigues(rvec, rmat);
  Eigen::Matrix3d R_armor2camera;
  cv::cv2eigen(rmat, R_armor2camera);
  Eigen::Matrix3d R_armor2gimbal = R_camera2gimbal_ * R_armor2camera;
  Eigen::Matrix3d R_armor2world = R_gimbal2world_ * R_armor2gimbal;
  armor.ypr_in_gimbal = tools::eulers(R_armor2gimbal, 2, 1, 0);
  armor.ypr_in_world = tools::eulers(R_armor2world, 2, 1, 0);

  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);

  auto yaw = armor.ypr_in_world[0];
  auto xyz_in_world = armor.xyz_in_world;

  auto sin_yaw = std::sin(yaw);
  auto cos_yaw = std::cos(yaw);

  auto sin_pitch = std::sin(pitch);
  auto cos_pitch = std::cos(pitch);

  // clang-format off
  const Eigen::Matrix3d _R_armor2world {
    {cos_yaw * cos_pitch, -sin_yaw, cos_yaw * sin_pitch},
    {sin_yaw * cos_pitch,  cos_yaw, sin_yaw * sin_pitch},
    {         -sin_pitch,        0,           cos_pitch}
  };
  // clang-format on

  // get R_armor2camera t_armor2camera
  const Eigen::Vector3d & t_armor2world = xyz_in_world;
  Eigen::Matrix3d _R_armor2camera =
    R_camera2gimbal_.transpose() * R_gimbal2world_.transpose() * _R_armor2world;
  Eigen::Vector3d t_armor2camera =
    R_camera2gimbal_.transpose() * (R_gimbal2world_.transpose() * t_armor2world - t_camera2gimbal_);

  // get rvec tvec
  cv::Vec3d _rvec;
  cv::Mat R_armor2camera_cv;
  cv::eigen2cv(_R_armor2camera, R_armor2camera_cv);
  cv::Rodrigues(R_armor2camera_cv, _rvec);
  cv::Vec3d _tvec(t_armor2camera[0], t_armor2camera[1], t_armor2camera[2]);

  // reproject
  std::vector<cv::Point2f> image_points;
  cv::projectPoints(object_points, _rvec, _tvec, camera_matrix_, distort_coeffs_, image_points);

  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  return error;
}

/**
 * @brief 对装甲板 yaw 角进行一维搜索优化
 * @param armor 待优化的装甲板（内部 yaw 将被更新）
 * @details 以云台 yaw 为中心，在给定角度范围内离散搜索 yaw 值，
 *          对每个 yaw 通过 armor_reprojection_error 计算重投影误差，
 *          选择误差最小的 yaw 作为优化结果。
 */
void Solver::optimize_yaw(Armor & armor) const
{
  Eigen::Vector3d gimbal_ypr = tools::eulers(R_gimbal2world_, 2, 1, 0);

  constexpr double SEARCH_RANGE = 140;  // degree
  auto yaw0 = tools::limit_rad(gimbal_ypr[0] - SEARCH_RANGE / 2 * CV_PI / 180.0);

  auto min_error = 1e10;
  auto best_yaw = armor.ypr_in_world[0];

  for (int i = 0; i < SEARCH_RANGE; i++) {
    double yaw = tools::limit_rad(yaw0 + i * CV_PI / 180.0);
    auto error = armor_reprojection_error(armor, yaw, (i - SEARCH_RANGE / 2) * CV_PI / 180.0);

    if (error < min_error) {
      min_error = error;
      best_yaw = yaw;
    }
  }

  armor.yaw_raw = armor.ypr_in_world[0];
  armor.ypr_in_world[0] = best_yaw;
}

/**
 * @brief 上海交大（SJTU）装甲板匹配代价函数
 * @param cv_refs 标准参考角点像素坐标
 * @param cv_pts 实际检测到的角点像素坐标
 * @param inclined 目标装甲板倾斜角（弧度）
 * @return 综合像素偏差与斜率偏差的代价值，数值越小匹配越好
 * @details 通过边长差、端点像素距离和边斜率差构造代价，
 *          并根据倾斜角调整像素误差与角度误差的权重。
 */
double Solver::SJTU_cost(
  const std::vector<cv::Point2f> & cv_refs, const std::vector<cv::Point2f> & cv_pts,
  const double & inclined) const
{
  // 上海交大（SJTU）代价函数：综合像素偏差和斜率差
  std::size_t size = cv_refs.size();
  std::vector<Eigen::Vector2d> refs;
  std::vector<Eigen::Vector2d> pts;
  for (std::size_t i = 0u; i < size; ++i) {
    refs.emplace_back(cv_refs[i].x, cv_refs[i].y);
    pts.emplace_back(cv_pts[i].x, cv_pts[i].y);
  }
  double cost = 0.;
  for (std::size_t i = 0u; i < size; ++i) {
    std::size_t p = (i + 1u) % size;
    // i - p 构成线段。过程：先移动起点，再补长度，再旋转
    Eigen::Vector2d ref_d = refs[p] - refs[i];  // 标准
    Eigen::Vector2d pt_d = pts[p] - pts[i];
    // 长度差代价 + 起点差代价(1 / 2)（0 度左右应该抛弃)
    double pixel_dis =  // dis 是指方差平面内到原点的距离
      (0.5 * ((refs[i] - pts[i]).norm() + (refs[p] - pts[p]).norm()) +
       std::fabs(ref_d.norm() - pt_d.norm())) /
      ref_d.norm();
    double angular_dis = ref_d.norm() * tools::get_abs_angle(ref_d, pt_d) / ref_d.norm();
    // 平方可能是为了配合 sin 和 cos
    // 弧度差代价（0 度左右占比应该大）
    double cost_i =
      tools::square(pixel_dis * std::sin(inclined)) +
      tools::square(angular_dis * std::cos(inclined)) * 2.0;  // DETECTOR_ERROR_PIXEL_BY_SLOPE
    // 重投影像素误差越大，越相信斜率
    cost += std::sqrt(cost_i);
  }
  return cost;
}

/**
 * @brief 计算给定 yaw 假设下装甲板的重投影误差
 * @param armor 目标装甲板（包含像素角点和世界坐标等信息）
 * @param yaw 世界系下假设的 yaw 角（弧度）
 * @param inclined 装甲板倾斜角（弧度），可用于 SJTU_cost（当前注释掉）
 * @return 在该 yaw 假设下的像素重投影误差（四点像素距离之和）
 * @details 调用 reproject_armor 将装甲板角点重投影回图像，
 *          与检测到的角点比较距离，用于 yaw 搜索的评价函数。
 */
double Solver::armor_reprojection_error(
  const Armor & armor, double yaw, const double & inclined) const
{
  // 根据假设的 yaw 重投影装甲板并计算像素误差
  auto image_points = reproject_armor(armor.xyz_in_world, yaw, armor.type, armor.name);
  auto error = 0.0;
  for (int i = 0; i < 4; i++) error += cv::norm(armor.points[i] - image_points[i]);
  // auto error = SJTU_cost(image_points, armor.points, inclined);

  return error;
}

/**
 * @brief 世界坐标点到像素坐标的转换
 * @param worldPoints 世界坐标系下的三维点集合
 * @return 对应的像素坐标集合（仅包含在相机前方的有效点）
 * @details 利用当前云台与相机外参计算世界到相机的位姿，将 3D 点变换到相机坐标系，
 *          过滤掉 z<=0 的点后，通过相机内参和畸变参数进行投影。
 */
std::vector<cv::Point2f> Solver::world2pixel(const std::vector<cv::Point3f> & worldPoints)
{
  Eigen::Matrix3d R_world2camera = R_camera2gimbal_.transpose() * R_gimbal2world_.transpose();
  Eigen::Vector3d t_world2camera = -R_camera2gimbal_.transpose() * t_camera2gimbal_;

  cv::Mat rvec;
  cv::Mat tvec;
  cv::eigen2cv(R_world2camera, rvec);
  cv::eigen2cv(t_world2camera, tvec);

  std::vector<cv::Point3f> valid_world_points;
  for (const auto & world_point : worldPoints) {
    Eigen::Vector3d world_point_eigen(world_point.x, world_point.y, world_point.z);
    Eigen::Vector3d camera_point = R_world2camera * world_point_eigen + t_world2camera;

    if (camera_point.z() > 0) {
      valid_world_points.push_back(world_point);
    }
  }
  // 如果没有有效点，返回空vector
  if (valid_world_points.empty()) {
    return std::vector<cv::Point2f>();
  }
  std::vector<cv::Point2f> pixelPoints;
  cv::projectPoints(valid_world_points, rvec, tvec, camera_matrix_, distort_coeffs_, pixelPoints);
  return pixelPoints;
}
}  // namespace auto_aim