#ifndef AUTO_AIM__ARMOR_HPP
#define AUTO_AIM__ARMOR_HPP

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace auto_aim
{
/**
 * @brief 装甲板颜色枚举
 */
enum Color
{
  red,        ///< 红色
  blue,       ///< 蓝色
  extinguish, ///< 熄灭（灰色）
  purple      ///< 紫色
};
const std::vector<std::string> COLORS = {"red", "blue", "extinguish", "purple"};  ///< 颜色名称字符串数组

/**
 * @brief 装甲板类型枚举
 */
enum ArmorType
{
  big,   ///< 大装甲板
  small  ///< 小装甲板
};
const std::vector<std::string> ARMOR_TYPES = {"big", "small"};  ///< 装甲板类型名称字符串数组

/**
 * @brief 装甲板名称枚举
 */
enum ArmorName
{
  one,       ///< 一号（英雄）
  two,       ///< 二号（工程）
  three,     ///< 三号（步兵）
  four,      ///< 四号（步兵）
  five,      ///< 五号（步兵）
  sentry,    ///< 哨兵
  outpost,   ///< 前哨站
  base,      ///< 基地
  not_armor  ///< 非装甲板
};
const std::vector<std::string> ARMOR_NAMES = {"one",    "two",     "three", "four",     "five",
                                              "sentry", "outpost", "base",  "not_armor"};  ///< 装甲板名称字符串数组

/**
 * @brief 装甲板优先级枚举
 */
enum ArmorPriority
{
  first = 1,  ///< 第一优先级
  second,     ///< 第二优先级
  third,      ///< 第三优先级
  forth,      ///< 第四优先级
  fifth       ///< 第五优先级
};

// clang-format off
/**
 * @brief 装甲板属性映射表
 * 
 * 将类别ID映射到(颜色, 名称, 类型)三元组
 * 用于从神经网络输出的类别ID获取装甲板的完整属性信息
 */
const std::vector<std::tuple<Color, ArmorName, ArmorType>> armor_properties = {
  {blue, sentry, small},     {red, sentry, small},     {extinguish, sentry, small},
  {blue, one, small},        {red, one, small},        {extinguish, one, small},
  {blue, two, small},        {red, two, small},        {extinguish, two, small},
  {blue, three, small},      {red, three, small},      {extinguish, three, small},
  {blue, four, small},       {red, four, small},       {extinguish, four, small},
  {blue, five, small},       {red, five, small},       {extinguish, five, small},
  {blue, outpost, small},    {red, outpost, small},    {extinguish, outpost, small},
  {blue, base, big},         {red, base, big},         {extinguish, base, big},      {purple, base, big},       
  {blue, base, small},       {red, base, small},       {extinguish, base, small},    {purple, base, small},    
  {blue, three, big},        {red, three, big},        {extinguish, three, big}, 
  {blue, four, big},         {red, four, big},         {extinguish, four, big},  
  {blue, five, big},         {red, five, big},         {extinguish, five, big}};
// clang-format on

/**
 * @brief 灯条结构体
 * 
 * 表示检测到的灯条信息，包含位置、角度、尺寸等属性
 */
struct Lightbar
{
  std::size_t id;                    ///< 灯条ID
  Color color;                       ///< 灯条颜色
  cv::Point2f center;                ///< 灯条中心点
  cv::Point2f top;                   ///< 灯条顶部中心点
  cv::Point2f bottom;                ///< 灯条底部中心点
  cv::Point2f top2bottom;            ///< 从顶部到底部的向量
  std::vector<cv::Point2f> points;   ///< 灯条关键点（顶部和底部）
  double angle;                       ///< 灯条角度（弧度）
  double angle_error;                 ///< 角度误差（与垂直方向的偏差）
  double length;                     ///< 灯条长度
  double width;                      ///< 灯条宽度
  double ratio;                       ///< 长宽比（length/width）
  cv::RotatedRect rotated_rect;       ///< 旋转矩形

  /**
   * @brief 构造函数，从旋转矩形创建灯条
   * @param rotated_rect 旋转矩形
   * @param id 灯条ID
   */
  Lightbar(const cv::RotatedRect & rotated_rect, std::size_t id);
  
  /**
   * @brief 默认构造函数
   */
  Lightbar() {};
};

/**
 * @brief 装甲板结构体
 * 
 * 表示检测到的装甲板信息，包含位置、类型、名称、3D坐标等属性
 */
struct Armor
{
  Color color;                        ///< 装甲板颜色
  Lightbar left, right;               ///< 左右灯条（传统检测方法使用）
  cv::Point2f center;                ///< 装甲板中心点（不是对角线交点，不能作为实际中心！）
  cv::Point2f center_norm;           ///< 归一化坐标中心点（范围[0,1]）
  std::vector<cv::Point2f> points;   ///< 装甲板四个角点（左上、右上、右下、左下）

  double ratio;                       ///< 两灯条的中点连线与长灯条的长度之比
  double side_ratio;                  ///< 长灯条与短灯条的长度之比
  double rectangular_error;           ///< 灯条和中点连线所成夹角与π/2的差值

  ArmorType type;                     ///< 装甲板类型（大/小）
  ArmorName name;                     ///< 装甲板名称
  ArmorPriority priority;             ///< 装甲板优先级
  int class_id;                       ///< 类别ID（神经网络输出）
  cv::Rect box;                       ///< 边界框
  cv::Mat pattern;                    ///< 装甲板图案图像
  double confidence;                  ///< 置信度
  bool duplicated;                    ///< 是否为重复检测

  Eigen::Vector3d xyz_in_gimbal;      ///< 云台坐标系下的3D坐标（单位：m）
  Eigen::Vector3d xyz_in_world;       ///< 世界坐标系下的3D坐标（单位：m）
  Eigen::Vector3d ypr_in_gimbal;      ///< 云台坐标系下的姿态角（单位：rad）
  Eigen::Vector3d ypr_in_world;       ///< 世界坐标系下的姿态角（单位：rad）
  Eigen::Vector3d ypd_in_world;       ///< 世界坐标系下的球坐标（yaw, pitch, distance）

  double yaw_raw;                     ///< 原始偏航角（单位：rad）

  /**
   * @brief 传统检测方法构造函数
   * @param left 左灯条
   * @param right 右灯条
   */
  Armor(const Lightbar & left, const Lightbar & right);
  
  /**
   * @brief 神经网络检测方法构造函数
   * @param class_id 类别ID
   * @param confidence 置信度
   * @param box 边界框
   * @param armor_keypoints 装甲板四个角点
   */
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints);
  
  /**
   * @brief 神经网络检测方法构造函数（带ROI偏移）
   * @param class_id 类别ID
   * @param confidence 置信度
   * @param box 边界框
   * @param armor_keypoints 装甲板四个角点（ROI坐标系）
   * @param offset ROI偏移量
   */
  Armor(
    int class_id, float confidence, const cv::Rect & box, std::vector<cv::Point2f> armor_keypoints,
    cv::Point2f offset);
  
  /**
   * @brief YOLOV5检测方法构造函数
   * @param color_id 颜色ID（0:蓝色, 1:红色, 2:熄灭）
   * @param num_id 数字ID（类别ID）
   * @param confidence 置信度
   * @param box 边界框
   * @param armor_keypoints 装甲板四个角点
   */
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints);
  
  /**
   * @brief YOLOV5检测方法构造函数（带ROI偏移）
   * @param color_id 颜色ID（0:蓝色, 1:红色, 2:熄灭）
   * @param num_id 数字ID（类别ID）
   * @param confidence 置信度
   * @param box 边界框
   * @param armor_keypoints 装甲板四个角点（ROI坐标系）
   * @param offset ROI偏移量
   */
  Armor(
    int color_id, int num_id, float confidence, const cv::Rect & box,
    std::vector<cv::Point2f> armor_keypoints, cv::Point2f offset);
};

}  // namespace auto_aim

#endif  // AUTO_AIM__ARMOR_HPP