#include "yolov8.hpp"

#include <fmt/chrono.h>
#include <omp.h>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <random>

#include "tasks/auto_aim/classifier/classifier.hpp"
#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"

namespace auto_aim
{
/**
 * @brief 构造函数，初始化YOLOV8检测器
 * @param config_path 配置文件路径
 * @param debug 是否开启调试模式
 */
YOLOV8::YOLOV8(const std::string & config_path, bool debug)
: classifier_(config_path), detector_(config_path), debug_(debug)
{
  auto yaml = YAML::LoadFile(config_path);

  model_path_ = yaml["yolov8_model_path"].as<std::string>();
  device_ = yaml["device"].as<std::string>();
  binary_threshold_ = yaml["threshold"].as<double>();
  min_confidence_ = yaml["min_confidence"].as<double>();
  int x = 0, y = 0, width = 0, height = 0;
  x = yaml["roi"]["x"].as<int>();
  y = yaml["roi"]["y"].as<int>();
  width = yaml["roi"]["width"].as<int>();
  height = yaml["roi"]["height"].as<int>();
  use_roi_ = yaml["use_roi"].as<bool>();
  roi_ = cv::Rect(x, y, width, height);
  offset_ = cv::Point2f(x, y);

  save_path_ = "imgs";
  std::filesystem::create_directory(save_path_);

  auto model = core_.read_model(model_path_);
  ov::preprocess::PrePostProcessor ppp(model);
  auto & input = ppp.input();

  input.tensor()
    .set_element_type(ov::element::u8)
    .set_shape({1, 416, 416, 3})
    .set_layout("NHWC")
    .set_color_format(ov::preprocess::ColorFormat::BGR);

  input.model().set_layout("NCHW");

  input.preprocess()
    .convert_element_type(ov::element::f32)
    .convert_color(ov::preprocess::ColorFormat::RGB)
    .scale(255.0);

  // TODO: ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY)
  model = ppp.build();
  compiled_model_ = core_.compile_model(
    model, device_, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
}

/**
 * @brief 从图像中检测所有装甲板
 * @param raw_img 输入的BGR彩色图像
 * @param frame_count 帧计数，用于调试显示
 * @return 检测到的装甲板列表
 */
std::list<Armor> YOLOV8::detect(const cv::Mat & raw_img, int frame_count)
{
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return std::list<Armor>();
  }

  cv::Mat bgr_img;
  if (use_roi_) {
    if (roi_.width == -1) {  // -1 表示该维度不裁切
      roi_.width = raw_img.cols;
    }
    if (roi_.height == -1) {  // -1 表示该维度不裁切
      roi_.height = raw_img.rows;
    }
    bgr_img = raw_img(roi_);
  } else {
    bgr_img = raw_img;
  }

  // 计算缩放比例，保持宽高比
  auto x_scale = static_cast<double>(416) / bgr_img.rows;
  auto y_scale = static_cast<double>(416) / bgr_img.cols;
  auto scale = std::min(x_scale, y_scale);
  auto h = static_cast<int>(bgr_img.rows * scale);
  auto w = static_cast<int>(bgr_img.cols * scale);

  // 图像预处理：缩放到416x416，并填充到模型输入尺寸
  auto input = cv::Mat(416, 416, CV_8UC3, cv::Scalar(0, 0, 0));
  auto roi = cv::Rect(0, 0, w, h);
  cv::resize(bgr_img, input(roi), {w, h});
  ov::Tensor input_tensor(ov::element::u8, {1, 416, 416, 3}, input.data);

  // 模型推理
  auto infer_request = compiled_model_.create_infer_request();
  infer_request.set_input_tensor(input_tensor);
  infer_request.infer();

  // 获取模型输出
  auto output_tensor = infer_request.get_output_tensor();
  auto output_shape = output_tensor.get_shape();
  cv::Mat output(output_shape[1], output_shape[2], CV_32F, output_tensor.data());

  return parse(scale, output, raw_img, frame_count);
}

/**
 * @brief 解析YOLO模型输出，提取装甲板信息
 * @param scale 图像缩放比例
 * @param output YOLO模型输出的特征图（会被转置修改）
 * @param bgr_img 原始BGR彩色图像
 * @param frame_count 帧计数，用于调试显示
 * @return 解析后的装甲板列表
 */
std::list<Armor> YOLOV8::parse(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // 转置输出矩阵以便处理
  // 解析每一行输出：xywh + 类别分数 + 关键点
  cv::transpose(output, output);

  std::vector<int> ids;
  std::vector<float> confidences;
  std::vector<cv::Rect> boxes;
  std::vector<std::vector<cv::Point2f>> armors_key_points;
  for (int r = 0; r < output.rows; r++) {
    auto xywh = output.row(r).colRange(0, 4);
    auto scores = output.row(r).colRange(4, 4 + class_num_);
    auto one_key_points = output.row(r).colRange(4 + class_num_, 14);

    std::vector<cv::Point2f> armor_key_points;

    // 获取最高置信度分数
    double score;
    cv::Point max_point;
    cv::minMaxLoc(scores, nullptr, &score, nullptr, &max_point);

    // 过滤低置信度检测结果
    if (score < score_threshold_) continue;

    // 提取边界框坐标并缩放回原图尺寸
    auto x = xywh.at<float>(0);
    auto y = xywh.at<float>(1);
    auto w = xywh.at<float>(2);
    auto h = xywh.at<float>(3);
    auto left = static_cast<int>((x - 0.5 * w) / scale);
    auto top = static_cast<int>((y - 0.5 * h) / scale);
    auto width = static_cast<int>(w / scale);
    auto height = static_cast<int>(h / scale);

    // 提取并缩放关键点坐标（4个角点）
    for (int i = 0; i < 4; i++) {
      float x = one_key_points.at<float>(0, i * 2 + 0) / scale;
      float y = one_key_points.at<float>(0, i * 2 + 1) / scale;
      cv::Point2f kp = {x, y};
      armor_key_points.push_back(kp);
    }
    ids.emplace_back(max_point.x);
    confidences.emplace_back(score);
    boxes.emplace_back(left, top, width, height);
    armors_key_points.emplace_back(armor_key_points);
  }

  // 应用NMS（非极大值抑制）去除重复检测
  std::vector<int> indices;
  cv::dnn::NMSBoxes(boxes, confidences, score_threshold_, nms_threshold_, indices);

  // 创建装甲板对象列表
  std::list<Armor> armors;
  for (const auto & i : indices) {
    sort_keypoints(armors_key_points[i]);
    if (use_roi_) {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i], offset_);
    } else {
      armors.emplace_back(ids[i], confidences[i], boxes[i], armors_key_points[i]);
    }
  }

  // 后处理：验证和过滤检测结果
  for (auto it = armors.begin(); it != armors.end();) {
    // 提取图案并分类
    it->pattern = get_pattern(bgr_img, *it);
    classifier_.classify(*it);

    // 检查名称和置信度
    if (!check_name(*it)) {
      it = armors.erase(it);
      continue;
    }

    // 获取类型并检查类型匹配
    it->type = get_type(*it);
    if (!check_type(*it)) {
      it = armors.erase(it);
      continue;
    }

    // 计算归一化中心点
    it->center_norm = get_center_norm(bgr_img, it->center);
    ++it;
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

/**
 * @brief 检查装甲板的名称识别结果和置信度
 * @param armor 待检查的装甲板对象
 * @return 是否为有效装甲板且置信度满足要求
 */
bool YOLOV8::check_name(const Armor & armor) const
{
  auto name_ok = armor.name != ArmorName::not_armor;
  auto confidence_ok = armor.confidence > min_confidence_;

  // 保存不确定的图案，用于分类器的迭代
  // if (name_ok && !confidence_ok) save(armor);

  return name_ok && confidence_ok;
}

/**
 * @brief 检查装甲板的类型与名称是否匹配
 * @param armor 待检查的装甲板对象
 * @return 类型与名称是否匹配（小装甲板不能是1号或基地，大装甲板不能是2号、哨兵或前哨站）
 */
bool YOLOV8::check_type(const Armor & armor) const
{
  auto name_ok = (armor.type == ArmorType::small)
                   ? (armor.name != ArmorName::one && armor.name != ArmorName::base)
                   : (armor.name != ArmorName::two && armor.name != ArmorName::sentry &&
                      armor.name != ArmorName::outpost);

  // 保存异常的图案，用于分类器的迭代
  // if (!name_ok) save(armor);

  return name_ok;
}

/**
 * @brief 根据装甲板的名称判断装甲板类型（大/小）
 * @param armor 装甲板对象
 * @return 装甲板类型（ArmorType::big 或 ArmorType::small）
 */
ArmorType YOLOV8::get_type(const Armor & armor)
{
  // 英雄、基地只能是大装甲板
  if (armor.name == ArmorName::one || armor.name == ArmorName::base) {
    return ArmorType::big;
  }

  // 工程、哨兵、前哨站只能是小装甲板
  if (
    armor.name == ArmorName::two || armor.name == ArmorName::sentry ||
    armor.name == ArmorName::outpost) {
    return ArmorType::small;
  }

  // 步兵假设为小装甲板
  return ArmorType::small;
}

/**
 * @brief 将像素坐标归一化到[0,1]范围
 * @param bgr_img 输入的BGR彩色图像
 * @param center 像素坐标中心点
 * @return 归一化后的坐标点（x/w, y/h）
 */
cv::Point2f YOLOV8::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  auto h = bgr_img.rows;
  auto w = bgr_img.cols;
  return {center.x / w, center.y / h};
}

/**
 * @brief 从图像中提取装甲板的图案ROI区域
 * @param bgr_img 输入的BGR彩色图像
 * @param armor 装甲板对象，包含角点信息
 * @return 提取的装甲板图案图像，如果ROI无效则返回空Mat
 */
cv::Mat YOLOV8::get_pattern(const cv::Mat & bgr_img, const Armor & armor) const
{
  // 延长灯条获得装甲板角点
  // 1.125 = 0.5 * armor_height / lightbar_length = 0.5 * 126mm / 56mm
  auto tl = (armor.points[0] + armor.points[3]) / 2 - (armor.points[3] - armor.points[0]) * 1.125;
  auto bl = (armor.points[0] + armor.points[3]) / 2 + (armor.points[3] - armor.points[0]) * 1.125;
  auto tr = (armor.points[2] + armor.points[1]) / 2 - (armor.points[2] - armor.points[1]) * 1.125;
  auto br = (armor.points[2] + armor.points[1]) / 2 + (armor.points[2] - armor.points[1]) * 1.125;

  auto roi_left = std::max<int>(std::min(tl.x, bl.x), 0);
  auto roi_top = std::max<int>(std::min(tl.y, tr.y), 0);
  auto roi_right = std::min<int>(std::max(tr.x, br.x), bgr_img.cols);
  auto roi_bottom = std::min<int>(std::max(bl.y, br.y), bgr_img.rows);
  auto roi_tl = cv::Point(roi_left, roi_top);
  auto roi_br = cv::Point(roi_right, roi_bottom);
  auto roi = cv::Rect(roi_tl, roi_br);

  // 检查ROI是否有效
  if (roi_left < 0 || roi_top < 0 || roi_right <= roi_left || roi_bottom <= roi_top) {
    // std::cerr << "Invalid ROI: " << roi << std::endl;
    return cv::Mat();  // 返回一个空的Mat对象
  }

  // 检查ROI是否超出图像边界
  if (roi_right > bgr_img.cols || roi_bottom > bgr_img.rows) {
    // std::cerr << "ROI out of image bounds: " << roi << " Image size: " << bgr_img.size()
    //           << std::endl;
    return cv::Mat();  // 返回一个空的Mat对象
  }

  return bgr_img(roi);
}

/**
 * @brief 保存装甲板图案到文件，用于分类器训练数据收集
 * @param armor 待保存的装甲板对象
 */
void YOLOV8::save(const Armor & armor) const
{
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  auto img_path = fmt::format("{}/{}_{}.jpg", save_path_, armor.name, file_name);
  cv::imwrite(img_path, armor.pattern);
}

/**
 * @brief 在调试模式下绘制检测结果
 * @param img 原始图像
 * @param armors 检测到的装甲板列表
 * @param frame_count 帧计数
 */
void YOLOV8::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  // 绘制帧计数
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  // 绘制每个检测到的装甲板
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {}", armor.confidence, ARMOR_NAMES[armor.name], ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }

  // 如果使用ROI，绘制ROI区域
  if (use_roi_) {
    cv::Scalar green(0, 255, 0);
    cv::rectangle(detection, roi_, green, 2);
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);  // 显示时缩小图片尺寸
  cv::imshow("detection", detection);
}

/**
 * @brief 对关键点进行排序，使其按顺序为：左上、右上、右下、左下
 * @param keypoints 待排序的关键点向量（输入输出参数，会被修改）
 */
void YOLOV8::sort_keypoints(std::vector<cv::Point2f> & keypoints)
{
  // 确保关键点数量为4
  if (keypoints.size() != 4) {
    std::cout << "beyond 4!!" << std::endl;
    return;
  }

  // 按y坐标排序，分为上下两组
  std::sort(keypoints.begin(), keypoints.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.y < b.y;
  });

  std::vector<cv::Point2f> top_points = {keypoints[0], keypoints[1]};
  std::vector<cv::Point2f> bottom_points = {keypoints[2], keypoints[3]};

  // 对上下两组分别按x坐标排序
  std::sort(top_points.begin(), top_points.end(), [](const cv::Point2f & a, const cv::Point2f & b) {
    return a.x < b.x;
  });

  std::sort(
    bottom_points.begin(), bottom_points.end(),
    [](const cv::Point2f & a, const cv::Point2f & b) { return a.x < b.x; });

  // 重新排列为：左上、右上、右下、左下
  keypoints[0] = top_points[0];     // top-left
  keypoints[1] = top_points[1];     // top-right
  keypoints[2] = bottom_points[1];  // bottom-right
  keypoints[3] = bottom_points[0];  // bottom-left
}

/**
 * @brief 后处理YOLO模型输出，解析检测结果
 * @param scale 图像缩放比例
 * @param output YOLO模型输出的特征图
 * @param bgr_img 原始BGR彩色图像
 * @param frame_count 帧计数，用于调试显示
 * @return 解析后的装甲板列表
 */
std::list<Armor> YOLOV8::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  return parse(scale, output, bgr_img, frame_count);
}

}  // namespace auto_aim