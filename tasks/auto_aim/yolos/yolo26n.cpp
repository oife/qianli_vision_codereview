#include "yolo26n.hpp"

#include <yaml-cpp/yaml.h>

#include <chrono>

#include "tools/img_tools/img_tools.hpp"
#include "tools/logger/logger.hpp"
#include "tools/math_tools/math_tools.hpp"

namespace auto_aim
{

YOLO26N::YOLO26N(const std::string & config_path, bool debug)
: debug_(debug), backend_(config_path)
{
  auto yaml = YAML::LoadFile(config_path);
  model_path_ = yaml["yolo26n_model_path"].as<std::string>();
  min_confidence_ = yaml["min_confidence"].as<double>();

  BackendConfig config;
  config.input_size = cv::Size(INPUT_SIZE, INPUT_SIZE);
  config.padding_color = cv::Scalar(114, 114, 114);
  config.center_pad = true;
  if (!backend_.init(model_path_, config)) {
    throw std::runtime_error("Backend initializing failed");
  }
  backend_ctx_ = backend_.create_ctx();
}

std::list<Armor> YOLO26N::detect(const cv::Mat & raw_img, int frame_count)
{
  auto detect_start = std::chrono::steady_clock::now();
  if (raw_img.empty()) {
    tools::logger()->warn("Empty img!, camera drop!");
    return {};
  }

  double scale = 1.0;
  cv::Mat output;
  if (!backend_.execute(raw_img, output, scale, backend_ctx_.get())) {
    tools::logger()->error("YOLO26N backend execution failed");
    return {};
  }
  auto execute_end = std::chrono::steady_clock::now();

  last_profile_.backend = backend_.get_last_profile();
  last_profile_.backend_execute_ms = last_profile_.backend.total_ms;

  auto armors = postprocess(scale, output, raw_img, frame_count);
  auto detect_end = std::chrono::steady_clock::now();
  last_profile_.postprocess_ms = tools::delta_time(detect_end, execute_end) * 1000.0;
  last_profile_.detect_total_ms = tools::delta_time(detect_end, detect_start) * 1000.0;

  return armors;
}

std::list<Armor> YOLO26N::parse(
  const float * data, int num_anchors, const cv::Mat & bgr_img, int frame_count)
{
  // 第一阶段：解析所有候选检测，准备 NMS
  struct RawDet {
    cv::Rect box;
    float confidence;
    int color_id;
    int name_id;
    std::vector<cv::Point2f> kpts;
    std::vector<float> vis;
  };
  std::vector<RawDet> raw_dets;
  std::vector<cv::Rect> nms_boxes;
  std::vector<float> nms_confs;

  for (int i = 0; i < num_anchors; i++) {
    // 置信度 = max(color_scores)
    float max_conf = -1.f;
    int max_color = 0;
    for (int c = 0; c < NUM_COLORS; c++) {
      float score = data[(4 + c) * num_anchors + i];
      if (score > max_conf) { max_conf = score; max_color = c; }
    }
    if (max_conf < score_threshold_) continue;

    // per-anchor 名称分类
    float max_name_score = -1.f;
    int max_name = 0;
    for (int n = 0; n < NUM_NAMES; n++) {
      float score = data[(7 + n) * num_anchors + i];
      if (score > max_name_score) { max_name_score = score; max_name = n; }
    }

    // bbox: cx,cy,w,h -> 映射回原图
    float cx = data[0 * num_anchors + i];
    float cy = data[1 * num_anchors + i];
    float w = data[2 * num_anchors + i];
    float h = data[3 * num_anchors + i];
    float x1 = (cx - w / 2.f - pad_x_) / scale_;
    float y1 = (cy - h / 2.f - pad_y_) / scale_;
    float x2 = (cx + w / 2.f - pad_x_) / scale_;
    float y2 = (cy + h / 2.f - pad_y_) / scale_;

    // 关键点（通道 15 开始）+ visibility
    // 原始顺序: k0, k1, k2, k3
    // 按yolov5的方式重排为: k0, k3, k2, k1 以匹配Solver期望的 tl, tr, br, bl
    std::vector<cv::Point2f> raw_kpts(NUM_KPTS);
    std::vector<float> raw_vis(NUM_KPTS);
    int visible_count = 0;
    for (int k = 0; k < NUM_KPTS; k++) {
      raw_kpts[k].x = (data[(15 + k * 3 + 0) * num_anchors + i] - pad_x_) / scale_;
      raw_kpts[k].y = (data[(15 + k * 3 + 1) * num_anchors + i] - pad_y_) / scale_;
      raw_vis[k] = data[(15 + k * 3 + 2) * num_anchors + i];
      if (raw_vis[k] > 0.5f) visible_count++;
    }
    // 重排: 0->0, 3->1, 2->2, 1->3 (与yolov5一致)
    std::vector<cv::Point2f> kpts = {raw_kpts[0], raw_kpts[3], raw_kpts[2], raw_kpts[1]};
    std::vector<float> vis = {raw_vis[0], raw_vis[3], raw_vis[2], raw_vis[1]};

    // 丢弃可见关键点 < 3 的检测
    if (visible_count < 3) continue;

    cv::Rect box((int)x1, (int)y1, (int)(x2 - x1), (int)(y2 - y1));
    raw_dets.push_back({box, max_conf, max_color, max_name, kpts, vis});
    nms_boxes.push_back(box);
    nms_confs.push_back(max_conf);
  }

  // NMS
  std::vector<int> indices;
  cv::dnn::NMSBoxes(nms_boxes, nms_confs, score_threshold_, nms_threshold_, indices);

  // 第二阶段：构建 Armor 对象
  std::list<Armor> armors;
  for (const auto & idx : indices) {
    const auto & det = raw_dets[idx];

    // color 映射: v2{0=blue,1=red,2=gray} -> Color{red=0,blue=1,extinguish=2}
    Color color;
    if (det.color_id == 0)
      color = Color::blue;
    else if (det.color_id == 1)
      color = Color::red;
    else
      color = Color::extinguish;

    // name 映射: v2{0=sentry,1=one,2=two,3=three,4=four,5=five,6=outpost,7=not_armor}
    ArmorName name;
    switch (det.name_id) {
      case 0: name = ArmorName::sentry; break;
      case 1: name = ArmorName::one; break;
      case 2: name = ArmorName::two; break;
      case 3: name = ArmorName::three; break;
      case 4: name = ArmorName::four; break;
      case 5: name = ArmorName::five; break;
      case 6: name = ArmorName::outpost; break;
      default: name = ArmorName::not_armor; break;
    }

    // 过滤 not_armor 和低置信度
    if (name == ArmorName::not_armor) continue;
    if (det.confidence < min_confidence_) continue;

    // ArmorType: name 推断 + 关键点宽高比辅助
    ArmorType type = infer_armor_type(name, det.kpts);

    // 构造 Armor（用 color_id/num_id 构造函数，再覆盖映射结果）
    Armor armor(det.color_id, det.name_id, det.confidence, det.box, det.kpts);
    armor.color = color;
    armor.name = name;
    armor.type = type;
    armor.kpt_visibility = det.vis;
    // 强制所有通过NMS的关键点为可见，避免Solver因visibility过滤导致PnP不稳定
    for (auto & v : armor.kpt_visibility) v = 1.0f;
    armor.center_norm = get_center_norm(bgr_img, armor.center);

    armors.push_back(armor);
  }

  if (debug_) draw_detections(bgr_img, armors, frame_count);

  return armors;
}

std::list<Armor> YOLO26N::postprocess(
  double scale, cv::Mat & output, const cv::Mat & bgr_img, int frame_count)
{
  // MultiThreadDetector 调用此接口，output 是 [27, 8400] 的 cv::Mat
  // scale 和 pad 需要从 output 的上下文重新计算
  scale_ = (float)scale;
  int new_w = (int)(bgr_img.cols * scale_);
  int new_h = (int)(bgr_img.rows * scale_);
  pad_x_ = (INPUT_SIZE - new_w) / 2.f;
  pad_y_ = (INPUT_SIZE - new_h) / 2.f;

  const float * data = (const float *)output.data;
  int num_anchors = output.cols;
  return parse(data, num_anchors, bgr_img, frame_count);
}

ArmorType YOLO26N::infer_armor_type(ArmorName name, const std::vector<cv::Point2f> & kpts) const
{
  // sentry/outpost/engineer 一定是小装甲板
  if (name == ArmorName::sentry || name == ArmorName::outpost || name == ArmorName::two)
    return ArmorType::small;

  // hero 一定是大装甲板
  if (name == ArmorName::one) return ArmorType::big;

  // 步兵 3/4/5：用关键点宽高比辅助判断
  // 宽 = 上边 + 下边平均，高 = 左边 + 右边平均
  auto top_len = cv::norm(kpts[0] - kpts[1]);
  auto bottom_len = cv::norm(kpts[3] - kpts[2]);
  auto left_len = cv::norm(kpts[0] - kpts[3]);
  auto right_len = cv::norm(kpts[1] - kpts[2]);
  auto avg_width = (top_len + bottom_len) / 2.0;
  auto avg_height = (left_len + right_len) / 2.0;
  double aspect_ratio = avg_width / (avg_height + 1e-6);

  return (aspect_ratio > BIG_ARMOR_RATIO_THRESH) ? ArmorType::big : ArmorType::small;
}

cv::Point2f YOLO26N::get_center_norm(const cv::Mat & bgr_img, const cv::Point2f & center) const
{
  return {center.x / bgr_img.cols, center.y / bgr_img.rows};
}

void YOLO26N::draw_detections(
  const cv::Mat & img, const std::list<Armor> & armors, int frame_count) const
{
  auto detection = img.clone();
  tools::draw_text(detection, fmt::format("[{}]", frame_count), {10, 30}, {255, 255, 255});
  for (const auto & armor : armors) {
    auto info = fmt::format(
      "{:.2f} {} {} {}", armor.confidence, COLORS[armor.color], ARMOR_NAMES[armor.name],
      ARMOR_TYPES[armor.type]);
    tools::draw_points(detection, armor.points, {0, 255, 0});
    tools::draw_text(detection, info, armor.center, {0, 255, 0});
  }
  cv::resize(detection, detection, {}, 0.5, 0.5);
  cv::imshow("detection", detection);
}

}  // namespace auto_aim
