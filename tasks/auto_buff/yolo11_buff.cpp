#include "yolo11_buff.hpp"

#include <fmt/core.h>

#include <algorithm>
#include <chrono>

namespace
{
constexpr float kConfidenceThreshold = 0.7F;
constexpr float kIouThreshold = 0.4F;
}

namespace auto_buff
{
YOLO11_BUFF::YOLO11_BUFF(const std::string & config) : backend_(config)
{
  auto yaml = YAML::LoadFile(config);
  model_path_ = yaml["model"].as<std::string>("");

  if (model_path_.empty()) {
    tools::logger()->warn("[YOLO11_BUFF] no model configured, buff detector disabled");
    return;
  }

  auto ext = std::filesystem::path(model_path_).extension().string();
  if (ext != ".onnx" && ext != ".engine") {
    tools::logger()->warn(
      "[YOLO11_BUFF] Jetson path requires ONNX/TensorRT engine, current model is {}. Buff detector disabled.",
      model_path_);
    return;
  }

  auto_aim::BackendConfig cfg;
  cfg.input_size = cv::Size(INPUT_SIZE, INPUT_SIZE);
  cfg.padding_color = cv::Scalar(0, 0, 0);
  if (!backend_.init(model_path_, cfg)) {
    tools::logger()->warn("[YOLO11_BUFF] failed to initialize backend with {}", model_path_);
    return;
  }

  enabled_ = true;
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_multicandidateboxes(cv::Mat & image)
{
  return infer(image, true);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::get_onecandidatebox(cv::Mat & image)
{
  return infer(image, false);
}

std::vector<YOLO11_BUFF::Object> YOLO11_BUFF::infer(cv::Mat & image, bool use_nms)
{
  const auto start = cv::getTickCount();
  if (!enabled_ || image.empty()) return {};

  double scale = 1.0;
  cv::Mat output;
  auto ctx = backend_.create_ctx();
  if (!backend_.execute(image, output, scale, ctx.get()) || output.empty()) return {};

  const int point_dim = std::max(2, (output.rows - 5) / NUM_POINTS);
  std::vector<cv::Rect> boxes;
  std::vector<float> confidences;
  std::vector<std::vector<cv::Point2f>> keypoints;

  for (int i = 0; i < output.cols; ++i) {
    const float score = output.at<float>(4, i);
    if (score < kConfidenceThreshold) continue;

    const float cx = output.at<float>(0, i);
    const float cy = output.at<float>(1, i);
    const float ow = output.at<float>(2, i);
    const float oh = output.at<float>(3, i);

    boxes.emplace_back(
      static_cast<int>((cx - 0.5F * ow) / scale),
      static_cast<int>((cy - 0.5F * oh) / scale),
      static_cast<int>(ow / scale),
      static_cast<int>(oh / scale));
    confidences.push_back(score);

    std::vector<cv::Point2f> pts;
    for (int k = 0; k < NUM_POINTS; ++k) {
      const int row = 5 + k * point_dim;
      if (row + 1 >= output.rows) break;
      pts.emplace_back(output.at<float>(row, i) / scale, output.at<float>(row + 1, i) / scale);
    }
    keypoints.push_back(std::move(pts));
  }

  std::vector<int> selected;
  if (use_nms) {
    cv::dnn::NMSBoxes(boxes, confidences, kConfidenceThreshold, kIouThreshold, selected);
  } else if (!confidences.empty()) {
    auto best = static_cast<int>(
      std::max_element(confidences.begin(), confidences.end()) - confidences.begin());
    selected.push_back(best);
  }

  std::vector<Object> results;
  for (const int index : selected) {
    Object obj;
    obj.rect = boxes[index];
    obj.prob = confidences[index];
    obj.kpt = keypoints[index];
    draw_result(image, obj);
    results.push_back(std::move(obj));
  }

  const float cost =
    (cv::getTickCount() - start) / static_cast<float>(cv::getTickFrequency());
  cv::putText(
    image, cv::format("FPS: %.2f", 1.0 / cost), cv::Point(20, 40), cv::FONT_HERSHEY_PLAIN, 2.0,
    cv::Scalar(255, 0, 0), 2, 8);
  return results;
}

void YOLO11_BUFF::draw_result(cv::Mat & image, const Object & obj) const
{
  cv::rectangle(image, obj.rect, cv::Scalar(255, 255, 255), 1, 8);
  const auto label = fmt::format("buff:{:.2f}", obj.prob);
  const auto text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, nullptr);
  const cv::Rect text_box(
    obj.rect.tl().x, obj.rect.tl().y - 15, text_size.width, text_size.height + 5);
  cv::rectangle(image, text_box, cv::Scalar(0, 255, 255), cv::FILLED);
  cv::putText(
    image, label, cv::Point(obj.rect.tl().x, obj.rect.tl().y - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5,
    cv::Scalar(0, 0, 0));
  for (size_t i = 0; i < obj.kpt.size(); ++i) {
    cv::circle(image, obj.kpt[i], 2, cv::Scalar(255, 255, 0), -1, cv::LINE_AA);
    cv::putText(
      image, std::to_string(i + 1), obj.kpt[i] + cv::Point2f(5, -5), cv::FONT_HERSHEY_SIMPLEX,
      0.5, cv::Scalar(255, 255, 0), 1, cv::LINE_AA);
  }
}

void YOLO11_BUFF::save(const std::string & program_name, const cv::Mat & image)
{
  std::filesystem::create_directories("../result");
  cv::imwrite(fmt::format("../result/{}.jpg", program_name), image);
}

}  // namespace auto_buff
