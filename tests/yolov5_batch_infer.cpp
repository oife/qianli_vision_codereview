/**
 * @file yolov5_batch_infer.cpp
 * @brief 使用原项目 YOLOv5 批量推理图片并保存 YOLO 格式标签
 */

#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

namespace fs = std::filesystem;

class YOLOv5Detector {
public:
    YOLOv5Detector(const std::string& model_path, const std::string& device) {
        ov::Core core;
        auto model = core.read_model(model_path);

        ov::preprocess::PrePostProcessor ppp(model);
        auto& input = ppp.input();
        input.tensor()
            .set_element_type(ov::element::u8)
            .set_shape({1, 640, 640, 3})
            .set_layout("NHWC")
            .set_color_format(ov::preprocess::ColorFormat::BGR);
        input.model().set_layout("NCHW");
        input.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ov::preprocess::ColorFormat::RGB)
            .scale(255.0);

        model = ppp.build();
        compiled_model_ = core.compile_model(
            model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        infer_request_ = compiled_model_.create_infer_request();
    }

    struct Detection {
        int color_id;      // 0-3: blue, red, extinguish, purple
        int name_id;       // 0-8: sentry, 1-5, outpost, base, not_armor
        float confidence;
        float cx, cy, w, h;  // 归一化坐标
        std::vector<std::pair<float, float>> keypoints;  // 4个关键点归一化坐标
    };

    std::vector<Detection> detect(const cv::Mat& image, float conf_thresh = 0.5f) {
        int orig_h = image.rows, orig_w = image.cols;

        // 预处理：缩放到 640x640
        float x_scale = 640.0f / orig_h;
        float y_scale = 640.0f / orig_w;
        float scale = std::min(x_scale, y_scale);
        int new_h = static_cast<int>(orig_h * scale);
        int new_w = static_cast<int>(orig_w * scale);

        cv::Mat input_img = cv::Mat::zeros(640, 640, CV_8UC3);
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(new_w, new_h));
        resized.copyTo(input_img(cv::Rect(0, 0, new_w, new_h)));

        // 推理
        ov::Tensor input_tensor(ov::element::u8, {1, 640, 640, 3}, input_img.data);
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();

        // 解析输出
        auto output_tensor = infer_request_.get_output_tensor();
        auto shape = output_tensor.get_shape();
        float* output = output_tensor.data<float>();
        int rows = shape[1], cols = shape[2];

        std::vector<Detection> detections;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<int> color_ids, name_ids;
        std::vector<std::vector<std::pair<float, float>>> all_kpts;

        for (int r = 0; r < rows; r++) {
            float* row = output + r * cols;
            float conf = sigmoid(row[8]);
            if (conf < conf_thresh) continue;

            // 关键点 (像素坐标)
            std::vector<std::pair<float, float>> kpts = {
                {row[0] / scale, row[1] / scale},
                {row[6] / scale, row[7] / scale},
                {row[4] / scale, row[5] / scale},
                {row[2] / scale, row[3] / scale}
            };

            // 计算边界框
            float min_x = kpts[0].first, max_x = kpts[0].first;
            float min_y = kpts[0].second, max_y = kpts[0].second;
            for (const auto& kp : kpts) {
                min_x = std::min(min_x, kp.first);
                max_x = std::max(max_x, kp.first);
                min_y = std::min(min_y, kp.second);
                max_y = std::max(max_y, kp.second);
            }

            // 颜色和名称分类
            int color_id = 0, name_id = 0;
            float max_color = row[9], max_name = row[13];
            for (int i = 1; i < 4; i++) {
                if (row[9 + i] > max_color) { max_color = row[9 + i]; color_id = i; }
            }
            for (int i = 1; i < 9; i++) {
                if (row[13 + i] > max_name) { max_name = row[13 + i]; name_id = i; }
            }

            boxes.emplace_back(min_x, min_y, max_x - min_x, max_y - min_y);
            confidences.push_back(conf);
            color_ids.push_back(color_id);
            name_ids.push_back(name_id);
            all_kpts.push_back(kpts);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.3f, indices);

        for (int idx : indices) {
            Detection det;
            det.color_id = color_ids[idx];
            det.name_id = name_ids[idx];
            det.confidence = confidences[idx];

            // 归一化坐标
            cv::Rect& box = boxes[idx];
            det.cx = (box.x + box.width / 2.0f) / orig_w;
            det.cy = (box.y + box.height / 2.0f) / orig_h;
            det.w = box.width / static_cast<float>(orig_w);
            det.h = box.height / static_cast<float>(orig_h);

            // 归一化关键点
            for (const auto& kp : all_kpts[idx]) {
                det.keypoints.emplace_back(kp.first / orig_w, kp.second / orig_h);
            }

            detections.push_back(det);
        }

        return detections;
    }

private:
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    float sigmoid(float x) {
        return x > 0 ? 1.0f / (1.0f + std::exp(-x)) : std::exp(x) / (1.0f + std::exp(x));
    }
};

// 将 color_id 和 name_id 映射到单一类别 ID
// 这里简化为: class_id = color_id (0-3)，你可以根据需要修改映射逻辑
int map_to_class_id(int color_id, int name_id) {
    // 方案1: 只用颜色作为类别 (4类)
    // return color_id;

    // 方案2: 颜色+名称组合 (最多 4*9=36 类)
    // return color_id * 9 + name_id;

    // 方案3: 只用名称作为类别 (9类)
    return name_id;
}

void save_yolo_label(const std::string& label_path,
                     const std::vector<YOLOv5Detector::Detection>& detections) {
    std::ofstream ofs(label_path);
    for (const auto& det : detections) {
        int class_id = map_to_class_id(det.color_id, det.name_id);

        // YOLO 格式: class_id cx cy w h [kp1_x kp1_y kp2_x kp2_y ...]
        ofs << class_id << " "
            << det.cx << " " << det.cy << " "
            << det.w << " " << det.h;

        // 添加关键点
        for (const auto& kp : det.keypoints) {
            ofs << " " << kp.first << " " << kp.second;
        }
        ofs << "\n";
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "assets/yolov5.xml";
    std::string images_dir = "roi_for_classification_extracted/images";
    std::string labels_dir = "roi_for_classification_extracted/labels";
    std::string device = "GPU";
    float conf_thresh = 0.5f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-m" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) images_dir = argv[++i];
        else if (arg == "-o" && i + 1 < argc) labels_dir = argv[++i];
        else if (arg == "-d" && i + 1 < argc) device = argv[++i];
        else if (arg == "-c" && i + 1 < argc) conf_thresh = std::stof(argv[++i]);
        else if (arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -m  模型路径 (default: assets/yolov5.xml)\n"
                      << "  -i  图片目录 (default: roi_for_classification_extracted/images)\n"
                      << "  -o  标签目录 (default: roi_for_classification_extracted/labels)\n"
                      << "  -d  设备 (default: GPU)\n"
                      << "  -c  置信度阈值 (default: 0.5)\n";
            return 0;
        }
    }

    // 创建标签目录
    fs::create_directories(labels_dir);

    std::cout << "加载模型: " << model_path << "\n";
    YOLOv5Detector detector(model_path, device);

    // 获取所有图片
    std::vector<std::string> image_files;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png") {
            image_files.push_back(entry.path().string());
        }
    }
    std::sort(image_files.begin(), image_files.end());

    std::cout << "共 " << image_files.size() << " 张图片\n";
    std::cout << "开始推理...\n";

    int count = 0, detected = 0;
    for (const auto& img_path : image_files) {
        cv::Mat image = cv::imread(img_path);
        if (image.empty()) continue;

        auto detections = detector.detect(image, conf_thresh);

        // 保存标签
        std::string filename = fs::path(img_path).stem().string();
        std::string label_path = labels_dir + "/" + filename + ".txt";
        save_yolo_label(label_path, detections);

        if (!detections.empty()) detected++;
        count++;

        if (count % 1000 == 0) {
            std::cout << "进度: " << count << "/" << image_files.size()
                      << " (" << (100.0 * count / image_files.size()) << "%)"
                      << " 检测到目标: " << detected << "\n";
        }
    }

    std::cout << "完成! 共处理 " << count << " 张图片, 检测到目标 " << detected << " 张\n";
    std::cout << "标签保存到: " << labels_dir << "\n";

    return 0;
}
