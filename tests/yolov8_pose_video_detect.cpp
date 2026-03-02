/**
 * @file yolov8_pose_video_detect.cpp
 * @brief YOLOv8n-Pose 视频检测，只标注关键点（红点）和连接线（绿色四边形）
 */

#include <chrono>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

struct Detection {
    float confidence;
    std::vector<cv::Point2f> keypoints;  // 4个关键点
    std::vector<bool> kpt_valid;         // 关键点是否有效
};

class YOLOv8PoseDetector {
public:
    YOLOv8PoseDetector(const std::string& model_path, const std::string& device) {
        ov::Core core;
        auto model = core.read_model(model_path);

        // 设置输入预处理（自动转换 FP32 -> FP16）
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32);
        ppp.input().preprocess().convert_element_type(ov::element::f16);
        ppp.output().postprocess().convert_element_type(ov::element::f32);
        model = ppp.build();

        compiled_model_ = core.compile_model(
            model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
        infer_request_ = compiled_model_.create_infer_request();
    }

    std::vector<Detection> detect(const cv::Mat& image, float conf_thresh = 0.25f) {
        int orig_w = image.cols, orig_h = image.rows;

        // 预处理
        cv::Mat input_image;
        float scale = std::min(640.0f / orig_w, 640.0f / orig_h);
        int new_w = static_cast<int>(orig_w * scale);
        int new_h = static_cast<int>(orig_h * scale);
        cv::resize(image, input_image, cv::Size(new_w, new_h));

        cv::Mat canvas = cv::Mat::zeros(640, 640, CV_8UC3);
        int pad_x = (640 - new_w) / 2, pad_y = (640 - new_h) / 2;
        input_image.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));

        cv::cvtColor(canvas, canvas, cv::COLOR_BGR2RGB);
        canvas.convertTo(canvas, CV_32F, 1.0 / 255.0);

        // HWC -> NCHW
        ov::Tensor input_tensor(ov::element::f32, {1, 3, 640, 640});
        float* input_data = input_tensor.data<float>();
        std::vector<cv::Mat> channels(3);
        cv::split(canvas, channels);
        for (int c = 0; c < 3; c++) {
            std::memcpy(input_data + c * 640 * 640, channels[c].data, 640 * 640 * sizeof(float));
        }

        // 推理
        infer_request_.set_input_tensor(input_tensor);
        infer_request_.infer();

        // 后处理
        const float* output = infer_request_.get_output_tensor().data<float>();
        return postprocess(output, orig_w, orig_h, scale, pad_x, pad_y, conf_thresh);
    }

private:
    ov::CompiledModel compiled_model_;
    ov::InferRequest infer_request_;

    std::vector<Detection> postprocess(const float* output, int orig_w, int orig_h,
                                        float scale, int pad_x, int pad_y, float conf_thresh) {
        std::vector<Detection> results;
        std::vector<cv::Rect> boxes;
        std::vector<float> confidences;
        std::vector<std::vector<cv::Point2f>> all_kpts;
        std::vector<std::vector<bool>> all_kpt_valid;

        // output shape: [1, 19, 8400] -> 遍历 8400 个 anchor
        for (int i = 0; i < 8400; i++) {
            float cx = output[0 * 8400 + i];
            float cy = output[1 * 8400 + i];
            float w = output[2 * 8400 + i];
            float h = output[3 * 8400 + i];
            float conf = output[4 * 8400 + i];

            if (conf < conf_thresh) continue;

            // 边界框 (用于 NMS)
            int x1 = static_cast<int>((cx - w / 2) * 640);
            int y1 = static_cast<int>((cy - h / 2) * 640);
            int bw = static_cast<int>(w * 640);
            int bh = static_cast<int>(h * 640);
            boxes.emplace_back(x1, y1, bw, bh);
            confidences.push_back(conf);

            // 关键点
            std::vector<cv::Point2f> kpts(4);
            std::vector<bool> kpt_valid(4, false);
            for (int j = 0; j < 4; j++) {
                float kx = output[(5 + j * 3 + 0) * 8400 + i];
                float ky = output[(5 + j * 3 + 1) * 8400 + i];
                if (kx > 0.001f && ky > 0.001f) {
                    float px = (kx * 640 - pad_x) / scale;
                    float py = (ky * 640 - pad_y) / scale;
                    px = std::max(0.0f, std::min(static_cast<float>(orig_w - 1), px));
                    py = std::max(0.0f, std::min(static_cast<float>(orig_h - 1), py));
                    kpts[j] = cv::Point2f(px, py);
                    kpt_valid[j] = true;
                }
            }
            all_kpts.push_back(kpts);
            all_kpt_valid.push_back(kpt_valid);
        }

        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, conf_thresh, 0.5f, indices);

        for (int idx : indices) {
            Detection det;
            det.confidence = confidences[idx];
            det.keypoints = all_kpts[idx];
            det.kpt_valid = all_kpt_valid[idx];
            results.push_back(det);
        }
        return results;
    }
};

void draw_keypoints(cv::Mat& image, const std::vector<Detection>& detections) {
    for (const auto& det : detections) {
        // 绘制绿色连接线 (按顺序: 0->1->2->3->0)
        int order[] = {0, 1, 2, 3, 0};
        for (int i = 0; i < 4; i++) {
            int idx1 = order[i], idx2 = order[i + 1];
            if (det.kpt_valid[idx1] && det.kpt_valid[idx2]) {
                cv::line(image, det.keypoints[idx1], det.keypoints[idx2],
                         cv::Scalar(0, 255, 0), 2);
            }
        }
        // 绘制红色关键点
        for (int i = 0; i < 4; i++) {
            if (det.kpt_valid[i]) {
                cv::circle(image, det.keypoints[i], 5, cv::Scalar(0, 0, 255), -1);
            }
        }
    }
}

int main(int argc, char* argv[]) {
    std::string model_path = "armor-yolov8n-pose_fp16.onnx";
    std::string input_path, output_path;
    std::string device = "GPU";
    float conf_thresh = 0.25f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "-i" || arg == "--input") && i + 1 < argc) input_path = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) output_path = argv[++i];
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) device = argv[++i];
        else if ((arg == "-c" || arg == "--conf") && i + 1 < argc) conf_thresh = std::stof(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " -i input.mp4 -o output.mp4 [options]\n"
                      << "  -m  模型路径\n  -i  输入视频\n  -o  输出视频\n"
                      << "  -d  设备 (CPU/GPU)\n  -c  置信度阈值\n";
            return 0;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "请指定输入和输出视频路径 (-i, -o)\n";
        return 1;
    }

    std::cout << "加载模型: " << model_path << "\n";
    YOLOv8PoseDetector detector(model_path, device);

    std::cout << "打开视频: " << input_path << "\n";
    cv::VideoCapture cap(input_path);
    if (!cap.isOpened()) { std::cerr << "无法打开视频\n"; return 1; }

    int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    std::cout << "视频信息: " << width << "x" << height << " @ " << fps << "fps, 共 " << total << " 帧\n";

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, cv::Size(width, height));

    cv::Mat frame;
    int count = 0;
    while (cap.read(frame)) {
        auto dets = detector.detect(frame, conf_thresh);
        draw_keypoints(frame, dets);
        writer.write(frame);
        if (++count % 100 == 0)
            std::cout << "处理进度: " << count << "/" << total << " (" << (100.0 * count / total) << "%)\n";
    }

    std::cout << "完成! 输出: " << output_path << "\n";
    return 0;
}
