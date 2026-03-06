/**
 * @file yolo26n_v2_video_detect.cpp
 * @brief YOLO26n-Pose+Name v2 视频检测 (OpenVINO C++)
 *
 * 模型 (无NMS, 单输出):
 * - 输入: [1,3,640,640] FP32
 * - output0: [1,27,8400]
 *   [0-3] cx,cy,w,h  [4-6] color(3)  [7-14] name(8)  [15-26] kpts(4x3)
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <openvino/openvino.hpp>
#include <opencv2/opencv.hpp>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int color_id;
    int name_id;
    float kpts[12];
};

static void nms(std::vector<Detection>& dets, float iou_thresh) {
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
    std::vector<bool> suppressed(dets.size(), false);
    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            float ix1 = std::max(dets[i].x1, dets[j].x1);
            float iy1 = std::max(dets[i].y1, dets[j].y1);
            float ix2 = std::min(dets[i].x2, dets[j].x2);
            float iy2 = std::min(dets[i].y2, dets[j].y2);
            float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
            float area_i = (dets[i].x2 - dets[i].x1) * (dets[i].y2 - dets[i].y1);
            float area_j = (dets[j].x2 - dets[j].x1) * (dets[j].y2 - dets[j].y1);
            float iou = inter / (area_i + area_j - inter + 1e-6f);
            if (iou > iou_thresh) suppressed[j] = true;
        }
    }
    std::vector<Detection> result;
    for (size_t i = 0; i < dets.size(); i++)
        if (!suppressed[i]) result.push_back(dets[i]);
    dets = std::move(result);
}

static std::vector<Detection> postprocess(const float* data, int num_anchors,
                                           float conf_thresh, float iou_thresh,
                                           float scale, float pad_x, float pad_y) {
    std::vector<Detection> dets;
    for (int i = 0; i < num_anchors; i++) {
        float max_conf = -1.f;
        int max_color = 0;
        for (int c = 0; c < 3; c++) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_conf) { max_conf = score; max_color = c; }
        }
        if (max_conf < conf_thresh) continue;

        // per-anchor 名称分类
        float max_name_score = -1.f;
        int max_name = 0;
        for (int n = 0; n < 8; n++) {
            float score = data[(7 + n) * num_anchors + i];
            if (score > max_name_score) { max_name_score = score; max_name = n; }
        }

        float cx = data[0 * num_anchors + i];
        float cy = data[1 * num_anchors + i];
        float w  = data[2 * num_anchors + i];
        float h  = data[3 * num_anchors + i];

        Detection det;
        det.x1 = (cx - w / 2.f - pad_x) / scale;
        det.y1 = (cy - h / 2.f - pad_y) / scale;
        det.x2 = (cx + w / 2.f - pad_x) / scale;
        det.y2 = (cy + h / 2.f - pad_y) / scale;
        det.confidence = max_conf;
        det.color_id = max_color;
        det.name_id = max_name;

        // 关键点从通道15开始
        for (int k = 0; k < 4; k++) {
            det.kpts[k * 3 + 0] = (data[(15 + k * 3 + 0) * num_anchors + i] - pad_x) / scale;
            det.kpts[k * 3 + 1] = (data[(15 + k * 3 + 1) * num_anchors + i] - pad_y) / scale;
            det.kpts[k * 3 + 2] = data[(15 + k * 3 + 2) * num_anchors + i];
        }
        dets.push_back(det);
    }
    nms(dets, iou_thresh);
    return dets;
}

int main(int argc, char* argv[]) {
    std::string model_path = "assets/best_fp32.onnx";
    std::string video_path = "raw_video.mp4";
    std::string output_path = "output_yolo26n_v2.mp4";
    std::string device = "GPU";
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "-i" || arg == "--input") && i + 1 < argc) video_path = argv[++i];
        else if ((arg == "-o" || arg == "--output") && i + 1 < argc) output_path = argv[++i];
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) device = argv[++i];
        else if ((arg == "-c" || arg == "--conf") && i + 1 < argc) conf_thresh = std::stof(argv[++i]);
    }

    const int INPUT_SIZE = 640;
    const char* color_names[] = {"blue", "red", "gray"};
    const cv::Scalar color_colors[] = {
        cv::Scalar(255, 0, 0),    // blue
        cv::Scalar(0, 0, 255),    // red
        cv::Scalar(128, 128, 128) // gray
    };
    const char* armor_names[] = {"sentry","1","2","3","4","5","outpost","N/A"};

    std::cout << "加载模型: " << model_path << "\n";
    ov::Core core;
    auto model = core.read_model(model_path);

    if (model->input(0).get_element_type() == ov::element::f16) {
        std::cout << "检测到 FP16 模型，添加类型转换\n";
        ov::preprocess::PrePostProcessor ppp(model);
        ppp.input().tensor().set_element_type(ov::element::f32);
        ppp.input().preprocess().convert_element_type(ov::element::f16);
        for (size_t i = 0; i < model->outputs().size(); i++)
            ppp.output(i).postprocess().convert_element_type(ov::element::f32);
        model = ppp.build();
    }

    auto compiled = core.compile_model(model, device,
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    auto infer_req = compiled.create_infer_request();

    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) { std::cerr << "无法打开视频\n"; return 1; }
    int frame_w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int frame_h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    int total_frames = (int)cap.get(cv::CAP_PROP_FRAME_COUNT);
    std::cout << "视频: " << frame_w << "x" << frame_h << " @ " << fps << " FPS, " << total_frames << " 帧\n";

    cv::VideoWriter writer(output_path, cv::VideoWriter::fourcc('m','p','4','v'), fps, {frame_w, frame_h});
    if (!writer.isOpened()) { std::cerr << "无法创建输出视频\n"; return 1; }

    float scale = std::min((float)INPUT_SIZE / frame_w, (float)INPUT_SIZE / frame_h);
    int new_w = (int)(frame_w * scale);
    int new_h = (int)(frame_h * scale);
    float pad_x = (INPUT_SIZE - new_w) / 2.f;
    float pad_y = (INPUT_SIZE - new_h) / 2.f;

    ov::Tensor input_tensor(ov::element::f32, {1, 3, INPUT_SIZE, INPUT_SIZE});
    infer_req.set_input_tensor(input_tensor);

    cv::Mat frame, resized;
    int frame_count = 0;
    double total_infer_ms = 0;
    auto t_start = std::chrono::steady_clock::now();

    while (cap.read(frame)) {
        frame_count++;

        // letterbox
        cv::resize(frame, resized, {new_w, new_h});
        cv::Mat padded(INPUT_SIZE, INPUT_SIZE, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect((int)pad_x, (int)pad_y, new_w, new_h)));

        // HWC BGR -> NCHW RGB
        float* input_data = input_tensor.data<float>();
        for (int y = 0; y < INPUT_SIZE; y++) {
            const uchar* row = padded.ptr<uchar>(y);
            for (int x = 0; x < INPUT_SIZE; x++) {
                int idx = y * INPUT_SIZE + x;
                input_data[0 * INPUT_SIZE * INPUT_SIZE + idx] = row[x * 3 + 2] / 255.f;
                input_data[1 * INPUT_SIZE * INPUT_SIZE + idx] = row[x * 3 + 1] / 255.f;
                input_data[2 * INPUT_SIZE * INPUT_SIZE + idx] = row[x * 3 + 0] / 255.f;
            }
        }

        auto t0 = std::chrono::steady_clock::now();
        infer_req.infer();
        auto t1 = std::chrono::steady_clock::now();
        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_infer_ms += infer_ms;

        auto out0 = infer_req.get_output_tensor(0);
        const float* det_data = out0.data<float>();
        int num_anchors = out0.get_shape()[2];

        auto dets = postprocess(det_data, num_anchors, conf_thresh, iou_thresh,
                                scale, pad_x, pad_y);

        for (const auto& det : dets) {
            cv::Point pts[4];
            bool visible[4];
            for (int k = 0; k < 4; k++) {
                pts[k] = cv::Point((int)det.kpts[k * 3], (int)det.kpts[k * 3 + 1]);
                visible[k] = det.kpts[k * 3 + 2] > 0.5f;
                if (visible[k])
                    cv::circle(frame, pts[k], 4, cv::Scalar(0, 0, 255), -1);
            }
            for (int k = 0; k < 4; k++) {
                int next = (k + 1) % 4;
                if (visible[k] && visible[next])
                    cv::line(frame, pts[k], pts[next], cv::Scalar(0, 255, 0), 2);
            }

            std::string label = std::string(color_names[det.color_id]) + " " +
                                armor_names[det.name_id] + " " +
                                std::to_string((int)(det.confidence * 100)) + "%";
            cv::putText(frame, label, cv::Point(pts[0].x, pts[0].y - 8),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color_colors[det.color_id], 2);
        }

        std::string info = "YOLO26n-v2 | " + std::to_string((int)(1000.0 / infer_ms)) + " FPS | " +
                           std::to_string(dets.size()) + " dets";
        cv::putText(frame, info, {10, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.7, {0, 255, 0}, 2);

        writer.write(frame);

        if (frame_count % 200 == 0) {
            double avg = total_infer_ms / frame_count;
            std::cout << "  [" << frame_count << "/" << total_frames << "] "
                      << "平均推理: " << std::fixed << std::setprecision(2) << avg << " ms\n";
        }
    }

    auto t_end = std::chrono::steady_clock::now();
    double wall_time = std::chrono::duration<double>(t_end - t_start).count();
    cap.release();
    writer.release();

    std::cout << "\n完成!\n";
    std::cout << "  处理帧数: " << frame_count << "\n";
    std::cout << "  总耗时: " << std::fixed << std::setprecision(2) << wall_time << " s\n";
    std::cout << "  平均推理: " << (total_infer_ms / frame_count) << " ms\n";
    std::cout << "  输出: " << output_path << "\n";

    return 0;
}
