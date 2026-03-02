/**
 * @file yolo26n_v2_infer_test.cpp
 * @brief YOLO26n-Pose+Name v2 推理性能测试 (OpenVINO C++)
 *
 * 模型 (无NMS, 单输出):
 * - 输入: [1,3,640,640] FP32
 * - output0: [1,27,8400]
 *   [0-3] cx,cy,w,h  [4-6] color(3)  [7-14] name(8)  [15-26] kpts(4x3)
 * - 所有 scores 已 sigmoid，bbox/kpts 已解码到 640 尺度
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <openvino/openvino.hpp>

struct Detection {
    float x1, y1, x2, y2;
    float confidence;
    int color_id;
    int name_id;
    float kpts[12];  // 4 keypoints * (x, y, vis)
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
                                           float conf_thresh, float iou_thresh) {
    // data 布局: [27, 8400]
    std::vector<Detection> dets;
    for (int i = 0; i < num_anchors; i++) {
        // color scores (已 sigmoid), 取 max 作为置信度
        float max_conf = -1.f;
        int max_color = 0;
        for (int c = 0; c < 3; c++) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_conf) { max_conf = score; max_color = c; }
        }
        if (max_conf < conf_thresh) continue;

        // name scores (已 sigmoid), 取 argmax
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
        det.x1 = cx - w / 2.f;
        det.y1 = cy - h / 2.f;
        det.x2 = cx + w / 2.f;
        det.y2 = cy + h / 2.f;
        det.confidence = max_conf;
        det.color_id = max_color;
        det.name_id = max_name;

        for (int k = 0; k < 4; k++) {
            det.kpts[k * 3 + 0] = data[(15 + k * 3 + 0) * num_anchors + i];
            det.kpts[k * 3 + 1] = data[(15 + k * 3 + 1) * num_anchors + i];
            det.kpts[k * 3 + 2] = data[(15 + k * 3 + 2) * num_anchors + i];
        }
        dets.push_back(det);
    }
    nms(dets, iou_thresh);
    return dets;
}

int main(int argc, char* argv[]) {
    std::string model_path = "assets/best_fp32.onnx";
    std::string device = "GPU";
    int warmup = 50;
    int iterations = 500;
    float conf_thresh = 0.25f;
    float iou_thresh = 0.45f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) model_path = argv[++i];
        else if ((arg == "-d" || arg == "--device") && i + 1 < argc) device = argv[++i];
        else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) warmup = std::stoi(argv[++i]);
        else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) iterations = std::stoi(argv[++i]);
        else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -m  模型路径 (default: assets/best_fp32.onnx)\n"
                      << "  -d  推理设备 CPU/GPU (default: GPU)\n"
                      << "  -w  预热次数 (default: 50)\n"
                      << "  -n  测试次数 (default: 500)\n";
            return 0;
        }
    }

    std::cout << "============================================================\n";
    std::cout << "YOLO26n-Pose+Name v2 推理性能测试 (无NMS, CPU后处理)\n";
    std::cout << "============================================================\n\n";

    ov::Core core;
    std::cout << "[1] 加载模型: " << model_path << "\n";
    auto model = core.read_model(model_path);

    for (size_t i = 0; i < model->inputs().size(); i++)
        std::cout << "    输入[" << i << "]: " << model->input(i).get_shape()
                  << ", " << model->input(i).get_element_type() << "\n";
    for (size_t i = 0; i < model->outputs().size(); i++)
        std::cout << "    输出[" << i << "]: " << model->output(i).get_shape()
                  << ", " << model->output(i).get_element_type() << "\n";

    std::cout << "\n[2] 编译到: " << device << "\n";
    auto compiled = core.compile_model(model, device,
        ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    auto infer_req = compiled.create_infer_request();

    // 随机输入
    ov::Tensor input_tensor(ov::element::f32, {1, 3, 640, 640});
    float* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < 1 * 3 * 640 * 640; i++)
        input_data[i] = static_cast<float>(rand()) / RAND_MAX;
    infer_req.set_input_tensor(input_tensor);

    // 预热
    std::cout << "\n[3] Warmup (" << warmup << ")...\n";
    for (int i = 0; i < warmup; i++) {
        infer_req.infer();
        auto out = infer_req.get_output_tensor(0);
        postprocess(out.data<float>(), out.get_shape()[2], conf_thresh, iou_thresh);
    }

    // 测试
    std::cout << "[4] Benchmarking (" << iterations << ")...\n";
    std::vector<double> infer_lat, nms_lat, total_lat;

    for (int i = 0; i < iterations; i++) {
        auto t0 = std::chrono::steady_clock::now();
        infer_req.infer();
        auto t1 = std::chrono::steady_clock::now();

        auto out = infer_req.get_output_tensor(0);
        auto dets = postprocess(out.data<float>(), out.get_shape()[2], conf_thresh, iou_thresh);
        auto t2 = std::chrono::steady_clock::now();

        infer_lat.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
        nms_lat.push_back(std::chrono::duration<double, std::milli>(t2 - t1).count());
        total_lat.push_back(std::chrono::duration<double, std::milli>(t2 - t0).count());

        if ((i + 1) % 100 == 0) {
            double avg_t = std::accumulate(total_lat.begin(), total_lat.end(), 0.0) / total_lat.size();
            std::cout << "  [" << (i+1) << "/" << iterations << "] "
                      << std::fixed << std::setprecision(2) << avg_t << " ms, "
                      << std::setprecision(1) << (1000.0/avg_t) << " FPS\n";
        }
    }

    // 统计
    auto stats = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq = 0; for (double l : v) sq += (l-avg)*(l-avg);
        return std::make_tuple(avg, v.front(), v.back(), v[v.size()/2],
                               v[(int)(v.size()*0.95)], v[(int)(v.size()*0.99)], std::sqrt(sq/v.size()));
    };

    auto [ai, mi, xi, p50i, p95i, p99i, si] = stats(infer_lat);
    auto [an, mn, xn, p50n, p95n, p99n, sn] = stats(nms_lat);
    auto [at, mt, xt, p50t, p95t, p99t, st] = stats(total_lat);

    // 验证
    auto out = infer_req.get_output_tensor(0);
    auto dets = postprocess(out.data<float>(), out.get_shape()[2], conf_thresh, iou_thresh);
    const char* colors[] = {"blue","red","gray"};
    const char* names[] = {"sentry","1","2","3","4","5","outpost","N/A"};

    std::cout << "\n[5] 验证: output0 " << out.get_shape()
              << ", NMS后 " << dets.size() << " 个检测\n";
    for (size_t i = 0; i < std::min(dets.size(), (size_t)3); i++) {
        auto& d = dets[i];
        std::cout << "    #" << i << ": " << colors[d.color_id] << " " << names[d.name_id]
                  << " conf=" << std::fixed << std::setprecision(3) << d.confidence << "\n";
    }

    std::cout << "\n============================================================\n";
    std::cout << "测试结果 (YOLO26n v2)\n";
    std::cout << "============================================================\n";
    std::cout << "  [纯推理 GPU]  " << std::setprecision(2) << ai << " ms | "
              << std::setprecision(1) << (1000.0/ai) << " FPS\n";
    std::cout << "    min/max: " << std::setprecision(2) << mi << "/" << xi
              << " ms, std: " << si << " ms\n";
    std::cout << "    P50/P95/P99: " << p50i << "/" << p95i << "/" << p99i << " ms\n";
    std::cout << "  [NMS CPU]     " << an << " ms\n";
    std::cout << "  [总计]        " << at << " ms | "
              << std::setprecision(1) << (1000.0/at) << " FPS\n";
    std::cout << "    min/max: " << std::setprecision(2) << mt << "/" << xt
              << " ms, std: " << st << " ms\n";
    std::cout << "    P50/P95/P99: " << p50t << "/" << p95t << "/" << p99t << " ms\n";
    std::cout << "============================================================\n";

    return 0;
}
