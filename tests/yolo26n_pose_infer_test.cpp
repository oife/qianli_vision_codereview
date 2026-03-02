/**
 * @file yolo26n_pose_infer_test.cpp
 * @brief YOLO26n-Pose+Name 推理性能测试 (OpenVINO C++)
 *
 * 模型特点 (无NMS版本):
 * - 双输出: output0 [1,19,8400] 原始检测, output1 [1,9] 名称分类
 * - 输入: [1,3,640,640] FP32
 * - 后处理: 需在CPU做NMS
 *
 * output0 每个候选框 19 维:
 *   [0-3] cx, cy, w, h
 *   [4-6] 3个颜色类别置信度 (blue/red/gray)
 *   [7-18] 4个关键点 (x,y,vis) * 4
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
    float kpts[12];  // 4 keypoints * (x, y, vis)
};

// CPU NMS
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
    for (size_t i = 0; i < dets.size(); i++) {
        if (!suppressed[i]) result.push_back(dets[i]);
    }
    dets = std::move(result);
}

// 后处理: 转置 [1,19,8400] -> 解析 -> 阈值过滤 -> NMS
static std::vector<Detection> postprocess(const float* data, int num_anchors, int dims,
                                           float conf_thresh, float iou_thresh) {
    // data 布局: [19, 8400], 需要按列读取 (第i个anchor的第j维 = data[j * num_anchors + i])
    std::vector<Detection> dets;
    for (int i = 0; i < num_anchors; i++) {
        // 取3个颜色类别的最大值作为置信度
        float max_conf = -1.f;
        int max_color = 0;
        for (int c = 0; c < 3; c++) {
            float score = data[(4 + c) * num_anchors + i];
            if (score > max_conf) {
                max_conf = score;
                max_color = c;
            }
        }
        if (max_conf < conf_thresh) continue;

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

        for (int k = 0; k < 4; k++) {
            det.kpts[k * 3 + 0] = data[(7 + k * 3 + 0) * num_anchors + i];  // x
            det.kpts[k * 3 + 1] = data[(7 + k * 3 + 1) * num_anchors + i];  // y
            det.kpts[k * 3 + 2] = data[(7 + k * 3 + 2) * num_anchors + i];  // vis
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
    std::cout << "YOLO26n-Pose+Name 推理性能测试 (无NMS, CPU后处理)\n";
    std::cout << "============================================================\n\n";

    // 初始化 OpenVINO
    std::cout << "[1] 初始化 OpenVINO...\n";
    ov::Core core;
    auto devices = core.get_available_devices();
    std::cout << "    可用设备: ";
    for (const auto& d : devices) std::cout << d << " ";
    std::cout << "\n";

    // 加载模型
    std::cout << "\n[2] 加载模型: " << model_path << "\n";
    auto model = core.read_model(model_path);

    // 显示输入输出信息
    std::cout << "    输入数量: " << model->inputs().size() << "\n";
    for (size_t i = 0; i < model->inputs().size(); i++) {
        auto input = model->input(i);
        std::cout << "    输入[" << i << "]: " << input.get_shape()
                  << ", " << input.get_element_type() << "\n";
    }
    std::cout << "    输出数量: " << model->outputs().size() << "\n";
    for (size_t i = 0; i < model->outputs().size(); i++) {
        auto output = model->output(i);
        std::cout << "    输出[" << i << "]: " << output.get_shape()
                  << ", " << output.get_element_type() << "\n";
    }

    // 编译模型
    std::cout << "\n[3] 编译模型到设备: " << device << "\n";
    ov::CompiledModel compiled_model = core.compile_model(
        model, device, ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
    ov::InferRequest infer_request = compiled_model.create_infer_request();

    // 准备输入数据
    std::cout << "\n[4] 准备输入数据...\n";
    std::cout << "    使用随机数据 (640x640)\n";
    ov::Tensor input_tensor(ov::element::f32, {1, 3, 640, 640});
    float* input_data = input_tensor.data<float>();
    for (size_t i = 0; i < 1 * 3 * 640 * 640; i++) {
        input_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    infer_request.set_input_tensor(input_tensor);

    // 预热
    std::cout << "\n[5] 开始性能测试...\n";
    std::cout << "Warmup (" << warmup << " iterations)...\n";
    for (int i = 0; i < warmup; i++) {
        infer_request.infer();
        auto out0 = infer_request.get_output_tensor(0);
        const float* data = out0.data<float>();
        int num_anchors = out0.get_shape()[2];  // 8400
        int dims = out0.get_shape()[1];         // 19
        postprocess(data, num_anchors, dims, conf_thresh, iou_thresh);
    }

    // 性能测试 - 分别计时推理和后处理
    std::cout << "Benchmarking (" << iterations << " iterations)...\n";
    std::vector<double> infer_latencies, total_latencies, nms_latencies;
    infer_latencies.reserve(iterations);
    total_latencies.reserve(iterations);
    nms_latencies.reserve(iterations);

    auto bench_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        // 推理计时
        auto t0 = std::chrono::steady_clock::now();
        infer_request.infer();
        auto t1 = std::chrono::steady_clock::now();

        // NMS后处理计时
        auto out0 = infer_request.get_output_tensor(0);
        const float* data = out0.data<float>();
        int num_anchors = out0.get_shape()[2];
        int dims = out0.get_shape()[1];
        auto dets = postprocess(data, num_anchors, dims, conf_thresh, iou_thresh);
        auto t2 = std::chrono::steady_clock::now();

        double infer_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        double nms_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        double total_ms = std::chrono::duration<double, std::milli>(t2 - t0).count();

        infer_latencies.push_back(infer_ms);
        nms_latencies.push_back(nms_ms);
        total_latencies.push_back(total_ms);

        if ((i + 1) % 100 == 0) {
            double avg_infer = std::accumulate(infer_latencies.begin(), infer_latencies.end(), 0.0) / infer_latencies.size();
            double avg_total = std::accumulate(total_latencies.begin(), total_latencies.end(), 0.0) / total_latencies.size();
            std::cout << "  [" << (i + 1) << "/" << iterations << "] "
                      << "推理: " << std::fixed << std::setprecision(2) << avg_infer
                      << " ms, 总计: " << avg_total << " ms, FPS: "
                      << std::setprecision(1) << (1000.0 / avg_total) << "\n";
        }
    }
    auto bench_end = std::chrono::steady_clock::now();
    double bench_time = std::chrono::duration<double>(bench_end - bench_start).count();

    // 统计函数
    auto calc_stats = [](std::vector<double>& v) {
        std::sort(v.begin(), v.end());
        double avg = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
        double sq_sum = 0;
        for (double l : v) sq_sum += (l - avg) * (l - avg);
        double std_dev = std::sqrt(sq_sum / v.size());
        return std::make_tuple(avg, v.front(), v.back(),
                               v[v.size() / 2], v[(int)(v.size() * 0.95)], v[(int)(v.size() * 0.99)], std_dev);
    };

    auto [avg_infer, min_infer, max_infer, p50_infer, p95_infer, p99_infer, std_infer] = calc_stats(infer_latencies);
    auto [avg_nms, min_nms, max_nms, p50_nms, p95_nms, p99_nms, std_nms] = calc_stats(nms_latencies);
    auto [avg_total, min_total, max_total, p50_total, p95_total, p99_total, std_total] = calc_stats(total_latencies);

    // 验证输出
    std::cout << "\n[6] 验证推理输出...\n";
    auto out0 = infer_request.get_output_tensor(0);
    auto out1 = infer_request.get_output_tensor(1);
    std::cout << "    output0 shape: " << out0.get_shape() << "\n";
    std::cout << "    output1 shape: " << out1.get_shape() << "\n";

    const float* data = out0.data<float>();
    int num_anchors = out0.get_shape()[2];
    int dims = out0.get_shape()[1];
    auto final_dets = postprocess(data, num_anchors, dims, conf_thresh, iou_thresh);
    std::cout << "    NMS后检测数: " << final_dets.size() << " (conf>" << conf_thresh << ", iou>" << iou_thresh << ")\n";

    // 名称分类
    const float* cls_data = out1.data<float>();
    int max_cls = 0;
    float max_score = cls_data[0];
    for (int i = 1; i < 9; i++) {
        if (cls_data[i] > max_score) { max_score = cls_data[i]; max_cls = i; }
    }
    const char* names[] = {"sentry","one","two","three","four","five","outpost","base","not_armor"};
    std::cout << "    名称分类: " << names[max_cls] << " (logit: " << max_score << ")\n";

    // 输出结果
    std::cout << "\n============================================================\n";
    std::cout << "测试结果\n";
    std::cout << "============================================================\n";
    std::cout << "  设备:           " << device << "\n";
    std::cout << "  总迭代次数:     " << iterations << "\n";
    std::cout << "  总耗时:         " << std::fixed << std::setprecision(2) << bench_time << " s\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  [纯推理 (GPU)]\n";
    std::cout << "    平均延迟:     " << std::setprecision(2) << avg_infer << " ms\n";
    std::cout << "    平均FPS:      " << std::setprecision(1) << (1000.0 / avg_infer) << "\n";
    std::cout << "    最小/最大:    " << std::setprecision(2) << min_infer << " / " << max_infer << " ms\n";
    std::cout << "    标准差:       " << std_infer << " ms\n";
    std::cout << "    P50/P95/P99:  " << p50_infer << " / " << p95_infer << " / " << p99_infer << " ms\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  [NMS后处理 (CPU)]\n";
    std::cout << "    平均延迟:     " << avg_nms << " ms\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  [推理+NMS 总计]\n";
    std::cout << "    平均延迟:     " << avg_total << " ms\n";
    std::cout << "    平均FPS:      " << std::setprecision(1) << (1000.0 / avg_total) << "\n";
    std::cout << "    最小/最大:    " << std::setprecision(2) << min_total << " / " << max_total << " ms\n";
    std::cout << "    标准差:       " << std_total << " ms\n";
    std::cout << "    P50/P95/P99:  " << p50_total << " / " << p95_total << " / " << p99_total << " ms\n";
    std::cout << "============================================================\n";

    return 0;
}
