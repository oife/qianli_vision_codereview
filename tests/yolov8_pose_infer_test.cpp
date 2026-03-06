/**
 * @file yolov8_pose_infer_test.cpp
 * @brief YOLOv8n-Pose FP16 模型推理性能测试 (OpenVINO C++)
 */

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>
#include <algorithm>
#include <openvino/openvino.hpp>

int main(int argc, char* argv[]) {
    std::string model_path = "armor-yolov8n-pose_fp16.onnx";
    std::string device = "GPU";
    int warmup = 50;
    int iterations = 500;

    // 解析命令行参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            device = argv[++i];
        } else if ((arg == "-w" || arg == "--warmup") && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--iterations") && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "  -m, --model      模型路径 (default: armor-yolov8n-pose_fp16.onnx)\n"
                      << "  -d, --device     推理设备 CPU/GPU (default: GPU)\n"
                      << "  -w, --warmup     预热次数 (default: 50)\n"
                      << "  -n, --iterations 测试次数 (default: 500)\n";
            return 0;
        }
    }

    std::cout << "============================================================\n";
    std::cout << "YOLOv8n-Pose FP16 推理性能测试 (C++ OpenVINO)\n";
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
    auto input = model->input();
    auto output = model->output();
    std::cout << "    输入: " << input.get_shape() << ", " << input.get_element_type() << "\n";
    std::cout << "    输出: " << output.get_shape() << ", " << output.get_element_type() << "\n";

    // 设置输入预处理（自动转换 FP32 -> FP16）
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_element_type(ov::element::f32);
    ppp.input().preprocess().convert_element_type(ov::element::f16);
    model = ppp.build();

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
    }

    // 性能测试
    std::cout << "Benchmarking (" << iterations << " iterations)...\n";
    std::vector<double> latencies;
    latencies.reserve(iterations);

    auto total_start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::steady_clock::now();
        infer_request.infer();
        auto end = std::chrono::steady_clock::now();
        double latency = std::chrono::duration<double, std::milli>(end - start).count();
        latencies.push_back(latency);

        if ((i + 1) % 100 == 0) {
            double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
            double fps = 1000.0 / avg;
            std::cout << "  [" << (i + 1) << "/" << iterations << "] "
                      << "Avg latency: " << std::fixed << std::setprecision(2) << avg
                      << " ms, FPS: " << std::setprecision(1) << fps << "\n";
        }
    }
    auto total_end = std::chrono::steady_clock::now();
    double total_time = std::chrono::duration<double>(total_end - total_start).count();

    // 计算统计数据
    std::sort(latencies.begin(), latencies.end());
    double avg_latency = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double min_latency = latencies.front();
    double max_latency = latencies.back();
    double p50 = latencies[iterations / 2];
    double p95 = latencies[static_cast<int>(iterations * 0.95)];
    double p99 = latencies[static_cast<int>(iterations * 0.99)];

    // 计算标准差
    double sq_sum = 0;
    for (double l : latencies) sq_sum += (l - avg_latency) * (l - avg_latency);
    double std_dev = std::sqrt(sq_sum / latencies.size());

    // 输出结果
    std::cout << "\n============================================================\n";
    std::cout << "测试结果\n";
    std::cout << "============================================================\n";
    std::cout << "  设备:           " << device << "\n";
    std::cout << "  总迭代次数:     " << iterations << "\n";
    std::cout << "  总耗时:         " << std::fixed << std::setprecision(2) << total_time << " s\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  平均帧率 (FPS): " << std::setprecision(1) << (iterations / total_time) << "\n";
    std::cout << "  平均延迟:       " << std::setprecision(2) << avg_latency << " ms\n";
    std::cout << "  最小延迟:       " << min_latency << " ms\n";
    std::cout << "  最大延迟:       " << max_latency << " ms\n";
    std::cout << "  标准差:         " << std_dev << " ms\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "  P50 延迟:       " << p50 << " ms\n";
    std::cout << "  P95 延迟:       " << p95 << " ms\n";
    std::cout << "  P99 延迟:       " << p99 << " ms\n";
    std::cout << "============================================================\n";

    return 0;
}
