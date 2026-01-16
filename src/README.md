# 源代码说明

本目录按功能分类存放主要可执行程序源代码文件。每个文件对应不同的功能模块或调试版本。

## 目录速览
- `standard/`：标准模式（含多线程与MPC变体）
- `auto_aim/`：自瞄调试相关
- `auto_buff/`：打符调试相关
- `sentry/`：哨兵与全向感知相关
- `uav/`：无人机模式相关

## standard（标准模式）
- `standard/standard.cpp`：标准自瞄程序；基础检测、追踪、瞄准，支持模式切换，使用CBoard通信。
- `standard/standard_mpc.cpp`：MPC规划版；自瞄/打符模式切换，使用Planner进行MPC，云台控制。
- `standard/mt_standard.cpp`：多线程版；自瞄与打符模式切换，多线程检测，CBoard通信。

## auto_aim（自瞄调试）
- `auto_aim/auto_aim_debug_mpc.cpp`：自瞄调试（MPC）；MPC轨迹规划、重投影可视化、绘图、云台控制。
- `auto_aim/mt_auto_aim_debug.cpp`：多线程自瞄调试；独立线程检测、重投影与绘图调试、模式切换、CBoard通信。

## auto_buff（打符调试）
- `auto_buff/auto_buff_debug_mpc.cpp`：打符调试（MPC）；MPC瞄准、能量机关可视化、绘图、云台控制。
- `auto_buff/auto_buff_debug.cpp`：打符调试（CBoard）；基础打符检测/瞄准，可视化调试，使用CBoard通信。

## sentry（哨兵与全向感知）
- `sentry/sentry.cpp`：哨兵标准；自瞄逻辑、全向感知决策（Decider）、ROS2通信、多相机支持、装甲板过滤与优先级。
- `sentry/sentry_debug.cpp`：哨兵调试；核心同上，增加重投影/绘图等可视化输出，显示追踪器与观测器数据。
- `sentry/sentry_bp.cpp`：哨兵标准备份版；与 `sentry.cpp` 类似，部分功能注释。
- `sentry/sentry_multithread.cpp`：哨兵多线程；多线程检测，全向感知器（Perceptron，4路USB相机），目标切换/丢失处理。

## uav（无人机）
- `uav/uav.cpp`：无人机模式；自瞄（auto_aim/outpost）与打符（小/大符）模式切换，自动模式管理，CBoard通信。
- `uav/uav_debug.cpp`：无人机调试；同上核心，增加重投影/绘图，可视化追踪器与观测器内部数据、卡方检验等。

## 使用说明
所有程序支持通过命令行参数指定配置文件路径：
```bash
./程序名 [配置文件路径]
```
多数程序默认使用 `configs/sentry.yaml` 或相应配置。使用 `--help` / `-h` 查看更多选项。

## 程序分类
- 调试版本：文件名含 `debug`，带可视化与详细输出
- 标准版本：生产使用的精简版
- 多线程版本：文件名含 `mt` 或 `multithread`
- MPC版本：使用模型预测控制（MPC）进行规划

