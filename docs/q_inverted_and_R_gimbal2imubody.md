# 四元数反向 (q_inverted) 与 R_gimbal2imubody 调试指南

> 背景：2024-03-14 调试无人机自瞄时发现的问题及解决方案

## 问题描述

在无人机上运行 `./build/hero configs/uav.yaml` 时：
- `hero_auto_aim` 窗口中红色框（aimer）和绿色框（EKF 预测）**位置正确**
- 但云台 yaw **突然大角度旋转**，不瞄准目标
- 尝试了 `calibration.yaml` 中全部 8 个 `R_gimbal2imubody`，均无法正常自瞄
- 其中 2 个能让显示正确，但标定偏角出现 ~170° 的异常值

同期，英雄机器人使用 `R_gimbal2imubody: [1, 0, 0, 0, -1, 0, 0, 0, 1]` 可以正常自瞄。

## 根因分析

### 1. 为什么显示正确但控制错误？

重投影显示是**闭环**过程：`R_gimbal2world` 把装甲板从云台系→世界系，`reproject_armor` 再用同一个 `R_gimbal2world` 的逆从世界系→相机系。误差互相抵消，所以显示永远正确。

而控制是**开环**：`aimer.cpp` 计算 `yaw = atan2(y, x)` 得到世界系绝对角，直接发给下位机。如果世界系错了，yaw 就是错的。

### 2. 为什么 8 个 R_gimbal2imubody 全部失败？

经电控 AI 分析无人机固件代码，发现：

| 项目 | 视觉代码假设 | 无人机固件实际 |
|------|-------------|---------------|
| **四元数 q 含义** | `R_sensor→world` | `R_world→sensor`（反向） |

**关键数学结论：** 如果四元数方向反了（`R` 变成 `R^T`），那么**不存在任何** `R_gimbal2imubody`（无论是 8 个对角矩阵还是全部 24 个合法旋转矩阵）能同时让 yaw 和 pitch 都正确。

证明：需要 `m1*m2 = -r1*r2`、`m1*m3 = -r1*r3`、`m2*m3 = -r2*r3` 同时成立，三式相乘得 `1 = -1`，矛盾。

这就是为什么 170° 偏角会出现——某些候选值能让 yaw 近似正确但 pitch 错误，或反之，但无法两者同时正确。

### 3. 为什么英雄能正常工作？

英雄和无人机使用**不同的电控固件**。英雄固件的四元数约定与视觉代码一致（`R_sensor→world`），所以不需要转置。

## 解决方案

### 新增 `q_inverted` 配置项

**commit:** `29e291b`

在配置文件中可选地添加：

```yaml
q_inverted: true
```

效果：在 `set_R_gimbal2world()` 中对四元数旋转矩阵做转置，将 `R_world→sensor` 转为 `R_sensor→world`。

**默认值为 false，不加此行则行为不变，现有配置文件无需修改。**

### 修改的文件

| 文件 | 改动 |
|------|------|
| `tasks/auto_aim/solver/solver.hpp` | 加 `bool q_inverted_` 成员 |
| `tasks/auto_aim/solver/solver.cpp` | 读配置 + 转置 |
| `tasks/auto_buff/buff_solver.hpp` | 加 `bool q_inverted_` 成员 |
| `tasks/auto_buff/buff_solver.cpp` | 读配置 + 转置 |
| `calibration/calibrate_handeye.cpp` | 支持转置 |
| `calibration/calibrate_robotworld_handeye.cpp` | 支持转置 |
| `tests/find_R_gimbal2imubody.cpp` | 新增诊断工具 |
| `CMakeLists.txt` | 注册诊断工具 |

## 后续操作步骤

### 第一步：编译

```bash
cd build && cmake .. && make hero find_R_gimbal2imubody calibrate_robotworld_handeye
```

### 第二步：在无人机配置文件中加 q_inverted

在无人机的 yaml 配置文件中加一行：

```yaml
q_inverted: true
```

同时在 `calibration.yaml`（用于标定的配置）中也加上这一行。

### 第三步：用诊断工具找到正确的 R_gimbal2imubody

```bash
./build/find_R_gimbal2imubody configs/uav.yaml
```

操作方法：
1. 让云台水平朝前，观察哪些候选值的 pitch ≈ 0, roll ≈ 0（标记为绿色 CANDIDATE）
2. 缓慢**左右转**云台，看哪些候选值**只有 yaw 变化**，pitch/roll 基本不动
3. 缓慢**上下抬低**云台，看哪些候选值**只有 pitch 变化**，yaw/roll 基本不动
4. 同时满足以上条件的就是正确的 R_gimbal2imubody
5. 按 `n`/`p` 翻页，每页 6 个候选值，共 24 个（4 页）

### 第四步：重新手眼标定

将找到的 `R_gimbal2imubody` 写入 `calibration.yaml`（确保 `q_inverted: true` 也在），然后：

```bash
# 拍标定照片（如果还没拍的话）
./build/capture -c configs/calibration.yaml

# 运行手眼标定
./build/calibrate_robotworld_handeye assets/img_with_q -c configs/calibration.yaml
```

标定结果应满足：
- 相机偏角 yaw/pitch/roll 全在 **±10° 以内**（绝不应出现 170°）
- 标定板距离合理

### 第五步：写入配置并测试

将标定输出的 `R_gimbal2imubody`、`R_camera2gimbal`、`t_camera2gimbal` 写入无人机配置文件，运行：

```bash
./build/hero configs/uav.yaml
```

验证：
- 显示窗口红绿框位置正确
- 云台跟踪目标，yaw 不发生大角度跳变
- 瞄准点稳定

## 电控固件差异对照表

以下是电控 AI 分析的**无人机固件**（不适用于英雄）的关键信息：

| 项目 | 无人机固件行为 |
|------|---------------|
| 四元数 q | `R_world→sensor`，wxyz 顺序，AHRS 初始化方向为零点 |
| 回传 yaw/pitch | 电机编码器原始值，**不是角度** |
| 接收 yaw | 相对进入视觉模式时编码器位置的偏移量 (rad) |
| 接收 pitch | 取反后与 IMU pitch 比较 |
| yaw_vel/yaw_acc | 已解析但**未使用** |

> **注意：** gimbal_test 已验证 yaw/pitch 控制正常，说明串口协议本身没有问题。
> yaw 大角度跳变纯粹是因为四元数反向导致世界系计算错误，进而 `atan2(y,x)` 产生了错误的绝对角。

## 各车型 R_gimbal2imubody 参考

| 车型 | R_gimbal2imubody | q_inverted |
|------|------------------|------------|
| hero | `[1, 0, 0, 0, -1, 0, 0, 0, 1]` | 不需要 |
| standard3/4 | `[1, 0, 0, 0, 1, 0, 0, 0, 1]` | 不需要 |
| sentry | `[1, 0, 0, 0, -1, 0, 0, 0, 1]` | 不需要 |
| uav | **待确定**（用 find_R_gimbal2imubody 工具找） | `true` |
