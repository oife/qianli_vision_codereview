# Calibration 标定工具

本目录包含用于相机和手眼标定的工具程序。

## 文件说明

### capture.cpp
数据采集程序，用于采集标定所需的图片和对应的IMU四元数数据。
- 从相机读取图像
- 从CAN总线或串口读取IMU数据
- 实时显示标定板识别结果和IMU欧拉角
- 按 `s` 键保存图片和对应的四元数，按 `q` 键退出
- 输出文件：`{序号}.jpg`（图片）和 `{序号}.txt`（四元数，格式为wxyz）

### calibrate_camera.cpp
相机内参标定程序，用于标定相机的内参矩阵和畸变系数。
- 从输入文件夹读取标定板图片
- 使用圆点网格标定板进行标定
- 计算相机内参矩阵（camera_matrix）和畸变系数（distort_coeffs）
- 输出重投影误差和标定结果（YAML格式）

### calibrate_handeye.cpp
手眼标定程序，用于标定相机与云台之间的相对位姿。
- 从输入文件夹读取图片和对应的IMU四元数
- 计算云台在世界坐标系中的位姿
- 使用OpenCV的`calibrateHandEye`函数进行手眼标定
- 输出相机到云台的旋转矩阵（R_camera2gimbal）和平移向量（t_camera2gimbal）
- 计算并显示相机相对理想情况的偏角

### calibrate_robotworld_handeye.cpp
机器人世界手眼标定程序，同时标定相机与云台的相对位姿以及标定板在世界坐标系中的位姿。
- 从输入文件夹读取图片和对应的IMU四元数
- 使用OpenCV的`calibrateRobotWorldHandEye`函数进行标定
- 同时输出相机到云台的位姿（R_camera2gimbal, t_camera2gimbal）和标定板到世界的位姿（R_board2world, t_board2world）
- 计算并显示相机偏角、标定板到世界坐标系原点的水平距离以及标定板的偏角

### split_video.cpp
视频分割工具，用于从视频文件中提取指定帧范围的视频和对应的文本数据。
- 读取AVI视频文件和对应的TXT文本文件（包含时间戳和四元数）
- 根据指定的起始帧和结束帧索引提取视频片段
- 输出提取的视频片段和对应的文本数据
- 用于从录制的视频中提取标定所需的数据

## 使用流程

1. **数据采集**：使用 `capture.cpp` 采集多组不同姿态下的标定板图片和IMU数据
2. **相机标定**：使用 `calibrate_camera.cpp` 标定相机内参
3. **手眼标定**：使用 `calibrate_handeye.cpp` 或 `calibrate_robotworld_handeye.cpp` 进行手眼标定
4. **（可选）视频处理**：使用 `split_video.cpp` 从录制的视频中提取标定数据

## 注意事项

- 所有程序都需要配置文件（默认路径：`configs/calibration.yaml`）
- 标定板使用圆点网格图案（默认尺寸：10列7行）
- 四元数输出顺序为 wxyz
- 确保采集数据时标定板在不同姿态下都能被正确识别

