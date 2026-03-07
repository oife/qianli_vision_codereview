#!/bin/bash
# 对比新旧版本 auto_aim_test 的 reprojection 输出
# 用法: bash tools/compare_pixel_update.sh [input_path] [config_path]
#
# 此脚本会：
# 1. 用当前(新)代码运行，保存 reprojection 视频
# 2. 临时回退 pixel-update 相关改动，用旧代码运行，保存 reprojection 视频
# 3. 恢复新代码
# 4. 用 Python 拼出左右对比视频

set -e

INPUT_PATH="${1:-assets/demo/demo}"
CONFIG_PATH="${2:-configs/camera_detect_yolo26n.yaml}"
OUTPUT_DIR="comparison_output"

mkdir -p "$OUTPUT_DIR"

echo "========== Step 1: 运行新版本（含 pixel update）=========="
./build/auto_aim_test "$INPUT_PATH" \
  --config-path="$CONFIG_PATH" \
  --save-video="${OUTPUT_DIR}/new_"

echo ""
echo "========== Step 2: 临时回退为旧版本（不含 pixel update）=========="
# 保存当前 pixel-update 相关文件
cp tasks/auto_aim/yolos/yolo26n.cpp    "${OUTPUT_DIR}/_backup_yolo26n.cpp"
cp tasks/auto_aim/target/target.cpp    "${OUTPUT_DIR}/_backup_target.cpp"
cp tasks/auto_aim/target/target.hpp    "${OUTPUT_DIR}/_backup_target.hpp"
cp tasks/auto_aim/tracker/tracker.cpp  "${OUTPUT_DIR}/_backup_tracker.cpp"
cp tasks/auto_aim/armor/armor.hpp      "${OUTPUT_DIR}/_backup_armor.hpp"
cp tools/extended_kalman_filter/extended_kalman_filter.cpp "${OUTPUT_DIR}/_backup_ekf.cpp"

# 回退到 pixel-update 之前的版本
git checkout HEAD~1 -- \
  tasks/auto_aim/yolos/yolo26n.cpp \
  tasks/auto_aim/target/target.cpp \
  tasks/auto_aim/target/target.hpp \
  tasks/auto_aim/tracker/tracker.cpp \
  tasks/auto_aim/armor/armor.hpp \
  tools/extended_kalman_filter/extended_kalman_filter.cpp

echo "重新编译旧版本..."
cmake --build build --target auto_aim_test -j$(nproc) 2>&1 | tail -3

echo "运行旧版本..."
./build/auto_aim_test "$INPUT_PATH" \
  --config-path="$CONFIG_PATH" \
  --save-video="${OUTPUT_DIR}/old_"

echo ""
echo "========== Step 3: 恢复新版本 =========="
cp "${OUTPUT_DIR}/_backup_yolo26n.cpp"   tasks/auto_aim/yolos/yolo26n.cpp
cp "${OUTPUT_DIR}/_backup_target.cpp"    tasks/auto_aim/target/target.cpp
cp "${OUTPUT_DIR}/_backup_target.hpp"    tasks/auto_aim/target/target.hpp
cp "${OUTPUT_DIR}/_backup_tracker.cpp"   tasks/auto_aim/tracker/tracker.cpp
cp "${OUTPUT_DIR}/_backup_armor.hpp"     tasks/auto_aim/armor/armor.hpp
cp "${OUTPUT_DIR}/_backup_ekf.cpp"       tools/extended_kalman_filter/extended_kalman_filter.cpp

echo "重新编译新版本..."
cmake --build build --target auto_aim_test -j$(nproc) 2>&1 | tail -3

# 清理备份
rm -f "${OUTPUT_DIR}"/_backup_*

echo ""
echo "========== Step 4: 生成对比视频 =========="
python3 tools/merge_comparison.py \
  "${OUTPUT_DIR}/old_reprojection.avi" \
  "${OUTPUT_DIR}/new_reprojection.avi" \
  "${OUTPUT_DIR}/comparison.avi"

echo ""
echo "完成！对比视频保存在: ${OUTPUT_DIR}/comparison.avi"
echo "  旧版本视频: ${OUTPUT_DIR}/old_reprojection.avi"
echo "  新版本视频: ${OUTPUT_DIR}/new_reprojection.avi"
echo ""
echo "也可以逐帧查看（按任意键前进，q退出）："
echo "  ./build/auto_aim_test $INPUT_PATH --config-path=$CONFIG_PATH --step"
