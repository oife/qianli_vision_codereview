#!/usr/bin/env python3
"""将两个视频拼成左右对比视频"""
import sys
import cv2
import numpy as np

def main():
    if len(sys.argv) < 4:
        print(f"用法: {sys.argv[0]} <left.avi> <right.avi> <output.avi>")
        sys.exit(1)

    left_path, right_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    
    cap_l = cv2.VideoCapture(left_path)
    cap_r = cv2.VideoCapture(right_path)
    
    if not cap_l.isOpened():
        print(f"无法打开: {left_path}")
        sys.exit(1)
    if not cap_r.isOpened():
        print(f"无法打开: {right_path}")
        sys.exit(1)
    
    w = int(cap_l.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap_l.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap_l.get(cv2.CAP_PROP_FPS) or 30
    total = int(min(cap_l.get(cv2.CAP_PROP_FRAME_COUNT), cap_r.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w * 2 + 4, h))
    
    frame_idx = 0
    while True:
        ret_l, frame_l = cap_l.read()
        ret_r, frame_r = cap_r.read()
        if not ret_l or not ret_r:
            break
        
        # 确保尺寸一致
        frame_r = cv2.resize(frame_r, (w, h))
        
        # 添加标签
        cv2.putText(frame_l, "OLD (no pixel update)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame_r, "NEW (with pixel update)", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 分隔线
        sep = np.full((h, 4, 3), 255, dtype=np.uint8)
        
        combined = np.hstack([frame_l, sep, frame_r])
        out.write(combined)
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"  处理帧: {frame_idx}/{total}")
    
    cap_l.release()
    cap_r.release()
    out.release()
    print(f"对比视频已保存: {out_path} ({frame_idx} 帧)")

if __name__ == "__main__":
    main()
