#!/usr/bin/env python3
"""
无前端联调：用本地图片模拟一次巡检会话
将 dataset 图片 push 到板子后，在板子上运行此脚本创建 session + captures
"""

import glob
import os
import sys

import cv2

from inspection_db import init_db, create_session, end_session, save_capture, DB_PATH

RESULTS_DIR = "/userdata/results"
LOCAL_IMAGES = [
    "/userdata/live_frame.jpg",
    "/userdata/testb.jpg",
]


def find_images():
    images = []
    for path in LOCAL_IMAGES:
        if os.path.exists(path):
            images.append(path)
    images.extend(sorted(glob.glob("/userdata/results/result_*.jpg"))[:5])
    return list(dict.fromkeys(images))


def mock_detection(image_path, index):
    img = cv2.imread(image_path)
    if img is None:
        return None
    h, w = img.shape[:2]
    if index % 2 == 0:
        return {
            "confidence": 0.86,
            "crack_count": 1,
            "crack_area": 2350,
            "max_width": 8.6,
            "total_length": 420,
            "severity": "MEDIUM",
            "suggestion": "Moderate crack, recommend repair",
            "result_json": {
                "model": "YOLOv11-RKNN",
                "bbox": [w // 8, h // 8, w * 3 // 4, h * 3 // 4],
                "area": 2350,
                "length": 420.0,
                "width": 8.6,
                "severity": "MEDIUM",
            },
        }
    return {
        "confidence": 0.0,
        "crack_count": 0,
        "crack_area": 0,
        "max_width": 0,
        "total_length": 0,
        "severity": "NONE",
        "suggestion": "No crack detected",
        "result_json": {},
    }


def main():
    init_db()
    images = find_images()
    if not images:
        print("未找到本地图片，请先 adb push 图片到 /userdata/live_frame.jpg")
        return 1

    inspector = os.environ.get("TEST_INSPECTOR", "zhangsan")
    location = os.environ.get("TEST_LOCATION", "building-a-east-wall")

    session_id, start_time = create_session(inspector, location)
    print(f"创建会话 session_id={session_id} inspector={inspector} location={location}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    for i, src in enumerate(images[:3]):
        img = cv2.imread(src)
        if img is None:
            continue
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_path = os.path.join(RESULTS_DIR, f"raw_{ts}_{i}.jpg")
        result_path = os.path.join(RESULTS_DIR, f"result_{ts}_{i}.jpg")
        cv2.imwrite(raw_path, img)
        cv2.imwrite(result_path, img)

        record = mock_detection(src, i)
        cid = save_capture(session_id, raw_path, result_path, record)
        print(f"  capture #{cid}: {os.path.basename(result_path)}")

    end_time = end_session(session_id)
    print(f"会话已结束 end_time={end_time}")
    print(f"\n下一步上传测试:")
    print(f"  BACKEND_URL=http://<PC_IP>:5000/upload_result python3 test_upload_inspection.py {session_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
