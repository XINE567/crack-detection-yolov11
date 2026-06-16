import json
import os
from pathlib import Path

import requests


# Flask 后端地址。本机测试使用 127.0.0.1；设备上传时改成电脑的局域网 IP。
UPLOAD_URL = "http://127.0.0.1:5000/upload_result"
# UPLOAD_URL = "http://192.168.146.181:5000/upload_result"

BASE_DIR = Path(__file__).resolve().parent

# 按顺序查找测试图片，避免文件名多了 .jpg 时脚本直接失败。
IMAGE_CANDIDATES = [
    BASE_DIR / "test_images" / "detected_001.jpg",
    BASE_DIR / "test_images" / "detected_001.jpg.jpg",
    BASE_DIR / "test.jpg",
]


def find_test_image():
    for image_path in IMAGE_CANDIDATES:
        if image_path.is_file():
            return image_path
    return None


def upload_test_result():
    image_path = find_test_image()
    if image_path is None:
        print("找不到测试图片，请把图片放到以下任意位置：")
        for candidate in IMAGE_CANDIDATES:
            print(f"- {candidate}")
        return

    image_field_name = image_path.name

    upload_data = {
        # workorders.id 当前是 MySQL INT，测试编号需要小于 2147483647。
        "session_id": 2026061501,
        "inspector": "admin",
        "location": "教学楼一楼东侧墙面",
        "inspection_start_time": "2026-06-15 14:30:00",
        "inspection_end_time": "2026-06-15 14:35:00",
        "capture_count": 1,
        "captures": [
            {
                "capture_id": "capture_001",
                "captured_at": "2026-06-15 14:31:20",
                "image_filename": image_field_name,
                "crack_count": 2,
                "crack_area": 1834.52,
                "max_width": 4.76,
                "total_length": 126.35,
                "severity": "中等",
                "suggestion": "建议对裂缝区域进行复检，并进行封闭修补。",
                "confidence": 0.917,
                "result_json": {
                    "model_name": "YOLOv8-crack",
                    "model_version": "v1.0",
                    "image_width": 1280,
                    "image_height": 720,
                    "detections": [
                        {
                            "class_name": "crack",
                            "confidence": 0.932,
                            "bbox": {
                                "x1": 253,
                                "y1": 146,
                                "x2": 611,
                                "y2": 288,
                            },
                        },
                        {
                            "class_name": "crack",
                            "confidence": 0.902,
                            "bbox": {
                                "x1": 675,
                                "y1": 314,
                                "x2": 940,
                                "y2": 512,
                            },
                        },
                    ],
                },
            }
        ],
    }

    try:
        with image_path.open("rb") as image_file:
            form_data = {
                "data": json.dumps(upload_data, ensure_ascii=False)
            }
            files = {
                image_field_name: (
                    image_path.name,
                    image_file,
                    "image/jpeg",
                )
            }

            print("正在上传检测数据...")
            print("接口地址：", UPLOAD_URL)
            print("测试图片：", image_path)
            print("上传内容：")
            print(json.dumps(upload_data, ensure_ascii=False, indent=2))

            response = requests.post(
                UPLOAD_URL,
                data=form_data,
                files=files,
                timeout=60,
            )

        print("\nHTTP 状态码：", response.status_code)

        try:
            print("服务器返回：")
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        except ValueError:
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("连接失败：请检查 Flask 服务是否已启动，以及 IP 地址和端口是否正确。")
    except requests.exceptions.Timeout:
        print("上传超时。")
    except requests.exceptions.RequestException as exc:
        print(f"请求异常：{exc}")


if __name__ == "__main__":
    upload_test_result()
