#!/usr/bin/env python3
"""
PC 端模拟前端 WebSocket 推流：
定时 adb push dataset 图片到 RK3588 的 /userdata/live_frame.jpg
"""

import argparse
import glob
import os
import subprocess
import sys
import time


DEFAULT_DATASET = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "dataset"
)
DEFAULT_TARGET = "/userdata/live_frame.jpg"


def run_cmd(cmd):
    print("[CMD]", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.returncode != 0:
        if result.stderr.strip():
            print(result.stderr.strip(), file=sys.stderr)
        raise RuntimeError(f"命令失败: {' '.join(cmd)}")
    return result


def collect_images(dataset_dir):
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    images = []
    for pattern in patterns:
        images.extend(glob.glob(os.path.join(dataset_dir, pattern)))
    images = sorted(set(images))
    if not images:
        raise FileNotFoundError(f"dataset 目录下没有图片: {dataset_dir}")
    return images


def main():
    parser = argparse.ArgumentParser(description="模拟前端实时推流到 RK3588")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="本地 dataset 目录")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="板端实时画面路径")
    parser.add_argument("--interval", type=float, default=1.0, help="推送间隔(秒)")
    parser.add_argument("--adb", default="adb", help="adb 命令路径")
    args = parser.parse_args()

    images = collect_images(args.dataset)
    print(f"找到 {len(images)} 张图片")
    print(f"推送目标: {args.target}")
    print(f"推送间隔: {args.interval}s")
    print("按 Ctrl+C 停止")

    run_cmd([args.adb, "shell", "mkdir", "-p", "/userdata"])

    index = 0
    try:
        while True:
            image_path = images[index % len(images)]
            index += 1
            print(f"[PUSH] {os.path.basename(image_path)}")
            run_cmd([args.adb, "push", image_path, args.target])
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\n已停止模拟推流")


if __name__ == "__main__":
    main()
