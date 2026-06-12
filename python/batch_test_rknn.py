#!/usr/bin/env python3
import os
import cv2
import numpy as np
from rknnlite.api import RKNNLite

def skeletonize(image):
    skeleton = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
        if cv2.countNonZero(image) == 0:
            break
    return skeleton

def process_image(img_path, rknn):
    img = cv2.imread(img_path)
    if img is None:
        print(f"  读取失败: {img_path}")
        return None

    orig_h, orig_w = img.shape[:2]

    # 预处理：正确格式！NHWC RGB [0, 255]
    img_resized = cv2.resize(img, (640, 640))
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32)
    blob = np.expand_dims(blob, axis=0)

    outputs = rknn.inference(inputs=[blob])
    det = outputs[0][0]
    protos = outputs[1][0]

    conf = det[4]
    best_idx = np.argmax(conf)
    best_conf = conf[best_idx]

    if best_conf < 0.1:
        print(f"  置信度太低: {best_conf:.4f}")
        return None

    cx, cy = det[0, best_idx], det[1, best_idx]
    w, h = det[2, best_idx], det[3, best_idx]

    scale_x = orig_w / 640
    scale_y = orig_h / 640
    cx_s, cy_s = cx * scale_x, cy * scale_y
    w_s, h_s = w * scale_x, h * scale_y

    x1 = max(0, int(cx_s - w_s/2))
    y1 = max(0, int(cy_s - h_s/2))
    x2 = min(orig_w, int(cx_s + w_s/2))
    y2 = min(orig_h, int(cy_s + h_s/2))

    mask_coef = det[6:37, best_idx]
    mask = np.zeros((160, 160), dtype=np.float32)
    for i in range(31):
        mask += protos[i] * mask_coef[i]
    mask = 1 / (1 + np.exp(-mask))
    mask = cv2.resize(mask, (640, 640))
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_binary = cv2.resize(mask_binary, (orig_w, orig_h))

    mask_cropped = np.zeros_like(mask_binary)
    mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
    mask_binary = mask_cropped

    kernel_close = np.ones((3, 3), np.uint8)
    mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
    mask_binary = skeletonize(mask_binary)

    result = img.copy()
    overlay = np.zeros_like(img)
    overlay[mask_binary > 128] = [0, 255, 0]
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
    cv2.putText(result, f"Conf: {best_conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return result, best_conf

def main():
    dataset_dir = "/userdata/dataset"
    output_dir = "/userdata/results"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("加载 RKNN 模型...")
    rknn = RKNNLite()
    ret = rknn.load_rknn("/userdata/yolo11_fp_v4.rknn")
    if ret != 0:
        print("加载 RKNN 模型失败!")
        exit(1)

    print("初始化运行环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("初始化运行环境失败!")
        exit(1)

    # 获取所有 jpg 图片
    images = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]
    images.sort()

    print(f"\n找到 {len(images)} 张图片，开始处理...\n")

    for i, img_name in enumerate(images):
        img_path = os.path.join(dataset_dir, img_name)
        base_name = os.path.splitext(img_name)[0]

        print(f"[{i+1}/{len(images)}] 处理: {img_name}")

        result = process_image(img_path, rknn)
        if result is not None:
            result_img, conf = result
            output_path = os.path.join(output_dir, f"{base_name}_result.jpg")
            cv2.imwrite(output_path, result_img)
            print(f"  -> 保存: {output_path} (conf={conf:.4f})")
        else:
            print(f"  -> 跳过")

    rknn.release()
    print(f"\n完成！结果保存在: {output_dir}")

if __name__ == "__main__":
    main()
