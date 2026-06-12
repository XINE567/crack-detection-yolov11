#!/usr/bin/env python3
"""在板端批量测试 RKNN 模型，NMS 多目标 + 各 bbox 内 mask 裁剪。"""
import os
import sys
import cv2
import numpy as np
from rknnlite.api import RKNNLite

OBJ_THRESH = 0.25
NMS_THRESH = 0.45

MODEL_PATH = "/userdata/best2.0.rknn"
DATASET_DIR = "/userdata/dataset"
OUTPUT_DIR = "/userdata/results_best2_nms"

COLORS = [
    (0, 255, 0),
    (0, 255, 255),
    (255, 128, 0),
    (255, 0, 255),
    (0, 128, 255),
    (128, 255, 0),
]


def nms_boxes(boxes, scores, iou_thresh=NMS_THRESH):
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep


def decode_mask(det_idx, det, protos, bbox, orig_h, orig_w):
    mask_coef = det[6:37, det_idx]
    mask = np.zeros((160, 160), dtype=np.float32)
    for i in range(31):
        mask += protos[i] * mask_coef[i]
    mask = 1 / (1 + np.exp(-mask))
    mask = cv2.resize(mask, (640, 640))
    mask_binary = (mask > 0.5).astype(np.uint8) * 255
    mask_binary = cv2.resize(mask_binary, (orig_w, orig_h))

    x1, y1, x2, y2 = bbox
    mask_cropped = np.zeros_like(mask_binary)
    mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]

    kernel = np.ones((3, 3), np.uint8)
    mask_cropped = cv2.morphologyEx(mask_cropped, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask_cropped


def postprocess(det, protos, orig_h, orig_w):
    conf = det[4]
    valid = np.where(conf >= OBJ_THRESH)[0]
    if valid.size == 0:
        return []

    scale_x = orig_w / 640.0
    scale_y = orig_h / 640.0

    boxes_640 = np.zeros((len(valid), 4), dtype=np.float32)
    scores = conf[valid]
    for i, idx in enumerate(valid):
        cx, cy = det[0, idx], det[1, idx]
        w, h = det[2, idx], det[3, idx]
        boxes_640[i] = [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

    keep = nms_boxes(boxes_640, scores)
    detections = []
    for k in keep:
        idx = valid[k]
        cx, cy = det[0, idx], det[1, idx]
        w, h = det[2, idx], det[3, idx]
        cx_s, cy_s = cx * scale_x, cy * scale_y
        w_s, h_s = w * scale_x, h * scale_y
        x1 = max(0, int(cx_s - w_s / 2))
        y1 = max(0, int(cy_s - h_s / 2))
        x2 = min(orig_w, int(cx_s + w_s / 2))
        y2 = min(orig_h, int(cy_s + h_s / 2))
        bbox = (x1, y1, x2, y2)
        mask = decode_mask(idx, det, protos, bbox, orig_h, orig_w)
        detections.append({
            "idx": int(idx),
            "conf": float(conf[idx]),
            "bbox": bbox,
            "mask": mask,
        })
    return detections


def draw_detections(img, detections):
    result = img.copy()
    combined_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for i, det in enumerate(detections):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = det["bbox"]
        mask = det["mask"]
        combined_mask = np.maximum(combined_mask, mask)

        overlay = np.zeros_like(img)
        overlay[mask > 128] = color
        result = cv2.addWeighted(result, 1.0, overlay, 0.35, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, color, 2)
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            result, f"#{i + 1} {det['conf']:.3f}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2,
        )

    cv2.putText(
        result, f"Detections: {len(detections)}",
        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
    )
    return result, combined_mask


def process_image(img_path, rknn):
    img = cv2.imread(img_path)
    if img is None:
        return None

    orig_h, orig_w = img.shape[:2]
    img_resized = cv2.resize(img, (640, 640))
    rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    blob = np.expand_dims(rgb.astype(np.float32), axis=0)

    outputs = rknn.inference(inputs=[blob])
    det = outputs[0][0]
    protos = outputs[1][0]

    detections = postprocess(det, protos, orig_h, orig_w)
    if not detections:
        return {"count": 0, "result": img, "mask": np.zeros((orig_h, orig_w), np.uint8)}

    result, combined_mask = draw_detections(img, detections)
    return {"count": len(detections), "result": result, "mask": combined_mask, "detections": detections}


def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print(f"加载模型: {MODEL_PATH}")
    rknn = RKNNLite()
    if rknn.load_rknn(MODEL_PATH) != 0:
        print("加载 RKNN 模型失败!")
        sys.exit(1)
    if rknn.init_runtime() != 0:
        print("初始化运行环境失败!")
        sys.exit(1)

    images = sorted(f for f in os.listdir(DATASET_DIR) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    print(f"找到 {len(images)} 张图片\n")

    summary_lines = []
    for i, img_name in enumerate(images):
        img_path = os.path.join(DATASET_DIR, img_name)
        base = os.path.splitext(img_name)[0]
        print(f"[{i + 1}/{len(images)}] {img_name}", end=" ")

        out = process_image(img_path, rknn)
        if out is None:
            print("-> 读取失败")
            summary_lines.append(f"{img_name}\tERROR")
            continue

        result_path = os.path.join(OUTPUT_DIR, f"{base}_result.jpg")
        mask_path = os.path.join(OUTPUT_DIR, f"{base}_mask.png")
        cv2.imwrite(result_path, out["result"])
        cv2.imwrite(mask_path, out["mask"])
        confs = [f"{d['conf']:.3f}" for d in out.get("detections", [])]
        print(f"-> {out['count']} 个目标 ({', '.join(confs) if confs else 'none'})")
        summary_lines.append(f"{img_name}\t{out['count']}\t{','.join(confs)}")

    summary_path = os.path.join(OUTPUT_DIR, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("image\tdetections\tconfidences\n")
        f.write("\n".join(summary_lines))
        f.write("\n")

    rknn.release()
    print(f"\n完成，结果目录: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
