#!/usr/bin/env python3
import sys
import os
import time
import numpy as np
import cv2

from rknn.api import RKNN

MASK_COEF_COUNT = 32
CONF_THRESHOLD = 0.5
LABEL_NAME = "crack"

def deqnt_affine_to_f32(qnt, zp, scale):
    return ((float(qnt) - float(zp)) * scale)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def generate_mask(proto_masks, mask_coefs, proto_width, proto_height, num_coefs, out_width, out_height):
    proto_len = proto_width * proto_height
    temp_mask = np.zeros(proto_len, dtype=np.float32)

    for c in range(num_coefs):
        coef = mask_coefs[c]
        temp_mask += coef * proto_masks[c * proto_len:(c + 1) * proto_len]

    min_val = temp_mask.min()
    max_val = temp_mask.max()
    range_val = max_val - min_val
    if range_val < 1e-6:
        range_val = 1e-6

    out_mask = np.zeros((out_height, out_width), dtype=np.uint8)

    for y in range(out_height):
        for x in range(out_width):
            px = x / out_width * proto_width
            py = y / out_height * proto_height
            x0, y0 = int(px), int(py)
            x1, y1 = min(x0 + 1, proto_width - 1), min(y0 + 1, proto_height - 1)
            fx, fy = px - x0, py - y0

            idx00 = y0 * proto_width + x0
            idx01 = y0 * proto_width + x1
            idx10 = y1 * proto_width + x0
            idx11 = y1 * proto_width + x1

            val = (temp_mask[idx00] * (1 - fx) * (1 - fy) +
                   temp_mask[idx01] * fx * (1 - fy) +
                   temp_mask[idx10] * (1 - fx) * fy +
                   temp_mask[idx11] * fx * fy)

            norm_val = (val - min_val) / range_val
            sigmoid_val = sigmoid(norm_val)
            out_mask[y, x] = int(sigmoid_val * 255)

    return out_mask

def resize_image(src, dst_w, dst_h):
    return cv2.resize(src, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)

def post_process_yolo11_seg(output, proto_masks, num_boxes, is_quant, zp, scale,
                            model_width, model_height, proto_width, proto_height, proto_channel):
    results = []

    output = output.transpose(1, 0)

    for i in range(num_boxes):
        if is_quant:
            output_i8 = output.astype(np.int8)
            conf = deqnt_affine_to_f32(output_i8[4 * num_boxes + i], zp, scale)
            mask_coefs = [deqnt_affine_to_f32(output_i8[(5 + c) * num_boxes + i], zp, scale) for c in range(MASK_COEF_COUNT)]
        else:
            conf = output[4 * num_boxes + i]
            mask_coefs = [output[(5 + c) * num_boxes + i] for c in range(MASK_COEF_COUNT)]

        if conf < CONF_THRESHOLD:
            continue

        if is_quant:
            x_center = deqnt_affine_to_f32(output_i8[0 * num_boxes + i], zp, scale)
            y_center = deqnt_affine_to_f32(output_i8[1 * num_boxes + i], zp, scale)
            width = deqnt_affine_to_f32(output_i8[2 * num_boxes + i], zp, scale)
            height = deqnt_affine_to_f32(output_i8[3 * num_boxes + i], zp, scale)
        else:
            x_center = output[0 * num_boxes + i]
            y_center = output[1 * num_boxes + i]
            width = output[2 * num_boxes + i]
            height = output[3 * num_boxes + i]

        x_center *= model_width
        y_center *= model_height
        width *= model_width
        height *= model_height

        x1 = max(0, x_center - width / 2)
        y1 = max(0, y_center - height / 2)
        x2 = min(model_width, x_center + width / 2)
        y2 = min(model_height, y_center + height / 2)

        mask = generate_mask(proto_masks, mask_coefs, proto_width, proto_height, proto_channel,
                           int(x2 - x1), int(y2 - y1))

        results.append({
            'box': (int(x1), int(y1), int(x2), int(y2)),
            'conf': float(conf),
            'mask': mask
        })

    return results

def draw_results(frame, results):
    for det in results:
        x1, y1, x2, y2 = det['box']
        conf = det['conf']
        mask = det['mask']

        box_w, box_h = x2 - x1, y2 - y1
        if box_w > 0 and box_h > 0:
            resized_mask = cv2.resize(mask, (box_w, box_h), interpolation=cv2.INTER_LINEAR)
            color_mask = np.zeros((box_h, box_w, 3), dtype=np.uint8)
            color_mask[resized_mask > 128] = [0, 200, 0]

            roi = frame[y1:y2, x1:x2]
            if roi.size > 0:
                frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1.0, color_mask, 0.5, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        text = f"{LABEL_NAME} {conf*100:.1f}%"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"{LABEL_NAME} @ ({x1}, {y1}, {x2}, {y2}) {conf:.3f}")

def main():
    if len(sys.argv) < 2:
        print(f"Usage: python {sys.argv[0]} <model_path> [camera_id]")
        print(f"  model_path: Path to .rknn model file")
        print(f"  camera_id: Camera ID (default: 0)")
        sys.exit(1)

    # 使用默认模型路径（可通过命令行参数覆盖）
    model_path = sys.argv[1] if len(sys.argv) > 1 else 'model/crack_detector.rknn'
    camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)

    print("Initializing RKNN model...")
    rknn = RKNN()

    ret = rknn.load_rknn(model_path)
    if ret != 0:
        print(f"Load RKNN model failed!")
        sys.exit(1)

    print("Initializing runtime...")
    ret = rknn.init_runtime()
    if ret != 0:
        print(f"Init runtime failed!")
        sys.exit(1)

    rknn_inputs = rknn.query()
    print(f"Model info: {rknn_inputs}")

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print(f"Failed to open camera {camera_id}")
        sys.exit(1)

    print("Camera opened. Press 'q' or ESC to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        input_img = resize_image(frame, 640, 640)

        start_time = time.time()
        outputs = rknn.inference(inputs=[input_img])
        inf_time = time.time() - start_time

        output = outputs[0]
        proto_masks_output = outputs[1] if len(outputs) > 1 else None

        is_quant = True
        zp = 0
        scale = 1.0

        num_boxes = output.shape[1]

        proto_masks = None
        proto_width, proto_height, proto_channel = 0, 0, 0

        if proto_masks_output is not None:
            proto_data = proto_masks_output[0]
            if is_quant:
                proto_masks = (proto_data.astype(np.float32) - zp) * scale
            else:
                proto_masks = proto_data

            proto_height, proto_width = proto_masks.shape[1], proto_masks.shape[2]
            proto_channel = proto_masks.shape[0]

        results = post_process_yolo11_seg(
            output,
            proto_masks, num_boxes, is_quant, zp, scale,
            640, 640, proto_width, proto_height, proto_channel
        )

        draw_results(frame, results)

        fps = 1.0 / inf_time if inf_time > 0 else 0
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Crack Detection", frame)

        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    rknn.release()
    print("Done.")

if __name__ == "__main__":
    main()
