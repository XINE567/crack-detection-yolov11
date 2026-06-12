import onnxruntime as ort
import numpy as np
import cv2
from rknn.api import RKNN

print("=" * 50)
print("对比 ONNX 和 RKNN 输出")
print("=" * 50)

# 读取和预处理图片
img = cv2.imread("model/testa.jpg")
orig_h, orig_w = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
blob = rgb.astype(np.float32) / 255.0
blob = blob.transpose(2, 0, 1)
blob = np.expand_dims(blob, axis=0)

print("\n1. 运行 ONNX 模型")
session = ort.InferenceSession("model/best_new_opset19.onnx", providers=['CPUExecutionProvider'])
onnx_outputs = session.run(None, {"images": blob})
onnx_det = onnx_outputs[0][0]
onnx_protos = onnx_outputs[1][0]

print(f"   ONNX det 形状: {onnx_det.shape}")
print(f"   ONNX protos 形状: {onnx_protos.shape}")

print("\n2. 运行 RKNN 模型 (PC 模拟)")
rknn = RKNN()
ret = rknn.load_rknn("model/yolo11_fp_v3.rknn")
if ret != 0:
    print("加载 RKNN 失败")
    exit(1)
ret = rknn.init_runtime()
if ret != 0:
    print("初始化失败")
    exit(1)
rknn_outputs = rknn.inference(inputs=[blob])
rknn_det = rknn_outputs[0][0]
rknn_protos = rknn_outputs[1][0]

print(f"   RKNN det 形状: {rknn_det.shape}")
print(f"   RKNN protos 形状: {rknn_protos.shape}")

print("\n" + "=" * 50)
print("3. 对比差异")
print("=" * 50)

# 对比检测头
print(f"\n检测头差异分析:")
diff_det = np.abs(onnx_det - rknn_det)
print(f"   max diff: {np.max(diff_det):.6f}")
print(f"   mean diff: {np.mean(diff_det):.6f}")

# 对比检测置信度
onnx_conf = onnx_det[4]
rknn_conf = rknn_det[4]
best_onnx_idx = np.argmax(onnx_conf)
best_rknn_idx = np.argmax(rknn_conf)
print(f"\n最高置信度对比:")
print(f"   ONNX: idx={best_onnx_idx}, conf={onnx_conf[best_onnx_idx]:.6f}")
print(f"   RKNN: idx={best_rknn_idx}, conf={rknn_conf[best_rknn_idx]:.6f}")

# 对比最佳检测的坐标
if best_onnx_idx < onnx_det.shape[1]:
    print(f"\nONNX 最佳检测的坐标:")
    cx, cy = onnx_det[0, best_onnx_idx], onnx_det[1, best_onnx_idx]
    w, h = onnx_det[2, best_onnx_idx], onnx_det[3, best_onnx_idx]
    print(f"   中心点: ({cx:.2f}, {cy:.2f}), 宽高: ({w:.2f}, {h:.2f})")

if best_rknn_idx < rknn_det.shape[1]:
    print(f"\nRKNN 最佳检测的坐标:")
    cx, cy = rknn_det[0, best_rknn_idx], rknn_det[1, best_rknn_idx]
    w, h = rknn_det[2, best_rknn_idx], rknn_det[3, best_rknn_idx]
    print(f"   中心点: ({cx:.2f}, {cy:.2f}), 宽高: ({w:.2f}, {h:.2f})")

# 对比原型差异
diff_protos = np.abs(onnx_protos - rknn_protos)
print(f"\nPrototypes 差异:")
print(f"   max diff: {np.max(diff_protos):.6f}")
print(f"   mean diff: {np.mean(diff_protos):.6f}")

rknn.release()
print("\n" + "=" * 50)
