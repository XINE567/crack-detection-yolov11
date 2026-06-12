import onnxruntime as ort
import numpy as np
import cv2

print("=" * 50)
print("分析 ONNX 模型输出")
print("=" * 50)

# 读取和预处理
img = cv2.imread("model/testa.jpg")
orig_h, orig_w = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
blob = rgb.astype(np.float32) / 255.0
blob = blob.transpose(2, 0, 1)
blob = np.expand_dims(blob, axis=0)

print(f"\n输入:")
print(f"   原始图片: {orig_w}x{orig_h}")
print(f"   输入形状: {blob.shape}")
print(f"   输入数据范围: [{np.min(blob):.4f}, {np.max(blob):.4f}]")

# 运行 ONNX
session = ort.InferenceSession("model/best_new_opset19.onnx", providers=['CPUExecutionProvider'])
outputs = session.run(None, {"images": blob})
det = outputs[0][0]
protos = outputs[1][0]

print(f"\n检测头 det:")
print(f"   形状: {det.shape}")
print(f"   数据范围: [{np.min(det):.4f}, {np.max(det):.4f}]")

# 查看前 10 个索引的置信度
conf = det[4]
print(f"\n前 20 个高置信度检测:")
top_indices = np.argsort(-conf)[:20]
for i, idx in enumerate(top_indices):
    cx, cy = det[0, idx], det[1, idx]
    w, h = det[2, idx], det[3, idx]
    c = det[4, idx]
    print(f"   [{i}] idx={idx}: ({cx:.2f}, {cy:.2f}) + ({w:.2f}, {h:.2f}), conf={c:.4f}")

# 查看 protos
print(f"\nPrototypes:")
print(f"   形状: {protos.shape}")
print(f"   数据范围: [{np.min(protos):.4f}, {np.max(protos):.4f}]")

print("\n" + "=" * 50)
