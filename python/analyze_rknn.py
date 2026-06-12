import cv2
import numpy as np
from rknnlite.api import RKNNLite

print("=" * 50)
print("分析 RKNN 模型输出")
print("=" * 50)

# 读取和预处理
img = cv2.imread("/userdata/testa.jpg")
orig_h, orig_w = img.shape[:2]
img_resized = cv2.resize(img, (640, 640))
# 官方是先转 RGB，然后 NHWC 格式！
rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
blob = rgb.astype(np.float32)  # 保持 [0,255], 形状 (640, 640, 3)！
blob = np.expand_dims(blob, axis=0)  # (1, 640, 640, 3) NHWC！

print(f"\n输入:")
print(f"   原始图片: {orig_w}x{orig_h}")
print(f"   输入形状: {blob.shape}")
print(f"   输入数据范围: [{np.min(blob):.4f}, {np.max(blob):.4f}]")

# 运行 RKNN
rknn = RKNNLite()
ret = rknn.load_rknn("/userdata/yolo11_fp_v4.rknn")
if ret != 0:
    print("加载 RKNN 模型失败!")
    exit(1)
print("\n初始化运行环境...")
ret = rknn.init_runtime()
if ret != 0:
    print("初始化失败!")
    exit(1)

print("\n推理中...")
outputs = rknn.inference(inputs=[blob])
print(f"原始输出: {len(outputs)}个")
print(f"   output[0] shape: {outputs[0].shape}")
print(f"   output[1] shape: {outputs[1].shape}")
det = outputs[0][0]
protos = outputs[1][0]

print(f"输出数量: {len(outputs)}")
print(f"检测头 det:")
print(f"   形状: {det.shape}")
print(f"   数据范围: [{np.min(det):.4f}, {np.max(det):.4f}]")

# 查看前 20 个索引的置信度
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

rknn.release()
print("\n" + "=" * 50)
