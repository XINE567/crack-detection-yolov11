import os
import cv2
import sys
import numpy as np

realpath = os.path.abspath(__file__)
_sep = os.path.sep
realpath = realpath.split(_sep)
sys.path.append(os.path.join(realpath[0]+_sep, *realpath[1:realpath.index('rknn_model_zoo')+1]))

print("=" * 50)
print("测试 yolo11.rknn 模型")
print("=" * 50)

# 读取图片
img = cv2.imread("model/testa.jpg")
orig_h, orig_w = img.shape[:2]
print(f"原图: {orig_w}x{orig_h}")

# 预处理
img_resized = cv2.resize(img, (640, 640))
rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
blob = rgb.astype(np.float32) / 255.0
blob = blob.transpose(2, 0, 1)
blob = np.expand_dims(blob, axis=0)

# 加载 RKNN 模型
from py_utils.rknn_executor import RKNN_model_container
model = RKNN_model_container("model/yolo11.rknn", target='rk3588')

# 推理
print("\n推理中...")
outputs = model.run([blob])
print(f"输出数量: {len(outputs)}")

det = outputs[0][0]   # (37, 8400)
protos = outputs[1][0]  # (32, 160, 160)
print(f"检测头: {det.shape}")
print(f"Prototypes: {protos.shape}")

# 解析结果
conf = det[4]
best_idx = np.argmax(conf)
best_conf = conf[best_idx]
print(f"\n最高置信度: {best_conf:.4f}")

# BBox
cx, cy = det[0, best_idx], det[1, best_idx]
w, h = det[2, best_idx], det[3, best_idx]
print(f"中心点: ({cx:.1f}, {cy:.1f})")
print(f"宽高: ({w:.1f}, {h:.1f})")

# 缩放到原图
scale_x = orig_w / 640
scale_y = orig_h / 640
cx_s, cy_s = cx * scale_x, cy * scale_y
w_s, h_s = w * scale_x, h * scale_y

x1 = max(0, int(cx_s - w_s/2))
y1 = max(0, int(cy_s - h_s/2))
x2 = min(orig_w, int(cx_s + w_s/2))
y2 = min(orig_h, int(cy_s + h_s/2))
print(f"BBox: ({x1}, {y1}, {x2}, {y2})")

# Mask 计算
mask_coef = det[6:37, best_idx]
mask = np.zeros((160, 160), dtype=np.float32)
for i in range(31):
    mask += protos[i] * mask_coef[i]
mask = 1 / (1 + np.exp(-mask))
mask = cv2.resize(mask, (640, 640))
mask_binary = (mask > 0.5).astype(np.uint8) * 255
mask_binary = cv2.resize(mask_binary, (orig_w, orig_h))

# 删除检测框外的掩码
mask_cropped = np.zeros_like(mask_binary)
mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
mask_binary = mask_cropped

# 骨架化算法：提取纹路的单像素骨架
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

# 先进行闭运算填充小空洞，保持连贯性
kernel_close = np.ones((3, 3), np.uint8)
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)

# 骨架化提取中心线
mask_binary = skeletonize(mask_binary)

# 绘制结果
result = img.copy()

# Mask填充
overlay = np.zeros_like(img)
overlay[mask_binary > 128] = [0, 255, 0]
result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)

# 轮廓
contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

# BBox
cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)

# 文字
cv2.putText(result, "YOLO11-RKNN Detection", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(result, f"Conf: {best_conf:.3f}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imwrite("model/resulta_rknn.jpg", result)
cv2.imwrite("model/resulta_rknn_mask.png", mask_binary)

# 释放资源
model.release()

print("\n" + "=" * 50)
print("结果已保存:")
print("  - model/resulta_rknn.jpg (可视化)")
print("  - model/resulta_rknn_mask.png (二值化 Mask)")
print("=" * 50)
