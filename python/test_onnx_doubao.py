import onnxruntime as ort
import numpy as np
import cv2

print("=" * 50)
print("测试 best_new_opset19.onnx 模型")
print("=" * 50)

# 轮廓平滑优化函数
def smooth_contour(mask_bin):
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    smooth = []
    for cnt in contours:
        eps = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) > 2:
            smooth.append(approx)
    return smooth

# 加载模型
session = ort.InferenceSession("model/best_new_opset19.onnx", providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
input_shape = session.get_inputs()[0].shape
print(f"\n输入: {input_name}, 形状: {input_shape}")

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

# 推理
print("\n推理中...")
outputs = session.run(None, {input_name: blob})
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

# Mask 计算 复刻PT官方处理逻辑
mask_coef = det[6:37, best_idx]
mask = np.zeros((160, 160), dtype=np.float32)
for i in range(31):
    mask += protos[i] * mask_coef[i]
mask = 1 / (1 + np.exp(-mask))
mask = cv2.resize(mask, (640, 640))
mask_full = cv2.resize(mask, (orig_w, orig_h))

# 平滑降噪+适配官方阈值，解决线条错位弯曲
mask_smooth = cv2.GaussianBlur(mask_full, (3, 3), 0)
mask_binary = (mask_smooth > 0.35).astype(np.uint8) * 255

# 严格裁剪只保留框内区域，彻底消除框外多余线条
mask_cropped = np.zeros_like(mask_binary)
mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
mask_binary = mask_cropped

# 形态学修补裂缝断线，保持原生走向不偏移
kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)

# 绘制结果 完整保留框和文字
result = img.copy()

# 淡色掩码填充
overlay = np.zeros_like(img)
overlay[mask_binary > 128] = [0, 255, 0]
result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)

# 平滑抗锯齿轮廓，线条连贯对齐缝隙
smooth_conts = smooth_contour(mask_binary)
cv2.drawContours(result, smooth_conts, -1, (0, 255, 0), 2, cv2.LINE_AA)

# 保留黄色检测框
cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)

# 保留原有文字标签
cv2.putText(result, "YOLOX-ONNX Detection", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(result, f"Conf: {best_conf:.3f}", (10, 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

cv2.imwrite("model/resulta_new.jpg", result)
cv2.imwrite("model/resulta_new_mask.png", mask_binary)

print("\n" + "=" * 50)
print("结果已保存:")
print("  - model/resulta_new.jpg (可视化)")
print("  - model/resulta_new_mask.png (二值化 Mask)")
print("=" * 50)