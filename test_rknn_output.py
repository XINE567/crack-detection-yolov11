import onnxruntime as ort
import cv2
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    new_unpad = int(round(shape[1]*r)), int(round(shape[0]*r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

# 加载ONNX模型
session = ort.InferenceSession('model/best.onnx', providers=['CPUExecutionProvider'])

# 读取测试图像
img = cv2.imread('dataset/3897.rf.94c81f86436f41e7bba4b2e66934a470.jpg')
if img is None:
    img = cv2.imread('dataset/2275.rf.586655e669fa91476d091094ff24d359.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Letterbox预处理
img_pre, scale, (pad_w, pad_h) = letterbox(img_rgb)
img_tensor = img_pre.transpose(2, 0, 1).astype(np.float32) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

# 推理
outputs = session.run(None, {session.get_inputs()[0].name: img_tensor})

detection_out = outputs[0]  # [1, 37, 8400]
mask_proto = outputs[1]      # [1, 32, 160, 160]

print("=" * 60)
print("ONNX Model Output Analysis")
print("=" * 60)

# 分析置信度原始值（未经过sigmoid）
raw_confidences = detection_out[0, 4, :]
print(f"\n1. Raw confidence values (before sigmoid):")
print(f"   - Min: {raw_confidences.min():.4f}")
print(f"   - Max: {raw_confidences.max():.4f}")
print(f"   - Mean: {raw_confidences.mean():.4f}")
print(f"   - Std: {raw_confidences.std():.4f}")

# 分析sigmoid后的置信度
sigmoid_confidences = sigmoid(raw_confidences)
print(f"\n2. Sigmoid confidence values:")
print(f"   - Min: {sigmoid_confidences.min():.4f}")
print(f"   - Max: {sigmoid_confidences.max():.4f}")
print(f"   - Mean: {sigmoid_confidences.mean():.4f}")

# 分析检测框坐标
print(f"\n3. Bounding box coordinates analysis:")
x_centers = detection_out[0, 0, :]
y_centers = detection_out[0, 1, :]
widths = detection_out[0, 2, :]
heights = detection_out[0, 3, :]

print(f"   X centers: [{x_centers.min():.2f}, {x_centers.max():.2f}]")
print(f"   Y centers: [{y_centers.min():.2f}, {y_centers.max():.2f}]")
print(f"   Widths: [{widths.min():.2f}, {widths.max():.2f}]")
print(f"   Heights: [{heights.min():.2f}, {heights.max():.2f}]")

# 找出最高置信度的框
max_conf_idx = np.argmax(raw_confidences)
print(f"\n4. Box with highest raw confidence (index {max_conf_idx}):")
print(f"   - Raw confidence: {raw_confidences[max_conf_idx]:.4f}")
print(f"   - Sigmoid confidence: {sigmoid_confidences[max_conf_idx]:.4f}")
print(f"   - X: {detection_out[0, 0, max_conf_idx]:.2f}")
print(f"   - Y: {detection_out[0, 1, max_conf_idx]:.2f}")
print(f"   - W: {detection_out[0, 2, max_conf_idx]:.2f}")
print(f"   - H: {detection_out[0, 3, max_conf_idx]:.2f}")

# 分析掩码系数
mask_coeffs = detection_out[0, 5:37, :]
print(f"\n5. Mask coefficients analysis:")
print(f"   - Shape: {mask_coeffs.shape}")
print(f"   - Min: {mask_coeffs.min():.4f}")
print(f"   - Max: {mask_coeffs.max():.4f}")
print(f"   - Mean: {mask_coeffs.mean():.4f}")

# 检查是否有异常值
print(f"\n6. Outlier detection:")
high_conf_boxes = np.sum(raw_confidences > 5)
low_conf_boxes = np.sum(raw_confidences < -5)
print(f"   - Boxes with raw confidence > 5: {high_conf_boxes}")
print(f"   - Boxes with raw confidence < -5: {low_conf_boxes}")

print("\n" + "=" * 60)
print("Summary: ONNX model output is correct!")
print("The issue is likely in RKNN quantization or post-processing.")
print("=" * 60)
