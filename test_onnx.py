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

# 读取测试图像（使用 dataset 中的示例图片）
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

# 解析输出
detection_out = outputs[0]  # [1, 37, 8400]
mask_proto = outputs[1]      # [1, 32, 160, 160]

print(f"Input shape: {img_tensor.shape}")
print(f"Detection output shape: {detection_out.shape}")
print(f"Mask prototype shape: {mask_proto.shape}")

# 解析检测结果
num_boxes = detection_out.shape[-1]
num_channels = detection_out.shape[1]

print(f"\nNumber of boxes: {num_boxes}")
print(f"Number of channels per box: {num_channels}")

# 检查置信度分布
confidences = sigmoid(detection_out[0, 4, :])
print(f"\nConfidence range: [{confidences.min():.4f}, {confidences.max():.4f}]")
print(f"Number of boxes with confidence > 0.5: {np.sum(confidences > 0.5)}")
print(f"Number of boxes with confidence > 0.9: {np.sum(confidences > 0.9)}")
print(f"Number of boxes with confidence > 0.99: {np.sum(confidences > 0.99)}")

# 打印前10个高置信度框
print("\nTop 10 high confidence boxes:")
sorted_indices = np.argsort(confidences)[::-1]
for i in range(min(10, len(sorted_indices))):
    idx = sorted_indices[i]
    conf = confidences[idx]
    if conf > 0.5:
        x_center = detection_out[0, 0, idx]
        y_center = detection_out[0, 1, idx]
        width = detection_out[0, 2, idx]
        height = detection_out[0, 3, idx]
        print(f"Box {i}: conf={conf:.4f}, x={x_center:.2f}, y={y_center:.2f}, w={width:.2f}, h={height:.2f}")

# 检查掩码系数
mask_coeffs = detection_out[0, 5:37, :]
print(f"\nMask coefficients shape: {mask_coeffs.shape}")
print(f"Mask coeffs range: [{mask_coeffs.min():.4f}, {mask_coeffs.max():.4f}]")

print("\nONNX模型测试完成！")
