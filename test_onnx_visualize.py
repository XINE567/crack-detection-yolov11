
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

def nms(boxes, confidences, threshold=0.45):
    if len(boxes) == 0:
        return []
    
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidences.tolist(), 0.5, threshold)
    if len(indices) > 0:
        return indices.flatten().tolist()
    return []

# 加载ONNX模型
session = ort.InferenceSession('model/best.onnx', providers=['CPUExecutionProvider'])

# 读取测试图像
img = cv2.imread('dataset/3897.rf.94c81f86436f41e7bba4b2e66934a470.jpg')
if img is None:
    img = cv2.imread('dataset/2275.rf.586655e669fa91476d091094ff24d359.jpg')

img_height, img_width = img.shape[:2]
img_copy = img.copy()

# Letterbox预处理
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_pre, scale, (pad_w, pad_h) = letterbox(img_rgb)
img_tensor = img_pre.transpose(2, 0, 1).astype(np.float32) / 255.0
img_tensor = np.expand_dims(img_tensor, axis=0)

# 推理
outputs = session.run(None, {session.get_inputs()[0].name: img_tensor})

detection_out = outputs[0]  # [1, 37, 8400]
mask_proto = outputs[1]      # [1, 32, 160, 160]

# 解析检测结果
num_channels = 37
num_boxes = detection_out.shape[-1]

boxes = []
confidences = []
mask_coeffs_list = []

for i in range(num_boxes):
    x_center = detection_out[0, 0, i]
    y_center = detection_out[0, 1, i]
    width = detection_out[0, 2, i]
    height = detection_out[0, 3, i]
    raw_conf = detection_out[0, 4, i]
    conf = sigmoid(raw_conf)
    
    if conf > 0.5:
        # 转换为原始图像坐标
        x_center = (x_center - pad_w) / scale
        y_center = (y_center - pad_h) / scale
        width = width / scale
        height = height / scale
        
        left = max(0, int(x_center - width / 2))
        top = max(0, int(y_center - height / 2))
        right = min(img_width, int(left + width))
        bottom = min(img_height, int(top + height))
        
        w = right - left
        h = bottom - top
        
        if w > 30 and h > 30 and w < 200 and h < 200:
            boxes.append([left, top, w, h])
            confidences.append(conf)
            mask_coeffs_list.append(detection_out[0, 5:37, i])

boxes = np.array(boxes)
confidences = np.array(confidences)

# NMS过滤
if len(boxes) > 0:
    keep_indices = nms(boxes, confidences, 0.5)
    boxes = boxes[keep_indices]
    confidences = confidences[keep_indices]
    mask_coeffs_list = [mask_coeffs_list[i] for i in keep_indices]

# 绘制检测框
for i, box in enumerate(boxes):
    left, top, w, h = box
    right = left + w
    bottom = top + h
    conf = confidences[i]
    
    # 绘制红色边框
    cv2.rectangle(img_copy, (left, top), (right, bottom), (0, 0, 255), 2)
    
    # 绘制置信度标签
    label = f"crack: {conf:.3f}"
    cv2.putText(img_copy, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 保存结果
output_path = 'output_onnx.jpg'
cv2.imwrite(output_path, img_copy)

print(f"ONNX 模型检测完成！")
print(f"检测到 {len(boxes)} 个裂纹")
for i, box in enumerate(boxes):
    print(f"  crack @ ({box[0]}, {box[1]}, {box[0]+box[2]}, {box[1]+box[3]}) conf={confidences[i]:.3f}")
print(f"结果已保存到: {output_path}")
