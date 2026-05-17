import onnxruntime as ort
import cv2
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def yolo_preprocess(img, input_size=(640, 640)):
    """YOLO官方预处理方式"""
    # 调整大小并保持宽高比
    h, w = img.shape[:2]
    r = min(input_size[0]/h, input_size[1]/w)
    new_h, new_w = int(h*r), int(w*r)
    
    # 双线性插值缩放
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # 创建填充图像
    pad_img = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    
    # 计算填充位置
    dw = (input_size[1] - new_w) // 2
    dh = (input_size[0] - new_h) // 2
    pad_img[dh:dh+new_h, dw:dw+new_w] = img_resized
    
    # BGR转RGB，归一化到0-1
    img_rgb = pad_img[..., ::-1].astype(np.float32) / 255.0
    
    # 转换为CHW格式
    img_tensor = img_rgb.transpose(2, 0, 1)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    
    return img_tensor, r, (dw, dh)

def draw_detections(img, boxes, confidences, class_names=None):
    """绘制检测结果"""
    for i, (box, conf) in enumerate(zip(boxes, confidences)):
        left, top, w, h = box
        right = left + w
        bottom = top + h
        
        # 绘制红色边框
        cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
        
        # 绘制置信度标签
        label = f"{conf:.3f}"
        if class_names:
            label = f"{class_names[0]}: {conf:.3f}"
        cv2.putText(img, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    return img

# 加载ONNX模型
session = ort.InferenceSession('model/best.onnx', providers=['CPUExecutionProvider'])

# 测试图片列表
test_images = [
    'dataset/3897.rf.94c81f86436f41e7bba4b2e66934a470.jpg',
    'dataset/2275.rf.586655e669fa91476d091094ff24d359.jpg',
    'dataset/1553.rf.3163d849e95a4c309e0090e43e39a20a.jpg'
]

for img_path in test_images:
    print(f"\n{'='*60}")
    print(f"Testing: {img_path}")
    print(f"{'='*60}")
    
    # 读取图像
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to load {img_path}")
        continue
    
    original_img = img.copy()
    img_height, img_width = img.shape[:2]
    
    # YOLO预处理
    img_tensor, scale, (pad_w, pad_h) = yolo_preprocess(img)
    
    # 推理
    outputs = session.run(None, {session.get_inputs()[0].name: img_tensor})
    
    detection_out = outputs[0]
    mask_proto = outputs[1]
    
    # 解析检测结果
    num_channels = 37
    num_boxes = detection_out.shape[-1]
    
    boxes = []
    confidences = []
    
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
            
            if w > 10 and h > 10:
                boxes.append([left, top, w, h])
                confidences.append(conf)
    
    # 按置信度排序
    if len(boxes) > 0:
        sorted_indices = np.argsort(confidences)[::-1]
        boxes = [boxes[i] for i in sorted_indices]
        confidences = [confidences[i] for i in sorted_indices]
        
        # 只保留前20个高置信度框
        boxes = boxes[:20]
        confidences = confidences[:20]
        
        # 绘制检测结果
        result_img = draw_detections(original_img, boxes, confidences, ['crack'])
        
        # 保存结果
        output_path = f'output_{img_path.split("/")[-1]}'
        cv2.imwrite(output_path, result_img)
        
        print(f"检测到 {len(boxes)} 个裂纹")
        for i in range(min(5, len(boxes))):
            print(f"  crack @ ({boxes[i][0]}, {boxes[i][1]}, {boxes[i][0]+boxes[i][2]}, {boxes[i][1]+boxes[i][3]}) conf={confidences[i]:.4f}")
        print(f"结果已保存到: {output_path}")
    else:
        print("未检测到裂纹")
