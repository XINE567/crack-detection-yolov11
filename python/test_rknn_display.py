import cv2
import numpy as np
from rknnlite.api import RKNNLite
import time

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

def main():
    print("=" * 50)
    print("YOLO11 图片检测 (RKNN Lite)")
    print("=" * 50)
    
    rknn = RKNNLite()
    ret = rknn.load_rknn("/userdata/yolo11_fp_v4.rknn")
    if ret != 0:
        print("加载 RKNN 模型失败!")
        exit(ret)
    
    print("初始化运行环境...")
    ret = rknn.init_runtime()
    if ret != 0:
        print("初始化运行环境失败!")
        exit(ret)
    
    print("加载测试图片...")
    img = cv2.imread("/userdata/testb.jpg")
    if img is None:
        print("无法读取测试图片!")
        exit(-1)
    
    orig_h, orig_w = img.shape[:2]
    
    cv2.namedWindow("YOLO11 Detection", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("YOLO11 Detection", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("\n按 'q' 退出...")
    
    while True:
        img_resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32)
        blob = np.expand_dims(blob, axis=0)
        
        outputs = rknn.inference(inputs=[blob])
        det = outputs[0][0]
        protos = outputs[1][0]
        
        conf = det[4]
        best_idx = np.argmax(conf)
        best_conf = conf[best_idx]
        
        result = img.copy()
        
        if best_conf > 0.1:
            cx, cy = det[0, best_idx], det[1, best_idx]
            w, h = det[2, best_idx], det[3, best_idx]
            
            scale_x = orig_w / 640
            scale_y = orig_h / 640
            cx_s, cy_s = cx * scale_x, cy * scale_y
            w_s, h_s = w * scale_x, h * scale_y
            
            x1 = max(0, int(cx_s - w_s/2))
            y1 = max(0, int(cy_s - h_s/2))
            x2 = min(orig_w, int(cx_s + w_s/2))
            y2 = min(orig_h, int(cy_s + h_s/2))
            
            mask_coef = det[6:37, best_idx]
            mask = np.zeros((160, 160), dtype=np.float32)
            for i in range(31):
                mask += protos[i] * mask_coef[i]
            mask = 1 / (1 + np.exp(-mask))
            mask = cv2.resize(mask, (640, 640))
            mask_binary = (mask > 0.5).astype(np.uint8) * 255
            mask_binary = cv2.resize(mask_binary, (orig_w, orig_h))
            
            mask_cropped = np.zeros_like(mask_binary)
            mask_cropped[y1:y2, x1:x2] = mask_binary[y1:y2, x1:x2]
            mask_binary = mask_cropped
            
            kernel_close = np.ones((3, 3), np.uint8)
            mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
            mask_binary = skeletonize(mask_binary)
            
            overlay = np.zeros_like(img)
            overlay[mask_binary > 128] = [0, 255, 0]
            result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
            contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(result, f"Conf: {best_conf:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        cv2.putText(result, "YOLO11-RKNN Display", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("YOLO11 Detection", result)
        
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    rknn.release()
    print("\n检测结束")

if __name__ == '__main__':
    main()