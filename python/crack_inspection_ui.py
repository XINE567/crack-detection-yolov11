import cv2
import numpy as np
from rknnlite.api import RKNNLite
import os
import time
import json
from datetime import datetime

class CrackInspectionUI:
    def __init__(self):
        self.rknn = None
        self.cap = None
        self.running = False
        self.detecting = False
        self.current_frame = None
        self.detection_result = None
        self.history = []
        self.scan_mode = 'manual'
        
        self.WINDOW_NAME = "智能墙面裂纹检测系统"
        
        self.status_text = "就绪"
        self.confidence = 0.0
        self.crack_count = 0
        self.total_scanned = 0

    def skeletonize(self, image):
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

    def load_model(self):
        try:
            self.rknn = RKNNLite()
            ret = self.rknn.load_rknn("/userdata/yolo11_fp_v4.rknn")
            if ret != 0:
                return False, "加载模型失败"
            ret = self.rknn.init_runtime()
            if ret != 0:
                return False, "初始化运行环境失败"
            return True, "模型加载成功"
        except Exception as e:
            return False, str(e)

    def detect_crack(self, frame):
        if self.rknn is None:
            return None
        
        orig_h, orig_w = frame.shape[:2]
        img_resized = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32)
        blob = np.expand_dims(blob, axis=0)
        
        try:
            outputs = self.rknn.inference(inputs=[blob])
            det = outputs[0][0]
            protos = outputs[1][0]
            
            conf = det[4]
            best_idx = np.argmax(conf)
            best_conf = conf[best_idx]
            
            if best_conf < 0.1:
                return {"confidence": 0.0, "bbox": None, "mask": None, "crack": False}
            
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
            mask_binary = self.skeletonize(mask_binary)
            
            return {
                "confidence": float(best_conf),
                "bbox": (x1, y1, x2, y2),
                "mask": mask_binary,
                "crack": best_conf >= 0.5
            }
        except Exception as e:
            print(f"检测错误: {e}")
            return None

    def draw_ui(self, frame):
        overlay = frame.copy()
        
        cv2.rectangle(overlay, (0, 0), (640, 60), (0, 0, 0), -1)
        cv2.putText(overlay, "智能墙面裂纹检测系统", (180, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)
        
        cv2.rectangle(overlay, (0, 500), (640, 70), (0, 0, 0), -1)
        
        cv2.putText(overlay, f"状态: {self.status_text}", (50, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(overlay, f"置信度: {self.confidence:.2f}", (200, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if self.confidence > 0.5 else (0, 0, 255), 1)
        cv2.putText(overlay, f"已检测: {self.total_scanned} | 裂纹: {self.crack_count}", (380, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.rectangle(overlay, (50, 560), (200, 610), (0, 255, 0) if not self.detecting else (0, 128, 0), -1)
        cv2.putText(overlay, "[S]开始检测", (60, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(overlay, (220, 560), (370, 610), (0, 0, 255), -1)
        cv2.putText(overlay, "[T]停止检测", (230, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(overlay, (390, 560), (540, 610), (255, 165, 0), -1)
        cv2.putText(overlay, "[C]拍照保存", (400, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.rectangle(overlay, (560, 560), (710, 610), (128, 0, 128), -1)
        cv2.putText(overlay, "[H]历史记录", (570, 588), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        mode_color = [(255,255,255),(128,128,128),(128,128,128)]
        if self.scan_mode == 'auto':
            mode_color = [(128,128,128),(255,255,255),(128,128,128)]
        elif self.scan_mode == 'continuous':
            mode_color = [(128,128,128),(128,128,128),(255,255,255)]
        
        cv2.putText(overlay, "[1]手动", (70, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color[0], 1)
        cv2.putText(overlay, "[2]自动", (190, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color[1], 1)
        cv2.putText(overlay, "[3]连续", (310, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, mode_color[2], 1)
        
        cv2.putText(overlay, "[ESC]退出", (500, 515), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
        
        return cv2.addWeighted(frame, 0.9, overlay, 0.1, 0)

    def draw_detection(self, frame, result):
        if result is None or not result["crack"]:
            return frame
        
        x1, y1, x2, y2 = result["bbox"]
        mask = result["mask"]
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        overlay = np.zeros_like(frame)
        overlay[mask > 128] = [0, 255, 0]
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"裂纹检测: {result['confidence']:.2%}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame

    def save_result(self, frame, result):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/userdata/results/crack_{timestamp}.jpg"
        
        os.makedirs("/userdata/results", exist_ok=True)
        cv2.imwrite(filename, frame)
        
        record = {
            "timestamp": timestamp,
            "filename": filename,
            "confidence": result["confidence"],
            "crack_found": result["crack"],
            "bbox": result["bbox"]
        }
        self.history.append(record)
        
        with open("/userdata/results/history.json", "w") as f:
            json.dump(self.history, f)
        
        print(f"结果已保存: {filename}")

    def show_history(self):
        history_win = "检测历史记录"
        cv2.namedWindow(history_win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(history_win, 640, 480)
        
        while True:
            history_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            cv2.putText(history_frame, "检测历史记录", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            y_pos = 70
            for i, record in enumerate(reversed(self.history[-10:])):
                status = "有裂纹" if record["crack_found"] else "无裂纹"
                color = (0, 0, 255) if record["crack_found"] else (0, 255, 0)
                text = f"{i+1}. {record['timestamp']} - {status} ({record['confidence']:.2f})"
                cv2.putText(history_frame, text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                y_pos += 30
            
            cv2.putText(history_frame, "按 ESC 返回", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 128, 128), 1)
            
            cv2.imshow(history_win, history_frame)
            
            key = cv2.waitKey(30)
            if key == 27:
                break
        
        cv2.destroyWindow(history_win)

    def handle_keyboard(self, key):
        if key == ord('s') or key == ord('S'):
            self.detecting = not self.detecting
            self.status_text = "检测中..." if self.detecting else "就绪"
            
        elif key == ord('t') or key == ord('T'):
            self.detecting = False
            self.status_text = "已停止"
            
        elif key == ord('c') or key == ord('C'):
            if self.current_frame is not None and self.detection_result:
                self.save_result(self.current_frame, self.detection_result)
                self.status_text = "图片已保存"
            
        elif key == ord('h') or key == ord('H'):
            self.show_history()
            
        elif key == ord('1'):
            self.scan_mode = 'manual'
            
        elif key == ord('2'):
            self.scan_mode = 'auto'
            
        elif key == ord('3'):
            self.scan_mode = 'continuous'
            
        elif key == 27:
            self.running = False

    def run(self):
        success, msg = self.load_model()
        if not success:
            print(f"模型加载失败: {msg}")
            return
        
        print("尝试打开摄像头...")
        camera_dev = '/dev/video-camera0'
        use_test_image = False
        
        if os.path.exists(camera_dev):
            self.cap = cv2.VideoCapture(camera_dev)
            if not self.cap.isOpened():
                print("摄像头打开失败，使用测试图片")
                use_test_image = True
        else:
            print("摄像头设备不存在，使用测试图片")
            use_test_image = True
        
        if use_test_image:
            self.current_frame = cv2.imread("/userdata/testb.jpg")
            if self.current_frame is None:
                print("测试图片也无法加载")
                return
            self.running = True
        else:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
            self.running = True
        
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, 710, 620)
        
        print("系统启动成功！")
        print("快捷键: S-开始检测 | T-停止检测 | C-拍照保存 | H-历史记录 | 1/2/3-切换模式 | ESC-退出")
        self.status_text = "就绪"
        
        last_detect_time = 0
        detect_interval = 1000
        
        while self.running:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break
                self.current_frame = cv2.resize(frame, (710, 500))
            else:
                self.current_frame = cv2.resize(cv2.imread("/userdata/testb.jpg"), (710, 500))
            
            current_time = cv2.getTickCount()
            elapsed_ms = (current_time - last_detect_time) / cv2.getTickFrequency() * 1000
            
            if self.detecting:
                if self.scan_mode == 'continuous' or \
                   (self.scan_mode == 'auto' and elapsed_ms >= detect_interval) or \
                   (self.scan_mode == 'manual' and elapsed_ms >= 5000):
                    self.detection_result = self.detect_crack(self.current_frame)
                    if self.detection_result:
                        self.confidence = self.detection_result["confidence"]
                        if self.detection_result["crack"]:
                            self.crack_count += 1
                        self.total_scanned += 1
                    last_detect_time = current_time
            
            display_frame = self.current_frame.copy()
            
            if self.detection_result:
                display_frame = self.draw_detection(display_frame, self.detection_result)
            
            display_frame = self.draw_ui(display_frame)
            
            cv2.imshow(self.WINDOW_NAME, display_frame)
            
            key = cv2.waitKey(1)
            if key != -1:
                self.handle_keyboard(key)
        
        if self.cap:
            self.cap.release()
        if self.rknn:
            self.rknn.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = CrackInspectionUI()
    app.run()