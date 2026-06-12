import cv2
import numpy as np
import os
import struct
import threading
from datetime import datetime
from rknnlite.api import RKNNLite

EV_SYN = 0x00
EV_KEY = 0x01
EV_ABS = 0x03
ABS_X = 0x00
ABS_Y = 0x01
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
BTN_TOUCH = 0x14a
BTN_TOOL_FINGER = 0x145

class TouchHandler:
    def __init__(self):
        self.touch_x = 0
        self.touch_y = 0
        self.touch_pressed = False
        self.input_dev = None
        self.running = True
        self.thread = None
        self.event_count = 0
        
    def open_touch_device(self):
        try:
            self.input_dev = open("/dev/input/event2", "rb")
            print("Opened touch device: /dev/input/event2 (GT9XX touchscreen)")
            return True
        except Exception as e:
            print(f"Failed to open touch device: {e}")
            return False
    
    def _read_loop(self):
        while self.running and self.input_dev:
            try:
                data = self.input_dev.read(24)
                if len(data) == 24:
                    _, _, typ, code, value = struct.unpack('llHHI', data)
                    self.event_count += 1
                    if self.event_count % 10 == 0:
                        print(f"Event: type={hex(typ)}, code={hex(code)}, value={value}")
                    
                    if typ == EV_ABS:
                        if code == ABS_X or code == ABS_MT_POSITION_X:
                            self.touch_x = value
                        elif code == ABS_Y or code == ABS_MT_POSITION_Y:
                            self.touch_y = value
                    elif typ == EV_KEY:
                        if code == BTN_TOUCH or code == BTN_TOOL_FINGER:
                            self.touch_pressed = (value == 1)
                            if self.touch_pressed:
                                print(f"Touch pressed at ({self.touch_x}, {self.touch_y})")
            except Exception as e:
                print(f"Read error: {e}")
                break
    
    def start(self):
        if self.input_dev:
            self.thread = threading.Thread(target=self._read_loop)
            self.thread.daemon = True
            self.thread.start()
            print("Touch thread started")
    
    def stop(self):
        self.running = False

class CrackDetector:
    def __init__(self, model_path):
        self.rknn = RKNNLite()
        print(f"Loading RKNN model: {model_path}")
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            print("Failed to load RKNN model!")
            return
        
        ret = self.rknn.init_runtime()
        if ret != 0:
            print("Failed to init runtime!")
            return
        print("RKNN model loaded successfully")
    
    def detect(self, img):
        orig_h, orig_w = img.shape[:2]
        
        img_resized = cv2.resize(img, (640, 640))
        rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32)
        blob = np.expand_dims(blob, axis=0)
        
        outputs = self.rknn.inference(inputs=[blob])
        
        det = outputs[0][0]
        protos = outputs[1][0]
        
        conf = det[4]
        best_idx = np.argmax(conf)
        best_conf = conf[best_idx]
        
        if best_conf < 0.25:
            return None, None, None
        
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
        
        kernel_close = np.ones((3, 3), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        mask_binary = skeletonize(mask_binary)
        
        return best_conf, (x1, y1, x2, y2), mask_binary, self.analyze_crack(mask_binary, (x1, y1, x2, y2))
    
    def analyze_crack(self, mask, bbox):
        if mask is None:
            return {"area": 0, "length": 0, "width": 0, "severity": "NONE", "suggestion": "No crack detected"}
        
        x1, y1, x2, y2 = bbox
        crack_area = cv2.countNonZero(mask)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        max_length = 0
        if contours:
            contour = max(contours, key=cv2.contourArea)
            if len(contour) > 1:
                max_dist = 0
                for i in range(len(contour)):
                    for j in range(i+1, len(contour)):
                        dist = np.sqrt((contour[i][0][0] - contour[j][0][0])**2 + 
                                      (contour[i][0][1] - contour[j][0][1])**2)
                        max_dist = max(max_dist, dist)
                max_length = max_dist
        
        bbox_area = (x2 - x1) * (y2 - y1)
        avg_width = crack_area / max_length if max_length > 0 else 0
        
        severity, suggestion = self.calculate_severity(crack_area, bbox_area, max_length, avg_width)
        
        return {
            "area": crack_area,
            "bbox_area": bbox_area,
            "length": max_length,
            "width": avg_width,
            "severity": severity,
            "suggestion": suggestion
        }
    
    def calculate_severity(self, crack_area, bbox_area, length, width):
        area_ratio = crack_area / bbox_area if bbox_area > 0 else 0
        
        if crack_area < 100:
            return "MINOR", "Surface scratch, no structural concern"
        elif crack_area < 500:
            return "LOW", "Minor crack, monitor periodically"
        elif crack_area < 2000 or length > 50:
            return "MEDIUM", "Moderate crack, recommend repair"
        elif crack_area < 5000 or length > 100 or width > 5:
            return "HIGH", "Severe crack, immediate attention needed"
        else:
            return "CRITICAL", "Critical damage, urgent repair required"
    
    def release(self):
        self.rknn.release()

def draw_detection(img, conf, bbox, mask, analysis=None):
    result = img.copy()
    
    if mask is not None:
        overlay = np.zeros_like(img)
        overlay[mask > 128] = [0, 255, 0]
        result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(result, f"Conf: {conf:.3f}", (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    if analysis:
        y_pos = 30
        cv2.putText(result, f"Area: {analysis['area']} px", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 20
        cv2.putText(result, f"Length: {analysis['length']:.1f} px", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 20
        cv2.putText(result, f"Width: {analysis['width']:.2f} px", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_pos += 20
        
        severity_colors = {
            "NONE": (0, 255, 0),
            "MINOR": (0, 255, 255),
            "LOW": (0, 255, 255),
            "MEDIUM": (0, 165, 255),
            "HIGH": (0, 0, 255),
            "CRITICAL": (255, 0, 255)
        }
        color = severity_colors.get(analysis['severity'], (255, 255, 255))
        cv2.putText(result, f"Severity: {analysis['severity']}", (10, y_pos), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return result

def show_history(display, screen_w, screen_h, touch):
    import glob
    
    history_display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    
    cv2.putText(history_display, "HISTORY", (400, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    
    back_btn = {"x1": 50, "y1": 1750, "x2": 450, "y2": 1880, "color": (60, 60, 60)}
    cv2.rectangle(history_display, (back_btn["x1"], back_btn["y1"]), 
                  (back_btn["x2"], back_btn["y2"]), back_btn["color"], -1)
    cv2.putText(history_display, "BACK", (180, 1820), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    result_files = sorted(glob.glob("/userdata/results/crack_*.jpg"), reverse=True)
    
    if len(result_files) == 0:
        cv2.putText(history_display, "NO HISTORY", (350, 500), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    else:
        row = 0
        col = 0
        for i, f in enumerate(result_files[:6]):
            img = cv2.imread(f)
            if img is not None:
                img = cv2.resize(img, (300, 200))
                x_off = 90 + col * 330
                y_off = 120 + row * 230
                history_display[y_off:y_off+200, x_off:x_off+300] = img
                
                filename = os.path.basename(f)
                cv2.putText(history_display, filename[:20], (x_off, y_off + 220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                col += 1
                if col >= 3:
                    col = 0
                    row += 1
    
    cv2.imshow("CRACK", history_display)
    cv2.waitKey(1)
    
    last_pressed = touch.touch_pressed
    while True:
        cv2.imshow("CRACK", history_display)
        key = cv2.waitKey(50) & 0xFF
        if key == 27:
            print("Back to main via ESC")
            break
        
        if touch.touch_pressed and not last_pressed:
            x, y = touch.touch_x, touch.touch_y
            print(f"History touch: ({x}, {y})")
            if back_btn["x1"] <= x <= back_btn["x2"] and back_btn["y1"] <= y <= back_btn["y2"]:
                print("Back to main")
                break
        
        last_pressed = touch.touch_pressed
    
def main():
    print("=== CRACK INSPECTION SYSTEM ===")
    
    frame = cv2.imread("/userdata/testb.jpg")
    if frame is None:
        print("Test image not found! Creating dummy image...")
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(frame, "NO IMAGE", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    print(f"Image loaded: {frame.shape}")
    
    detector = CrackDetector("/userdata/yolo11_fp_v4.rknn")
    
    SCREEN_W = 1080
    SCREEN_H = 1920
    
    cv2.namedWindow("CRACK", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CRACK", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    inspector_name = "TEST_USER"
    location = "WALL_A1"
    inspection_time = ""
    detecting = False
    detection_result = None
    
    buttons = [
        {"name": "START", "x1": 50, "y1": 1650, "x2": 280, "y2": 1800, "color": (0,200,0)},
        {"name": "SAVE", "x1": 310, "y1": 1650, "x2": 540, "y2": 1800, "color": (255,140,0)},
        {"name": "HISTORY", "x1": 570, "y1": 1650, "x2": 800, "y2": 1800, "color": (128,0,128)},
        {"name": "EXIT", "x1": 830, "y1": 1650, "x2": 1060, "y2": 1800, "color": (80,80,80)},
    ]
    
    def check_button_click(x, y):
        for btn in buttons:
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                return btn["name"]
        return None
    
    touch = TouchHandler()
    touch.open_touch_device()
    touch.start()
    
    print("Ready! Click buttons on screen or use keyboard:")
    print("s=START/STOP, c=SAVE, h=HISTORY, ESC=EXIT")
    
    last_touch_state = False
    display_frame = frame.copy()
    
    while True:
        display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        
        frame_h, frame_w = display_frame.shape[:2]
        scale = min((SCREEN_W - 60)/frame_w, 800/frame_h)
        new_w, new_h = int(frame_w * scale), int(frame_h * scale)
        frame_resized = cv2.resize(display_frame, (new_w, new_h))
        x_off = (SCREEN_W - new_w) // 2
        y_off = 380
        display[y_off:y_off+new_h, x_off:x_off+new_w] = frame_resized
        
        cv2.rectangle(display, (0, 0), (SCREEN_W, 80), (30, 30, 30), -1)
        cv2.putText(display, "CRACK INSPECTION SYSTEM", (280, 55), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        
        cv2.rectangle(display, (30, 85), (SCREEN_W-30, 250), (40, 40, 40), -1)
        cv2.putText(display, "INSPECTION INFO", (400, 130), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        cv2.putText(display, "INSPECTOR:", (60, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
        cv2.rectangle(display, (280, 170), (SCREEN_W-60, 215), (60,60,60), -1)
        cv2.putText(display, inspector_name[:25], (300, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.putText(display, "LOCATION:", (60, 265), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
        cv2.rectangle(display, (280, 250), (SCREEN_W-60, 295), (60,60,60), -1)
        cv2.putText(display, location[:25], (300, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.putText(display, "TIME:", (60, 345), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200,200,200), 2)
        cv2.rectangle(display, (280, 330), (SCREEN_W-60, 375), (60,60,60), -1)
        cv2.putText(display, inspection_time[:25] if inspection_time else "AUTO", (300, 365), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        
        cv2.rectangle(display, (30, 1200), (SCREEN_W-30, 1600), (40, 40, 40), -1)
        cv2.putText(display, "DETECTION RESULTS", (380, 1260), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 2)
        
        status_text = "SCANNING" if detecting else "READY"
        status_color = (0,255,0) if detecting else (255,255,255)
        cv2.putText(display, f"STATUS: {status_text}", (60, 1320), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, status_color, 2)
        
        if detection_result:
            conf, bbox, mask, analysis = detection_result
            
            conf_color = (0,255,0) if conf>0.5 else (0,0,255)
            cv2.putText(display, f"CONFIDENCE: {conf:.2f}", (60, 1380), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
            
            result_text = "CRACK DETECTED" if conf > 0.25 else "NO CRACK"
            result_color = (0,0,255) if conf > 0.25 else (0,255,0)
            cv2.putText(display, f"RESULT: {result_text}", (500, 1380), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, result_color, 2)
            
            if analysis:
                cv2.putText(display, f"AREA: {analysis['area']} px", (60, 1440), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                cv2.putText(display, f"LENGTH: {analysis['length']:.1f} px", (400, 1440), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                cv2.putText(display, f"WIDTH: {analysis['width']:.2f} px", (750, 1440), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,0), 2)
                
                severity_colors = {
                    "NONE": (0, 255, 0),
                    "MINOR": (0, 255, 255),
                    "LOW": (0, 255, 255),
                    "MEDIUM": (0, 165, 255),
                    "HIGH": (0, 0, 255),
                    "CRITICAL": (255, 0, 255)
                }
                sev_color = severity_colors.get(analysis['severity'], (255,255,255))
                cv2.putText(display, f"SEVERITY: {analysis['severity']}", (60, 1510), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, sev_color, 2)
                
                suggestion_display = analysis['suggestion'][:40] + "..." if len(analysis['suggestion']) > 40 else analysis['suggestion']
                cv2.putText(display, f"SUGGESTION: {suggestion_display}", (60, 1570), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200,200,255), 2)
        else:
            cv2.putText(display, "CONFIDENCE: --", (60, 1380), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(display, "RESULT: --", (500, 1380), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            cv2.putText(display, "AREA: -- | LENGTH: -- | WIDTH: --", (60, 1440), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200,200,200), 2)
        
        cv2.rectangle(display, (0, 1600), (SCREEN_W, SCREEN_H), (25, 25, 25), -1)
        
        for btn in buttons:
            color = btn["color"] if btn["name"] != "START" or not detecting else (0,100,0)
            cv2.rectangle(display, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), color, -1)
            cv2.rectangle(display, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (200,200,200), 2)
            
            text = btn["name"] if btn["name"] != "START" or not detecting else "STOP"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
            tx = btn["x1"] + (btn["x2"] - btn["x1"]) // 2 - text_size[0] // 2
            ty = btn["y1"] + (btn["y2"] - btn["y1"]) // 2 + text_size[1] // 2
            cv2.putText(display, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)
        
        cv2.imshow("CRACK", display)
        
        if touch.touch_pressed and not last_touch_state:
            print(f"\n=== TOUCH EVENT ===")
            print(f"Touch detected at: ({touch.touch_x}, {touch.touch_y})")
            print(f"Screen resolution: {SCREEN_W}x{SCREEN_H}")
            
            btn_name = check_button_click(touch.touch_x, touch.touch_y)
            if btn_name:
                print(f"Button clicked: {btn_name}")
                
                if btn_name == "START":
                    detecting = not detecting
                    print(f"Detecting: {detecting}")
                    if detecting:
                        inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        print(f"Inspection Time: {inspection_time}")
                        print("Running detection...")
                        result = detector.detect(frame)
                        if result is not None and len(result) == 4:
                            conf, bbox, mask, analysis = result
                            detection_result = (conf, bbox, mask, analysis)
                            display_frame = draw_detection(frame, conf, bbox, mask, analysis)
                            print(f"Detection complete: conf={conf:.3f}")
                            print(f"Analysis: Area={analysis['area']}, Length={analysis['length']:.1f}, Width={analysis['width']:.2f}")
                            print(f"Severity: {analysis['severity']} - {analysis['suggestion']}")
                        else:
                            detection_result = None
                            display_frame = frame.copy()
                            print("No crack detected")
                    else:
                        detection_result = None
                        display_frame = frame.copy()
                elif btn_name == "SAVE":
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"/userdata/results/crack_{timestamp}.jpg"
                    os.makedirs("/userdata/results", exist_ok=True)
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")
                elif btn_name == "HISTORY":
                    print("History clicked")
                    show_history(display, SCREEN_W, SCREEN_H, touch)
                elif btn_name == "EXIT":
                    print("Exit pressed")
                    touch.stop()
                    detector.release()
                    break
            else:
                print("No button clicked - checking all button areas:")
                for btn in buttons:
                    in_x = btn["x1"] <= touch.touch_x <= btn["x2"]
                    in_y = btn["y1"] <= touch.touch_y <= btn["y2"]
                    print(f"  {btn['name']}: x[{btn['x1']}-{btn['x2']}]={in_x}, y[{btn['y1']}-{btn['y2']}]={in_y}")
        
        last_touch_state = touch.touch_pressed
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == 27:
            print("Exit pressed")
            touch.stop()
            detector.release()
            break
        elif key == ord('s') or key == ord('S'):
            detecting = not detecting
            print(f"Detecting: {detecting}")
            if detecting:
                print("Running detection...")
                result = detector.detect(frame)
                if result is not None and len(result) == 4:
                    conf, bbox, mask, analysis = result
                    detection_result = (conf, bbox, mask, analysis)
                    display_frame = draw_detection(frame, conf, bbox, mask, analysis)
                    print(f"Detection complete: conf={conf:.3f}")
                    print(f"Analysis: Area={analysis['area']}, Length={analysis['length']:.1f}, Width={analysis['width']:.2f}")
                    print(f"Severity: {analysis['severity']} - {analysis['suggestion']}")
                else:
                    detection_result = None
                    display_frame = frame.copy()
                    print("No crack detected")
            else:
                detection_result = None
                display_frame = frame.copy()
        elif key == ord('c') or key == ord('C'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"/userdata/results/crack_{timestamp}.jpg"
            os.makedirs("/userdata/results", exist_ok=True)
            cv2.imwrite(filename, display_frame)
            print(f"Saved: {filename}")
        elif key == ord('h') or key == ord('H'):
            print("History clicked")
        elif key != 255:
            print(f"Key pressed: {chr(key)} ({key})")
    
    cv2.destroyAllWindows()
    print("System closed")

if __name__ == '__main__':
    main()