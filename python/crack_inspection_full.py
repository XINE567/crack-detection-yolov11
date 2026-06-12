import cv2
import numpy as np
import os
import struct
import threading
import time
from datetime import datetime

from inspection_db import (
    init_db,
    create_session,
    end_session,
    save_capture,
    get_session,
    get_session_captures,
    mark_capture_uploaded,
)
from upload_client import upload_inspection_session, DEFAULT_BACKEND_URL
from frontend_client import FrontendClient, DEFAULT_FRONTEND_WS_URL, DEFAULT_STREAM_FPS

LIVE_FRAME_PATH = "/userdata/live_frame.jpg"
RESULTS_DIR = "/userdata/results"
BACKEND_URL = os.environ.get("BACKEND_URL", DEFAULT_BACKEND_URL)
FRONTEND_WS_URL = os.environ.get("FRONTEND_WS_URL", DEFAULT_FRONTEND_WS_URL)
FRONTEND_FPS = int(os.environ.get("FRONTEND_FPS", str(DEFAULT_STREAM_FPS)))
RESULT_FLASH_SEC = 1.0
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

title_font = None

try:
    from rknnlite.api import RKNNLite
except:
    print("RKNNLite not found, running in demo mode")
    class RKNNLite:
        def load_rknn(self, path): return 0
        def init_runtime(self): return 0
        def inference(self, inputs): return [np.zeros((1,37,8400)), np.zeros((1,32,160,160))]
        def release(self): pass
    class CrackDetector:
        def __init__(self, path): print(f"Demo mode: {path}")
        def detect(self, img): return (0.85, (100,100,400,400), np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8), {"area":500,"bbox_area":90000,"length":100,"width":5,"severity":"MEDIUM","suggestion":"Recommend repair"})
        def release(self): pass

class VirtualKeyboard:
    def __init__(self, screen_w=1080, screen_h=1920):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.key_buttons = []
        self.current_text = ""
        self.active = False
        
        self.key_w = 100
        self.key_h = 85
        self.key_gap = 8
        
        self.rows_y = [1420, 1515, 1610, 1705]
        
        self._build_keys()
    
    def _build_keys(self):
        self.key_buttons = []
        
        rows = [
            ("1234567890", self.rows_y[0]),
            ("QWERTYUIOP", self.rows_y[1]),
            ("ASDFGHJKL", self.rows_y[2]),
        ]
        
        for row_text, y in rows:
            n = len(row_text)
            total_w = n * self.key_w + (n - 1) * self.key_gap
            start_x = (self.screen_w - total_w) // 2
            
            for i, key in enumerate(row_text):
                x1 = start_x + i * (self.key_w + self.key_gap)
                x2 = x1 + self.key_w
                self.key_buttons.append({
                    "key": key, "x1": int(x1), "y1": int(y), "x2": int(x2), "y2": int(y + self.key_h)
                })
        
        row4_y = self.rows_y[3]
        
        back_x1 = 30
        self.key_buttons.append({
            "key": "BACK", "x1": int(back_x1), "y1": int(row4_y), 
            "x2": int(back_x1 + 130), "y2": int(row4_y + self.key_h)
        })
        
        space_x1 = 175
        self.key_buttons.append({
            "key": "SPACE", "x1": int(space_x1), "y1": int(row4_y), 
            "x2": int(space_x1 + 520), "y2": int(row4_y + self.key_h)
        })
        
        clear_x1 = 710
        self.key_buttons.append({
            "key": "CLEAR", "x1": int(clear_x1), "y1": int(row4_y), 
            "x2": int(clear_x1 + 130), "y2": int(row4_y + self.key_h)
        })
        
        enter_x1 = 855
        self.key_buttons.append({
            "key": "ENTER", "x1": int(enter_x1), "y1": int(row4_y), 
            "x2": int(enter_x1 + 195), "y2": int(row4_y + self.key_h)
        })
    
    def draw(self, frame):
        if not self.active:
            return
        
        for btn in self.key_buttons:
            x1, y1, x2, y2 = int(btn["x1"]), int(btn["y1"]), int(btn["x2"]), int(btn["y2"])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 65), -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 120, 130), 2)
            
            if btn["key"] == "BACK":
                text = "DEL"
            elif btn["key"] == "ENTER":
                text = "OK"
            elif btn["key"] == "CLEAR":
                text = "CLR"
            elif btn["key"] == "SPACE":
                text = "SPACE"
            else:
                text = btn["key"]
            
            font_scale = 1.4
            thickness = 2
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            tx = x1 + (x2 - x1) // 2 - text_size[0] // 2
            ty = y1 + (y2 - y1) // 2 + text_size[1] // 2
            
            color = (255, 255, 255)
            if btn["key"] == "BACK":
                color = (255, 180, 80)
            elif btn["key"] == "ENTER":
                color = (80, 255, 80)
            elif btn["key"] == "CLEAR":
                color = (255, 80, 80)
            
            cv2.putText(frame, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
    
    def handle_click(self, x, y):
        if not self.active:
            return None
        
        for btn in self.key_buttons:
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                if btn["key"] == "BACK":
                    self.current_text = self.current_text[:-1]
                    return "backspace"
                elif btn["key"] == "SPACE":
                    self.current_text += " "
                    return "space"
                elif btn["key"] == "ENTER":
                    return "enter"
                elif btn["key"] == "CLEAR":
                    self.current_text = ""
                    return "clear"
                elif len(btn["key"]) == 1:
                    self.current_text += btn["key"].lower()
                    return "letter"
        
        return None
    
    def get_text(self):
        return self.current_text
    
    def set_text(self, text):
        self.current_text = text
    
    def clear(self):
        self.current_text = ""
    
    def activate(self):
        self.active = True
    
    def deactivate(self):
        self.active = False


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
        
        kernel_close = np.ones((3, 3), np.uint8)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel_close, iterations=1)
        
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
                contour_points = contour.reshape(-1, 2)
                min_x, min_y = contour_points.min(axis=0)
                max_x, max_y = contour_points.max(axis=0)
                max_length = max(max_x - min_x, max_y - min_y)
        
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


def draw_input_screen(display, inspector_name, location, active_input, virtual_keyboard):
    SCREEN_W = 1080
    SCREEN_H = 1920

    if display.dtype != np.uint8 or len(display.shape) != 3 or display.shape[2] != 3:
        display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
    
    display[:, :] = [20, 20, 25]

    cv2.rectangle(display, (0, 0), (SCREEN_W, 200), (25, 25, 35), -1)
    
    title1 = "SMART INSPECTION SYSTEM"
    title2 = "Crack Detection"
    
    t1_size = cv2.getTextSize(title1, cv2.FONT_HERSHEY_SIMPLEX, 1.8, 3)[0]
    t1_x = (SCREEN_W - t1_size[0]) // 2
    cv2.putText(display, title1, (t1_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 220, 255), 3)
    
    t2_size = cv2.getTextSize(title2, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 2)[0]
    t2_x = (SCREEN_W - t2_size[0]) // 2
    cv2.putText(display, title2, (t2_x, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 200, 255), 2)

    cv2.rectangle(display, (0, 200), (SCREEN_W, 900), (35, 35, 45), -1)
    cv2.putText(display, "Enter Inspection Info", (280, 260),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

    inspector_box = {"x1": 80, "y1": 330, "x2": 1000, "y2": 480}
    if active_input == "inspector":
        cv2.rectangle(display, (inspector_box["x1"], inspector_box["y1"]),
                      (inspector_box["x2"], inspector_box["y2"]), (0, 100, 180), -1)
        cv2.rectangle(display, (inspector_box["x1"], inspector_box["y1"]),
                      (inspector_box["x2"], inspector_box["y2"]), (0, 150, 220), 4)
        display_text = virtual_keyboard.get_text() if virtual_keyboard.active else inspector_name
    else:
        cv2.rectangle(display, (inspector_box["x1"], inspector_box["y1"]),
                      (inspector_box["x2"], inspector_box["y2"]), (60, 60, 80), -1)
        cv2.rectangle(display, (inspector_box["x1"], inspector_box["y1"]),
                      (inspector_box["x2"], inspector_box["y2"]), (100, 100, 120), 2)
        display_text = inspector_name

    cv2.putText(display, "Inspector:", (100, 305), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 2)
    text_color = (200, 200, 200) if not display_text else (255, 255, 255)
    display_text = display_text if display_text else "Tap to enter..."
    cv2.putText(display, display_text[:22], (110, 420), cv2.FONT_HERSHEY_SIMPLEX, 1.6, text_color, 2)

    location_box = {"x1": 80, "y1": 580, "x2": 1000, "y2": 730}
    if active_input == "location":
        cv2.rectangle(display, (location_box["x1"], location_box["y1"]),
                      (location_box["x2"], location_box["y2"]), (0, 100, 180), -1)
        cv2.rectangle(display, (location_box["x1"], location_box["y1"]),
                      (location_box["x2"], location_box["y2"]), (0, 150, 220), 4)
        display_text = virtual_keyboard.get_text() if virtual_keyboard.active else location
    else:
        cv2.rectangle(display, (location_box["x1"], location_box["y1"]),
                      (location_box["x2"], location_box["y2"]), (60, 60, 80), -1)
        cv2.rectangle(display, (location_box["x1"], location_box["y1"]),
                      (location_box["x2"], location_box["y2"]), (100, 100, 120), 2)
        display_text = location

    cv2.putText(display, "Location:", (100, 545), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (200, 200, 200), 2)
    text_color = (200, 200, 200) if not display_text else (255, 255, 255)
    display_text = display_text if display_text else "Tap to enter..."
    cv2.putText(display, display_text[:22], (110, 670), cv2.FONT_HERSHEY_SIMPLEX, 1.6, text_color, 2)

    cv2.rectangle(display, (0, 780), (SCREEN_W, 950), (30, 30, 40), -1)
    cv2.putText(display, "Tap input box, then use keyboard below", (180, 870),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (130, 130, 160), 2)

    if virtual_keyboard.active:
        virtual_keyboard.draw(display)

    can_start = inspector_name.strip() and location.strip()

    start_btn_w = 600
    start_btn_h = 120
    start_btn_x1 = (SCREEN_W - start_btn_w) // 2
    start_btn_x2 = start_btn_x1 + start_btn_w
    start_btn_y1 = 1800
    start_btn_y2 = start_btn_y1 + start_btn_h
    
    if can_start:
        btn_color = (0, 160, 60)
    else:
        btn_color = (50, 80, 60)
    
    cv2.rectangle(display, (start_btn_x1, start_btn_y1),
                  (start_btn_x2, start_btn_y2), btn_color, -1)
    cv2.rectangle(display, (start_btn_x1, start_btn_y1),
                  (start_btn_x2, start_btn_y2), (180, 180, 180), 4)

    btn_text = "START INSPECTION" if can_start else "COMPLETE INFO"
    text_size = cv2.getTextSize(btn_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
    tx = (start_btn_x1 + start_btn_x2) // 2 - text_size[0] // 2
    ty = (start_btn_y1 + start_btn_y2) // 2 + text_size[1] // 2
    cv2.putText(display, btn_text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    return display, inspector_box, location_box, {"x1": start_btn_x1, "y1": start_btn_y1, "x2": start_btn_x2, "y2": start_btn_y2}


def read_live_frame(fallback=None, frontend_client=None, prefer_frontend=False):
    if frontend_client is not None:
        frame = frontend_client.get_frame()
        if frame is not None:
            return frame
        if prefer_frontend:
            return None

    if not prefer_frontend:
        frame = cv2.imread(LIVE_FRAME_PATH)
        if frame is not None:
            return frame
    if fallback is not None:
        return fallback.copy()
    return None


def connect_frontend(frontend_client):
    if frontend_client is None:
        return False
    return frontend_client.start()


def disconnect_frontend(frontend_client):
    if frontend_client is not None:
        frontend_client.stop()


def _py(val):
    if hasattr(val, "item"):
        return val.item()
    return val


def build_detection_record(conf, bbox, mask, analysis):
    return {
        "confidence": float(_py(conf)) if conf is not None else 0,
        "crack_count": 1 if conf and conf > 0.25 else 0,
        "crack_area": int(_py(analysis.get("area", 0))) if analysis else 0,
        "max_width": float(_py(analysis.get("width", 0))) if analysis else 0,
        "total_length": int(_py(analysis.get("length", 0))) if analysis else 0,
        "severity": analysis.get("severity", "NONE") if analysis else "NONE",
        "suggestion": analysis.get("suggestion", "No crack detected") if analysis else "No crack detected",
        "result_json": {
            "bbox": [int(_py(v)) for v in bbox] if bbox else [],
            "area": int(_py(analysis.get("area", 0))) if analysis else 0,
            "length": float(_py(analysis.get("length", 0))) if analysis else 0,
            "width": float(_py(analysis.get("width", 0))) if analysis else 0,
            "severity": analysis.get("severity", "NONE") if analysis else "NONE",
            "model": "YOLOv11-RKNN",
        },
    }


def persist_capture(session_id, source_frame, result_frame, conf, bbox, mask, analysis):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(RESULTS_DIR, f"raw_{timestamp}.jpg")
    result_path = os.path.join(RESULTS_DIR, f"result_{timestamp}.jpg")

    cv2.imwrite(raw_path, source_frame)
    cv2.imwrite(result_path, result_frame)

    record = build_detection_record(conf, bbox, mask, analysis)
    capture_id = save_capture(session_id, raw_path, result_path, record)
    print(f"Saved capture #{capture_id}: {result_path}")
    return capture_id, result_path


def draw_button(display, btn):
    cv2.rectangle(display, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), btn["color"], -1)
    cv2.rectangle(display, (btn["x1"], btn["y1"]), (btn["x2"], btn["y2"]), (200, 200, 200), 2)
    text_size = cv2.getTextSize(btn["name"], cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
    tx = btn["x1"] + (btn["x2"] - btn["x1"]) // 2 - text_size[0] // 2
    ty = btn["y1"] + (btn["y2"] - btn["y1"]) // 2 + text_size[1] // 2
    cv2.putText(display, btn["name"], (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)


def draw_info_panel(display, inspector_name, location, inspection_time, extra_lines=None):
    screen_w = display.shape[1]
    cv2.rectangle(display, (0, 0), (screen_w, 80), (30, 30, 30), -1)
    cv2.putText(display, "CRACK INSPECTION SYSTEM", (280, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)

    cv2.rectangle(display, (30, 85), (screen_w - 30, 250), (40, 40, 40), -1)
    cv2.putText(display, "INSPECTION INFO", (400, 130),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    cv2.putText(display, "INSPECTOR:", (60, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.rectangle(display, (280, 170), (screen_w - 60, 215), (60, 60, 60), -1)
    cv2.putText(display, inspector_name[:25], (300, 205), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(display, "LOCATION:", (60, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.rectangle(display, (280, 250), (screen_w - 60, 295), (60, 60, 60), -1)
    cv2.putText(display, location[:25], (300, 285), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(display, "INSPECTION TIME:", (60, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
    cv2.rectangle(display, (320, 330), (screen_w - 60, 375), (60, 60, 60), -1)
    cv2.putText(display, inspection_time[:30], (340, 365), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    if extra_lines:
        y = 420
        for line, color in extra_lines:
            cv2.putText(display, line, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            y += 35


def embed_preview(display, frame, top=380, max_h=780):
    if frame is None:
        cv2.putText(display, "WAITING FOR LIVE FRAME...", (220, 700),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 180, 255), 2)
        return

    screen_w = display.shape[1]
    frame_h, frame_w = frame.shape[:2]
    scale = min((screen_w - 60) / frame_w, max_h / frame_h)
    new_w, new_h = int(frame_w * scale), int(frame_h * scale)
    frame_resized = cv2.resize(frame, (new_w, new_h))
    x_off = (screen_w - new_w) // 2
    y_off = top
    display[y_off:y_off + new_h, x_off:x_off + new_w] = frame_resized


def draw_analysis_panel(display, analysis, conf=None, title="DETECTION RESULTS"):
    screen_w = display.shape[1]
    cv2.rectangle(display, (30, 1180), (screen_w - 30, 1580), (40, 40, 40), -1)
    cv2.putText(display, title, (380, 1230), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    if analysis:
        conf_color = (0, 255, 0) if conf and conf > 0.5 else (0, 0, 255)
        cv2.putText(display, f"CONFIDENCE: {conf:.2f}" if conf is not None else "CONFIDENCE: --",
                    (60, 1290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, conf_color, 2)
        cv2.putText(display, f"AREA: {analysis['area']} px", (60, 1350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(display, f"LENGTH: {analysis['length']:.1f} px", (400, 1350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(display, f"WIDTH: {analysis['width']:.2f} px", (750, 1350),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)
        cv2.putText(display, f"SEVERITY: {analysis['severity']}", (60, 1420),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        suggestion = analysis["suggestion"][:42] + "..." if len(analysis["suggestion"]) > 42 else analysis["suggestion"]
        cv2.putText(display, f"SUGGESTION: {suggestion}", (60, 1480),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 255), 2)
    else:
        cv2.putText(display, "NO CRACK DETECTED", (320, 1350),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)


def upload_session_captures(session_id):
    from datetime import datetime

    session = get_session(session_id)
    if not session:
        return 0, ["会话不存在"], None

    if not session.get("end_time"):
        end_session(session_id)
        session = get_session(session_id)

    captures = get_session_captures(session_id)
    pending = [c for c in captures if not c.get("uploaded")]
    if not pending:
        return 0, ["没有待上传的记录"], None

    saved_count, messages, summary, uploaded_ids, resp = upload_inspection_session(
        session, captures, BACKEND_URL
    )
    print(summary)

    for capture_id in uploaded_ids:
        mark_capture_uploaded(capture_id)

    return saved_count, messages, resp
    
def main():
    print("=== CRACK INSPECTION SYSTEM ===")
    init_db()

    fallback_frame = cv2.imread("/userdata/testb.jpg")
    if fallback_frame is None:
        fallback_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(fallback_frame, "NO IMAGE", (400, 360), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    detector = CrackDetector("/userdata/yolo11_fp_v4.rknn")

    SCREEN_W = 1080
    SCREEN_H = 1920

    cv2.namedWindow("CRACK", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("CRACK", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    app_state = "input"
    inspector_name = ""
    location = ""
    active_input = None
    virtual_keyboard = VirtualKeyboard(SCREEN_W, SCREEN_H)

    session_id = None
    inspection_time = ""
    frontend_connected = False
    capture_count = 0
    session_captures = []
    selected_capture_idx = 0

    live_frame = fallback_frame.copy()
    flash_frame = None
    flash_until = 0.0
    flash_analysis = None
    flash_conf = None
    status_message = ""

    start_btn = None

    frontend_client = FrontendClient(FRONTEND_WS_URL, FRONTEND_FPS)

    touch = TouchHandler()
    touch.open_touch_device()
    touch.start()

    print("=== Smart Inspection System ===")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"Frontend WS: {FRONTEND_WS_URL}")

    last_touch_state = False

    def hit_button(x, y, buttons):
        for btn in buttons:
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                return btn["name"]
        return None

    while True:
        display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)

        if app_state == "input":
            display, inspector_box, location_box, start_btn = draw_input_screen(
                display, inspector_name, location, active_input, virtual_keyboard
            )

            if touch.touch_pressed and not last_touch_state:
                x, y = touch.touch_x, touch.touch_y

                if virtual_keyboard.active:
                    result = virtual_keyboard.handle_click(x, y)
                    if result == "enter":
                        if active_input == "inspector":
                            inspector_name = virtual_keyboard.get_text()
                        elif active_input == "location":
                            location = virtual_keyboard.get_text()
                        virtual_keyboard.deactivate()
                        virtual_keyboard.clear()
                        active_input = None
                else:
                    if inspector_box["x1"] <= x <= inspector_box["x2"] and \
                       inspector_box["y1"] <= y <= inspector_box["y2"]:
                        active_input = "inspector"
                        virtual_keyboard.set_text(inspector_name)
                        virtual_keyboard.activate()
                    elif location_box["x1"] <= x <= location_box["x2"] and \
                         location_box["y1"] <= y <= location_box["y2"]:
                        active_input = "location"
                        virtual_keyboard.set_text(location)
                        virtual_keyboard.activate()
                    elif start_btn["x1"] <= x <= start_btn["x2"] and \
                         start_btn["y1"] <= y <= start_btn["y2"]:
                        if inspector_name.strip() and location.strip():
                            session_id, inspection_time = create_session(inspector_name, location)
                            connect_frontend(frontend_client)
                            frontend_connected = True
                            capture_count = 0
                            session_captures = []
                            app_state = "inspecting"
                            status_message = "LIVE PREVIEW"
                            print(f"Inspection started: session={session_id}")

        elif app_state == "inspecting":
            frontend_connected = frontend_client.is_streaming()
            has_frame = frontend_client.get_frame() is not None

            if time.time() < flash_until and flash_frame is not None:
                preview = flash_frame
            else:
                frame = read_live_frame(fallback_frame, frontend_client, prefer_frontend=True)
                if frame is not None:
                    live_frame = frame
                preview = live_frame

            if has_frame:
                stream_status = "LIVE"
            elif frontend_connected:
                stream_status = "WAIT FRAME"
            else:
                stream_status = "CONNECTING..."

            err = frontend_client.get_error()
            if err and not has_frame:
                stream_status = "NO SIGNAL"

            extra = [
                (f"STREAM: {stream_status}", (0, 255, 0) if has_frame else (0, 140, 255)),
                (f"FRAMES: {frontend_client.get_frame_count()}", (255, 255, 255)),
                (f"CAPTURES: {capture_count}", (255, 255, 255)),
            ]
            draw_info_panel(display, inspector_name, location, inspection_time, extra)
            embed_preview(display, preview, top=470, max_h=680)
            draw_analysis_panel(display, flash_analysis, flash_conf, "LAST CAPTURE")

            buttons = [
                {"name": "CAPTURE", "x1": 80, "y1": 1650, "x2": 360, "y2": 1800, "color": (0, 180, 0)},
                {"name": "FINISH", "x1": 390, "y1": 1650, "x2": 690, "y2": 1800, "color": (0, 120, 220)},
                {"name": "EXIT", "x1": 720, "y1": 1650, "x2": 1000, "y2": 1800, "color": (80, 80, 80)},
            ]

            cv2.rectangle(display, (0, 1600), (SCREEN_W, SCREEN_H), (25, 25, 25), -1)
            for btn in buttons:
                draw_button(display, btn)

            if touch.touch_pressed and not last_touch_state:
                btn_name = hit_button(touch.touch_x, touch.touch_y, buttons)
                if btn_name == "CAPTURE" and time.time() >= flash_until:
                    source = live_frame.copy()
                    print("Capture requested...")
                    result = detector.detect(source)
                    if result is not None and len(result) == 4:
                        conf, bbox, mask, analysis = result
                        result_img = draw_detection(source, conf, bbox, mask, analysis)
                        flash_conf = conf
                        flash_analysis = analysis
                    else:
                        result_img = source.copy()
                        flash_conf = None
                        flash_analysis = None
                        conf, bbox, mask, analysis = None, None, None, None

                    flash_frame = result_img
                    flash_until = time.time() + RESULT_FLASH_SEC
                    persist_capture(session_id, source, result_img, conf, bbox, mask, analysis)
                    capture_count += 1
                    session_captures = get_session_captures(session_id)
                    status_message = "CAPTURE SAVED"
                elif btn_name == "FINISH":
                    end_session(session_id)
                    disconnect_frontend(frontend_client)
                    frontend_connected = False
                    session_captures = get_session_captures(session_id)
                    app_state = "finished"
                    status_message = f"FINISHED - {capture_count} CAPTURES"
                    print(f"Inspection finished: session={session_id}, captures={capture_count}")
                elif btn_name == "EXIT":
                    if session_id is not None:
                        end_session(session_id)
                    disconnect_frontend(frontend_client)
                    touch.stop()
                    detector.release()
                    break

        elif app_state == "finished":
            draw_info_panel(display, inspector_name, location, inspection_time, [
                (status_message, (0, 255, 255)),
                ("INSPECTION COMPLETED", (255, 255, 255)),
            ])

            buttons = [
                {"name": "RESULTS", "x1": 60, "y1": 900, "x2": 340, "y2": 1040, "color": (120, 60, 180)},
                {"name": "UPLOAD", "x1": 380, "y1": 900, "x2": 700, "y2": 1040, "color": (0, 140, 200)},
                {"name": "NEW", "x1": 740, "y1": 900, "x2": 1020, "y2": 1040, "color": (0, 160, 60)},
                {"name": "EXIT", "x1": 340, "y1": 1650, "x2": 740, "y2": 1800, "color": (80, 80, 80)},
            ]

            cv2.putText(display, f"TOTAL CAPTURES: {len(session_captures)}", (260, 520),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (255, 255, 255), 2)
            if status_message.startswith("UPLOAD"):
                cv2.putText(display, status_message[:60], (120, 620),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            for btn in buttons:
                draw_button(display, btn)

            if touch.touch_pressed and not last_touch_state:
                btn_name = hit_button(touch.touch_x, touch.touch_y, buttons)
                if btn_name == "RESULTS":
                    selected_capture_idx = 0
                    app_state = "results"
                elif btn_name == "UPLOAD":
                    pending_count = len([c for c in session_captures if not c.get("uploaded")])
                    saved_count, messages, resp = upload_session_captures(session_id)
                    if resp and resp.get("code") == 200:
                        status_message = f"UPLOAD OK: {resp.get('saved_count', saved_count)}/{pending_count}"
                    else:
                        err = messages[0] if messages else "UPLOAD FAIL"
                        status_message = f"UPLOAD FAIL: {err[:40]}"
                    for line in messages:
                        print(line)
                    session_captures = get_session_captures(session_id)
                elif btn_name == "NEW":
                    app_state = "input"
                    session_id = None
                    inspector_name = ""
                    location = ""
                    capture_count = 0
                    session_captures = []
                    status_message = ""
                elif btn_name == "EXIT":
                    touch.stop()
                    detector.release()
                    break

        elif app_state == "results":
            draw_info_panel(display, inspector_name, location, inspection_time, [
                (f"RESULT {selected_capture_idx + 1}/{max(len(session_captures), 1)}", (0, 255, 255)),
            ])

            if session_captures:
                capture = session_captures[selected_capture_idx]
                img = cv2.imread(capture["result_path"])
                embed_preview(display, img, top=430, max_h=620)

                analysis = {
                    "area": capture.get("crack_area", 0),
                    "length": capture.get("total_length", 0),
                    "width": capture.get("max_width", 0),
                    "severity": capture.get("severity", "NONE"),
                    "suggestion": capture.get("suggestion", ""),
                }
                draw_analysis_panel(display, analysis, capture.get("confidence"), "CAPTURE DETAIL")
                cv2.putText(display, f"TIME: {capture.get('captured_at', '')}", (60, 1540),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
                upload_tag = "UPLOADED" if capture.get("uploaded") else "NOT UPLOADED"
                upload_color = (0, 255, 0) if capture.get("uploaded") else (0, 140, 255)
                cv2.putText(display, upload_tag, (700, 1540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, upload_color, 2)
            else:
                cv2.putText(display, "NO CAPTURES", (380, 700), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)

            buttons = [
                {"name": "PREV", "x1": 60, "y1": 1650, "x2": 300, "y2": 1800, "color": (70, 70, 90)},
                {"name": "NEXT", "x1": 330, "y1": 1650, "x2": 570, "y2": 1800, "color": (70, 70, 90)},
                {"name": "BACK", "x1": 600, "y1": 1650, "x2": 1020, "y2": 1800, "color": (80, 80, 80)},
            ]
            cv2.rectangle(display, (0, 1600), (SCREEN_W, SCREEN_H), (25, 25, 25), -1)
            for btn in buttons:
                draw_button(display, btn)

            if touch.touch_pressed and not last_touch_state:
                btn_name = hit_button(touch.touch_x, touch.touch_y, buttons)
                if btn_name == "PREV" and session_captures:
                    selected_capture_idx = (selected_capture_idx - 1) % len(session_captures)
                elif btn_name == "NEXT" and session_captures:
                    selected_capture_idx = (selected_capture_idx + 1) % len(session_captures)
                elif btn_name == "BACK":
                    app_state = "finished"

        last_touch_state = touch.touch_pressed
        cv2.imshow("CRACK", display)

        key = cv2.waitKey(30 if app_state == "inspecting" else 50) & 0xFF
        if key == 27:
            if app_state == "results":
                app_state = "finished"
            elif app_state == "inspecting" and session_id is not None:
                end_session(session_id)
                disconnect_frontend(frontend_client)
            touch.stop()
            detector.release()
            break

    cv2.destroyAllWindows()
    print("System closed")

if __name__ == '__main__':
    main()