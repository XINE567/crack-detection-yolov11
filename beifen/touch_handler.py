import struct
import threading

class TouchHandler:
    def __init__(self, device_path="/dev/input/event2"):
        self.touch_x = 0
        self.touch_y = 0
        self.touch_pressed = False
        self.input_dev = None
        self.running = True
        self.thread = None
        self.device_path = device_path
        
    def open(self):
        try:
            self.input_dev = open(self.device_path, "rb")
            print(f"Touch device opened: {self.device_path}")
            return True
        except Exception as e:
            print(f"Failed to open touch device: {e}")
            return False
    
    def _read_loop(self):
        EV_ABS = 0x03
        EV_KEY = 0x01
        ABS_X = 0x00
        ABS_Y = 0x01
        ABS_MT_POSITION_X = 0x35
        ABS_MT_POSITION_Y = 0x36
        BTN_TOUCH = 0x14a
        BTN_TOOL_FINGER = 0x145
        
        while self.running and self.input_dev:
            try:
                data = self.input_dev.read(24)
                if len(data) == 24:
                    _, _, typ, code, value = struct.unpack('llHHI', data)
                    
                    if typ == EV_ABS:
                        if code == ABS_X or code == ABS_MT_POSITION_X:
                            self.touch_x = value
                        elif code == ABS_Y or code == ABS_MT_POSITION_Y:
                            self.touch_y = value
                    elif typ == EV_KEY:
                        if code == BTN_TOUCH or code == BTN_TOOL_FINGER:
                            self.touch_pressed = (value == 1)
            except:
                break
    
    def start(self):
        if self.input_dev:
            self.thread = threading.Thread(target=self._read_loop)
            self.thread.daemon = True
            self.thread.start()
            print("Touch handler started")
    
    def stop(self):
        self.running = False
        if self.input_dev:
            self.input_dev.close()
    
    def get_position(self):
        return self.touch_x, self.touch_y
    
    def is_pressed(self):
        return self.touch_pressed
    
    def check_button_click(self, buttons):
        x, y = self.touch_x, self.touch_y
        for btn in buttons:
            if btn["x1"] <= x <= btn["x2"] and btn["y1"] <= y <= btn["y2"]:
                return btn["name"]
        return None

class Button:
    def __init__(self, name, x1, y1, x2, y2, color):
        self.name = name
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.color = color
    
    def to_dict(self):
        return {
            "name": self.name,
            "x1": self.x1,
            "y1": self.y1,
            "x2": self.x2,
            "y2": self.y2,
            "color": self.color
        }

def create_button(name, x1, y1, x2, y2, color):
    return Button(name, x1, y1, x2, y2, color)

def draw_button(img, button, text=None):
    x1, y1 = button.x1, button.y1
    x2, y2 = button.x2, button.y2
    color = button.color
    
    cv2 = __import__('cv2')
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    
    if text is None:
        text = button.name
    
    tx = x1 + (x2 - x1) // 2 - len(text) * 15
    ty = y1 + (y2 - y1) // 2 + 15
    cv2.putText(img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

def draw_buttons(img, buttons, text_map=None):
    for btn in buttons:
        text = text_map[btn.name] if text_map and btn.name in text_map else btn.name
        draw_button(img, btn, text)