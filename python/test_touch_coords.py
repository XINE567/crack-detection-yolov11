import struct
import threading
import cv2
import numpy as np

EV_ABS = 0x03
EV_KEY = 0x01
ABS_X = 0x00
ABS_Y = 0x01
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
BTN_TOUCH = 0x14a
BTN_TOOL_FINGER = 0x145

class TouchTester:
    def __init__(self):
        self.touch_x = 0
        self.touch_y = 0
        self.touch_pressed = False
        self.input_dev = None
        self.running = True
        self.min_x = 9999
        self.max_x = 0
        self.min_y = 9999
        self.max_y = 0
        
    def open(self):
        try:
            self.input_dev = open("/dev/input/event2", "rb")
            print("Opened touch device")
            return True
        except Exception as e:
            print(f"Failed: {e}")
            return False
    
    def _read_loop(self):
        while self.running and self.input_dev:
            try:
                data = self.input_dev.read(24)
                if len(data) == 24:
                    _, _, typ, code, value = struct.unpack('llHHI', data)
                    
                    if typ == EV_ABS:
                        if code == ABS_X or code == ABS_MT_POSITION_X:
                            self.touch_x = value
                            self.min_x = min(self.min_x, value)
                            self.max_x = max(self.max_x, value)
                        elif code == ABS_Y or code == ABS_MT_POSITION_Y:
                            self.touch_y = value
                            self.min_y = min(self.min_y, value)
                            self.max_y = max(self.max_y, value)
                    elif typ == EV_KEY:
                        if code == BTN_TOUCH or code == BTN_TOOL_FINGER:
                            self.touch_pressed = (value == 1)
                            if self.touch_pressed:
                                print(f"Touch: ({self.touch_x}, {self.touch_y})")
                                print(f"Range X: [{self.min_x}, {self.max_x}]")
                                print(f"Range Y: [{self.min_y}, {self.max_y}]")
            except:
                break
    
    def start(self):
        if self.input_dev:
            t = threading.Thread(target=self._read_loop)
            t.daemon = True
            t.start()
    
    def stop(self):
        self.running = False

def main():
    print("=== Touch Coordinate Tester ===")
    print("Touch the screen to see coordinates...")
    
    tester = TouchTester()
    if not tester.open():
        print("Failed to open touch device")
        return
    
    tester.start()
    
    SCREEN_W = 1080
    SCREEN_H = 1920
    
    cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("TEST", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    while True:
        display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        
        cv2.putText(display, "TOUCH COORDINATE TESTER", (250, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        cv2.putText(display, f"X: {tester.touch_x}", (100, 300), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
        cv2.putText(display, f"Y: {tester.touch_y}", (100, 400), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        
        cv2.putText(display, f"Range X: [{tester.min_x}, {tester.max_x}]", (100, 550), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
        cv2.putText(display, f"Range Y: [{tester.min_y}, {tester.max_y}]", (100, 620), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 1)
        
        cv2.rectangle(display, (80, 1150), (500, 1280), (0, 255, 0), -1)
        cv2.putText(display, "TEST AREA", (150, 1220), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        cv2.imshow("TEST", display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
    
    tester.stop()
    cv2.destroyAllWindows()
    print("Done")

if __name__ == '__main__':
    main()