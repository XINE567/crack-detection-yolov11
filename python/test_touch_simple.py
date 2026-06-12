import struct
import threading
import time

EV_ABS = 0x03
EV_KEY = 0x01
ABS_X = 0x00
ABS_Y = 0x01
ABS_MT_POSITION_X = 0x35
ABS_MT_POSITION_Y = 0x36
BTN_TOUCH = 0x14a
BTN_TOOL_FINGER = 0x145

touch_x = 0
touch_y = 0
touch_pressed = False
running = True

def read_touch():
    global touch_x, touch_y, touch_pressed
    try:
        with open('/dev/input/event2', 'rb') as f:
            print('Touch device opened!')
            while running:
                data = f.read(24)
                if len(data) == 24:
                    _, _, typ, code, value = struct.unpack('llHHI', data)
                    
                    if typ == EV_ABS:
                        if code == ABS_X or code == ABS_MT_POSITION_X:
                            touch_x = value
                        elif code == ABS_Y or code == ABS_MT_POSITION_Y:
                            touch_y = value
                    elif typ == EV_KEY:
                        if code == BTN_TOUCH or code == BTN_TOOL_FINGER:
                            touch_pressed = (value == 1)
                            
    except Exception as e:
        print(f'Error: {e}')

print("Starting touch test...")
t = threading.Thread(target=read_touch)
t.daemon = True
t.start()

print("Listening for touch events (press Ctrl+C to exit)...")
last_pressed = False

try:
    while True:
        if touch_pressed and not last_pressed:
            print(f"Touch detected: ({touch_x}, {touch_y})")
            last_pressed = True
        elif not touch_pressed:
            last_pressed = False
        
        time.sleep(0.1)
except KeyboardInterrupt:
    running = False
    print("Exiting...")
