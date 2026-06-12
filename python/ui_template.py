import cv2
import numpy as np
from touch_handler import TouchHandler, create_button

def main():
    print("=== RK3588 UI Template ===")
    
    SCREEN_W = 1080
    SCREEN_H = 1920
    
    cv2.namedWindow("UI", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("UI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    touch = TouchHandler()
    touch.open()
    touch.start()
    
    buttons = [
        create_button("START", 80, 1150, 500, 1280, (0, 255, 0)),
        create_button("SAVE", 560, 1150, 1000, 1280, (255, 165, 0)),
        create_button("EXIT", 560, 1330, 1000, 1460, (60, 60, 60)),
    ]
    
    last_pressed = False
    running = True
    
    while running:
        display = np.zeros((SCREEN_H, SCREEN_W, 3), dtype=np.uint8)
        
        cv2.putText(display, "YOUR UI HERE", (350, 500), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        for btn in buttons:
            cv2.rectangle(display, (btn.x1, btn.y1), (btn.x2, btn.y2), btn.color, -1)
            tx = btn.x1 + (btn.x2 - btn.x1) // 2 - len(btn.name) * 15
            ty = btn.y1 + (btn.y2 - btn.y1) // 2 + 15
            cv2.putText(display, btn.name, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        cv2.imshow("UI", display)
        
        if touch.is_pressed() and not last_pressed:
            x, y = touch.get_position()
            print(f"Touch: ({x}, {y})")
            
            btn_name = touch.check_button_click(buttons)
            if btn_name:
                print(f"Button: {btn_name}")
                
                if btn_name == "START":
                    print("Do something...")
                elif btn_name == "SAVE":
                    print("Save something...")
                elif btn_name == "EXIT":
                    running = False
        
        last_pressed = touch.is_pressed()
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            running = False
    
    touch.stop()
    cv2.destroyAllWindows()
    print("Done")

if __name__ == '__main__':
    main()