import cv2
import numpy as np

print("Test started")

screen_w = 1080
screen_h = 1920

display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

cv2.putText(display, "TEST", (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 4)
cv2.putText(display, "Inspector:", (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(display, "Location:", (100, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
cv2.putText(display, "Status: Ready", (100, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

cv2.rectangle(display, (100, 600), (500, 750), (0, 255, 0), -1)
cv2.putText(display, "START", (200, 700), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

cv2.namedWindow("TEST", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("TEST", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

print("Showing window")
cv2.imshow("TEST", display)
cv2.waitKey(0)
cv2.destroyAllWindows()
print("Test done")