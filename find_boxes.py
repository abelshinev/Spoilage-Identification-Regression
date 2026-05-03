import cv2
import numpy as np

img = cv2.imread('spoilage_t0.jpeg')
if img is None:
    print("Could not read image")
    exit(1)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

boxes = []
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    area = w * h
    if area > 4000: # Filter out small noise
        boxes.append({"x1": x, "y1": y, "x2": x+w, "y2": y+h, "x": x, "w": w, "h": h})

boxes.sort(key=lambda b: b['x'])

for i, b in enumerate(boxes):
    roi = img[b['y1']:b['y2'], b['x1']:b['x2']]
    avg_color = np.mean(roi, axis=(0,1))
    print(f"Blob {i}: x1={b['x1']}, y1={b['y1']}, x2={b['x2']}, y2={b['y2']}, w={b['w']}, h={b['h']}, color_bgr={avg_color}")
