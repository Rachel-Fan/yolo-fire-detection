import cv2
from pathlib import Path

def test_camera():
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            print(f"No camera found at index {index}")
        else:
            print(f"Camera found at index {index}")
            cap.release()
            break
        index += 1
        cap.release()
        if index > 10:  # You can adjust this limit based on how many cameras you expect
            print("Tested up to index 10, no cameras found.")
            break

test_camera()
