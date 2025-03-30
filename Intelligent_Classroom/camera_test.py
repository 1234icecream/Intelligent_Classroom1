import cv2
from config.settings import Settings

def test_camera():
    print("可用的摄像头索引:")
    for i in range(3):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"索引 {i}: 可用 (分辨率: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)})")
            cap.release()
        else:
            print(f"索引 {i}: 不可用")

if __name__ == "__main__":
    test_camera()