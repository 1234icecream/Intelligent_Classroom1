import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings:
    # 路径配置
    FACE_DATA_DIR = os.path.join(BASE_DIR, "data/face_data")
    CLASS_PHOTOS_DIR = os.path.join(BASE_DIR, "data/class_photos")
    ATTENDANCE_DIR = os.path.join(BASE_DIR, "data/attendance_records")
    CAPTCHA_DIR = os.path.join(BASE_DIR, "temp/captcha")

    # 模型路径配置（关键修复）
    DLIB_SHAPE_PREDICTOR = os.path.join(BASE_DIR, "models/dlib/shape_predictor_68_face_landmarks.dat")
    DLIB_FACE_RECOGNITION_MODEL = os.path.join(BASE_DIR,
                                               "models/dlib/dlib_face_recognition_resnet_model_v1.dat")  # 注意属性名一致性
    YOLO_MODEL = os.path.join(BASE_DIR, "models/yolo/learn.pt")

    # 系统参数
    MAX_STUDENTS = 30
    CAMERA_INDEX = 1