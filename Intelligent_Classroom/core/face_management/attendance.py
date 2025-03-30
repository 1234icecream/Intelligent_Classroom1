import os
import csv
import time
import cv2
import numpy as np
import dlib
from config.settings import Settings
from config.constants import AttendanceStatus


class FaceAttendance:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        self.sp = dlib.shape_predictor(Settings.DLIB_SHAPE_PREDICTOR)
        self.facerec = dlib.face_recognition_model_v1(Settings.DLIB_FACE_RECOGNITION)
        self.known_encodings, self.known_names = self._load_known_faces()

    def _load_known_faces(self):
        encodings, names = [], []
        for class_name in os.listdir(Settings.FACE_DATA_DIR):
            class_path = os.path.join(Settings.FACE_DATA_DIR, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith('.csv'):
                    name = filename.split('_')[0]
                    with open(os.path.join(class_path, filename), 'r') as f:
                        encoding = list(map(float, next(csv.reader(f))))
                    encodings.append(encoding)
                    names.append(name)
        return encodings, names

    def take_attendance(self, timeout=30):
        """执行考勤并返回识别结果"""
        cap = cv2.VideoCapture(0)
        recognized = set()
        start_time = time.time()

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.detector(rgb_frame)

            for face in faces:
                shape = self.sp(rgb_frame, face)
                encoding = self.facerec.compute_face_descriptor(rgb_frame, shape)
                distances = np.linalg.norm(self.known_encodings - np.array(encoding), axis=1)
                min_dist_idx = np.argmin(distances)

                if distances[min_dist_idx] < 0.6:  # 阈值设为0.6
                    recognized.add(self.known_names[min_dist_idx])

            cv2.imshow("Attendance Taking (Press 'q' to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return recognized
