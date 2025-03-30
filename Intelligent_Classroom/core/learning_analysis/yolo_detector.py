from ultralytics import YOLO
import cv2
import time
from config.settings import Settings
from config.constants import LearningState


class LearningStateDetector:
    def __init__(self):
        self.model = YOLO(Settings.YOLO_MODEL)
        self.state_mapping = {
            0: LearningState.NORMAL,
            1: LearningState.SLEEPING,
            2: LearningState.RAISING_HAND,
            3: LearningState.PHONE,
            4: LearningState.NOTE_TAKING
        }

    def detect_states(self, timeout=300):
        """实时检测学习状态"""
        cap = cv2.VideoCapture(0)
        start_time = time.time()
        state_counts = {state: 0 for state in LearningState.__dict__.values()}

        while time.time() - start_time < timeout:
            ret, frame = cap.read()
            results = self.model(frame)[0]

            for box in results.boxes:
                class_id = int(box.cls)
                state = self.state_mapping.get(class_id, LearningState.NORMAL)
                state_counts[state] += 1
                cv2.putText(frame, state, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            cv2.imshow("Learning State Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        return state_counts