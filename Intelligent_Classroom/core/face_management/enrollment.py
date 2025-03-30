import os
import cv2
import dlib
import numpy as np
import logging
from datetime import datetime
from config.settings import Settings
from utils.file_utils import ensure_dir

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceEnrollment:
    """人脸信息录入核心类（稳定版）"""

    def __init__(self):
        """安全初始化模型和资源"""
        self.detector = None
        self.sp = None
        self.facerec = None
        self._init_models()

    def _init_models(self):
        """分步初始化模型，防止单点失败"""
        try:
            # 1. 验证模型文件存在
            self._validate_model_files()

            # 2. 初始化检测器（最可能失败的点）
            self.detector = dlib.get_frontal_face_detector()

            # 3. 初始化关键点检测器
            self.sp = dlib.shape_predictor(Settings.DLIB_SHAPE_PREDICTOR)

            # 4. 初始化识别模型
            self.facerec = dlib.face_recognition_model_v1(Settings.DLIB_FACE_RECOGNITION_MODEL)

            # 5. 创建目录
            ensure_dir(Settings.FACE_DATA_DIR)
            ensure_dir(Settings.CLASS_PHOTOS_DIR)

            logger.info("所有模型初始化成功")
        except Exception as e:
            logger.error(f"模型初始化失败: {str(e)}")
            self._release_resources()
            raise

    def _validate_model_files(self):
        """验证模型文件完整性"""
        required_files = {
            "人脸关键点检测模型": Settings.DLIB_SHAPE_PREDICTOR,
            "人脸识别模型": Settings.DLIB_FACE_RECOGNITION_MODEL
        }

        missing_files = []
        for name, path in required_files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")

        if missing_files:
            raise FileNotFoundError(f"缺少模型文件:\n" + "\n".join(missing_files))

    def enroll_student(self, student_name: str) -> bool:
        """
        安全的人脸录入流程
        :param student_name: 只允许字母数字和中文字符
        :return: 是否成功
        """
        if not self._validate_name(student_name):
            logger.error("无效的学生姓名格式")
            return False

        cap = None
        try:
            cap = self._init_camera()
            if not cap.isOpened():
                raise RuntimeError("摄像头初始化失败")

            while True:
                ret, frame = self._capture_frame(cap)
                if not ret:
                    continue

                # 人脸检测与交互
                rgb, faces = self._detect_faces(frame)
                if len(faces) != 1:
                    self._display_guidance(frame, faces)
                    continue

                if self._confirm_capture(frame):
                    return self._save_face_data(rgb, faces[0], student_name)

        except Exception as e:
            logger.error(f"录入过程出错: {str(e)}")
            return False
        finally:
            self._release_camera(cap)
            cv2.destroyAllWindows()

    def _init_camera(self):
        cap = cv2.VideoCapture(Settings.CAMERA_INDEX, cv2.CAP_DSHOW)  # 强制使用DirectShow
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)  # 限制帧率
        return cap
    def _capture_frame(self, cap):
        """捕获帧并进行基本验证"""
        ret, frame = cap.read()
        if not ret:
            logger.warning("帧捕获失败，尝试重新初始化摄像头...")
            cap.release()
            cap = cv2.VideoCapture(Settings.CAMERA_INDEX)
            return False, None
        return True, frame

    def _validate_name(self, name: str) -> bool:
        """严格的姓名验证"""
        if not name or len(name) > 30:
            return False
        return all(c.isalnum() or '\u4e00' <= c <= '\u9fff' for c in name)

    def _detect_faces(self, frame):
        """带降级策略的人脸检测"""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 策略1：灰度图像检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)

        # 策略2：如果失败尝试RGB
        if len(faces) == 0:
            faces = self.detector(rgb)

        return rgb, faces

    def _display_guidance(self, frame, faces):
        """可视化引导界面"""
        status = "未检测到人脸" if len(faces) == 0 else f"检测到 {len(faces)} 个人脸"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, "请保持单人正对摄像头", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, "按空格键捕获 | ESC退出", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        for face in faces:
            cv2.rectangle(frame, (face.left(), face.top()),
                          (face.right(), face.bottom()), (0, 0, 255), 2)

        cv2.imshow("人脸录入", frame)
        cv2.waitKey(1)

    def _confirm_capture(self, frame) -> bool:
        """带视觉反馈的捕获确认"""
        cv2.putText(frame, "确认捕获？按空格键确认", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "ESC重新检测", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("人脸录入", frame)

        while True:
            key = cv2.waitKey(1)
            if key == 32:  # 空格键
                return True
            elif key == 27:  # ESC键
                return False

    def _save_face_data(self, rgb, face, student_name) -> bool:
        """原子化保存操作"""
        try:
            # 1. 计算特征
            shape = self.sp(rgb, face)
            encoding = np.array(self.facerec.compute_face_descriptor(rgb, shape))

            # 2. 准备存储路径
            class_dir = self._get_class_dir()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs(class_dir, exist_ok=True)

            # 3. 保存特征值
            np.save(os.path.join(class_dir, f"{student_name}_{timestamp}.npy"), encoding)

            # 4. 保存照片
            photo_dir = os.path.join(Settings.CLASS_PHOTOS_DIR, os.path.basename(class_dir))
            os.makedirs(photo_dir, exist_ok=True)
            cv2.imwrite(os.path.join(photo_dir, f"{student_name}.jpg"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            logger.info(f"成功保存 {student_name} 的数据到 {class_dir}")
            return True

        except Exception as e:
            logger.error(f"数据保存失败: {str(e)}")
            # 清理可能产生的半成品文件
            if 'photo_dir' in locals():
                temp_file = os.path.join(photo_dir, f"{student_name}.jpg")
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            return False

    def _get_class_dir(self):
        """智能分配班级目录"""
        # 现有班级检查
        for class_id in os.listdir(Settings.FACE_DATA_DIR):
            class_dir = os.path.join(Settings.FACE_DATA_DIR, class_id)
            if len(os.listdir(class_dir)) < Settings.MAX_STUDENTS:
                return class_dir

        # 新建班级
        new_class = f"class_{len(os.listdir(Settings.FACE_DATA_DIR)) + 1}"
        new_dir = os.path.join(Settings.FACE_DATA_DIR, new_class)
        os.makedirs(new_dir)
        return new_dir

    def _release_camera(self, cap):
        """安全释放摄像头"""
        if cap and cap.isOpened():
            cap.release()
            logger.debug("摄像头资源已释放")

    def _release_resources(self):
        """清理所有模型资源"""
        self.detector = None
        self.sp = None
        self.facerec = None
        logger.debug("模型资源已释放")


# 测试用例
if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.DEBUG)
        logger.info("开始人脸录入测试...")

        enroll = FaceEnrollment()
        test_name = "test_学生_" + datetime.now().strftime("%H%M%S")

        if enroll.enroll_student(test_name):
            logger.info("测试成功！")
        else:
            logger.error("测试失败")

    except Exception as e:
        logger.critical(f"测试崩溃: {str(e)}", exc_info=True)