import cv2
import base64
import time
from enhanced_mask_detector import MaskDetector


class Camera:
    def __init__(self, camera_index=None):
        self.current_index = camera_index
        self.cap = self._open_camera(self.current_index)
        if self.cap is None:
            raise RuntimeError(" Не удалось открыть камеру ни одним backend'ом")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.detector = MaskDetector(alpha=0.8)
        self.last_frame = None

    @staticmethod
    def list_cameras(max_id=3):
        available = []
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
        ]
        for i in range(max_id):
            for backend, name in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        available.append((i, w, h, name))
                        cap.release()
                        break
                    cap.release()
        return available

    def _open_camera(self, camera_index=None):
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
        indices = [camera_index] if camera_index is not None else [0, 1, 2]
        
        for i in indices:
            for backend, name in backends:
                cap = cv2.VideoCapture(i, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret and frame is not None and frame.size > 0:
                        print(f" Камера {i} открыта (backend: {name})")
                        self.current_index = i
                        return cap
                    cap.release()
        return None

    def switch_camera(self, new_index):
        if self.cap:
            self.cap.release()
        self.cap = self._open_camera(new_index)
        return self.cap is not None

    def close(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        """Возвращает ОБРАБОТАННЫЙ кадр и сохраняет его в last_frame"""
        if not self.cap or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None or frame.size == 0:
            return None
        try:
            processed = self.detector(frame)
            self.last_frame = processed.copy()
            return processed
        except Exception as e:
            print(f" Ошибка обработки: {e}")
            self.last_frame = frame.copy()
            return frame

    @staticmethod
    def encode_frame(frame, dsizex=640, dsizey=360, ext='.jpg'):
        if frame is None or frame.size == 0:
            return b""
        try:
            resized = cv2.resize(frame, (dsizex, dsizey))
            ret, buf = cv2.imencode(ext, resized, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            return buf.tobytes() if ret else b""
        except Exception:
            return b""

    def frame_to_bytes(self, frame):
        return self.encode_frame(frame)

    def frame_to_base64(self, frame):
        if frame is None:
            return ""
        try:
            return base64.b64encode(self.encode_frame(frame)).decode('utf-8')
        except Exception:
            return ""