import cv2
import base64


class Reader:
    def __init__(self, resolution=(1920, 1080)):
        self.cap = self.get_working_camera()
        if self.cap is None:
            raise RuntimeError("No working camera found, exiting")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self.last_frame = None

    @staticmethod
    def get_working_camera(a=0, b=1000):
        backends = [
            (cv2.CAP_DSHOW, "DSHOW"),
            (cv2.CAP_MSMF, "MSMF"),
            (cv2.CAP_ANY, "ANY"),
        ]
        for i in range(a, b):
            # for backend, name in backends:
            #     cap = cv2.VideoCapture(i, backend)
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    print(f"Working camera offset: {i} (backend: {None})")
                    return cap
                cap.release()
        return None

    def __del__(self):
        self.close()

    def close(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.last_frame = frame
        return frame


class Writer:
    def __init__(self, path="out.avi", format=(640, 360), fps=60):
        self.out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, format)

    def write(self, frame):
        try:
            self.out.write(frame)
            return True
        except Exception as e:
            raise e
        return False

