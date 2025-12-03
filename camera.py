import cv2
import base64

class Reader:
    def __init__(self):
        self.cap = self.get_working_camera()
        if self.cap is None:
            print("No working camera found, exiting")
            exit(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        self.last_frame = None
    
    @staticmethod
    def get_working_camera(a = 0, b=1000):
        for i in range(a, b):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print("Working camera offset:",i)
                return cap
        return None

    def __del__(self):
        print("releasing camera")
        if self.cap is not None:
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
        self.out = out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'XVID'), fps, format)

    def write(self, frame):
        self.out.write(frame)

    
