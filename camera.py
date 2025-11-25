import cv2
import base64
import enhanced_mask_detector as emd

class Camera:
    def __init__(self):
        self.cap = self.get_working_camera()
        self.detector = emd.main(self.cap)
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
        self.cap.release()
        cv2.destroyAllWindows()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        self.last_frame = frame
        return frame

    @staticmethod
    def encode_frame(frame, dsizex=640, dsizey=360, ext='.jpg'):
        frame = cv2.resize(frame, (dsizex, dsizey))
        ret, img = cv2.imencode(ext, frame)
        return img

    def frame_to_base64(self, frame):
        return base64.b64encode(self.encode_frame(frame)).decode('utf-8')

    def frame_to_bytes(self, frame):
        return self.encode_frame(frame).tobytes()

    def frame_to_webformat(self, frame):
        return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + self.frame_to_bytes(frame) + b'\r\n\r\n')

    def process_frame(self, frame):
        return self.detector(frame)

    def get_video_stream(self):
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield self.get_frame()

    def video_to_webformat(self):
        for i in self.get_video_stream():
            yield self.frame_to_webformat(self.process_frame(i))

if __name__ == "__main__":
    cam = Camera()

    for i in cam.get_video_stream():
        cv2.imshow("Camera: ", i)

    
