from camera import Reader, Writer
import cv2 
import base64

class FrameProcessor:
    def __init__(self):
        pass

    @staticmethod
    def resize_frame(frame, format=(640, 360)):
        frame = cv2.resize(frame, format)
        return frame

    def encode_frame(self, frame, dsizex=640, dsizey=360, ext='.jpg'):
        if frame is None or frame.size == 0:
            return []
        frame = self.resize_frame(frame, (dsizex, dsizey))
        ret, img = cv2.imencode(ext, frame)
        if not ret:
            raise Exception('Error with encoding frame')
        return img

    def frame_to_base64(self, frame):
        if frame is None:
            return ''
        try:
            return base64.b64encode(self.encode_frame(frame)).decode('utf-8')
        except:
            return ''

    def frame_to_bytes(self, frame):
        buf = self.encode_frame(frame)
        if len(buf) == 0:
            return b""
        return buf.tobytes()

    def frame_to_webformat(self, frame):
        return (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + self.frame_to_bytes(frame) + b'\r\n\r\n')



class VideoProcessor(FrameProcessor):
    def __init__(self, src):
        super().__init__()
        self.cam = src

    
    def get_video_stream(self):
        while True:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            yield self.cam.get_frame()

    def video_to_webformat(self):
        for i in self.get_video_stream():
            yield self.frame_to_webformat(self.process_frame(i))
