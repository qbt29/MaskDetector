import camera
import videoProcessor
import requests
import threading
import datetime
import enhanced_mask_detector as EMD


class CameraProcessor():
    def __init__(self, url="http://127.0.0.1:8000/api/new_frame"):
        self.video_src = videoProcessor.VideoProcessor(camera.Reader())
        self.writer = camera.Writer(path=datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S.avi"))
        self.url = url
        self.detector = EMD.main(self.video_src.cam)

    
    def send_new_frame(self, frame):
        r = requests.post(url=self.url, json={"image": f"data:image/jpeg;base64,{self.video_src.frame_to_base64(frame=frame)}"})
        print(r.content)

    def process_frame(self, frame):
        return self.detector(frame)

    def video_process(self):
        for i in self.video_src.get_video_stream():
            i = self.process_frame(i)
            threading.Thread(target=self.send_new_frame, args=(i,)).start()
            self.writer.write(self.video_src.resize_frame(i, (640, 360)))

if __name__ == '__main__':
    CP = CameraProcessor()
    CP.video_process()