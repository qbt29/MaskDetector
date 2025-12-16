import camera
import videoProcessor
import requests
import threading
import datetime
import time


class CameraProcessor():
    def __init__(self, url="http://127.0.0.1:8000/api/new_frame/2", filename=datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S.avi"), isWrite=False, isSend = False, reader = camera.Reader, writer = camera.Writer, detector=None):
        self.video_src = videoProcessor.VideoProcessor(reader()) if reader is not None else None
        self.writer = writer(path=filename) if writer is not None else None
        self.url = url
        self.detector = detector() if detector is not None else None
        self.isSenderOnline = False
        self.isWriterOnline = False
        self.isWrite = isWrite
        self.isSend = isSend
        self.writerQueue = []
        self.senderQueue = []
        self.filename = filename

    def send_new_frame(self):
        last_sleep = 1
        while len(self.senderQueue) > 0:
            print("Current sender queue size: ", len(self.senderQueue))
            frame, detections = self.senderQueue[0]
            try:
                r = requests.post(url=self.url, json={"image": f"data:image/jpeg;base64,{self.video_src.frame_to_base64(frame=frame)}", "detections": detections})
                if r.status_code == 200 and r.json()["success"]:
                    self.senderQueue = self.senderQueue[1:]
                    last_sleep = 1
            except:
                time.sleep(last_sleep)
                last_sleep *= 2
        self.isSenderOnline = False

    def process_frame(self, frame):
        if self.detector is not None:
            return self.detector.detect(frame)
        return frame, {'detections': {i : 0 for i in ['Mask OK', 'No Mask', 'Wrong Mask']}, 'quantity': 0}

    def save_frame(self):
        last_sleep = 1
        while len(self.writerQueue) > 0:
            if self.writer is not None:
                if self.writer.write(self.writerQueue[0]):
                    self.writerQueue.pop(0)
                    last_sleep = 1
                else:
                    time.sleep(last_sleep)
                    last_sleep *= 2
            else:
                self.writerQueue = []
        self.isWriterOnline = False

    def video_process(self):
        for i in self.video_src.get_video_stream():
            i = self.video_src.resize_frame(i, (640, 360))
            frame, detections = self.process_frame(i)
            if self.isWrite:
                self.writerQueue.append(frame)
            if self.isSend:
                self.senderQueue.append(frame)
            if not self.isWriterOnline and self.isWrite:
                self.isWriterOnline = True
                threading.Thread(target=self.save_frame, args=()).start()
            if not self.isSenderOnline and self.isSend:
                self.isWriterOnline = True
                threading.Thread(target=self.send_new_frame, args=()).start()

if __name__ == '__main__':
    CP = CameraProcessor(isSend=True, writer=None, url="http://127.0.0.1:8000/api/new_frame/10")
    CP.video_process()