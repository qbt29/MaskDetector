import camera
import videoProcessor
import requests
import threading
import datetime
import time



class CameraProcessor():
    def __init__(self, url="http://127.0.0.1:8000/api/new_frame/1", filename=datetime.datetime.now().strftime("%Y.%m.%d %H:%M:%S.avi"), isWrite=False, isSend = False, reader = camera.Reader, writer = camera.Writer, detector=None):
        self.video_src = videoProcessor.VideoProcessor(reader)
        self.writer = writer(path=filename)
        self.url = url
        self.detector = detector
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
            frame = self.senderQueue[0]
            try:
                r = requests.post(url=self.url, json={"image": f"data:image/jpeg;base64,{self.video_src.frame_to_base64(frame=frame)}"})
                if r.status_code == 200 and r.json()["success"]:
                    self.senderQueue = self.senderQueue[1:]
                    last_sleep = 1
            except:
                time.sleep(last_sleep)
                last_sleep *= 2
        self.isSenderOnline = False

    def process_frame(self, frame):
        if self.detector is not None:
            return self.detector(frame)
        return frame

    def save_frame(self):
        last_sleep = 1
        while len(self.writerQueue) > 0:
            if self.writer.write(self.senderQueue[0]):
                self.senderQueue.pop(0)
                last_sleep = 1
            else:
                time.sleep(last_sleep)
                last_sleep *= 2
        self.isWriterOnline = False

    def video_process(self):
        for i in self.video_src.get_video_stream():
            # i = self.process_frame(i)
            if self.isWrite:
                self.writerQueue.append(self.video_src.resize_frame(i, (640, 360)))
            if self.isSend:
                self.senderQueue.append(i)
            if not self.isWriterOnline and self.isWrite:
                self.isWriterOnline = True
                threading.Thread(target=self.save_frame, args=()).start()
            if not self.isSenderOnline and self.isSend:
                self.isWriterOnline = True
                threading.Thread(target=self.send_new_frame, args=()).start()

if __name__ == '__main__':
    CP = CameraProcessor(isSend=True)
    CP.video_process()