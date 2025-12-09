import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from videoProcessor import FrameProcessor
from cameraProcessor import CameraProcessor
import cv2
import json
import base64
import numpy as np
import time

router = fastapi.APIRouter()
router.prefix='/api'

FP = FrameProcessor()
CP = CameraProcessor(url=None, writer=None, reader=None)

global last_frames, last_frames64
last_frames = {}
last_frames64 = {}

def last_frame_streaming(cam_id):
    global last_frames
    while True:
        yield last_frames.get(cam_id, b'')
        time.sleep(0.1)

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


@router.get('/current_frame/{cam_id}')
async def current_frame(cam_id: int):
    global last_frames
    return Response(content=last_frames.get(cam_id, b''), media_type='image/jpeg')

@router.get('/video_feed/{cam_id}')
async def video_feed(cam_id: int):
    return StreamingResponse(last_frame_streaming(cam_id), media_type="multipart/x-mixed-replace;boundary=frame")

@router.post('/new_frame/{cam_id}')
async def new_frame(request: Request, cam_id: int):
    global last_frames64, last_frames
    prev = last_frames64.get(cam_id, "")
    prev_frame = last_frames.get(cam_id, b"")
    try:
        last_frames64[cam_id] = (await request.json())['image']
        last_frames[cam_id] = FP.frame_to_webformat(CP.process_frame(readb64(last_frames64.get(cam_id, ""))))
        return {'success': True}
    except Exception as e:
        print(e)
        last_frames64[cam_id] = prev
        last_frames[cam_id] = prev_frame
        return {'success': False}

@router.get('/current_frame64/{cam_id}')
async def current_frame64(cam_id):
    global last_frames64
    return {"image": f"data:image/jpeg;base64,{last_frames64.get(cam_id, '')}"}

@router.get('/get_detections')
async def get_detections(request: Request):
    r = await request.json()
    return {'offset': r.get('offset', 0)-1, 'limit': r.get('limit', 10)+1}