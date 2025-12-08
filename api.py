import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from videoProcessor import FrameProcessor
import cv2
import json
import base64
import numpy as np
import time

router = fastapi.APIRouter()
router.prefix='/api'

FP = FrameProcessor()

global last_frame, last_frame64, changed
last_frame = []
last_frame64 = []
changed = False

def last_frame_streaming():
    global last_frame, last_frame64, changed
    while True:
        if changed:
            changed = False
            last_frame = FP.frame_to_webformat(readb64(last_frame64))
        yield last_frame
        time.sleep(0.1)

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


@router.get('/current_frame')
async def current_frame():
    return Response(content=last_frame, media_type='image/jpeg')

@router.get('/video_feed')
async def video_feed():
    return StreamingResponse(last_frame_streaming(), media_type="multipart/x-mixed-replace;boundary=frame")

@router.post('/new_frame')
async def new_frame(request: Request):
    global last_frame64, changed
    prev = last_frame64
    try:
        last_frame64 = (await request.json())['image']
        changed = True
        return {'status': True}
    except Exception as e:
        print(e)
        last_frame64 = prev
        return {'status': False}

@router.get('/current_frame64')
async def current_frame64():
    global last_frame64
    return {"image": f"data:image/jpeg;base64,{last_frame64}"}


@router.get('/get_detections')
async def get_detections(request: Request):
    r = await request.json()
    # print(offset, limit)
    return {'offset': r.get('offset', 0)-1, 'limit': r.get('limit', 10)+1}