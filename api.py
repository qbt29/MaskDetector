import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse
from videoProcessor import FrameProcessor
from cameraProcessor import CameraProcessor
# import enhanced_mask_detector as EMD
import cv2
import json
import base64
import numpy as np
import time

router = fastapi.APIRouter()
router.prefix='/api'

FP = FrameProcessor()
# CP = CameraProcessor(url=None, writer=None, reader=None, detector=EMD.MaskDetector().detect)
CP = CameraProcessor(url=None, writer=None, reader=None, detector=None)

global last_frames, last_frames64, cameras
cameras = {}
last_frames = {}
last_frames64 = {}

def last_frame_streaming(cam_id):
    global cameras
    while True:
        yield cameras.get(cam_id, {}).get('last_frame', b'')
        time.sleep(0.1)

def readb64(uri):
   encoded_data = uri.split(',')[1]
   nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
   img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
   return img


@router.get('/current_frame/{cam_id}')
async def current_frame(cam_id: int):
    global cameras
    return Response(content=cameras.get(cam_id, {}).get('last_frame', b''), media_type='image/jpeg')

@router.get('/video_feed/{cam_id}')
async def video_feed(cam_id: int):
    global cameras
    if cam_id in cameras.keys():
        return StreamingResponse(last_frame_streaming(cam_id), media_type="multipart/x-mixed-replace;boundary=frame")
    return None

@router.post('/new_frame/{cam_id}')
async def new_frame(request: Request, cam_id: int):
    global cameras
    prev = cameras.get(cam_id, {}).get('last_frame64', "")
    prev_frame = cameras.get(cam_id, {}).get('last_frame', "")
    try:
        cameras[cam_id]['last_frame64'] = (await request.json())['image']
        cameras[cam_id]['last_frame'] = FP.frame_to_webformat(CP.process_frame(readb64(cameras.get(cam_id, {}).get('last_frame64', ''))))
        return {'success': True}
    except Exception as e:
        print(e)
        cameras[cam_id]['last_frame64'] = prev
        cameras[cam_id]['last_frame'] = prev_frame
        return {'success': False}

@router.get('/current_frame64/{cam_id}')
async def current_frame64(cam_id):
    global cameras
    return {"image": f"data:image/jpeg;base64,{cameras.get(cam_id, {}).get('last_frame64', '')}"}

@router.get('/get_detections')
async def get_detections(request: Request):
    r = await request.json()
    return {'offset': r.get('offset', 0)-1, 'limit': r.get('limit', 10)+1}

@router.get('/list_cameras')
async def list_cameras(request: Request):
    global cameras
    arr = [{'id': id, 'ts': cameras[id].get('ts', int(time.time()))} for id in cameras.keys()]
    return {'cameras': arr}