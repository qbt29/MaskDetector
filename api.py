import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from videoProcessor import FrameProcessor
from cameraProcessor import CameraProcessor
import enhanced_mask_detector as EMD
import cv2
import base64
import numpy as np
import time
import os
router = fastapi.APIRouter()
router.prefix='/api'

FP = FrameProcessor()
CP = CameraProcessor(url=None, writer=None, reader=None, detector=EMD.MaskDetector().detect)
# CP = CameraProcessor(url=None, writer=None, reader=None, detector=None)

global cameras
cameras: dict = {}

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
    if 'screenshots' not in os.listdir():
        os.mkdir('screenshots')
    name = f'screenshots/mask_screenshot_{cam_id}_{int(time.time())}.jpg'
    cv2.imwrite(name, readb64(cameras.get(cam_id, {}).get('last_frame64', '')))
    return FileResponse(path=name, media_type='image/jpeg', filename=name)

@router.get('/video_feed/{cam_id}')
async def video_feed(cam_id: int):
    global cameras
    if cam_id in cameras.keys():
        return StreamingResponse(last_frame_streaming(cam_id), media_type="multipart/x-mixed-replace;boundary=frame")
    return None

@router.post('/new_frame/{cam_id}')
async def new_frame(request: Request, cam_id: int):
    global cameras
    if cam_id < 0:
        return {'success': False, 'reason':'Camera ID is negative'}
    if cam_id not in cameras.keys():
        # cameras[cam_id] = {'EMD': None}
        # cameras[cam_id] = {'CP':CameraProcessor(url=None, writer=None, reader=None, detector=None)}
        cameras[cam_id] = {'EMD': EMD.MaskDetector()}
        cameras[cam_id] = {'CP':CameraProcessor(url=None, writer=None, reader=None, detector=cameras[cam_id]['EMD'].detector)}
    ip = request.client.host
    last_ip = cameras[cam_id].get('ip', 0)
    last_ts = cameras[cam_id].get('ts', 0)
    prev = cameras[cam_id].get('last_frame64', "")
    prev_frame = cameras[cam_id].get('last_frame', "")
    if ip != last_ip and int(time.time()) - last_ts <= 600:
        return {'success': False, 'reason': 'This cameraID is already in use'}
    try:
        cameras[cam_id]['last_frame64'] = (await request.json())['image']
        cameras[cam_id]['last_frame'] = FP.frame_to_webformat(CP.process_frame(readb64(cameras.get(cam_id, {}).get('last_frame64', ''))))
        cameras[cam_id]['ip'] = ip
        cameras[cam_id]['ts'] = int(time.time())
        return {'success': True}
    except Exception as e:
        print(e)
        cameras[cam_id]['last_frame64'] = prev
        cameras[cam_id]['last_frame'] = prev_frame
        return {'success': False, 'reason': 'Internal Server Error'}

@router.get('/current_frame64/{cam_id}')
async def current_frame64(cam_id: int):
    global cameras
    return {"image": f"data:image/jpeg;base64,{cameras.get(cam_id, {}).get('last_frame64', '')}"}

@router.get('/get_detections/{cam_id}')
async def get_detections(request: Request, cam_id: int):
    r = await request.json()
    return {'offset': r.get('offset', 0)-1, 'limit': r.get('limit', 10)+1}

@router.get('/reset_tracks/{cam_id}')
async def reset_tracks(request: Request, cam_id: int):
    try:
        if cam_id == -1:
            for cam in cameras.keys():
                if cameras[cam]['EMD'] is None:
                    continue
                cameras[cam]['EMD'].reset_trackers()
        else:
            if cameras[cam_id]['EMD'] is not None:
                cameras[cam_id]['EMD'].reset_trackers()
        return {'success': True}
    except:
        return {'success': False, 'reason': 'Internal Server Error'}

@router.get('/list_cameras')
async def list_cameras(request: Request):
    global cameras
    arr = [{'id': id, 'isAlive': 0 if int(time.time())-cameras[id].get('ts', 0) > 600 else 1 if int(time.time())-cameras[id].get('ts', 0) > 60 else 2} for id in sorted(cameras.keys())]
    return {'cameras': arr}