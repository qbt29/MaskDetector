import fastapi
from fastapi import Request, Response
from fastapi.responses import StreamingResponse
import camera

router = fastapi.APIRouter()
router.prefix='/api'
cam = camera.Camera()



@router.get('/current_frame')
async def current_frame(request: Request):
    return Response(content=cam.frame_to_bytes(cam.last_frame), media_type='image/jpeg') if cam.last_frame is not None else None

@router.get('/video_feed')
async def video_feed(request: Request):
    return StreamingResponse(cam.video_to_webformat(), media_type="multipart/x-mixed-replace;boundary=frame")

@router.get('/current_frame64')
async def current_frame64(request: Request):
    return {"image": f"data:image/jpeg;base64,{cam.frame_to_base64(cam.last_frame)}"}
