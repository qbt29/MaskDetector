import fastapi
from fastapi import Response, Request
from fastapi.responses import JSONResponse

router = fastapi.APIRouter()
router.prefix = '/api'
cam = None 


@router.get('/current_frame')
async def current_frame():
    if cam is None:
        return Response(status_code=503)
    frame = cam.get_frame() 
    if frame is None:
        return Response(status_code=204)
    return Response(
        content=cam.frame_to_bytes(frame),
        media_type='image/jpeg',
        headers={"Cache-Control": "no-store, max-age=0"}
    )


@router.get('/current_frame64')
async def current_frame64():
    if cam is None or cam.last_frame is None:
        return {"image": ""}
    try:
        b64 = cam.frame_to_base64(cam.last_frame)
        return {"image": f"data:image/jpeg;base64,{b64}"}
    except Exception:
        return {"image": ""}


@router.get('/status')
async def status():
    return JSONResponse({
        "status": "ok" if cam else "error",
        "camera_index": getattr(cam, "current_index", None),
        "cameras_available": cam.list_cameras() if cam else []
    })


@router.post('/switch_camera')
async def switch_camera(request: Request):
    try:
        data = await request.json()
        idx = int(data.get("camera_id", 0))
        success = cam.switch_camera(idx) if cam else False
        return {"status": "success" if success else "error", "camera": idx}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@router.post('/reset_tracks')
async def reset_tracks():
    if cam and hasattr(cam, 'detector'):
        cam.detector.reset_tracks()
        return {"status": "success"}
    return {"status": "error", "message": "Tracks not available"}