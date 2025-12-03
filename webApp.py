from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi import Request
from api import router
import uvicorn
# import camera

app = FastAPI()
templates = Jinja2Templates(directory="templates")
# cam = camera.Camera()
@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", context={"request": request})

@app.get("/next_page")
async def next_page():
    return "This is next page"

# @app.get('/video_feed')
# async def video_feed(request: Request):
#     return StreamingResponse(cam.process_video(), media_type="multipart/x-mixed-replace;boundary=frame")

app.include_router(router)
uvicorn.run(app, host="0.0.0.0", port=8000)
