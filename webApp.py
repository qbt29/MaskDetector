from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi import Request
from api import router
import uvicorn
import os, time

app = FastAPI()
templates = Jinja2Templates(directory="templates")

async def clear_old_screenshots():
    if 'screenshots' not in os.listdir():
        return
    for filename in os.listdir('screenshots'):
        if filename.endswith('.jpg'):
            if os.path.getctime(f'screenshots/{filename}') + 60 < time.time():
                os.remove(f'screenshots/{filename}')


@app.middleware("http")
async def middleware(request: Request, call_next):
    await clear_old_screenshots()
    response = await call_next(request)
    return response

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