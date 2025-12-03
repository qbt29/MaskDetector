# webApp.py
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
import camera
import api

# ─── ФИЛЬТР ДЛЯ ТИХИХ ЭНДПОИНТОВ ─────────────────────────────
class QuietFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        return not any(endpoint in msg for endpoint in [
            "/api/current_frame",
            "/api/status",
            "/favicon.ico"
        ])

logging.getLogger("uvicorn.access").addFilter(QuietFilter())
# ───────────────────────────────────────────────────────────

templates = Jinja2Templates(directory="templates")
logger = logging.getLogger("app")

@asynccontextmanager
async def lifespan(app: FastAPI):
    cam = None
    try:
        logger.info("📷 Инициализация камеры...")
        cam = camera.Camera()
        api.cam = cam
        logger.info("✅ Камера успешно инициализирована")
        yield
    except Exception as e:
        logger.error(f"❌ ОШИБКА ИНИЦИАЛИЗАЦИИ КАМЕРЫ: {e}")
        api.cam = None
        yield
    finally:
        if cam:
            logger.info("📷 Завершение работы камеры...")
            cam.close()
            logger.info("✅ Камера освобождена")

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

app.include_router(api.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "webApp:app",
        host="127.0.0.1",
        port=8000,
        log_level="info",  # или "warning" для ещё более чистого вывода
        reload=False
    )