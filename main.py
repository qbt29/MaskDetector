from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn
import os

app = FastAPI()

# Создаем папки если их нет
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
os.makedirs("screenshots", exist_ok=True)

# Настраиваем шаблоны
templates = Jinja2Templates(directory="templates")

# Монтируем статические файлы
app.mount("/static", StaticFiles(directory="static"), name="static")

# Импортируем роутер API - исправленный импорт
try:
    from api import router as api_router
    app.include_router(api_router)
    print("✓ API router подключен")
except ImportError as e:
    print(f"⚠️ Ошибка импорта API: {e}")
    # Создаем простой роутер для теста
    from fastapi import APIRouter
    api_router = APIRouter(prefix="/api")
    
    @api_router.get("/test")
    async def test():
        return {"status": "ok", "message": "API работает"}
    
    app.include_router(api_router)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Главная страница"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/test")
async def test():
    """Тестовый эндпоинт"""
    return {"status": "ok", "message": "Сервер работает!"}

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🎥 Mask Detection Web Server")
    print("="*50)
    print("Доступные URL:")
    print("  http://127.0.0.1:8000      - Веб-интерфейс")
    print("  http://127.0.0.1:8000/test - Тест API")
    print("  http://127.0.0.1:8000/api/test - Тест API роутера")
    print("\nЗапуск сервера...")
    print("Нажмите Ctrl+C для остановки\n")
    
    uvicorn.run(
        app, 
        host="127.0.0.1", 
        port=8000,
        log_level="info",
        reload=True  # Автоматическая перезагрузка при изменениях
    )