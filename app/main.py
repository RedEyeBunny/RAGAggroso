from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.routes import documents, qa, health
from app.database import Base, engine

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.mount("/static", StaticFiles(directory="app/static"), name="static")

templates = Jinja2Templates(directory="app/templates")

app.include_router(documents.router, prefix="/documents")
app.include_router(qa.router, prefix="/qa")
app.include_router(health.router)

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

from app.routes.health import health_check

@app.get("/status")
def status_page(request: Request):
    health = health_check()
    return templates.TemplateResponse("status.html", {
        "request": request,
        "status": health
    })
