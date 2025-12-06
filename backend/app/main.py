from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.gzip import GZipMiddleware
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.dependencies import get_db
from app.db.database import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: load ML models later (face, voice, text)
    yield
    # TODO: cleanup resources here if needed


app = FastAPI(
    title="RIVION API",
    version="1.0.0",
    lifespan=lifespan,
)

# Ensure ORM models have tables (no-op if already created by schema.sql)
Base.metadata.create_all(bind=engine)

# ---------- Middleware ----------

origins = settings.cors_origins or ["http://localhost:5173"]

app.add_middleware(GZipMiddleware, minimum_size=1000)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Health Endpoints ----------


@app.get("/api/health")
async def health_check() -> dict:
    return {"status": "ok", "service": "rivion-backend"}


@app.get("/api/health/db")
def db_health_check(db: Session = Depends(get_db)) -> dict:
    db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "connected"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
