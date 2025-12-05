# TODO: implement
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZIPMiddleware

from app.config import settings  # type: ignore
from app.middleware import ErrorHandlingMiddleware  # type: ignore
from app.routers import auth, chat, emotion, dashboard, health  # type: ignore
from app.db.database import Base, engine  # type: ignore


# Create DB tables on startup (for simple setups; in prod use Alembic)
Base.metadata.create_all(bind=engine)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context.

    Use this for:
    - Loading ML models into memory once
    - Initializing connection pools / caches
    - Cleaning up resources on shutdown
    """
    # TODO: initialize shared ML services, caches, etc.
    yield
    # TODO: gracefully close connections, flush logs, etc.


app = FastAPI(
    title="RIVION API",
    version="1.0.0",
    lifespan=lifespan,
)


# Middleware configuration
app.add_middleware(GZIPMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,  # e.g. ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ErrorHandlingMiddleware)


# Router registration
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])
app.include_router(emotion.router, prefix="/api/emotion", tags=["emotion"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])
app.include_router(health.router, prefix="/api", tags=["health"])


@app.get("/")
async def root() -> dict:
    """Simple root endpoint to verify API is running."""
    return {"message": "RIVION backend is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
