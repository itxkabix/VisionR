from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.config import settings


# ---------- SQLAlchemy Base ----------

class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# ---------- Engine & Session ----------

engine = create_engine(
    settings.database_url,
    echo=False,         # set True to see SQL in logs
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
)
