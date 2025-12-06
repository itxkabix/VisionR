from collections.abc import Generator
from sqlalchemy.orm import Session

from app.db.database import SessionLocal


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a SQLAlchemy session and
    ensures it's closed after each request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
