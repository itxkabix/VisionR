# backend/app/models/session.py
from __future__ import annotations

from datetime import datetime
from typing import List, Optional
import uuid

from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import func, Index

from .base import Base
from .user import User  # for type reference


class Session(Base):
    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
    )

    session_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    session_end: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    total_stress_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    dominant_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    burnout_risk_level: Mapped[Optional[str]] = mapped_column(
        String(20), nullable=True
    )
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    user: Mapped[User] = relationship(back_populates="sessions")
    messages: Mapped[List["Message"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )
    emotion_logs: Mapped[List["EmotionLog"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Session id={self.id} user_id={self.user_id}>"


Index(
    "idx_sessions_user_date",
    Session.user_id,
    Session.session_start.desc(),
)
