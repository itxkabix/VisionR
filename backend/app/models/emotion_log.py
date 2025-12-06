# backend/app/models/emotion_log.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import func, Index

from .base import Base
from .session import Session  # type reference


class EmotionLog(Base):
    __tablename__ = "emotion_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        server_default=func.gen_random_uuid(),
    )

    session_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sessions.id", ondelete="CASCADE"),
        nullable=False,
    )

    timestamp: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Face
    face_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    face_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    face_arousal: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    face_valence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    face_features: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Voice
    voice_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    voice_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voice_stress: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    voice_tone: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    voice_features: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Text
    text_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    text_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    text_sentiment: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    text_keywords: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    text_risk_level: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)

    # Fusion
    final_emotion: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    final_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    stress_score: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    hidden_stress_detected: Mapped[Optional[bool]] = mapped_column(Boolean, nullable=True)

    # Relationship
    session: Mapped[Session] = relationship(back_populates="emotion_logs")

    def __repr__(self) -> str:
        return f"<EmotionLog id={self.id} session_id={self.session_id} timestamp={self.timestamp}>"

# Indexes
Index(
    "idx_emotion_logs_session_time",
    EmotionLog.session_id,
    EmotionLog.timestamp.desc(),
)
Index(
    "idx_emotion_logs_session",
    EmotionLog.session_id,
)
