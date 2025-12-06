# backend/app/models/message.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
import uuid

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy import func

from .base import Base
from .session import Session  # type reference


class Message(Base):
    __tablename__ = "messages"

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

    sender: Mapped[str] = mapped_column(String(10), nullable=False)  # 'USER' or 'AI'
    content: Mapped[str] = mapped_column(Text, nullable=False)
    emotion_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    tokens_used: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    response_time_ms: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relationships
    session: Mapped[Session] = relationship(back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id} session_id={self.session_id} sender={self.sender}>"
