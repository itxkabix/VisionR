# FILE: backend/app/schemas/chat.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, Field


class StartSessionRequest(BaseModel):
    """Payload to start a new chat session."""
    topic: Optional[str] = Field(
        default=None,
        description="Optional session topic, e.g., 'work stress' or 'exams'",
    )


class ChatMessageRequest(BaseModel):
    """Payload when user sends a chat message."""
    session_id: UUID
    content: str = Field(..., min_length=1, description="User's message text")


class ChatMessageResponse(BaseModel):
    """
    Response when a chat message is processed.
    Includes both stored message info and AI reply.
    """

    message_id: UUID
    ai_response: str
    emotion_label: Optional[str] = None
    stress_score: Optional[int] = Field(
        default=None,
        description="Optional stress score (0–100) estimated at this message",
    )
    created_at: datetime
