# FILE: backend/app/schemas/auth.py
from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    """Incoming payload for user registration."""

    email: EmailStr = Field(..., description="Unique email for the user")
    username: str = Field(..., min_length=3, max_length=100)
    password: str = Field(..., min_length=6, max_length=128)
    privacy_consent: bool = Field(
        ..., description="Whether the user agrees to emotion data processing"
    )


class LoginRequest(BaseModel):
    """Incoming payload for user login."""

    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Basic user information returned to the client."""

    id: UUID
    email: EmailStr
    username: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    privacy_consent: bool
    created_at: datetime

    class Config:
        from_attributes = True
