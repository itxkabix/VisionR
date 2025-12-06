# backend/app/models/base.py
from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, Boolean, Float, Integer, String, Text, ForeignKey, func
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base declarative class for all models."""
    pass
