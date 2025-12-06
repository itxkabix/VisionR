# FILE: backend/app/schemas/emotion.py
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# --- Face Emotion ---


class FaceEmotionRequest(BaseModel):
    """Request to analyze a single video frame."""
    frame_base64: str = Field(
        ..., description="Base64-encoded image frame captured from webcam"
    )


class FaceEmotionResponse(BaseModel):
    """Face emotion analysis result."""

    emotion: str = Field(..., description="Predicted facial emotion label")
    confidence: float = Field(..., ge=0.0, le=1.0)
    arousal: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Approximate arousal level from facial cues",
    )
    valence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Approximate valence from facial cues",
    )
    landmarks: Optional[List[List[float]]] = Field(
        default=None,
        description="Optional simplified facial landmarks (x,y) list",
    )


# --- Voice Emotion ---


class VoiceEmotionRequest(BaseModel):
    """Request to analyze voice emotion from an audio chunk."""

    audio_base64: str = Field(
        ..., description="Base64-encoded audio data (e.g., 2–5 seconds)"
    )
    duration_ms: int = Field(..., gt=0, description="Duration of the audio in ms")


class VoiceEmotionResponse(BaseModel):
    """Voice emotion and stress analysis result."""

    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    stress_level: float = Field(
        ..., ge=0.0, le=1.0, description="Relative vocal stress level (0–1)"
    )
    tone: str = Field(..., description="High-level tone label, e.g., calm or stressed")


# --- Text Emotion ---


class TextEmotionRequest(BaseModel):
    """Request to analyze emotion from text."""
    text: str = Field(..., min_length=1)


class TextEmotionResponse(BaseModel):
    """Text emotion and sentiment analysis result."""

    emotion: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sentiment: float = Field(
        ...,
        ge=-1.0,
        le=1.0,
        description="Sentiment score from -1 (very negative) to +1 (very positive)",
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Extracted salient keywords (e.g., 'exhausted', 'deadline')",
    )
    risk_level: str = Field(
        ...,
        description="Risk level classification: NORMAL, MILD, MODERATE, SEVERE",
    )


# --- Fusion ---


class FusionRequest(BaseModel):
    """
    Request to fuse multimodal emotion predictions.
    All modalities are optional to allow partial fusion (e.g., face + text only).
    """

    face_emotion: Optional[str] = None
    face_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    voice_emotion: Optional[str] = None
    voice_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    text_emotion: Optional[str] = None
    text_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class FusionResponse(BaseModel):
    """Final fused emotion and stress outcome."""

    final_emotion: str
    stress_score: int = Field(..., ge0=0, le=100)  # note: will fix below
    hidden_stress_detected: bool


# Fix small typo: Pydantic uses ge/le, not ge0
FusionResponse.model_fields["stress_score"].ge = 0  # type: ignore
FusionResponse.model_fields["stress_score"].le = 100  # type: ignore
