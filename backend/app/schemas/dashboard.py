# FILE: backend/app/schemas/dashboard.py
from __future__ import annotations

from typing import Dict, List, Any
from pydantic import BaseModel, Field


class StressTrendPoint(BaseModel):
    """Single point in a stress trend timeline."""
    date: str = Field(..., description="ISO date string (YYYY-MM-DD)")
    stress: float = Field(..., description="Average or representative stress score")
    emotion: str = Field(..., description="Dominant emotion for that date")


class DashboardSummaryResponse(BaseModel):
    """
    High-level dashboard summary across a given date range.
    Used by /api/dashboard/summary.
    """

    current_stress: float = Field(..., description="Most recent stress score")
    burnout_risk: str = Field(..., description="LOW / MEDIUM / HIGH")
    avg_stress_trend: List[StressTrendPoint] = Field(
        default_factory=list,
        description="Time-series of stress values used for line charts",
    )
    emotion_distribution: Dict[str, float] = Field(
        default_factory=dict,
        description="Distribution of emotions over the selected range",
    )
    recommendations: List[str] = Field(
        default_factory=list,
        description="Simple textual recommendations for the user",
    )
