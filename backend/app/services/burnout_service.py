# FILE: backend/app/services/burnout_service.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import List, Literal, Optional
import math


RiskLevel = Literal["LOW", "MEDIUM", "HIGH"]


@dataclass
class SessionStats:
    """
    Aggregated statistics for a single chat session.

    Attributes:
        date: Calendar date of the session.
        stress_score: Overall stress for the session (0–100).
        dominant_emotion: Most frequent or final emotion label for the session.
    """
    date: date
    stress_score: int
    dominant_emotion: str


@dataclass
class BurnoutRiskResult:
    """
    Output of the burnout risk algorithm.

    Attributes:
        risk_level: LOW / MEDIUM / HIGH
        score: Continuous risk score in [0,100]
        avg_stress: Mean stress across sessions
        recent_mean: Mean stress of recent half
        early_mean: Mean stress of early half
        negative_ratio: Fraction of sessions with negative dominant emotion
        sessions_per_week: Approximate help-seeking frequency
        factors: List of key explanatory flags
    """
    risk_level: RiskLevel
    score: int
    avg_stress: float
    early_mean: float
    recent_mean: float
    negative_ratio: float
    sessions_per_week: float
    factors: List[str]


def calculate_burnout_risk(
    sessions_history: List[SessionStats],
) -> BurnoutRiskResult:
    """
    Calculate burnout risk from a history of sessions.
    """
    if not sessions_history:
        # No history: treat as LOW with score 0
        return BurnoutRiskResult(
            risk_level="LOW",
            score=0,
            avg_stress=0.0,
            early_mean=0.0,
            recent_mean=0.0,
            negative_ratio=0.0,
            sessions_per_week=0.0,
            factors=[],
        )

    # 1. Sort by date (ascending)
    sessions = sorted(sessions_history, key=lambda s: s.date)
    N = len(sessions)

    # Extract stress and emotions
    stresses = [float(s.stress_score) for s in sessions]
    emotions = [s.dominant_emotion.lower() for s in sessions]

    # 2. Average stress
    avg_stress = sum(stresses) / N
    S_avg = avg_stress / 100.0

    # 3. Early vs recent means (trend)
    if N >= 2:
        split_index = N // 2
        early = stresses[:split_index]
        recent = stresses[split_index:]
        early_mean = sum(early) / len(early)
        recent_mean = sum(recent) / len(recent)
    else:
        # Only one session; no real trend, treat as stable
        early_mean = recent_mean = avg_stress

    delta = (recent_mean - early_mean) / 100.0  # in [-1,1]

    if delta <= 0:
        S_trend = 0.0
    else:
        # Positive delta → risk; 30-point increase (0.3) → full risk (1.0)
        S_trend = min(1.0, delta / 0.3)

    # 4. Negative emotion ratio
    negative_labels = {"sad", "angry", "fearful", "disgusted", "stressed"}
    neg_count = sum(1 for e in emotions if e in negative_labels)
    negative_ratio = neg_count / N if N > 0 else 0.0
    S_neg = negative_ratio

    # 5. Session frequency (sessions/week)
    first_day = sessions[0].date
    last_day = sessions[-1].date
    delta_days = max((last_day - first_day).days, 1)  # avoid 0
    sessions_per_day = N / delta_days
    sessions_per_week = sessions_per_day * 7.0

    # Map to [0,1] assuming 0–7+ sessions per week
    S_freq = min(1.0, sessions_per_week / 7.0)

    # 6. Combine subscores with weights
    w_avg = 0.40
    w_trend = 0.25
    w_neg = 0.25
    w_freq = 0.10

    S_burnout = (
        w_avg * S_avg
        + w_trend * S_trend
        + w_neg * S_neg
        + w_freq * S_freq
    )

    score = int(round(100.0 * S_burnout))

    # 7. Map to discrete risk level
    if score < 40:
        risk_level: RiskLevel = "LOW"
    elif score <= 70:
        risk_level = "MEDIUM"
    else:
        risk_level = "HIGH"

    # 8. Explanatory factors
    factors: List[str] = []
    if avg_stress >= 70.0:
        factors.append("high_average_stress")
    if (recent_mean - early_mean) >= 10.0:
        factors.append("increasing_stress_trend")
    if negative_ratio >= 0.6:
        factors.append("persistent_negative_mood")
    if sessions_per_week >= 4.0:
        factors.append("frequent_help_seeking")

    return BurnoutRiskResult(
        risk_level=risk_level,
        score=score,
        avg_stress=avg_stress,
        early_mean=early_mean,
        recent_mean=recent_mean,
        negative_ratio=negative_ratio,
        sessions_per_week=sessions_per_week,
        factors=factors,
    )
