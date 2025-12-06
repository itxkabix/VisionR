# FILE: backend/app/services/dashboard_service.py (example usage)
from datetime import date
from sqlalchemy.orm import Session as DbSession
from app.models.session import Session
from app.services.burnout_service import (
    SessionStats,
    calculate_burnout_risk,
    BurnoutRiskResult,
)

def get_user_burnout_risk(db: DbSession, user_id: str) -> BurnoutRiskResult:
    sessions = (
        db.query(Session)
        .filter(Session.user_id == user_id)
        .order_by(Session.session_start.desc())
        .limit(10)
        .all()
    )

    stats: List[SessionStats] = []
    for s in sessions:
        stats.append(
            SessionStats(
                date=s.session_start.date(),
                stress_score=int(s.total_stress_score or 0),
                dominant_emotion=s.dominant_emotion or "neutral",
            )
        )

    return calculate_burnout_risk(stats)
