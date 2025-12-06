# FILE: backend/tests/integration/test_chat_message_endpoint.py
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.db.database import Base, engine
from app.db.session import get_db

# NOTE: In a real setup, configure a separate test database and fixture
# that sets up and tears down tables for each test session.
Base.metadata.create_all(bind=engine)

client = TestClient(app)


@pytest.fixture
def auth_headers() -> dict:
  """
  Create a test user and return Authorization headers with a valid JWT.
  For simplicity here, we assume there's already a known JWT.
  In a real test, we would:
    1. POST /api/auth/register
    2. POST /api/auth/login
  and return the token from there.
  """
  # TODO: Replace with dynamic token generation or register+login flow
  token = "TEST_JWT_TOKEN"
  return {"Authorization": f"Bearer {token}"}


def test_chat_message_flow(auth_headers: dict) -> None:
  """
  Integration test for /api/chat/message:
    1. Start a session
    2. Send a user message
    3. Check that AI response and emotion metadata are present
  """

  # 1. Start a new session
  start_resp = client.post(
    "/api/chat/session/start",
    json={"topic": "test-session"},
    headers=auth_headers,
  )
  assert start_resp.status_code == 201
  session_id = start_resp.json()["session_id"]

  # 2. Send a user message
  message_payload = {
    "session_id": session_id,
    "content": "I feel very stressed and exhausted lately.",
  }

  msg_resp = client.post(
    "/api/chat/message",
    json=message_payload,
    headers=auth_headers,
  )

  assert msg_resp.status_code == 201
  body = msg_resp.json()

  # Basic structure checks
  assert "message_id" in body
  assert "ai_response" in body
  assert "emotion_detected" in body

  # Emotion structure: depends on schema, but commonly:
  emotion = body["emotion_detected"]
  assert "text_emotion" in emotion
  assert "stress_score" in emotion

  # 3. Optionally, verify that the message was persisted:
  history_resp = client.get(
    f"/api/emotion/history/{session_id}",
    headers=auth_headers,
  )
  assert history_resp.status_code == 200
  history = history_resp.json()
  assert "emotions" in history
  assert len(history["emotions"]) >= 1
