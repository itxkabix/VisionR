# FILE: backend/tests/unit/test_face_emotion_service.py
import numpy as np
import pytest

from app.services.face_emotion_service import FaceEmotionService


@pytest.fixture(scope="module")
def face_service() -> FaceEmotionService:
  # In real tests you might mock DeepFace + MediaPipe to avoid heavy loading.
  return FaceEmotionService()


def create_dummy_frame(color: int = 0) -> np.ndarray:
  """
  Create a dummy RGB frame (480x640) for testing.
  In real tests, you would load actual images from disk.
  """
  return np.full((480, 640, 3), color, dtype=np.uint8)


def test_detect_emotion_no_face(face_service: FaceEmotionService) -> None:
  """
  When there is no face in the frame, the service should return
  an "unknown" emotion (or similar) and low confidence.
  """
  frame = create_dummy_frame()  # plain black frame, no face

  result = face_service.detect_emotion(frame)

  assert result.emotion_label in {"unknown", "neutral"}
  assert result.confidence <= 0.2
  assert result.landmarks == [] or result.landmarks is None


def test_detect_emotion_happy_like_frame(face_service: FaceEmotionService) -> None:
  """
  With an actual "happy" test image the model should predict 'happy' with
  reasonable confidence. Here we stub by checking type/shape because we
  do not have real image fixtures in this example.
  """
  frame = create_dummy_frame(color=200)  # bright-ish dummy frame

  result = face_service.detect_emotion(frame)

  # Basic sanity checks on structure
  assert isinstance(result.emotion_label, str)
  assert 0.0 <= result.confidence <= 1.0
  assert 0.0 <= result.arousal <= 1.0
  assert -1.0 <= result.valence <= 1.0
  # Landmarks should be a list of (x, y) pairs or similar
  if result.landmarks is not None:
    assert isinstance(result.landmarks, list)
