from typing import Any, Dict, Optional

import numpy as np

# from deepface import DeepFace
# import mediapipe as mp
# import cv2


class FaceEmotionService:
    """
    Service responsible for:
    - Detecting faces in frames
    - Extracting facial landmarks / regions
    - Running emotion classification using DeepFace (or similar)
    - Computing arousal / valence approximations

    This class should be instantiated once at startup and reused.
    """

    def __init__(self) -> None:
        # TODO: initialize MediaPipe, DeepFace models, etc.
        # self.face_detector = mp.solutions.face_detection.FaceDetection(...)
        # self.emotion_model = DeepFace.build_model("Emotion")
        pass

    def detect_emotion(
        self,
        frame: np.ndarray,
        *,
        timestamp_ms: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run face detection + emotion classification on a single video frame.

        Args:
            frame: RGB or BGR image as NumPy array (H, W, 3).
            timestamp_ms: Optional timestamp from client, used for logging.

        Returns:
            Dictionary with keys such as:
            {
                "face_emotion": "happy" | "sad" | ... | "unknown",
                "confidence": float,
                "arousal": float,   # 0.0 - 1.0
                "valence": float,   # 0.0 - 1.0
                "landmarks": [...], # optional simplified representation
                "has_face": bool,
                "timestamp_ms": int | None,
            }
        """
        # TODO: implement detection & classification
        raise NotImplementedError("FaceEmotionService.detect_emotion is not implemented yet.")

    def detect_primary_face_region(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract the primary face region from the frame (e.g., largest bounding box).

        Args:
            frame: Full image.

        Returns:
            Cropped face image or None if no face is detected.
        """
        # TODO: implement face cropping using MediaPipe / OpenCV
        raise NotImplementedError(
            "FaceEmotionService.detect_primary_face_region is not implemented yet."
        )
