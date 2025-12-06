from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Literal, Any


EmotionModality = Literal["face", "voice", "text"]


@dataclass
class BaseEmotionResult:
    """
    Unified per-modality emotion result.

    This is the same shape assumed from FaceEmotionService, VoiceEmotionService,
    and TextEmotionService.
    """
    modality: EmotionModality
    emotion: Optional[str]
    confidence: Optional[float]

    # Optional modality-specific signals
    stress_score: Optional[float] = None     # 0–1 for voice/text if available
    sentiment: Optional[float] = None        # -1 to 1 (text)
    arousal: Optional[float] = None          # 0–1 (face, maybe voice)
    valence: Optional[float] = None          # 0–1 (face/text)
    risk_level: Optional[str] = None         # NORMAL / MILD / MODERATE / SEVERE

    # Raw / low-level signals
    features: Dict[str, Any] = field(default_factory=dict)

    # Optional timestamp
    timestamp_ms: Optional[int] = None


@dataclass
class FusionResult:
    """
    Final fused multimodal emotion result for a given time window.
    """
    final_emotion: Optional[str]
    final_confidence: Optional[float]
    stress_score: int  # 0–100
    hidden_stress_detected: bool

    # For debugging/explainability
    modality_weights: Dict[str, float] = field(default_factory=dict)
    emotion_scores: Dict[str, float] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)


class FusionService:
    """
    Fusion engine combining face, voice, and text modality results into:

    - final_emotion (label + confidence)
    - stress_score in [0, 100]
    - hidden_stress_detected (bool)
    """

    def __init__(self) -> None:
        # Base importance weights (can be tuned or loaded from config)
        self.base_attention_weights = {
            "face": 0.35,
            "voice": 0.30,
            "text": 0.35,
        }

        # Base weights for stress components
        self.base_stress_weights = {
            "face": 0.35,   # arousal
            "voice": 0.40,  # vocal stress
            "text": 0.25,   # text negativity
        }

        # Canonical emotion label set used for fusion voting.
        # This can be adapted to match training labels.
        self.emotion_labels: List[str] = [
            "happy",
            "sad",
            "angry",
            "surprised",
            "fearful",
            "disgusted",
            "neutral",
        ]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def fuse_emotions(
        self,
        face_result: Optional[BaseEmotionResult],
        voice_result: Optional[BaseEmotionResult],
        text_result: Optional[BaseEmotionResult],
    ) -> FusionResult:
        """
        Fuse per-modality results into a final emotion and stress estimate.

        Steps:
        1. Collect available modalities and compute attention weights.
        2. Build per-modality probability distributions over labels.
        3. Compute fused emotion score for each label and pick argmax.
        4. Calculate stress_score (0–100).
        5. Detect hidden_stress.
        """
        results = [r for r in (face_result, voice_result, text_result) if r is not None]

        if not results:
            # No modalities available: return defaults
            return FusionResult(
                final_emotion=None,
                final_confidence=None,
                stress_score=0,
                hidden_stress_detected=False,
                modality_weights={},
                emotion_scores={},
                debug_info={"reason": "no_modalities"},
            )

        # 1. Attention weights
        attention = self._compute_attention_weights(results)

        # 2. Per-modality distributions (score vectors)
        modality_dists: Dict[str, Dict[str, float]] = {}
        for r in results:
            modality_dists[r.modality] = self._get_emotion_distribution(r)

        # 3. Fused emotion scores
        fused_scores: Dict[str, float] = {label: 0.0 for label in self.emotion_labels}
        for label in self.emotion_labels:
            for r in results:
                w_m = attention.get(r.modality, 0.0)
                p_mL = modality_dists[r.modality].get(label, 0.0)
                fused_scores[label] += w_m * p_mL

        # Normalize fused_scores just in case (not strictly necessary)
        total_score = sum(fused_scores.values()) or 1.0
        for label in fused_scores:
            fused_scores[label] /= total_score

        final_emotion = max(fused_scores, key=fused_scores.get)
        final_confidence = fused_scores[final_emotion]

        # 4. Stress score
        stress_score = self.calculate_stress_score(
            face_result=face_result,
            voice_result=voice_result,
            text_result=text_result,
        )

        # 5. Hidden stress detection
        hidden_stress = self._detect_hidden_stress(
            face_result=face_result,
            voice_result=voice_result,
            text_result=text_result,
        )

        debug_info = {
            "attention_weights": attention,
            "modality_distributions": modality_dists,
        }

        return FusionResult(
            final_emotion=final_emotion,
            final_confidence=final_confidence,
            stress_score=stress_score,
            hidden_stress_detected=hidden_stress,
            modality_weights=attention,
            emotion_scores=fused_scores,
            debug_info=debug_info,
        )

    def calculate_stress_score(
        self,
        face_result: Optional[BaseEmotionResult],
        voice_result: Optional[BaseEmotionResult],
        text_result: Optional[BaseEmotionResult],
    ) -> int:
        """
        Compute stress_score in [0, 100] from:

        - face arousal (0–1)
        - voice stress_score (0–1)
        - text negativity (derived from sentiment in [-1, 1])
        - text risk_level (for boosting)
        """
        components: Dict[str, float] = {}

        # Face arousal
        if face_result and face_result.arousal is not None:
            components["face"] = float(face_result.arousal)

        # Voice vocal stress
        if voice_result and voice_result.stress_score is not None:
            # assume voice_result.stress_score already in [0,1]
            components["voice"] = float(voice_result.stress_score)

        # Text sentiment -> negativity
        text_risk_level = None
        if text_result:
            text_risk_level = (text_result.risk_level or "NORMAL").upper()
            if text_result.sentiment is not None:
                s = float(text_result.sentiment)  # -1..1
                negativity = (1.0 - s) / 2.0     # -> 0..1
                components["text"] = negativity

        if not components:
            return 0

        # Normalize stress weights over available components
        active_keys = list(components.keys())
        weight_sum = sum(self.base_stress_weights[k] for k in active_keys)
        weights = {
            k: self.base_stress_weights[k] / weight_sum for k in active_keys
        }

        # Weighted average
        raw_stress = sum(weights[k] * components[k] for k in active_keys)

        # Boost based on text risk level
        boosted = raw_stress
        if text_risk_level == "MODERATE":
            boosted = min(1.0, raw_stress + 0.15)
        elif text_risk_level == "SEVERE":
            boosted = min(1.0, raw_stress + 0.30)

        score_0_100 = int(round(boosted * 100))
        return max(0, min(100, score_0_100))

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _compute_attention_weights(
        self,
        results: List[BaseEmotionResult],
    ) -> Dict[str, float]:
        """
        Compute attention weights a_m = w_m * c_m and normalize over available
        modalities.
        """
        raw: Dict[str, float] = {}
        for r in results:
            if r.confidence is None:
                continue
            base_w = self.base_attention_weights.get(r.modality, 0.0)
            raw_val = base_w * float(r.confidence)
            if raw_val > 0:
                raw[r.modality] = raw_val

        if not raw:
            # fallback: equal weights
            n = len(results)
            return {r.modality: 1.0 / n for r in results}

        total = sum(raw.values())
        return {m: val / total for m, val in raw.items()}

    def _get_emotion_distribution(self, result: BaseEmotionResult) -> Dict[str, float]:
        """
        Retrieve or approximate an emotion probability distribution for a modality.

        Prefers `result.features["emotion_scores"]` if present, otherwise uses a
        delta distribution on the predicted emotion.
        """
        scores = result.features.get("emotion_scores")
        dist: Dict[str, float] = {label: 0.0 for label in self.emotion_labels}

        if isinstance(scores, dict):
            # Map known labels, ignore unknown
            total = 0.0
            for label, val in scores.items():
                if label in dist:
                    dist[label] = float(val)
                    total += float(val)
            # Normalize if possible
            if total > 0:
                for label in dist:
                    dist[label] /= total
                return dist

        # Fallback: delta distribution on predicted label
        if result.emotion in dist:
            dist[result.emotion] = 1.0
        else:
            # unknown label: keep uniform or all zeros (here: uniform)
            n = len(dist)
            for label in dist:
                dist[label] = 1.0 / n

        return dist

    def _detect_hidden_stress(
        self,
        face_result: Optional[BaseEmotionResult],
        voice_result: Optional[BaseEmotionResult],
        text_result: Optional[BaseEmotionResult],
    ) -> bool:
        """
        Hidden stress condition (heuristic):

        - Face appears positive/neutral:
          valence >= 0.6 OR emotion in {happy, neutral}
        AND
        - At least one of:
          - text negativity >= 0.6
          - voice stress >= 0.6
          - text risk_level in {MODERATE, SEVERE}
        """
        # Face positivity
        face_positive = False
        if face_result:
            if face_result.valence is not None and face_result.valence >= 0.6:
                face_positive = True
            if face_result.emotion in {"happy", "neutral"}:
                face_positive = True

        if not face_positive:
            return False

        # Text negativity and risk
        text_negativity = 0.0
        text_risk = "NORMAL"
        if text_result:
            if text_result.sentiment is not None:
                text_negativity = (1.0 - float(text_result.sentiment)) / 2.0
            if text_result.risk_level:
                text_risk = text_result.risk_level.upper()

        # Voice stress
        voice_stress = 0.0
        if voice_result and voice_result.stress_score is not None:
            voice_stress = float(voice_result.stress_score)

        stressed_by_text = text_negativity >= 0.6
        stressed_by_voice = voice_stress >= 0.6
        severe_risk = text_risk in {"MODERATE", "SEVERE"}

        return stressed_by_text or stressed_by_voice or severe_risk
