# text_emotion_service.py
from __future__ import annotations

from typing import Any, Dict, List

from .base_emotion import BaseEmotionResult

# Pseudocode imports:
# from transformers import pipeline
# import spacy
# from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


class TextEmotionService:
    """
    Uses HuggingFace emotion model + spaCy + sentiment rules
    to produce a rich text emotion representation.
    """

    def __init__(self) -> None:
        # TODO: real initialization
        # self.emotion_classifier = pipeline(
        #     "text-classification",
        #     model="j-hartmann/emotion-english-distilroberta-base",
        #     return_all_scores=True,
        # )
        # self.nlp = spacy.load("en_core_web_sm")
        # self.sentiment_analyzer = SentimentIntensityAnalyzer()
        self._risk_keywords_severe = ["suicide", "kill myself", "end it all"]
        self._risk_keywords_moderate = ["hopeless", "worthless", "broken", "can't go on"]
        self._risk_keywords_mild = ["exhausted", "burnt out", "overwhelmed", "stressed"]

    def analyze_text(self, text: str, timestamp_ms: int | None = None) -> BaseEmotionResult:
        """
        Main entry to analyze a single text message.

        Steps:
        1. Run emotion classifier to obtain label + confidences.
        2. Run sentiment analysis (VADER).
        3. Run spaCy NER and keyword extraction.
        4. Compute risk level based on rules.
        5. Wrap into BaseEmotionResult.
        """
        # 1. Emotion classification (pseudo)
        # scores = self.emotion_classifier(text)[0]  # list[{"label":..., "score":...}]
        # Map to {label: score}
        # emotion_scores = {s["label"]: float(s["score"]) for s in scores}
        # emotion_label = max(emotion_scores, key=emotion_scores.get)
        # confidence = emotion_scores[emotion_label]

        # Dummy example:
        emotion_scores = {"joy": 0.1, "sadness": 0.6, "anger": 0.1, "fear": 0.1, "love": 0.05, "surprise": 0.05}
        emotion_label = max(emotion_scores, key=emotion_scores.get)
        confidence = float(emotion_scores[emotion_label])

        # 2. Sentiment (pseudo)
        # vs = self.sentiment_analyzer.polarity_scores(text)
        # sentiment = float(vs["compound"])  # -1..1
        sentiment = -0.4  # example

        # 3. Entities & keywords (pseudo)
        entities: List[str] = []
        keywords: List[str] = []

        # Real implementation example:
        # doc = self.nlp(text)
        # entities = [ent.text for ent in doc.ents]
        # keywords = [chunk.text for chunk in doc.noun_chunks]

        # 4. Risk level
        risk_level = self._detect_risk_level(text=text, sentiment=sentiment, emotion_label=emotion_label)

        # 5. Build BaseEmotionResult
        features: Dict[str, Any] = {
            "emotion_scores": emotion_scores,
            "entities": entities,
            "keywords": keywords,
        }

        return BaseEmotionResult(
            modality="text",
            emotion=emotion_label,
            confidence=confidence,
            sentiment=sentiment,
            risk_level=risk_level,
            features=features,
            timestamp_ms=timestamp_ms,
        )

    def _detect_risk_level(self, text: str, sentiment: float, emotion_label: str) -> str:
        """
        Heuristic rules for risk level:

        - SEVERE: presence of suicidal/self-harm terms.
        - MODERATE: very negative sentiment + strong sadness/fear terms.
        - MILD: negative sentiment + stress/burnout terms.
        - NORMAL: otherwise.
        """
        lowered = text.lower()

        # SEVERE: explicit suicidal intent
        for phrase in self._risk_keywords_severe:
            if phrase in lowered:
                return "SEVERE"

        # MODERATE: very negative + sadness/fear
        if sentiment < -0.6 and emotion_label in {"sadness", "fear"}:
            return "MODERATE"

        # MILD: negative sentiment + burnout/stress mentions
        if sentiment < -0.2:
            for phrase in self._risk_keywords_mild + self._risk_keywords_moderate:
                if phrase in lowered:
                    return "MILD"

        return "NORMAL"
