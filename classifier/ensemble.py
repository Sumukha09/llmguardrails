import os
import re
import sys
import time

import yaml


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classifier.injection_classifier import InjectionClassifier
from classifier.toxic_classifier import ToxicClassifier


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
JAILBREAK_PATTERNS = [
    r"\bDAN\b",
    r"do anything now",
    r"no restrictions",
    r"no ethical guidelines",
    r"no content policy",
    r"no safety (filters?|rules?)",
    r"without (any )?(restrictions|limits|filters)",
    r"pretend (you have no|there are no)",
    r"act as (if you have no|a (different|new) ai)",
    r"you are now (a |an )?(different|unrestricted|evil)",
    r"jailbreak(ed|gpt)?",
    r"(escaped?|bypassed?) (your )?(constraints?|restrictions?|safety)",
]
PII_KEYWORDS = [
    "credit card",
    "ssn",
    "social security",
    "phone number",
    "bank account",
    "password",
    "leaked",
    "personal data",
]
VIOLENCE_KEYWORDS = [
    "kill",
    "suicide",
    "hurt myself",
    "hurt someone",
    "make a bomb",
    "make a weapon",
    "explosive",
    "how to murder",
]
INDIRECT_JAILBREAK_KEYWORDS = [
    "roleplay",
    "translate",
    "for a fictional",
    "explain illegal",
    "list criminal",
    "construction",
]
BOOST_SCORE = 0.6
UNSAFE_TOXIC_LABELS = {"H", "SH", "V", "S", "HR"}


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def matches_any_pattern(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def contains_any_keyword(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


class Ensemble:
    def __init__(self):
        config = load_config()
        thresholds = config["thresholds"]
        ensemble_config = config.get("ensemble", {})
        scale = ensemble_config.get("scale", {})

        self.injection_clf = InjectionClassifier(threshold=thresholds["injection"])
        self.toxic_clf = ToxicClassifier(threshold=thresholds["toxic"])
        self.scale_inj = scale.get("injection", 1.0)
        self.scale_tox = scale.get("toxic", 0.85)
        self.ensemble_threshold = ensemble_config.get("threshold", 0.5)

    @staticmethod
    def _to_unsafe_score(result: dict, unsafe_labels: set[str]) -> float:
        return result["score"] if result["label"] in unsafe_labels else 1.0 - result["score"]

    @staticmethod
    def _latency_ms(start_time: float) -> float:
        return round((time.time() - start_time) * 1000, 2)

    def _precheck_jailbreak(self, text: str, start_time: float) -> dict | None:
        # Regex precheck covers role-play jailbreaks the classifier under-scores.
        if not matches_any_pattern(text, JAILBREAK_PATTERNS):
            return None

        return {
            "verdict": "unsafe",
            "category": "jailbreak",
            "confidence": 1.0,
            "model_used": "jailbreak_precheck",
            "latency_ms": self._latency_ms(start_time),
            "injection_score": None,
            "toxic_score": None,
        }

    def _boost_low_confidence_scores(self, text: str, injection_score: float, toxic_score: float) -> tuple[float, float]:
        if max(injection_score, toxic_score) >= 0.15:
            return injection_score, toxic_score

        lowered_text = text.lower()

        if contains_any_keyword(lowered_text, PII_KEYWORDS) or contains_any_keyword(
            lowered_text, INDIRECT_JAILBREAK_KEYWORDS
        ):
            injection_score = max(injection_score, BOOST_SCORE)

        if contains_any_keyword(lowered_text, VIOLENCE_KEYWORDS):
            toxic_score = max(toxic_score, BOOST_SCORE)

        return injection_score, toxic_score

    def _resolve_unsafe_category(
        self,
        injection_score: float,
        toxic_score: float,
        injection_result: dict,
        toxic_result: dict,
    ) -> tuple[str, str]:
        if injection_score >= toxic_score:
            category = injection_result.get("category", "injection")
            return ("injection" if category == "safe" else category, "ProtectAI/deberta-v3-base-prompt-injection-v2")

        category = toxic_result.get("category", "toxic")
        return ("toxic" if category == "safe" else category, "KoalaAI/Text-Moderation")

    def classify(self, text: str) -> dict:
        start_time = time.time()
        jailbreak_result = self._precheck_jailbreak(text, start_time)
        if jailbreak_result is not None:
            return jailbreak_result

        injection_result = self.injection_clf.predict(text)
        toxic_result = self.toxic_clf.predict(text)

        injection_score = self._to_unsafe_score(injection_result, {"INJECTION"})
        toxic_score = self._to_unsafe_score(toxic_result, UNSAFE_TOXIC_LABELS)
        injection_score, toxic_score = self._boost_low_confidence_scores(text, injection_score, toxic_score)

        final_score = max(
            min(injection_score * self.scale_inj, 1.0),
            min(toxic_score * self.scale_tox, 1.0),
        )
        verdict = "unsafe" if final_score >= self.ensemble_threshold else "safe"

        if verdict == "unsafe":
            category, model_used = self._resolve_unsafe_category(
                injection_score,
                toxic_score,
                injection_result,
                toxic_result,
            )
        else:
            category = "safe"
            model_used = "both"

        return {
            "verdict": verdict,
            "category": category,
            "confidence": round(final_score, 4),
            "model_used": model_used,
            "latency_ms": self._latency_ms(start_time),
            "injection_score": round(injection_score, 4),
            "toxic_score": round(toxic_score, 4),
        }



