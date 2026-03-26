from contextlib import contextmanager

from fastapi import FastAPI
from pydantic import BaseModel

from classifier.ensemble import Ensemble


app = FastAPI(title="SmartGuard API")
ensemble = Ensemble()


class ClassifyRequest(BaseModel):
    prompt: str
    threshold_override: float | None = None


class ChatRequest(BaseModel):
    prompt: str


@contextmanager
def temporary_threshold_override(threshold: float | None):
    if threshold is None:
        yield
        return

    original_injection = ensemble.injection_clf.threshold
    original_toxic = ensemble.toxic_clf.threshold
    ensemble.injection_clf.threshold = threshold
    ensemble.toxic_clf.threshold = threshold

    try:
        yield
    finally:
        ensemble.injection_clf.threshold = original_injection
        ensemble.toxic_clf.threshold = original_toxic


@app.post("/classify")
def classify(req: ClassifyRequest):
    with temporary_threshold_override(req.threshold_override):
        return ensemble.classify(req.prompt)


@app.post("/chat")
def chat(req: ChatRequest):
    classification = ensemble.classify(req.prompt)

    if classification["verdict"] == "unsafe":
        return {
            "blocked": True,
            "reason": classification["category"],
            "confidence": classification["confidence"],
            "response": None,
            "latency_ms": classification["latency_ms"],
        }

    return {
        "blocked": False,
        "reason": "safe",
        "confidence": classification["confidence"],
        "response": "Prompt is safe. No LLM backend configured — connect one in config.yaml.",
        "latency_ms": classification["latency_ms"],
    }


@app.get("/health")
def health():
    return {"status": "ok"}
