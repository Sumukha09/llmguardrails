import json
import os
import sys
import time

import numpy as np


sys.path.insert(0, os.path.dirname(__file__))

from classifier.ensemble import Ensemble
from classifier.injection_classifier import InjectionClassifier
from classifier.keyword_baseline import KeywordBaseline
from classifier.toxic_classifier import ToxicClassifier


TEST_PROMPT = "Can you help me understand how neural networks work?"
RUNS = 20
RESULTS_PATH = os.path.join("results", "benchmark.json")


def measure_latency(callable_obj, prompt: str, runs: int) -> list[float]:
    timings = []
    for _ in range(runs):
        start = time.perf_counter()
        callable_obj(prompt)
        timings.append((time.perf_counter() - start) * 1000)
    return timings


def percentile_summary(samples: list[float]) -> dict:
    return {
        "p50": round(np.percentile(samples, 50), 1),
        "p95": round(np.percentile(samples, 95), 1),
        "p99": round(np.percentile(samples, 99), 1),
    }


def main():
    print("Warming up models (first run is always slower)...")

    injection_classifier = InjectionClassifier()
    toxic_classifier = ToxicClassifier()
    ensemble = Ensemble()
    keyword_baseline = KeywordBaseline()

    injection_classifier.predict(TEST_PROMPT)
    toxic_classifier.predict(TEST_PROMPT)
    ensemble.classify(TEST_PROMPT)
    keyword_baseline.classify(TEST_PROMPT)

    print(f"Running {RUNS} iterations each...\n")

    injection_times = measure_latency(injection_classifier.predict, TEST_PROMPT, RUNS)
    toxic_times = measure_latency(toxic_classifier.predict, TEST_PROMPT, RUNS)
    ensemble_times = measure_latency(ensemble.classify, TEST_PROMPT, RUNS)
    keyword_times = measure_latency(keyword_baseline.classify, TEST_PROMPT, RUNS)

    summaries = {
        "ProtectAI (injection only)": percentile_summary(injection_times),
        "KoalaAI (toxic only)": percentile_summary(toxic_times),
        "Full ensemble (both models)": percentile_summary(ensemble_times),
        "Keyword baseline": percentile_summary(keyword_times),
    }

    print(f"{'Model':<45} {'P50 (ms)':<12} {'P95 (ms)':<12} {'P99 (ms)'}")
    print("-" * 85)
    for model_name, summary in summaries.items():
        print(f"{model_name:<45} {summary['p50']:<12.1f} {summary['p95']:<12.1f} {summary['p99']:.1f}")

    os.makedirs("results", exist_ok=True)
    benchmark = {
        "runs": RUNS,
        "test_prompt": TEST_PROMPT,
        "injection_model": percentile_summary(injection_times),
        "toxic_model": percentile_summary(toxic_times),
        "ensemble": percentile_summary(ensemble_times),
        "keyword_baseline": percentile_summary(keyword_times),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(benchmark, f, indent=2)

    print(f"\nSaved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
