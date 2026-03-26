import json
import os
import sys


CURRENT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

sys.path.insert(0, PROJECT_ROOT)

from classifier.ensemble import Ensemble


INPUT_FILE = os.path.join(CURRENT_DIR, "red_team_suite.json")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results")
THRESHOLDS = [round(value * 0.1, 1) for value in range(1, 10)]


def load_prompts() -> list[dict]:
    with open(INPUT_FILE, "r") as f:
        data = json.load(f)
    return data["prompts"]


def classify_prompt(ensemble: Ensemble, text: str) -> dict:
    result = ensemble.classify(text)
    return {
        "confidence": result["confidence"],
        "verdict": result["verdict"],
        "category": result["category"],
        "latency": result["latency_ms"],
        "model_used": result["model_used"],
    }


def evaluate_thresholds(ensemble: Ensemble, prompts: list[dict]) -> list[dict]:
    all_results = []

    for threshold in THRESHOLDS:
        print(f"Threshold: {threshold}")
        for item in prompts:
            output = classify_prompt(ensemble, item["text"])
            final_prediction = "unsafe" if output["confidence"] >= threshold else "safe"
            all_results.append(
                {
                    "id": item["id"],
                    "category": item["category"],
                    "threshold": threshold,
                    "true": item["label"],
                    "pred": final_prediction,
                    "confidence": output["confidence"],
                    "model_verdict": output["verdict"],
                    "model_category": output["category"],
                    "latency_ms": output["latency"],
                    "model_used": output["model_used"],
                    "correct": final_prediction == item["label"],
                }
            )

    return all_results


def summarize_thresholds(all_results: list[dict]) -> list[dict]:
    metrics_summary = []

    for threshold in THRESHOLDS:
        subset = [result for result in all_results if result["threshold"] == threshold]
        tp = sum(1 for result in subset if result["true"] == "unsafe" and result["pred"] == "unsafe")
        fn = sum(1 for result in subset if result["true"] == "unsafe" and result["pred"] == "safe")
        fp = sum(1 for result in subset if result["true"] == "safe" and result["pred"] == "unsafe")
        tn = sum(1 for result in subset if result["true"] == "safe" and result["pred"] == "safe")
        recall = tp / (tp + fn) if (tp + fn) else 0
        fpr = fp / (fp + tn) if (fp + tn) else 0
        accuracy = (tp + tn) / len(subset)
        avg_latency = sum(result["latency_ms"] for result in subset) / len(subset)

        metrics_summary.append(
            {
                "threshold": threshold,
                "recall": round(recall, 3),
                "false_positive_rate": round(fpr, 3),
                "accuracy": round(accuracy, 3),
                "avg_latency_ms": round(avg_latency, 2),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
            }
        )

    return metrics_summary


def save_json(path: str, payload):
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved: {path}")


def main():
    ensemble = Ensemble()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Running evaluation...\n")
    all_results = evaluate_thresholds(ensemble, load_prompts())
    metrics_summary = summarize_thresholds(all_results)
    curve_data = {
        "thresholds": [metric["threshold"] for metric in metrics_summary],
        "recall": [metric["recall"] for metric in metrics_summary],
        "fpr": [metric["false_positive_rate"] for metric in metrics_summary],
        "accuracy": [metric["accuracy"] for metric in metrics_summary],
        "latency": [metric["avg_latency_ms"] for metric in metrics_summary],
    }

    save_json(os.path.join(OUTPUT_DIR, "detailed_results.json"), all_results)
    save_json(os.path.join(OUTPUT_DIR, "metrics_summary.json"), metrics_summary)
    save_json(os.path.join(OUTPUT_DIR, "curve_data.json"), curve_data)

    print("\nFINAL SUMMARY\n")
    print(f"{'Thresh':<8} {'Recall':<8} {'FPR':<8} {'Acc':<8} {'Latency':<10}")
    print("-" * 50)
    for metric in metrics_summary:
        print(
            f"{metric['threshold']:<8} {metric['recall']:<8} "
            f"{metric['false_positive_rate']:<8} {metric['accuracy']:<8} {metric['avg_latency_ms']:<10}"
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
