import json
import os


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results", "metrics_summary.json")
MIN_RECALL = 0.80
MAX_FPR = 0.20


def load_metrics() -> list[dict]:
    with open(RESULTS_PATH, "r") as f:
        return json.load(f)


def score_threshold(metric: dict) -> float:
    return metric["accuracy"] - metric["false_positive_rate"] + metric["recall"]


def print_thresholds(metrics: list[dict]):
    print("\nTHRESHOLD ANALYSIS\n")
    print(f"{'Thresh':<8} {'Recall':<8} {'FPR':<8} {'Acc':<8} {'PASS?'}")
    print("-" * 50)

    for metric in metrics:
        passes = metric["recall"] >= MIN_RECALL and metric["false_positive_rate"] <= MAX_FPR
        status = "PASS" if passes else "FAIL"
        print(
            f"{metric['threshold']:<8} {metric['recall']:<8} "
            f"{metric['false_positive_rate']:<8} {metric['accuracy']:<8} {status}"
        )


def print_valid_thresholds(metrics: list[dict]):
    valid_thresholds = [
        metric
        for metric in metrics
        if metric["recall"] >= MIN_RECALL and metric["false_positive_rate"] <= MAX_FPR
    ]

    if not valid_thresholds:
        print("\nNO THRESHOLD MEETS ALL REQUIREMENTS")
        print("You need to tune your model.")
        return

    print("\nTHRESHOLDS THAT MEET ALL REQUIREMENTS:\n")
    for metric in valid_thresholds:
        print(metric)


def main():
    metrics = load_metrics()
    best_threshold = max(metrics, key=score_threshold)

    print_thresholds(metrics)
    print("\nBEST THRESHOLD (BALANCED)\n")
    print(best_threshold)
    print_valid_thresholds(metrics)


if __name__ == "__main__":
    main()
