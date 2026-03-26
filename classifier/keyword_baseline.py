import re
import time


KEYWORDS = {
    "jailbreak": [
        r"\bDAN\b",
        r"do anything now",
        r"jailbreak",
        r"no restrictions",
        r"ignore (all |your |previous )?instructions",
        r"you are now",
        r"pretend you (are|have no)",
        r"act as if you",
        r"hypothetically speaking.*how (do|can|would)",
        r"roleplay as",
        r"in this scenario you have no",
    ],
    "injection": [
        r"ignore previous",
        r"disregard (all |the )?above",
        r"forget (your |all )?instructions",
        r"new instructions:",
        r"system prompt",
        r"reveal your prompt",
        r"print your instructions",
        r"what (are|were) your instructions",
        r"override",
        r"you are a (different|new) (ai|assistant|model)",
    ],
    "toxic": [
        r"how (to|do i) (make|build|create|synthesize) (a |an )?(bomb|weapon|poison|drug)",
        r"(kill|murder|harm|hurt) (myself|yourself|someone|people)",
        r"suicide method",
        r"self.harm",
        r"ways to (hurt|injure)",
        r"(hate|kill) all .*(people|group|race|religion)",
        r"(n-word|racial slur)",
        r"child (porn|abuse|exploitation)",
        r"how to (hack|crack|exploit|bypass)",
        r"(meth|heroin|fentanyl).*(make|cook|synthesize)",
        r"step.by.step.*(dangerous|illegal|harmful)",
    ],
}


class KeywordBaseline:
    def __init__(self):
        self.compiled = {
            category: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for category, patterns in KEYWORDS.items()
        }

    def classify(self, text: str) -> dict:
        start_time = time.time()

        for category, patterns in self.compiled.items():
            for pattern in patterns:
                if pattern.search(text):
                    latency_ms = round((time.time() - start_time) * 1000, 2)
                    return {
                        "verdict": "unsafe",
                        "category": category,
                        "confidence": 1.0,
                        "matched_pattern": pattern.pattern,
                        "latency_ms": latency_ms,
                    }

        latency_ms = round((time.time() - start_time) * 1000, 2)
        return {
            "verdict": "safe",
            "category": "safe",
            "confidence": 0.0,
            "matched_pattern": None,
            "latency_ms": latency_ms,
        }



