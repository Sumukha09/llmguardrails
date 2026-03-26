import json
import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from classifier.ensemble import Ensemble


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
DETAILED_RESULTS_PATH = os.path.join(RESULTS_DIR, "detailed_results.json")
METRICS_PATH = os.path.join(RESULTS_DIR, "metrics_summary.json")
CURVE_PATH = os.path.join(RESULTS_DIR, "curve_data.json")
SECTION_HEADER_HTML = '<div class="sg-section">{label}</div>'
EMPTY_STATE_HTML = """
<div style="
    height:300px;
    display:flex;
    align-items:center;
    justify-content:center;
    border:1px dashed #1e2a3a;
    border-radius:6px;
    font-family:'JetBrains Mono',monospace;
    font-size:0.75rem;
    color:#1e2a3a;
    letter-spacing:2px;
">
    AWAITING INPUT
</div>
"""
APP_STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}

.stApp {
    background: #0a0a0f;
}

.sg-header {
    padding: 2rem 0 1.5rem 0;
    border-bottom: 1px solid #1e2a3a;
    margin-bottom: 2rem;
}

.sg-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    letter-spacing: -0.5px;
    color: #f0f4ff;
    margin: 0;
}

.sg-title span {
    color: #3b82f6;
}

.sg-subtitle {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: #4a5568;
    margin-top: 0.4rem;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0f1117;
    border-bottom: 1px solid #1e2a3a;
    gap: 0;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    letter-spacing: 1px;
    color: #4a5568;
    padding: 0.8rem 1.5rem;
    border-bottom: 2px solid transparent;
    text-transform: uppercase;
}

.stTabs [aria-selected="true"] {
    color: #3b82f6 !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

[data-testid="metric-container"] {
    background: #0f1117;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 1rem;
}

[data-testid="metric-container"] label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #4a5568 !important;
}

[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.6rem;
    color: #f0f4ff;
}

.verdict-unsafe {
    background: linear-gradient(135deg, #1a0a0a, #2d1010);
    border: 1px solid #dc2626;
    border-left: 4px solid #dc2626;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    color: #fca5a5;
    font-size: 1.1rem;
    letter-spacing: 1px;
    margin: 1rem 0;
}

.verdict-safe {
    background: linear-gradient(135deg, #0a1a0f, #0d2418);
    border: 1px solid #16a34a;
    border-left: 4px solid #16a34a;
    border-radius: 6px;
    padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    color: #86efac;
    font-size: 1.1rem;
    letter-spacing: 1px;
    margin: 1rem 0;
}

.sg-section {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #3b82f6;
    margin: 1.5rem 0 0.8rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #1e2a3a;
}

.score-bar-wrap {
    background: #0f1117;
    border: 1px solid #1e2a3a;
    border-radius: 6px;
    padding: 1.2rem;
    margin-bottom: 0.8rem;
}

.score-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #4a5568;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}

.score-value {
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1.4rem;
    color: #f0f4ff;
}

.stTextArea textarea {
    background: #0f1117 !important;
    border: 1px solid #1e2a3a !important;
    border-radius: 6px !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stTextArea textarea:focus {
    border-color: #3b82f6 !important;
    box-shadow: 0 0 0 1px #3b82f6 !important;
}

.stButton button {
    background: #1d4ed8 !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.6rem 2rem !important;
    transition: background 0.2s !important;
}

.stButton button:hover {
    background: #2563eb !important;
}

.stSlider [data-baseweb="slider"] {
    padding: 0.5rem 0;
}

.stDataFrame {
    border: 1px solid #1e2a3a !important;
    border-radius: 6px !important;
}

.stProgress > div > div {
    background: #1e2a3a;
    border-radius: 3px;
}

.stProgress > div > div > div {
    background: #3b82f6 !important;
    border-radius: 3px;
}

.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 1px !important;
    color: #4a5568 !important;
    background: #0f1117 !important;
    border: 1px solid #1e2a3a !important;
}

.stAlert {
    border-radius: 4px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
}

.stCaption {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #4a5568 !important;
}

.tag-pass {
    display: inline-block;
    background: #052e16;
    color: #86efac;
    border: 1px solid #16a34a;
    border-radius: 3px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
}

.tag-fail {
    display: inline-block;
    background: #1a0a0a;
    color: #fca5a5;
    border: 1px solid #dc2626;
    border-radius: 3px;
    padding: 2px 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 1px;
}
</style>
"""


st.set_page_config(
    page_title="SmartGuard",
    page_icon="S",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(APP_STYLES, unsafe_allow_html=True)
st.markdown(
    """
    <div class="sg-header">
        <div class="sg-title">Smart<span>Guard</span></div>
        <div class="sg-subtitle">LLM Input / Output Firewall · Ensemble Classifier</div>
    </div>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_ensemble():
    return Ensemble()


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def render_section(label: str):
    st.markdown(SECTION_HEADER_HTML.format(label=label), unsafe_allow_html=True)


def render_verdict(result: dict):
    if result["verdict"] == "unsafe":
        st.markdown(
            f"""
            <div class="verdict-unsafe">
                BLOCKED · {result['category'].upper()}
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        """
        <div class="verdict-safe">
            SAFE · PASSED THROUGH
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model_score(label: str, score: float | None, skipped_message: str | None = None):
    if score is None:
        st.markdown(
            f"""
            <div class="score-bar-wrap">
                <div class="score-label">{label}</div>
                <div class="score-value" style="color:#4a5568">NOT RUN</div>
                <div style="font-family:'JetBrains Mono',monospace;font-size:0.7rem;color:#374151;margin-top:0.3rem;">
                    {skipped_message}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        f"""
        <div class="score-bar-wrap">
            <div class="score-label">{label}</div>
            <div class="score-value">{score:.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(min(float(score), 1.0))


def render_live_classifier(ensemble: Ensemble):
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        render_section("Prompt Input")
        user_prompt = st.text_area(
            label="",
            height=140,
            value="",
            placeholder="Enter any prompt to test the firewall...",
        )

        render_section("Firewall Strictness")
        user_threshold = st.slider(
            label="",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
        )
        toxic_threshold = round(max(0.1, user_threshold - 0.1), 2)
        st.markdown(
            f"""
            <div style="font-family:'JetBrains Mono',monospace; font-size:0.72rem; color:#4a5568; margin-top:0.3rem;">
                INJECTION THRESHOLD&nbsp;&nbsp;<span style="color:#3b82f6">{user_threshold}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                TOXIC THRESHOLD&nbsp;&nbsp;<span style="color:#3b82f6">{toxic_threshold}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                lower = stricter
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        classify_button = st.button("CLASSIFY PROMPT", type="primary", use_container_width=True)

    with col_right:
        render_section("Result")
        if not classify_button:
            st.markdown(EMPTY_STATE_HTML, unsafe_allow_html=True)
            return

        if not user_prompt.strip():
            st.warning("Enter a prompt first.")
            return

        ensemble.injection_clf.threshold = user_threshold
        ensemble.toxic_clf.threshold = toxic_threshold
        result = ensemble.classify(user_prompt)

        render_verdict(result)
        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Verdict", result["verdict"].upper())
        metric_2.metric("Confidence", f"{result['confidence']:.4f}")
        metric_3.metric("Latency", f"{result['latency_ms']} ms")

        render_section("Model Scores")
        render_model_score("ProtectAI - Injection / Jailbreak", result.get("injection_score") or 0.0)
        render_model_score(
            "KoalaAI - Toxic Content",
            result.get("toxic_score"),
            "ProtectAI caught it first - toxic model skipped",
        )


def load_default_metrics(metrics: list[dict]) -> dict:
    return next((metric for metric in metrics if metric["threshold"] == 0.5), metrics[0])


def render_pass_fail_tag(passes: bool, text: str):
    class_name = "tag-pass" if passes else "tag-fail"
    st.markdown(f'<span class="{class_name}">{text}</span>', unsafe_allow_html=True)


def render_aggregate_view():
    if not os.path.exists(METRICS_PATH):
        st.warning("Run `python red_team/run_red_team.py` first to generate results.")
        return

    metrics = load_json(METRICS_PATH)
    default_metrics = load_default_metrics(metrics)
    tp = default_metrics["tp"]
    fp = default_metrics["fp"]
    fn = default_metrics["fn"]
    tn = default_metrics["tn"]
    total = tp + fp + fn + tn
    block_rate = round(default_metrics["recall"] * 100, 1)
    fp_rate = round(default_metrics["false_positive_rate"] * 100, 1)
    accuracy = round(default_metrics["accuracy"] * 100, 1)

    render_section("Overview - Threshold 0.5 (Default)")
    metric_columns = st.columns(6)
    metric_columns[0].metric("Total Prompts", total)
    metric_columns[1].metric("Blocked", tp, delta="attacks caught")
    metric_columns[2].metric("Missed", fn, delta="attacks slipped")
    metric_columns[3].metric("False Positives", fp, delta="benign blocked")
    metric_columns[4].metric("Block Rate", f"{block_rate}%")
    metric_columns[5].metric("Accuracy", f"{accuracy}%")

    st.markdown("<br>", unsafe_allow_html=True)
    pass_fail_columns = st.columns(2)
    with pass_fail_columns[0]:
        render_pass_fail_tag(
            block_rate > 80,
            f"BLOCK RATE {block_rate}% - {'PASSES' if block_rate > 80 else 'BELOW'} >80% TARGET",
        )
    with pass_fail_columns[1]:
        render_pass_fail_tag(
            fp_rate < 20,
            f"FP RATE {fp_rate}% - {'PASSES' if fp_rate < 20 else 'ABOVE'} <20% TARGET",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    render_section("Block Rate by Category")
    if os.path.exists(DETAILED_RESULTS_PATH):
        detailed_results = load_json(DETAILED_RESULTS_PATH)
        threshold_results = [result for result in detailed_results if result["threshold"] == 0.5]
        for category in ["jailbreak", "injection", "toxic"]:
            unsafe_results = [
                result
                for result in threshold_results
                if result["category"] == category and result["true"] == "unsafe"
            ]
            blocked_results = [result for result in unsafe_results if result["pred"] == "unsafe"]
            rate = round(len(blocked_results) / len(unsafe_results) * 100, 1) if unsafe_results else 0
            st.progress(
                rate / 100,
                text=f"{category.upper()}  {rate}%  ({len(blocked_results)}/{len(unsafe_results)} blocked)",
            )

    render_section("All Thresholds - Full Summary")
    table_rows = []
    for metric in metrics:
        passes = metric["recall"] >= 0.80 and metric["false_positive_rate"] <= 0.20
        table_rows.append(
            {
                "Threshold": metric["threshold"],
                "Recall %": round(metric["recall"] * 100, 1),
                "FP Rate %": round(metric["false_positive_rate"] * 100, 1),
                "Accuracy %": round(metric["accuracy"] * 100, 1),
                "Latency ms": metric["avg_latency_ms"],
                "PS Pass": "PASS" if passes else "FAIL",
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_threshold_chart(curve: dict):
    thresholds = curve["thresholds"]
    recalls = [value * 100 for value in curve["recall"]]
    fp_rates = [value * 100 for value in curve["fpr"]]

    fig, ax = plt.subplots(figsize=(11, 5))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#0a0a0f")
    ax.plot(
        thresholds,
        recalls,
        color="#3b82f6",
        marker="o",
        linewidth=2.5,
        markersize=6,
        label="Recall / Block Rate (%)",
        zorder=3,
    )
    ax.plot(
        thresholds,
        fp_rates,
        color="#ef4444",
        marker="x",
        linewidth=2.5,
        markersize=8,
        markeredgewidth=2,
        label="False Positive Rate (%)",
        zorder=3,
    )
    ax.axhline(80, color="#3b82f6", linestyle="--", alpha=0.35, linewidth=1.2, label=">80% recall target")
    ax.axhline(20, color="#ef4444", linestyle="--", alpha=0.35, linewidth=1.2, label="<20% FP target")
    ax.fill_between(
        thresholds,
        [0 for _ in recalls],
        [min(value, 80) for value in recalls],
        alpha=0.04,
        color="#3b82f6",
    )
    ax.set_xlabel("Threshold", color="#4a5568", fontsize=11)
    ax.set_ylabel("Percentage (%)", color="#4a5568", fontsize=11)
    ax.set_title("Accuracy vs Strictness Curve", color="#e2e8f0", fontsize=13, fontweight="bold", pad=15)
    ax.tick_params(colors="#4a5568")
    ax.spines["bottom"].set_color("#1e2a3a")
    ax.spines["left"].set_color("#1e2a3a")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.08, color="#e2e8f0")
    ax.legend(fontsize=9, facecolor="#0f1117", edgecolor="#1e2a3a", labelcolor="#e2e8f0")
    st.pyplot(fig)


def render_recommended_threshold(curve: dict):
    thresholds = curve["thresholds"]
    recalls = [value * 100 for value in curve["recall"]]
    fp_rates = [value * 100 for value in curve["fpr"]]
    valid_thresholds = [
        (threshold, recall, fp_rate)
        for threshold, recall, fp_rate in zip(thresholds, recalls, fp_rates)
        if recall >= 80 and fp_rate <= 20
    ]

    if not valid_thresholds:
        st.markdown(
            """
            <div style="
                background:#1a0a0a;border:1px solid #dc2626;border-left:4px solid #dc2626;
                border-radius:6px;padding:1.2rem 1.5rem;
                font-family:'JetBrains Mono',monospace;color:#fca5a5;font-size:0.8rem;
            ">
                No single threshold satisfies both targets simultaneously.<br>
                Document this trade-off in your failure analysis.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    threshold, recall, fp_rate = valid_thresholds[-1]
    st.markdown(
        f"""
        <div style="
            background: #0a1520;
            border: 1px solid #1d4ed8;
            border-left: 4px solid #3b82f6;
            border-radius: 6px;
            padding: 1.2rem 1.5rem;
            font-family: 'JetBrains Mono', monospace;
            margin-top: 0.5rem;
        ">
            <span style="color:#4a5568;font-size:0.7rem;letter-spacing:2px;">RECOMMENDED THRESHOLD</span><br>
            <span style="color:#f0f4ff;font-size:1.8rem;font-family:'Syne',sans-serif;font-weight:700;">{round(float(threshold), 2)}</span>
            <span style="color:#4a5568;font-size:0.8rem;margin-left:1rem;">
                Recall: <span style="color:#3b82f6">{round(recall, 1)}%</span>
                &nbsp;·&nbsp;
                FP Rate: <span style="color:#ef4444">{round(fp_rate, 1)}%</span>
                &nbsp;·&nbsp;
                Both PS targets satisfied
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_threshold_curve():
    if not os.path.exists(CURVE_PATH):
        st.warning("Run `python red_team/run_red_team.py` first.")
        return

    curve = load_json(CURVE_PATH)
    render_section("Recall vs False Positive Rate - Threshold 0.1 to 0.9")
    render_threshold_chart(curve)

    render_section("Threshold Decision Table")
    thresholds = curve["thresholds"]
    recalls = [value * 100 for value in curve["recall"]]
    fp_rates = [value * 100 for value in curve["fpr"]]
    rows = []
    for threshold, recall, fp_rate in zip(thresholds, recalls, fp_rates):
        rows.append(
            {
                "Threshold": round(threshold, 2),
                "Recall %": round(recall, 1),
                "FP Rate %": round(fp_rate, 1),
                "PS Pass": "Both targets met" if recall >= 80 and fp_rate <= 20 else "FAIL",
            }
        )
    st.dataframe(rows, use_container_width=True, hide_index=True)

    render_section("Recommended Deployment Threshold")
    render_recommended_threshold(curve)


ensemble = load_ensemble()
tab_live, tab_aggregate, tab_curve = st.tabs(
    ["LIVE CLASSIFIER", "AGGREGATE VIEW", "THRESHOLD CURVE"]
)

with tab_live:
    render_live_classifier(ensemble)

with tab_aggregate:
    render_aggregate_view()

with tab_curve:
    render_threshold_curve()
