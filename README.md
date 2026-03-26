# SmartGuard — LLM Guardrail Firewall

A real-time safety layer that sits between users and LLMs, blocking prompt injections, jailbreaks, toxic content, and PII leaks — **before** they reach the model.

---

## Quick Start (4 commands)

```bash
git clone(https://github.com/Sumukha09/llmguardrails.git) && cd llmguardrails
python -m venv .venv && .venv\Scripts\activate        
pip install -r requirements.txt
python red_team/evaluate_guardrail.py                
```

> **Optional — Launch the dashboard or API:**
> ```bash
> streamlit run dashboard/app.py          

---

## Track Choice Justification  Track A (Build)

We chose **Track A** and built a multi-layer ensemble firewall using two specialist transformer models plus heuristic boosters:

| Layer | Model / Technique | Purpose |
|---|---|---|
| 1. Regex pre-check | 12 compiled patterns | Catches obvious DAN-style jailbreaks in <1 ms |
| 2. Injection classifier | `ProtectAI/deberta-v3-base-prompt-injection-v2` | Prompt injection & jailbreak detection |
| 3. Toxicity classifier | `KoalaAI/Text-Moderation` | Hate, self-harm, violence, sexual content |
| 4. Keyword boost | 50+ keyword/regex patterns | Fills blind spots for PII, violence, indirect jailbreaks |

### Why this architecture?

- **ProtectAI's DeBERTa-v3** uses disentangled attention, outperforming BERT/RoBERTa on adversarial sentence classification. It is the top-ranked model on HuggingFace for prompt-injection detection.
- **KoalaAI/Text-Moderation** is trained on OpenAI's moderation taxonomy and explicitly covers self-harm — a category both `martin-ha/toxic-comment` and `unitary/toxic-bert` missed during our evaluation.
- **Keyword baseline alone failed** 6/10 jailbreaks in red-team testing due to indirect phrasing. It serves only as a boost layer for low-confidence model scores, not as a standalone classifier.
- **Ensemble scoring** uses bias-corrected max-aggregation with configurable per-model scaling factors (`config.yaml`) to balance precision and recall without retraining.



---

## P95 Latency Results

Measured on CPU over 20 iterations (after warm-up). Run `python benchmark.py` to reproduce on your hardware.

| Component | P50 (ms) | P95 (ms) | P99 (ms) |
|---|---|---|---|
| ProtectAI (injection only) | 179.8 | **199.4** | 217.5 |
| KoalaAI (toxic only) | 55.1 | **56.8** | 56.8 |
| **Full ensemble (both models)** | **237.8** | **265.2** | **268.8** |
| Keyword baseline | <1 | **<1** | <1 |

> ✅ **P95 ensemble latency: 265.2 ms** — well under the 500 ms target.

---

## Evaluation Results

At the operating threshold of 0.4 (from `config.yaml`):

| Metric | Value |
|---|---|
| **Recall** | 0.857 |
| **False Positive Rate** | 0.059 |
| **Accuracy** | 0.889 |
| **Avg Latency** | 282 ms |

Full evaluation results are saved to `results/` after running the red-team suite.

---

## Project Structure

```
llmguardrails/
├── classifier/
│   ├── ensemble.py             
│   ├── injection_classifier.py
│   ├── toxic_classifier.py   
│   └── keyword_baseline.py    
├── api/
│   └── main.py                
├── dashboard/
│   └── app.py                 
├── red_team/
│   ├── red_team_suite.json     
│   ├── evaluate_guardrail.py   
│   └── analyze_results.py      
├── results/                    
├── config.yaml               
├── benchmark.py               
```

---

## API Endpoints


| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/classify` | Classify any prompt as safe/unsafe |
| `POST` | `/chat` | Classify, then forward to LLM if safe |
| `GET` | `/health` | Health check |

---

## Configuration

All thresholds and model settings are in `config.yaml`:

```yaml
thresholds:
  injection: 0.5    
  toxic: 0.4       

ensemble:
  scale:
    injection: 1.0  
    toxic: 0.85    
  threshold: 0.4    
```
