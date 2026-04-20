# 🔁 Prompt Repetition in LLMs — N-Repetition Experiment

> Replication and extension of [arXiv:2512.14982](https://arxiv.org/abs/2512.14982) — *"Does Repeating a Prompt Help?"*  
> Testing the effect of prompting a language model with N copies of the same query on accuracy, latency, and token efficiency.

---

## 📌 Overview

This project investigates whether **repeating a prompt N times** before generation improves LLM accuracy on reasoning and retrieval tasks. Building on the referenced paper's findings for N=2, this experiment extends the analysis to **N=1 through N=8**, across four diverse datasets and multiple metrics.

The core hypothesis: re-reading a question forces the model to attend to it more carefully, similar to how a human re-reads complex instructions before answering.

---

## 🏗️ Architecture

```
Notebook (Colab)
│
├── 1. Setup & Model Loading
│   └── TinyLlama-1.1B-Chat-v1.0 (HuggingFace Transformers)
│       ├── AutoTokenizer
│       └── AutoModelForCausalLM (device_map="auto", greedy decode)
│
├── 2. Inference Engine
│   ├── make_n_repetition(query, n)   ← Core prompt builder
│   │   ├── n=1  → query (baseline)
│   │   ├── n=2  → query + query
│   │   └── n>2  → query + "Let me repeat..." × (n-1)
│   ├── run_inference(prompt)         ← Tokenize → Generate → Decode
│   └── extract_letter(text)         ← MCQ answer parser (regex)
│
├── 3. Dataset Construction
│   ├── NameIndex   (SMALL,  30 samples)  — Positional retrieval
│   ├── ARC-Challenge (MEDIUM, 75 samples) — Science MCQ
│   ├── OpenBookQA  (LARGE, 150 samples)  — Commonsense + Science MCQ
│   └── MiddleMatch (MEDIUM, 75 samples)  — Relational positional retrieval
│
├── 4. Experiment Runner
│   └── run_repetition_experiment(dataset, rep_levels=[1..8])
│       └── Per sample: build prompt → infer → check → log metrics
│           Metrics: accuracy, latency, response length, input tokens, token efficiency
│
├── 5. Threshold & Peak Analysis
│   └── find_threshold(rep_results)
│       ├── Peak N       — highest accuracy repetition level
│       ├── Threshold N  — first N where acc drops >3% from peak
│       └── Saturation N — first N where gain <1% over previous
│
└── 6. Visualization & Reporting
    ├── Master Plot (matplotlib + seaborn)
    │   ├── Accuracy vs N (main result, annotated peaks)
    │   ├── Latency vs N
    │   ├── Input Token Length vs N
    │   ├── Token Efficiency vs N
    │   └── Accuracy Gain Heatmap (vs Baseline N=1)
    ├── Per-Dataset Case Study Report (printed)
    └── Metrics & Methods Glossary (printed)
```

---

## 📦 Datasets

| Dataset | Size | Task Type | Source |
|---|---|---|---|
| **NameIndex** | 30 samples | Positional name retrieval (find the 25th of 50 names) | Generated |
| **ARC-Challenge** | 75 samples | Science MCQ (4-choice) | HuggingFace `ai2_arc` |
| **OpenBookQA** | 150 samples | Commonsense + Science MCQ (4-choice) | HuggingFace `openbookqa` |
| **MiddleMatch** | 75 samples | Relational retrieval (find name between two anchors) | Generated |

---

## 📊 Metrics Tracked

| Metric | Description |
|---|---|
| **Accuracy** | Fraction of correct answers per repetition level |
| **Avg Latency** | Mean inference time per sample (seconds) |
| **Avg Input Tokens** | Mean tokenized prompt length (grows linearly with N) |
| **Token Efficiency** | Accuracy / Avg Input Tokens — quality per token spent |
| **Peak N** | Repetition level achieving maximum accuracy |
| **Threshold N** | First N after peak where accuracy drops >3% |
| **Saturation N** | First N where improvement over previous level is <1% |

---

## ⚙️ Key Design Decisions

- **Greedy decoding** (`do_sample=False`) for deterministic, reproducible outputs
- **Truncation** at 2048 tokens to stay within model context window
- `MAX_NEW_TOKENS = 20` — sufficient for single-letter MCQ or short name answers
- **Regex-based answer extraction** with three fallback patterns (explicit format → leading letter → first match)
- **Random seed fixed** (`seed=42`) across NumPy and Python random for reproducibility

---

## 🔁 Prompt Repetition Strategy

```
N=1: <query>                          ← Baseline (no repetition)
N=2: <query>\n<query>                 ← Paper's method
N=3: <query>\nLet me repeat that (repetition 2):\n<query>\nLet me repeat that one more time:\n<query>
...
N=k: k copies with escalating "Let me repeat..." prefixes
```

---

## 🛠️ Tech Stack

| Component | Library/Tool |
|---|---|
| Model | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` |
| Framework | `transformers`, `torch` |
| Datasets | `datasets` (HuggingFace Hub) |
| Visualization | `matplotlib`, `seaborn` |
| Utilities | `scikit-learn`, `scipy`, `numpy` |
| Environment | Google Colab (T4 GPU recommended) |

---

## 🚀 Getting Started

```bash
# 1. Install dependencies
pip install -q transformers accelerate datasets scipy matplotlib seaborn scikit-learn

# 2. Open the notebook in Google Colab and run cells sequentially:
#    Cell 1 → Install
#    Cell 2 → Load Model
#    Cell 3 → Inference Engine + N-Repetition Strategy
#    Cell 4 → Build Datasets
#    Cell 5 → Run Experiment (may take 30–90 min depending on GPU)
#    Cell 6 → Threshold Analysis
#    Cell 7 → Master Plot
#    Cell 8 → Case Study Report
#    Cell 9 → Glossary
```

> ⚠️ **Note:** Running all 4 datasets × 8 repetition levels on a T4 GPU takes approximately **60–90 minutes**. Reduce `SMALL_N`, `MEDIUM_N`, or `LARGE_N` for faster iteration.

---

## 📈 Expected Output

- **`n_repetition_master_plot.png`** — 5-panel visualization saved to working directory
- Printed threshold/peak analysis for each dataset
- Per-dataset case study with result tables
- Metrics glossary with plain-English explanations

---

## 📄 Reference

This work replicates and extends:

> *"Does Repeating a Prompt Help LLMs?"* — arXiv:2512.14982

---

## 📁 Repository Structure

```
.
├── Untitled3_(2).ipynb   ← Main experiment notebook
├── README.md             ← This file
└── n_repetition_master_plot.png  ← Generated after running experiment
```

---

## 👤 Author

**Gautam** — B.Tech CSE (AI), IIIT Ranchi  
ML Research Intern | Applied GenAI Engineer
