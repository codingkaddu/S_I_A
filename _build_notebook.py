"""
_build_notebook.py
Programmatically generates notebook.ipynb so all string escaping is handled
correctly by the json module rather than raw string embedding.
Run: python _build_notebook.py
"""
import json
from pathlib import Path


def cell(cell_type: str, source: str, **kwargs):
    if cell_type == "markdown":
        return {
            "cell_type": "markdown",
            "metadata": {},
            "source": source,
        }
    else:
        return {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source,
        }


cells = []

# ── Intro markdown ────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
# Support Integrity Auditor (SIA) — Stage 1: Self-Supervised Pseudo-Labeling Pipeline

This notebook implements a **self-supervised pseudo-labeling pipeline** for detecting priority mismatches in customer support tickets.  
Two independent signals are engineered without using human labels, then fused to derive an objective `inferred_severity`.  
Finally, that inferred label is compared against the human-assigned `Priority_Level` to produce a binary `is_mismatch` flag for downstream model training.

---
**Pipeline Stages**

| Stage | Description |
|-------|-------------|
| A | Data Loading & Column Audit |
| B | Signal 1 — Rule-Based NLP Urgency Scoring |
| C | Signal 2 — Resolution-Time Anomaly Proxy |
| D | Pseudo-Label Fusion & Thresholding |
| E | Mismatch Flagging |
| F | Ablation Data Extraction |
"""))

# ── 0: Imports ─────────────────────────────────────────────────────────────────
cells.append(cell("markdown", "## 0 · Imports & Global Configuration"))

cells.append(cell("code", """\
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")
tqdm.pandas()

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(r"C:\\Users\\Rudra\\Desktop\\Mars\\archive")
PRIMARY  = DATA_DIR / "customer_support_tickets.csv"
ENHANCED = DATA_DIR / "enhanced_customer_support_data.csv"

# ── Signal Fusion Weights ─────────────────────────────────────────────────────
W_NLP = 0.55   # weight for Rule-Based NLP Urgency (Signal 1)
W_RES = 0.45   # weight for Resolution-Time Proxy  (Signal 2)

# ── Severity Thresholds (composite score in [0, 1]) ───────────────────────────
THRESH = {
    "Critical": 0.75,
    "High":     0.50,
    "Medium":   0.25,
    # < 0.25 => Low
}

print("Configuration loaded.")
print(f"  Signal weights -> NLP: {W_NLP} | Resolution: {W_RES}")
"""))

# ── Stage A ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage A · Data Loading & Column Audit

Load the primary CSV and verify that the core columns required for the pipeline are present.
Column names in this dataset use underscores (e.g. `Ticket_Subject`, `Priority_Level`) which are
normalised to a consistent lowercase mapping for the rest of the notebook.
"""))

cells.append(cell("code", """\
# ── Load primary dataset ──────────────────────────────────────────────────────
df = pd.read_csv(PRIMARY)
print(f"Loaded  : {PRIMARY.name}")
print(f"Shape   : {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"\\nRaw columns:\\n{df.columns.tolist()}")

# ── Normalise column names ─────────────────────────────────────────────────────
df.columns = [c.strip() for c in df.columns]

# ── Column mapping (actual -> canonical internal names) ───────────────────────
COL_MAP = {
    "Ticket_Subject":        "ticket_subject",
    "Ticket_Description":    "ticket_description",
    "Priority_Level":        "ticket_priority",
    "Resolution_Time_Hours": "resolution_time",
    "Issue_Category":        "ticket_type",
}
df.rename(columns=COL_MAP, inplace=True)

REQUIRED = list(COL_MAP.values())
missing  = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns after mapping: {missing}")

print("\\nAll required columns present after mapping:")
for orig, canon in COL_MAP.items():
    print(f"  {orig:30s} -> {canon}")

# ── Quick sanity check ────────────────────────────────────────────────────────
print(f"\\nPriority_Level distribution:")
print(df["ticket_priority"].value_counts())
print(f"\\nTicket_Type distribution:")
print(df["ticket_type"].value_counts())
print(f"\\nResolution_Time_Hours - summary:")
print(df["resolution_time"].describe())
"""))

cells.append(cell("code", """\
# ── Cross-check with enhanced file ───────────────────────────────────────────
df_enh = pd.read_csv(ENHANCED)
df_enh.columns = [c.strip() for c in df_enh.columns]

print(f"Enhanced file shape  : {df_enh.shape}")
print(f"Enhanced file columns: {df_enh.columns.tolist()}")
print("\\nColumn audit complete - proceeding with primary dataset.")

# Show first 3 rows of core columns
df[REQUIRED].head(3)
"""))

# ── Stage B ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage B · Signal 1 — Rule-Based NLP Urgency Scoring

A lexical urgency scorer is built using three tiers of crisis indicators:

| Tier | Weight | Examples |
|------|--------|----------|
| **Critical** | 1.0 | breach, fraud, ransomware, legal, lawsuit |
| **High** | 0.7 | broken, fail, crash, outage, corrupted |
| **Escalation** | 0.4 | immediate, asap, urgent, escalate |

Basic **negation handling** suppresses matches that are preceded by negation tokens
within a 3-word window (e.g. a phrase like 'not broken' does not trigger the 'broken' keyword).
"""))

cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 1 : Rule-Based NLP Urgency Scorer
# ─────────────────────────────────────────────────────────────────────────────

# Tiered keyword dictionaries with associated urgency scores
CRISIS_KEYWORDS = {
    # Tier 1: Critical / Legal / Security
    "breach":          1.0,
    "data breach":     1.0,
    "leak":            1.0,
    "fraud":           1.0,
    "ransomware":      1.0,
    "malware":         1.0,
    "hacked":          1.0,
    "unauthorized":    1.0,
    "illegal":         1.0,
    "lawsuit":         1.0,
    "legal":           0.95,
    "compliance":      0.90,
    "gdpr":            0.90,
    "stolen":          0.95,
    "compromised":     0.95,
    # Tier 2: Operational Failures
    "broken":          0.75,
    "fail":            0.75,
    "failed":          0.75,
    "failure":         0.75,
    "crash":           0.80,
    "crashed":         0.80,
    "outage":          0.85,
    "down":            0.70,
    "unavailable":     0.70,
    "corrupted":       0.80,
    "lost":            0.65,
    "deleted":         0.70,
    "missing":         0.60,
    "error":           0.55,
    "critical":        0.85,
    "severe":          0.80,
    # Tier 3: Escalation / Time Pressure
    "immediate":       0.50,
    "immediately":     0.50,
    "asap":            0.50,
    "urgent":          0.55,
    "urgently":        0.55,
    "escalate":        0.45,
    "escalation":      0.45,
    "deadline":        0.40,
    "overdue":         0.45,
    "not working":     0.65,
    "blocked":         0.50,
    "stuck":           0.45,
    # Tier 4: Financial Impact
    "overcharged":     0.60,
    "double charge":   0.65,
    "payment failed":  0.70,
    "billing error":   0.65,
}

# Negation tokens
NEGATION_TOKENS = {
    "not", "no", "never", "without", "neither", "nor",
    "don't", "doesn't", "didn't", "won't", "wasn't",
    "isn't", "aren't", "haven't", "hasn't", "wouldn't",
    "couldn't", "shouldn't", "cannot",
}

NEGATION_WINDOW = 3   # words before keyword to check for negation


def _is_negated(tokens: list, match_start_idx: int) -> bool:
    window_start = max(0, match_start_idx - NEGATION_WINDOW)
    window = tokens[window_start:match_start_idx]
    return any(t in NEGATION_TOKENS for t in window)


def compute_nlp_urgency_score(text: str) -> float:
    \"\"\"
    Score the urgency of a support ticket text using CRISIS_KEYWORDS.
    Applies longest-match-first for multi-word phrases and negation suppression.
    Returns float in [0.0, 1.0].
    \"\"\"
    if not isinstance(text, str) or not text.strip():
        return 0.0

    lowered = text.lower()
    cleaned = re.sub(r"[^\\w\\s']", " ", lowered)
    tokens  = cleaned.split()

    accumulated_score = 0.0
    matched_spans = []

    # Sort by phrase length descending -> multi-word phrases matched first
    sorted_kw = sorted(CRISIS_KEYWORDS.items(), key=lambda kv: len(kv[0].split()), reverse=True)

    for phrase, weight in sorted_kw:
        phrase_tokens = phrase.split()
        phrase_len    = len(phrase_tokens)

        for i in range(len(tokens) - phrase_len + 1):
            window = tokens[i : i + phrase_len]
            if window == phrase_tokens:
                span = (i, i + phrase_len)
                # Skip if any token already consumed
                if any(s[0] <= j < s[1] for j in range(*span) for s in matched_spans):
                    continue
                if _is_negated(tokens, i):
                    continue
                accumulated_score += weight
                matched_spans.append(span)

    # tanh normalisation: smooth squash to [0, 1]
    score = float(np.tanh(accumulated_score / 2.0))
    return round(score, 6)


# ── Apply to dataset ──────────────────────────────────────────────────────────
print("Computing NLP urgency scores ...")
df["text_combined"] = (
    df["ticket_subject"].fillna("") + " " + df["ticket_description"].fillna("")
)
df["signal1_nlp_urgency"] = df["text_combined"].progress_apply(compute_nlp_urgency_score)

print(f"\\nSignal 1 (NLP Urgency) - descriptive stats:")
print(df["signal1_nlp_urgency"].describe())

print("\\nSample rows with highest NLP urgency:")
df[["ticket_subject", "ticket_priority", "signal1_nlp_urgency"]] \\
    .sort_values("signal1_nlp_urgency", ascending=False) \\
    .head(5)
"""))

# ── Stage C ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage C · Signal 2 — Resolution-Time Anomaly Proxy

A resolution-time anomaly score is derived **per ticket type** (relative to that category's
distribution), not as an absolute value.

**Rationale:** A 72-hour resolution is normal for a complex `Technical` ticket but is objectively
anomalous for a `General Inquiry`. Flagging absolute outliers within each category captures cases
where the system under-estimated priority.

Method: **Modified Z-score** (Iglewicz & Hoaglin, 1993) using the median and MAD, which is robust
to skewed resolution-time distributions.
"""))

cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL 2 : Resolution-Time Anomaly Proxy
# ─────────────────────────────────────────────────────────────────────────────

def compute_resolution_stats(df_: pd.DataFrame, time_col: str, group_col: str) -> pd.DataFrame:
    \"\"\"Compute per-group resolution-time statistics (median, MAD, IQR, upper fence).\"\"\"
    def stats(g):
        vals = g[time_col].dropna()
        med  = vals.median()
        mad  = (vals - med).abs().median()
        q1   = vals.quantile(0.25)
        q3   = vals.quantile(0.75)
        iqr  = q3 - q1
        return pd.Series({
            "count":       len(vals),
            "mean":        vals.mean(),
            "median":      med,
            "mad":         mad,
            "q1":          q1,
            "q3":          q3,
            "iqr":         iqr,
            "upper_fence": q3 + 1.5 * iqr,
        })

    return df_.groupby(group_col).apply(stats).reset_index()


resolution_stats = compute_resolution_stats(df, "resolution_time", "ticket_type")
print("Resolution-time statistics per ticket type:")
print(resolution_stats.to_string(index=False))
"""))

cells.append(cell("code", """\
def compute_resolution_anomaly_score(row: pd.Series, stats_map: dict) -> float:
    \"\"\"
    Compute a [0, 1] anomaly score using the modified Z-score within ticket type.
    Modified Z-score = 0.6745 * (x - median) / MAD
    Only positive deviations (longer than expected) indicate under-prioritisation.
    \"\"\"
    ttype = row["ticket_type"]
    rt    = row["resolution_time"]

    if pd.isna(rt) or ttype not in stats_map:
        return 0.0

    s   = stats_map[ttype]
    med = s["median"]
    mad = s["mad"]

    if mad == 0:
        mod_z = 0.0
    else:
        mod_z = 0.6745 * (rt - med) / mad

    mod_z = max(0.0, mod_z)

    # Normalise: mod_z of 3.5 -> score ~0.78; 7+ -> approaches 1
    score = float(np.tanh(mod_z / 4.5))
    return round(score, 6)


# Build lookup dict for vectorised apply
stats_map = resolution_stats.set_index("ticket_type").to_dict(orient="index")

print("Computing resolution anomaly scores ...")
df["signal2_resolution_anomaly"] = df.progress_apply(
    compute_resolution_anomaly_score, axis=1, stats_map=stats_map
)

print(f"\\nSignal 2 (Resolution Anomaly) - descriptive stats:")
print(df["signal2_resolution_anomaly"].describe())

print("\\nSample rows with highest resolution anomaly:")
df[["ticket_type", "resolution_time", "ticket_priority", "signal2_resolution_anomaly"]] \\
    .sort_values("signal2_resolution_anomaly", ascending=False) \\
    .head(5)
"""))

# ── Stage D ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage D · Pseudo-Label Fusion & Thresholding

The two signals are combined into a single **composite severity score**:

    composite = w_nlp * Signal1 + w_res * Signal2

The composite score (bounded in [0, 1]) is then thresholded into four severity buckets:

| Composite Score | inferred_severity |
|-----------------|-------------------|
| >= 0.75         | Critical          |
| >= 0.50         | High              |
| >= 0.25         | Medium            |
| < 0.25          | Low               |
"""))

cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# STAGE D : Pseudo-Label Fusion
# ─────────────────────────────────────────────────────────────────────────────

df["composite_score"] = (
    W_NLP * df["signal1_nlp_urgency"] +
    W_RES * df["signal2_resolution_anomaly"]
).clip(0.0, 1.0)


def score_to_severity(score: float) -> str:
    \"\"\"Map a composite [0, 1] score to a discrete severity label.\"\"\"
    if score >= THRESH["Critical"]:
        return "Critical"
    elif score >= THRESH["High"]:
        return "High"
    elif score >= THRESH["Medium"]:
        return "Medium"
    else:
        return "Low"


df["inferred_severity"] = df["composite_score"].apply(score_to_severity)

print("Composite score - descriptive stats:")
print(df["composite_score"].describe())

print("\\nInferred severity distribution:")
print(df["inferred_severity"].value_counts())

print("\\nSample fusion results:")
df[["ticket_priority", "signal1_nlp_urgency", "signal2_resolution_anomaly",
    "composite_score", "inferred_severity"]].head(8)
"""))

# ── Stage E ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage E · Mismatch Flagging

The human-assigned `ticket_priority` (Low / Medium / High / Critical) is compared to the
objectively derived `inferred_severity`.

- If they **agree**    -> `is_mismatch = 0`
- If they **disagree** -> `is_mismatch = 1`

This binary flag is the **pseudo-label** used for downstream supervised training in Stage 2.
"""))

cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# STAGE E : Mismatch Flagging
# ─────────────────────────────────────────────────────────────────────────────

# Normalise the human priority labels to Title Case
df["ticket_priority_norm"] = df["ticket_priority"].str.strip().str.title()

# Binary mismatch flag
df["is_mismatch"] = (df["ticket_priority_norm"] != df["inferred_severity"]).astype(int)

print("is_mismatch - binary class distribution:")
mismatch_counts = df["is_mismatch"].value_counts()
print(mismatch_counts)
mismatch_rate = df["is_mismatch"].mean() * 100
print(f"\\nOverall mismatch rate : {mismatch_rate:.2f}%")

# Direction of mismatch: under-prioritised vs over-prioritised
SEV_ORDER = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
df["priority_rank"]  = df["ticket_priority_norm"].map(SEV_ORDER)
df["inferred_rank"]  = df["inferred_severity"].map(SEV_ORDER)
df["mismatch_delta"] = df["inferred_rank"] - df["priority_rank"]

under_pri = (df["mismatch_delta"] > 0).sum()
over_pri  = (df["mismatch_delta"] < 0).sum()
print(f"\\nMismatch breakdown:")
print(f"  Under-prioritised tickets (inferred > assigned) : {under_pri:,}")
print(f"  Over-prioritised tickets  (inferred < assigned) : {over_pri:,}")

# Cross-tab
print("\\nConfusion matrix - Human Priority vs Inferred Severity:")
pd.crosstab(
    df["ticket_priority_norm"],
    df["inferred_severity"],
    rownames=["Human Priority"],
    colnames=["Inferred Severity"]
)
"""))

# ── Stage F ───────────────────────────────────────────────────────────────────
cells.append(cell("markdown", """\
---
## Stage F · Ablation Data Extraction

Two ablation checks before handoff to Stage 2:

1. **Pairwise Signal Agreement** — percentage of tickets where Signal 1 and Signal 2 produce
   the same discrete severity bucket independently.

2. **Binary Class Distribution** — verify that `is_mismatch` imbalance is quantified so that
   Stage 2 can select an appropriate resampling or loss-weighting strategy.
"""))

cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# STAGE F : Ablation Data Extraction
# ─────────────────────────────────────────────────────────────────────────────

# Convert each raw signal to a discrete severity bucket independently
df["severity_from_signal1"] = df["signal1_nlp_urgency"].apply(score_to_severity)
df["severity_from_signal2"] = df["signal2_resolution_anomaly"].apply(score_to_severity)

# ── Pairwise Agreement ────────────────────────────────────────────────────────
pairwise_agree = (df["severity_from_signal1"] == df["severity_from_signal2"]).mean() * 100
print("=" * 55)
print("  PAIRWISE SIGNAL AGREEMENT (Signal 1 vs Signal 2)")
print("=" * 55)
print(f"  Agreement    : {pairwise_agree:.2f}%")
print(f"  Disagreement : {100 - pairwise_agree:.2f}%")
print()

# ── Binary Class Distribution ─────────────────────────────────────────────────
print("=" * 55)
print("  BINARY CLASS DISTRIBUTION  (is_mismatch)")
print("=" * 55)
dist  = df["is_mismatch"].value_counts().sort_index()
total = len(df)
for label, count in dist.items():
    lname = "No mismatch" if label == 0 else "Mismatch   "
    print(f"  is_mismatch = {label}  ({lname}) : {count:>6,}  ({count/total*100:5.2f}%)")
print()

majority = dist.max()
minority = dist.min()
ratio    = majority / minority if minority > 0 else float("inf")
print(f"  Imbalance ratio (majority : minority) ~= {ratio:.2f} : 1")
if ratio > 10:
    print("  WARNING: High imbalance - consider SMOTE or class-weighted loss in Stage 2.")
else:
    print("  OK: Imbalance manageable - standard class-weighting should suffice.")
print()

# Per-type mismatch breakdown
print("Mismatch rate per Ticket Type:")
print(
    df.groupby("ticket_type")["is_mismatch"]
    .agg(["sum", "count", "mean"])
    .rename(columns={"sum": "mismatches", "count": "total", "mean": "mismatch_rate"})
    .assign(mismatch_rate=lambda x: (x["mismatch_rate"] * 100).round(2))
    .sort_values("mismatch_rate", ascending=False)
    .to_string()
)
"""))

# ── Export ─────────────────────────────────────────────────────────────────────
cells.append(cell("code", """\
# ─────────────────────────────────────────────────────────────────────────────
# Export pseudo-labelled dataset for Stage 2
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_COLS = [
    "ticket_subject",
    "ticket_description",
    "text_combined",
    "ticket_type",
    "ticket_priority",
    "resolution_time",
    "signal1_nlp_urgency",
    "signal2_resolution_anomaly",
    "composite_score",
    "inferred_severity",
    "is_mismatch",
]

out_path = DATA_DIR.parent / "pseudo_labelled_tickets.csv"
df[OUTPUT_COLS].to_csv(out_path, index=False)

print(f"Pseudo-labelled dataset saved to:")
print(f"  {out_path}")
print(f"  Shape: {df[OUTPUT_COLS].shape[0]:,} rows x {len(OUTPUT_COLS)} columns")
print(f"\\nStage 1 pipeline complete. Ready for Stage 2 (Supervised Training).")
"""))

cells.append(cell("markdown", """\
---
*End of Stage 1 — Support Integrity Auditor (SIA)*
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────────
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(r"C:\Users\Rudra\Desktop\Mars\notebook.ipynb")
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"notebook.ipynb written to {out}")

# Quick validation
with open(out, encoding="utf-8") as f:
    nb_check = json.load(f)

code_cells = [c for c in nb_check["cells"] if c["cell_type"] == "code"]
print(f"Validated: {len(nb_check['cells'])} total cells, {len(code_cells)} code cells")

# Compile each code cell
import ast
for i, c in enumerate(code_cells):
    src = "".join(c["source"])
    compile(src, f"cell_{i}", "exec")
print("All code cells compile without syntax errors.")
