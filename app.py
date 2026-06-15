"""
app.py -- Support Integrity Auditor (SIA) - Streamlit Dashboard
================================================================
Interactive web UI for mismatch detection on customer support tickets.

Features:
  - Single-ticket form input with real-time inference
  - Batch CSV upload with full pipeline processing
  - Evidence Dossier viewer per flagged ticket
  - Priority Mismatch Dashboard (distribution, types, top signals)
  - Severity Delta Heatmap across categories and channels

Usage:
    streamlit run app.py
"""

import json
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st

warnings.filterwarnings("ignore")

# Import SIA inference components from predict.py
sys.path.insert(0, str(Path(__file__).parent))
from predict import (
    SIAPredictor,
    build_evidence_dossier,
    fuse_text,
    compute_nlp_urgency,
    score_to_severity,
    SEV_ORDER,
)


# =============================================================================
# CONFIG & CONSTANTS
# =============================================================================

MODEL_DIR = Path(r"C:\Users\Rudra\Desktop\Mars\models\final")
SEVERITY_COLORS = {
    "Low": "#4CAF50",
    "Medium": "#FF9800",
    "High": "#F44336",
    "Critical": "#9C27B0",
}

MISMATCH_TYPE_COLORS = {
    "Hidden Crisis": "#E53935",
    "False Alarm": "#FB8C00",
}


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="SIA - Support Integrity Auditor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .main { font-family: 'Inter', sans-serif; }

    .sia-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .sia-header h1 {
        margin: 0 0 0.3rem 0;
        font-weight: 700;
        font-size: 2rem;
        letter-spacing: -0.5px;
    }
    .sia-header p {
        margin: 0;
        opacity: 0.8;
        font-weight: 300;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card h3 {
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    .metric-card p {
        margin: 0.3rem 0 0 0;
        font-size: 0.8rem;
        color: #6c757d;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .dossier-card {
        background: #fff;
        border: 1px solid #e0e0e0;
        border-left: 4px solid #E53935;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    .dossier-card.consistent {
        border-left-color: #4CAF50;
    }

    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
    }
    .badge-crisis { background: #E53935; }
    .badge-alarm  { background: #FB8C00; }
    .badge-ok     { background: #4CAF50; }

    .evidence-json {
        background: #1e1e1e;
        color: #d4d4d4;
        padding: 1rem;
        border-radius: 8px;
        font-family: 'Cascadia Code', 'Fira Code', monospace;
        font-size: 0.8rem;
        max-height: 500px;
        overflow-y: auto;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f0f2f6, #e3e6ea);
        border-radius: 10px;
        padding: 0.8rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# MODEL LOADING (cached)
# =============================================================================

@st.cache_resource
def load_model():
    """Load the SIA predictor (cached across reruns)."""
    try:
        from predict import SIAPredictor
        return SIAPredictor(MODEL_DIR)
    except Exception:
        from predict import HeuristicPredictor
        return HeuristicPredictor(MODEL_DIR)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def process_single_ticket(
    predictor: SIAPredictor,
    subject: str,
    description: str,
    category: str,
    channel: str,
    priority: str,
    ticket_id: str = "LIVE-001",
) -> dict:
    """Run inference on a single ticket and return result with dossier."""
    fused = fuse_text(channel, category, subject, description)
    preds = predictor.predict_batch([fused])
    pred = preds[0]

    result = {
        "ticket_id": ticket_id,
        "predicted_label": pred["predicted_label"],
        "predicted_class": "mismatch" if pred["predicted_label"] == 1 else "consistent",
        "confidence": pred["confidence"],
        "prob_mismatch": pred["prob_mismatch"],
        "dossier": None,
    }

    if pred["predicted_label"] == 1:
        result["dossier"] = build_evidence_dossier(
            ticket_id=ticket_id,
            subject=subject,
            description=description,
            category=category,
            channel=channel,
            assigned_priority=priority,
            prediction=pred,
        )

    return result


def process_batch_csv(predictor: SIAPredictor, df: pd.DataFrame) -> list[dict]:
    """Process a DataFrame of tickets and return list of result dicts."""
    # Detect columns
    col_map = {}
    for col in df.columns:
        cl = col.strip().lower().replace(" ", "_")
        if cl in ("ticket_id", "ticketid"):
            col_map["ticket_id"] = col
        elif cl in ("ticket_subject", "subject"):
            col_map["subject"] = col
        elif cl in ("ticket_description", "description"):
            col_map["description"] = col
        elif cl in ("issue_category", "ticket_type", "category"):
            col_map["category"] = col
        elif cl in ("ticket_channel", "channel"):
            col_map["channel"] = col
        elif cl in ("priority_level", "ticket_priority", "priority", "assigned_priority"):
            col_map["priority"] = col

    def safe_get(row, key, default="Unknown"):
        if key in col_map:
            val = row.get(col_map[key], default)
            return str(val).strip() if pd.notna(val) else default
        return default

    # Fuse texts
    fused_texts = []
    for _, row in df.iterrows():
        fused_texts.append(fuse_text(
            safe_get(row, "channel", "Unknown"),
            safe_get(row, "category", "Unknown"),
            safe_get(row, "subject", ""),
            safe_get(row, "description", ""),
        ))

    # Run inference
    predictions = predictor.predict_batch(fused_texts, batch_size=16)

    # Build results
    results = []
    for idx, (_, row) in enumerate(df.iterrows()):
        pred = predictions[idx]
        ticket_id = safe_get(row, "ticket_id", f"ROW-{idx}")
        priority = safe_get(row, "priority", "Unknown")

        result = {
            "ticket_id": ticket_id,
            "subject": safe_get(row, "subject", ""),
            "category": safe_get(row, "category", "Unknown"),
            "channel": safe_get(row, "channel", "Unknown"),
            "assigned_priority": priority,
            "predicted_label": pred["predicted_label"],
            "predicted_class": "mismatch" if pred["predicted_label"] == 1 else "consistent",
            "confidence": pred["confidence"],
            "prob_mismatch": pred["prob_mismatch"],
            "dossier": None,
        }

        if pred["predicted_label"] == 1:
            result["dossier"] = build_evidence_dossier(
                ticket_id=ticket_id,
                subject=safe_get(row, "subject", ""),
                description=safe_get(row, "description", ""),
                category=safe_get(row, "category", "Unknown"),
                channel=safe_get(row, "channel", "Unknown"),
                assigned_priority=priority,
                prediction=pred,
            )

        results.append(result)

    return results


# =============================================================================
# DASHBOARD COMPONENTS
# =============================================================================

def render_header():
    st.markdown("""
    <div class="sia-header">
        <h1>&#128737; Support Integrity Auditor</h1>
        <p>AI-powered priority mismatch detection for customer support tickets</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 🟢 SIA Engine: Active")


def render_metrics_row(results: list[dict]):
    """Display key summary metrics."""
    total = len(results)
    mismatches = sum(1 for r in results if r["predicted_label"] == 1)
    consistent = total - mismatches
    mismatch_rate = (mismatches / max(total, 1)) * 100

    hidden = sum(
        1 for r in results
        if r.get("dossier") and r["dossier"]["mismatch_type"] == "Hidden Crisis"
    )
    false_alarm = sum(
        1 for r in results
        if r.get("dossier") and r["dossier"]["mismatch_type"] == "False Alarm"
    )
    avg_conf = np.mean([r["confidence"] for r in results]) * 100 if results else 0

    cols = st.columns(6)
    metrics = [
        (f"{total:,}", "Total Tickets"),
        (f"{mismatches:,}", "Mismatches Flagged"),
        (f"{consistent:,}", "Consistent"),
        (f"{mismatch_rate:.1f}%", "Mismatch Rate"),
        (f"{hidden:,}", "Hidden Crises"),
        (f"{avg_conf:.1f}%", "Avg Confidence"),
    ]
    for col, (val, label) in zip(cols, metrics):
        with col:
            st.markdown(
                f'<div class="metric-card"><h3>{val}</h3><p>{label}</p></div>',
                unsafe_allow_html=True,
            )


def render_mismatch_dashboard(results: list[dict]):
    """Priority Mismatch Dashboard with distribution charts."""
    mismatched = [r for r in results if r["predicted_label"] == 1]
    if not mismatched:
        st.info("No mismatches detected in this batch.")
        return

    st.markdown("### Priority Mismatch Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        # Mismatch type distribution
        st.markdown("#### Mismatch Type Distribution")
        types = [r["dossier"]["mismatch_type"] for r in mismatched if r.get("dossier")]
        type_df = pd.DataFrame({"Mismatch Type": types})
        type_counts = type_df["Mismatch Type"].value_counts().reset_index()
        type_counts.columns = ["Mismatch Type", "Count"]
        st.bar_chart(type_counts.set_index("Mismatch Type"))

    with col2:
        # Flagged tickets by assigned priority
        st.markdown("#### Flagged Tickets by Assigned Priority")
        priorities = [r.get("assigned_priority", "Unknown") for r in mismatched]
        pri_df = pd.DataFrame({"Priority": priorities})
        pri_counts = pri_df["Priority"].value_counts().reset_index()
        pri_counts.columns = ["Priority", "Count"]
        st.bar_chart(pri_counts.set_index("Priority"))

    # Top contributing signals
    st.markdown("#### Top Contributing Crisis Keywords")
    all_keywords = []
    for r in mismatched:
        if r.get("dossier") and r["dossier"].get("feature_evidence"):
            for kw in r["dossier"]["feature_evidence"].get("matched_crisis_keywords", []):
                if not kw["negated"]:
                    all_keywords.append(kw["keyword"])

    if all_keywords:
        kw_series = pd.Series(all_keywords)
        kw_counts = kw_series.value_counts().head(15).reset_index()
        kw_counts.columns = ["Keyword", "Frequency"]
        st.bar_chart(kw_counts.set_index("Keyword"))
    else:
        st.info("No crisis keywords found in flagged tickets.")


def render_severity_heatmap(results: list[dict]):
    """Severity Delta Heatmap across categories and channels."""
    mismatched = [r for r in results if r["predicted_label"] == 1 and r.get("dossier")]
    if not mismatched:
        st.info("No mismatch data available for heatmap.")
        return

    st.markdown("### Severity Delta Heatmap")
    st.caption("Average severity delta by ticket category and channel. "
               "Positive = under-prioritised (Hidden Crisis), Negative = over-prioritised (False Alarm).")

    rows = []
    for r in mismatched:
        d = r["dossier"]
        rows.append({
            "Category": r.get("category", d.get("feature_evidence", {}).get("category", "Unknown")),
            "Channel": r.get("channel", d.get("feature_evidence", {}).get("channel", "Unknown")),
            "Severity Delta": d["severity_delta"],
        })

    heatmap_df = pd.DataFrame(rows)

    if heatmap_df.empty:
        st.info("No data for heatmap.")
        return

    # Pivot for heatmap
    pivot = heatmap_df.pivot_table(
        values="Severity Delta",
        index="Category",
        columns="Channel",
        aggfunc="mean",
        fill_value=0,
    ).round(2)

    # Display as a styled dataframe (heatmap colors)
    def color_delta(val):
        if val > 0.5:
            return "background-color: #FFCDD2; color: #B71C1C; font-weight: 600;"
        elif val > 0:
            return "background-color: #FFF3E0; color: #E65100;"
        elif val < -0.5:
            return "background-color: #C8E6C9; color: #1B5E20; font-weight: 600;"
        elif val < 0:
            return "background-color: #E8F5E9; color: #2E7D32;"
        return "background-color: #F5F5F5; color: #757575;"

    styled = pivot.style.map(color_delta).format("{:.2f}")
    st.dataframe(styled, use_container_width=True)

    # Summary stats
    col1, col2, col3 = st.columns(3)
    with col1:
        worst_cat = heatmap_df.groupby("Category")["Severity Delta"].mean().idxmax()
        st.metric("Most Under-Prioritised Category", worst_cat)
    with col2:
        worst_ch = heatmap_df.groupby("Channel")["Severity Delta"].mean().idxmax()
        st.metric("Most Under-Prioritised Channel", worst_ch)
    with col3:
        avg_delta = heatmap_df["Severity Delta"].mean()
        st.metric("Avg Severity Delta", f"{avg_delta:+.2f}")


def render_dossier_viewer(results: list[dict]):
    """Expandable Evidence Dossier viewer for each flagged ticket."""
    mismatched = [r for r in results if r["predicted_label"] == 1 and r.get("dossier")]

    if not mismatched:
        st.info("No Evidence Dossiers to display.")
        return

    st.markdown(f"### Evidence Dossiers ({len(mismatched):,} flagged tickets)")

    for r in mismatched[:50]:  # cap at 50 for performance
        d = r["dossier"]
        badge_class = "badge-crisis" if d["mismatch_type"] == "Hidden Crisis" else "badge-alarm"
        badge_text = d["mismatch_type"]

        with st.expander(
            f"{d['ticket_id']} | {d['assigned_priority']} -> {d['inferred_severity']} "
            f"| {d['mismatch_type']} | Conf: {d['confidence']:.1%}",
            expanded=False,
        ):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Assigned Priority:** `{d['assigned_priority']}`")
                st.markdown(f"**Inferred Severity:** `{d['inferred_severity']}`")
            with col2:
                st.markdown(f"**Mismatch Type:** `{d['mismatch_type']}`")
                st.markdown(f"**Severity Delta:** `{d['severity_delta']:+d}`")
            with col3:
                st.markdown(f"**Confidence:** `{d['confidence']:.4f}`")
                kw_count = d["feature_evidence"].get("keyword_count", 0)
                st.markdown(f"**Active Keywords:** `{kw_count}`")

            # Constraint analysis
            st.markdown("**Constraint Analysis:**")
            for flag in d.get("constraint_analysis", []):
                st.warning(flag)

            # Keywords
            keywords = d["feature_evidence"].get("matched_crisis_keywords", [])
            if keywords:
                st.markdown("**Matched Crisis Keywords:**")
                kw_df = pd.DataFrame(keywords)
                st.dataframe(kw_df, use_container_width=True, hide_index=True)

            # Full JSON
            st.markdown("**Full Evidence Dossier (JSON):**")
            st.code(json.dumps(d, indent=2), language="json")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    render_header()

    # Sidebar
    with st.sidebar:
        st.markdown("## Settings")
        mode = st.radio(
            "Inference Mode",
            ["Single Ticket", "Batch CSV Upload"],
            index=0,
        )
        st.markdown("---")


    # Load model
    predictor = load_model()
    if predictor is None:
        st.error(
            "Could not load the SIA model. Please ensure the model files exist at:\n\n"
            f"`{MODEL_DIR}`\n\n"
            "Run `python train_pipeline.py` first to train and save the model."
        )
        return

    # =========================================================================
    # SINGLE TICKET MODE
    # =========================================================================
    if mode == "Single Ticket":
        st.markdown("### Analyse a Single Ticket")

        with st.form("single_ticket_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                ticket_id = st.text_input("Ticket ID", value="LIVE-001")
                channel = st.selectbox(
                    "Channel",
                    ["Email", "Chat", "Web Form", "Phone", "Social Media"],
                    index=0,
                )
            with col2:
                category = st.selectbox(
                    "Category",
                    ["Technical", "Billing", "Account", "General Inquiry", "Fraud"],
                    index=0,
                )
                priority = st.selectbox(
                    "Assigned Priority",
                    ["Low", "Medium", "High", "Critical"],
                    index=0,
                )
            with col3:
                subject = st.text_input("Subject", value="")

            description = st.text_area(
                "Description",
                height=120,
                placeholder="Enter the full ticket description here...",
            )

            submitted = st.form_submit_button(
                "Analyse Ticket", type="primary", use_container_width=True
            )

        if submitted:
            if not description.strip():
                st.warning("Please enter a ticket description.")
                return

            with st.spinner("Running SIA inference..."):
                result = process_single_ticket(
                    predictor, subject, description, category, channel, priority, ticket_id,
                )

            st.markdown("---")

            # Verdict
            if result["predicted_label"] == 1:
                mtype = result["dossier"]["mismatch_type"]
                st.error(
                    f"**MISMATCH DETECTED** | Type: **{mtype}** | "
                    f"Confidence: **{result['confidence']:.1%}**"
                )
            else:
                st.success(
                    f"**CONSISTENT** | No priority mismatch detected | "
                    f"Confidence: **{result['confidence']:.1%}**"
                )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Prediction", result["predicted_class"].upper())
            with col2:
                st.metric("Mismatch Probability", f"{result['prob_mismatch']:.2%}")
            with col3:
                st.metric("Confidence", f"{result['confidence']:.2%}")

            # Evidence Dossier
            if result.get("dossier"):
                st.markdown("### Evidence Dossier")
                d = result["dossier"]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Assigned Priority:** `{d['assigned_priority']}`")
                    st.markdown(f"**Inferred Severity:** `{d['inferred_severity']}`")
                    st.markdown(f"**Mismatch Type:** `{d['mismatch_type']}`")
                    st.markdown(f"**Severity Delta:** `{d['severity_delta']:+d}`")

                with col2:
                    st.markdown("**Constraint Analysis:**")
                    for flag in d.get("constraint_analysis", []):
                        st.warning(flag)

                keywords = d["feature_evidence"].get("matched_crisis_keywords", [])
                if keywords:
                    st.markdown("**Matched Crisis Keywords:**")
                    st.dataframe(pd.DataFrame(keywords), use_container_width=True, hide_index=True)

                st.markdown("**Full Evidence Dossier (JSON):**")
                st.code(json.dumps(d, indent=2), language="json")

    # =========================================================================
    # BATCH CSV MODE
    # =========================================================================
    else:
        st.markdown("### Batch CSV Upload")
        st.caption(
            "Upload a CSV with columns: `Ticket_ID`, `Ticket_Subject`, `Ticket_Description`, "
            "`Issue_Category`, `Ticket_Channel`, `Priority_Level`"
        )

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.info(f"Loaded {len(df):,} tickets from uploaded CSV")

            # Preview
            with st.expander("Preview uploaded data", expanded=False):
                st.dataframe(df.head(10), use_container_width=True)

            if st.button("Run Batch Inference", type="primary", use_container_width=True):
                with st.spinner(f"Processing {len(df):,} tickets..."):
                    results = process_batch_csv(predictor, df)

                st.session_state["batch_results"] = results

        # Display results if available
        if "batch_results" in st.session_state:
            results = st.session_state["batch_results"]

            st.markdown("---")

            # Summary metrics
            render_metrics_row(results)
            st.markdown("")

            # Tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs([
                "Dashboard",
                "Severity Heatmap",
                "Evidence Dossiers",
                "Export",
            ])

            with tab1:
                render_mismatch_dashboard(results)

            with tab2:
                render_severity_heatmap(results)

            with tab3:
                render_dossier_viewer(results)

            with tab4:
                st.markdown("### Export Results")

                # Build export data
                mismatched = [r for r in results if r.get("dossier")]
                export_data = {
                    "metadata": {
                        "total_tickets": len(results),
                        "mismatches_flagged": len(mismatched),
                        "mismatch_rate_pct": round(
                            len(mismatched) / max(len(results), 1) * 100, 2
                        ),
                    },
                    "dossiers": [r["dossier"] for r in mismatched],
                }

                export_json = json.dumps(export_data, indent=2, ensure_ascii=False)
                st.download_button(
                    label="Download Evidence Dossiers (JSON)",
                    data=export_json,
                    file_name="sia_evidence_dossiers.json",
                    mime="application/json",
                    use_container_width=True,
                )

                # CSV summary export
                summary_rows = []
                for r in results:
                    row = {
                        "ticket_id": r["ticket_id"],
                        "predicted_class": r["predicted_class"],
                        "confidence": r["confidence"],
                        "prob_mismatch": r["prob_mismatch"],
                    }
                    if r.get("dossier"):
                        row["mismatch_type"] = r["dossier"]["mismatch_type"]
                        row["assigned_priority"] = r["dossier"]["assigned_priority"]
                        row["inferred_severity"] = r["dossier"]["inferred_severity"]
                        row["severity_delta"] = r["dossier"]["severity_delta"]
                    summary_rows.append(row)

                summary_df = pd.DataFrame(summary_rows)
                csv_buffer = summary_df.to_csv(index=False)
                st.download_button(
                    label="Download Summary (CSV)",
                    data=csv_buffer,
                    file_name="sia_predictions_summary.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
