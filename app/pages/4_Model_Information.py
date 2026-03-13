from datetime import datetime
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    METADATA_PATH,
    PIPELINE_PATH,
    apply_theme,
    compute_model_performance,
    load_metadata,
    validate_artifacts,
)

st.set_page_config(page_title="Model Information", layout="wide")

apply_theme()
validate_artifacts()

metadata = load_metadata()
metrics, summary = compute_model_performance()

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Model Reference And Monitoring</div>
        <h1>Model Information</h1>
        <p>
            Reference view for model validation performance, input coverage, stored artifacts,
            and operational use considerations within the fraud triage workflow.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Validation Performance")
metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
metric_col_1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
metric_col_2.metric("Precision", f"{metrics['precision']:.3f}")
metric_col_3.metric("Recall", f"{metrics['recall']:.3f}")
metric_col_4.metric("F1 Score", f"{metrics['f1']:.3f}")
metric_col_5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.markdown(
    """
    <div class="section-card">
        <div class="card-label">Dataset Profile</div>
        <h3>Dataset And Feature Coverage</h3>
        <p>
            Summary statistics describing the available data records, observed fraud prevalence,
            and total feature coverage used by the triage system.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

coverage_col_1, coverage_col_2, coverage_col_3 = st.columns(3)
coverage_col_1.metric("Total Records", int(summary["record_count"]))
coverage_col_2.metric("Fraud Positive Rate", f"{summary['fraud_rate']:.2%}")
coverage_col_3.metric("Total Input Features", int(summary["feature_count"]))

st.progress(
    float(summary["fraud_rate"]),
    text=f"Observed fraud prevalence in the reference dataset: {summary['fraud_rate']:.2%}",
)

st.markdown(
    """
    <div class="result-section-header">
        <div class="card-label">Input Schema</div>
        <h2>Feature Inventory</h2>
        <p>Review the numeric and categorical inputs expected by the fraud triage pipeline.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

feature_col_1, feature_col_2 = st.columns(2, gap="large")

with feature_col_1:
    st.markdown(
        """
        <div class="section-card">
            <h3>Numeric Fields</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    numeric_df = pd.DataFrame({"Field": metadata["numeric_columns"]})
    st.dataframe(numeric_df, use_container_width=True, hide_index=True)

with feature_col_2:
    st.markdown(
        """
        <div class="section-card">
            <h3>Categorical Fields</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    categorical_df = pd.DataFrame({"Field": metadata["categorical_columns"]})
    st.dataframe(categorical_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="result-section-header">
        <div class="card-label">Stored Files</div>
        <h2>Artifact Details</h2>
        <p>Inspect the current persisted model artifacts used by the application.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

def describe_artifact(path: Path) -> dict:
    file_stats = path.stat()
    return {
        "Artifact": path.name,
        "Location": str(path.relative_to(APP_DIR.parents[0])),
        "Size KB": round(file_stats.st_size / 1024, 2),
        "Last Modified": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }

artifact_df = pd.DataFrame(
    [
        describe_artifact(PIPELINE_PATH),
        describe_artifact(METADATA_PATH),
    ]
)
st.dataframe(artifact_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="section-card">
        <div class="card-label">Operational Guidance</div>
        <h3>Responsible Use Note</h3>
        <p>
            This model is intended to support analyst triage rather than replace human judgment.
            Predictions should always be interpreted alongside policy context, claims evidence,
            investigative workflow, and compliance requirements before final action is taken.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)