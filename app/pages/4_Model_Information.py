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
        <h1>Model Information</h1>
        <p>Reference view for model versioning, coverage, and current validation performance.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = st.columns(5)
metric_col_1.metric("Accuracy", f"{metrics['accuracy']:.3f}")
metric_col_2.metric("Precision", f"{metrics['precision']:.3f}")
metric_col_3.metric("Recall", f"{metrics['recall']:.3f}")
metric_col_4.metric("F1 Score", f"{metrics['f1']:.3f}")
metric_col_5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

st.markdown("## Dataset And Feature Coverage")

coverage_col_1, coverage_col_2, coverage_col_3 = st.columns(3)
coverage_col_1.metric("Total Records", int(summary["record_count"]))
coverage_col_2.metric("Fraud Positive Rate", f"{summary['fraud_rate']:.2%}")
coverage_col_3.metric("Total Input Features", int(summary["feature_count"]))

st.progress(
    float(summary["fraud_rate"]),
    text=f"Observed fraud prevalence in dataset: {summary['fraud_rate']:.2%}",
)

st.markdown("## Feature Inventory")

feature_col_1, feature_col_2 = st.columns(2)
with feature_col_1:
    st.markdown("### Numeric Fields")
    numeric_df = pd.DataFrame({"Field": metadata["numeric_columns"]})
    st.dataframe(numeric_df, use_container_width=True, hide_index=True)

with feature_col_2:
    st.markdown("### Categorical Fields")
    categorical_df = pd.DataFrame({"Field": metadata["categorical_columns"]})
    st.dataframe(categorical_df, use_container_width=True, hide_index=True)

st.markdown("## Artifact Details")


def describe_artifact(path: Path) -> dict:
    file_stats = path.stat()
    return {
        "Artifact": path.name,
        "Location": str(path.relative_to(APP_DIR.parents[0])),
        "Size KB": round(file_stats.st_size / 1024, 2),
        "Last Modified": datetime.fromtimestamp(file_stats.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
    }


artifact_df = pd.DataFrame([
    describe_artifact(PIPELINE_PATH),
    describe_artifact(METADATA_PATH),
])
st.dataframe(artifact_df, use_container_width=True, hide_index=True)

st.markdown(
    """
    <div class="section-card">
        <h3>Operational Notes</h3>
        <p>This model is intended for analyst triage support. Predictions should be reviewed with policy,
        claims, and compliance context before final decisions are made.</p>
    </div>
    """,
    unsafe_allow_html=True,
)
