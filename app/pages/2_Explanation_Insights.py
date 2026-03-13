from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    apply_theme,
    assess_claim,
    load_claim_logs,
    load_metadata,
    load_pipeline,
    render_risk_summary,
    render_shap_contributions,
    restore_input_from_log_row,
    validate_artifacts,
)

st.set_page_config(page_title="Explanation And Insights", layout="wide")

apply_theme()
validate_artifacts()

pipeline = load_pipeline()
metadata = load_metadata()

st.markdown(
    """
    <div class="hero-panel">
        <h1>Explanation And Insights</h1>
        <p>Understand why a claim was flagged by inspecting feature contributions and directional impact.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

logs_df = load_claim_logs()
latest_assessment = st.session_state.get("latest_assessment")

if logs_df.empty and latest_assessment is None:
    st.info("No assessed claims found yet. Run an assessment first on the Fraud Risk Assessment page.")

if not logs_df.empty:
    st.markdown("## Load Claim From History")
    claim_options = logs_df["claim_id"].astype(str).tolist()
    selected_claim_id = st.selectbox("Claim record", options=claim_options)

    if st.button("Load Selected Claim", use_container_width=False):
        selected_row = logs_df.loc[logs_df["claim_id"].astype(str) == selected_claim_id].iloc[0]
        selected_input = restore_input_from_log_row(selected_row, metadata)
        assessment = assess_claim(pipeline, metadata, selected_input)
        assessment["claim_id"] = str(selected_row.get("claim_id", assessment["claim_id"]))
        assessment["timestamp"] = str(selected_row.get("timestamp", assessment["timestamp"]))

        st.session_state["latest_assessment"] = assessment
        latest_assessment = assessment
        st.success(f"Loaded claim {selected_claim_id} for explanation.")

if latest_assessment:
    st.markdown("## Risk Summary")
    render_risk_summary(latest_assessment)

    st.markdown("## SHAP Contribution View")
    render_shap_contributions(latest_assessment, limit=12)

    shap_df = pd.DataFrame(latest_assessment.get("shap_contributions", []))
    if not shap_df.empty:
        risk_up = shap_df.sort_values(by="Impact", ascending=False).head(3)
        risk_down = shap_df.sort_values(by="Impact", ascending=True).head(3)

        insight_col_1, insight_col_2 = st.columns(2)
        with insight_col_1:
            st.markdown("### Main Drivers Increasing Fraud Score")
            for _, row in risk_up.iterrows():
                st.write(f"- {row['Feature']}: {row['Impact']:.4f}")

        with insight_col_2:
            st.markdown("### Main Drivers Decreasing Fraud Score")
            for _, row in risk_down.iterrows():
                st.write(f"- {row['Feature']}: {row['Impact']:.4f}")
else:
    st.info("No active assessment selected. Load a claim from history or run a new assessment.")
