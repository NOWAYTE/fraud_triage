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
    populate_form_state,
    restore_input_from_log_row,
    validate_artifacts,
)

st.set_page_config(page_title="Claim History Logs", layout="wide")

apply_theme()
validate_artifacts()

pipeline = load_pipeline()
metadata = load_metadata()
logs_df = load_claim_logs()

st.markdown(
    """
    <div class="hero-panel">
        <h1>Claim History Logs</h1>
        <p>Review prior assessments, filter by risk profile, and reopen records for detailed analysis.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if logs_df.empty:
    st.info("No claim logs available yet. Submit an assessment to start a history trail.")
    st.stop()

risk_options = sorted(logs_df["risk_band"].dropna().astype(str).unique().tolist())
selected_risks = st.multiselect("Risk band filter", options=risk_options, default=risk_options)
min_probability = st.slider("Minimum fraud probability", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

filtered_df = logs_df.copy()
if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_band"].astype(str).isin(selected_risks)]
filtered_df = filtered_df[filtered_df["fraud_probability"].astype(float) >= min_probability]

st.markdown("## Historical Assessments")
if filtered_df.empty:
    st.warning("No records match the selected filters.")
else:
    display_columns = [
        "timestamp",
        "claim_id",
        "fraud_probability",
        "risk_band",
        "recommended_action",
    ]
    existing_columns = [col for col in display_columns if col in filtered_df.columns]

    st.dataframe(
        filtered_df[existing_columns],
        use_container_width=True,
        hide_index=True,
    )

    claim_selection = st.selectbox(
        "Inspect claim details",
        options=filtered_df["claim_id"].astype(str).tolist(),
    )

    selected_row = filtered_df.loc[filtered_df["claim_id"].astype(str) == claim_selection].iloc[0]
    selected_input = restore_input_from_log_row(selected_row, metadata)

    st.markdown("## Claim Snapshot")
    snapshot_df = pd.DataFrame(
        {
            "Input Field": list(selected_input.keys()),
            "Value": list(selected_input.values()),
        }
    )
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

    action_col_1, action_col_2 = st.columns(2)

    if action_col_1.button("Open Claim In Assessment Form", use_container_width=True):
        populate_form_state(selected_input, metadata)
        st.success("Claim values loaded into the assessment form state. Open Fraud Risk Assessment from sidebar.")

    if action_col_2.button("Load Claim For Explanation", type="primary", use_container_width=True):
        assessment = assess_claim(pipeline, metadata, selected_input)
        assessment["claim_id"] = str(selected_row.get("claim_id", assessment["claim_id"]))
        assessment["timestamp"] = str(selected_row.get("timestamp", assessment["timestamp"]))
        st.session_state["latest_assessment"] = assessment
        st.success("Claim loaded into insights context. Open Explanation And Insights from sidebar.")
