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
        <div class="hero-eyebrow">Operational Review Workspace</div>
        <h1>Claim History Logs</h1>
        <p>
            Review prior fraud assessments, filter claims by risk profile, and reopen records
            for reassessment or explanation.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if logs_df.empty:
    st.markdown(
        """
        <div class="section-card empty-state-card">
            <h3>No claim logs available</h3>
            <p>
                Submit a fraud assessment first to begin building a historical triage trail
                for operational review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

st.markdown(
    """
    <div class="section-card">
        <div class="card-label">Filters</div>
        <h3>Refine Historical Records</h3>
        <p>Filter historical assessments by risk band and minimum fraud probability.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

filter_col_1, filter_col_2 = st.columns([1.3, 1])

risk_options = sorted(logs_df["risk_band"].dropna().astype(str).unique().tolist())

with filter_col_1:
    selected_risks = st.multiselect(
        "Risk band filter",
        options=risk_options,
        default=risk_options,
    )

with filter_col_2:
    min_probability = st.slider(
        "Minimum fraud probability",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.01,
    )

filtered_df = logs_df.copy()
if selected_risks:
    filtered_df = filtered_df[filtered_df["risk_band"].astype(str).isin(selected_risks)]
filtered_df = filtered_df[filtered_df["fraud_probability"].astype(float) >= min_probability]

high_risk_count = 0
avg_probability = 0.0
if not filtered_df.empty:
    high_risk_count = int((filtered_df["risk_band"].astype(str) == "High").sum())
    avg_probability = float(filtered_df["fraud_probability"].astype(float).mean())

st.markdown("### History Summary")
summary_col_1, summary_col_2, summary_col_3, summary_col_4 = st.columns(4)
summary_col_1.metric("Total Records", int(len(logs_df)))
summary_col_2.metric("Filtered Records", int(len(filtered_df)))
summary_col_3.metric("High-Risk Claims", high_risk_count)
summary_col_4.metric("Average Fraud Probability", f"{avg_probability:.2%}")

st.markdown(
    """
    <div class="result-section-header">
        <div class="card-label">Assessment Table</div>
        <h2>Historical Assessments</h2>
        <p>Browse previously assessed claims and select a record for deeper review.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if filtered_df.empty:
    st.markdown(
        """
        <div class="section-card empty-state-card">
            <h3>No matching records</h3>
            <p>No claim records match the selected filters. Adjust the criteria to continue.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
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

    st.markdown(
        """
        <div class="section-card">
            <div class="card-label">Selected Record</div>
            <h3>Inspect Claim Details</h3>
            <p>Select a claim from the filtered records to review its captured inputs and available actions.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    claim_selection = st.selectbox(
        "Claim record",
        options=filtered_df["claim_id"].astype(str).tolist(),
    )

    selected_row = filtered_df.loc[
        filtered_df["claim_id"].astype(str) == claim_selection
    ].iloc[0]
    selected_input = restore_input_from_log_row(selected_row, metadata)

    snapshot_col_1, snapshot_col_2, snapshot_col_3 = st.columns(3)
    snapshot_col_1.metric("Claim ID", str(selected_row.get("claim_id", "N/A")))
    snapshot_col_2.metric(
        "Risk Band",
        str(selected_row.get("risk_band", "N/A")),
    )
    snapshot_col_3.metric(
        "Fraud Probability",
        f"{float(selected_row.get('fraud_probability', 0.0)):.2%}",
    )

    st.markdown("### Claim Snapshot")
    snapshot_df = pd.DataFrame(
        {
            "Input Field": list(selected_input.keys()),
            "Value": list(selected_input.values()),
        }
    )
    st.dataframe(snapshot_df, use_container_width=True, hide_index=True)

    action_col_1, action_col_2 = st.columns(2)

    if action_col_1.button("Reopen In Assessment Form", use_container_width=True):
        populate_form_state(selected_input, metadata)
        st.success(
            "Claim values loaded into the assessment workflow. Open Fraud Risk Assessment from the sidebar."
        )

    if action_col_2.button("Send To Explanation Workspace", type="primary", use_container_width=True):
        assessment = assess_claim(pipeline, metadata, selected_input)
        assessment["claim_id"] = str(selected_row.get("claim_id", assessment["claim_id"]))
        assessment["timestamp"] = str(selected_row.get("timestamp", assessment["timestamp"]))
        st.session_state["latest_assessment"] = assessment
        st.success(
            "Claim loaded into the explanation workspace. Open Explanation And Insights from the sidebar."
        )