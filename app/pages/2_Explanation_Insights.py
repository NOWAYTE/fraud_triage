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
logs_df = load_claim_logs()
latest_assessment = st.session_state.get("latest_assessment")

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Model Interpretability Workspace</div>
        <h1>Explanation And Insights</h1>
        <p>
            Understand why a claim was assigned a particular fraud risk score by reviewing
            model outputs, SHAP-based feature contributions, and directional risk drivers.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

if logs_df.empty and latest_assessment is None:
    st.markdown(
        """
        <div class="section-card empty-state-card">
            <h3>No assessed claims available</h3>
            <p>
                Run a claim through the Fraud Risk Assessment workflow first, or load a previously
                assessed claim from history once records are available.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if not logs_df.empty:
    st.markdown(
        """
        <div class="section-card">
            <div class="card-label">Claim Selection</div>
            <h3>Load Claim For Explanation</h3>
            <p>Select a previously assessed claim record to inspect its fraud score and feature-level reasoning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    select_col, button_col = st.columns([3, 1])
    with select_col:
        claim_options = logs_df["claim_id"].astype(str).tolist()
        selected_claim_id = st.selectbox("Claim record", options=claim_options, label_visibility="visible")

    with button_col:
        st.markdown("<div style='height: 1.8rem;'></div>", unsafe_allow_html=True)
        load_clicked = st.button("Load Claim", use_container_width=True)

    if load_clicked:
        selected_row = logs_df.loc[logs_df["claim_id"].astype(str) == selected_claim_id].iloc[0]
        selected_input = restore_input_from_log_row(selected_row, metadata)
        assessment = assess_claim(pipeline, metadata, selected_input)
        assessment["claim_id"] = str(selected_row.get("claim_id", assessment["claim_id"]))
        assessment["timestamp"] = str(selected_row.get("timestamp", assessment["timestamp"]))

        st.session_state["latest_assessment"] = assessment
        latest_assessment = assessment
        st.success(f"Claim {selected_claim_id} loaded successfully for explanation.")

if latest_assessment:
    st.markdown(
        """
        <div class="result-section-header">
            <div class="card-label">Assessment Context</div>
            <h2>Risk Summary</h2>
            <p>Review the current fraud triage outcome before inspecting the model drivers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_risk_summary(latest_assessment)

    st.markdown(
        """
        <div class="result-section-header">
            <div class="card-label">Feature Attribution</div>
            <h2>SHAP Contribution View</h2>
            <p>Inspect the most influential features contributing to the final fraud risk score.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_shap_contributions(latest_assessment, limit=12)

    shap_df = pd.DataFrame(latest_assessment.get("shap_contributions", []))
    if not shap_df.empty:
        risk_up = shap_df.sort_values(by="Impact", ascending=False).head(3)
        risk_down = shap_df.sort_values(by="Impact", ascending=True).head(3)

        st.markdown("### Directional Risk Drivers")
        insight_col_1, insight_col_2 = st.columns(2, gap="large")

        with insight_col_1:
            st.markdown(
                """
                <div class="section-card">
                    <div class="card-label">Risk Escalation</div>
                    <h3>Signals Increasing Fraud Risk</h3>
                """,
                unsafe_allow_html=True,
            )
            for _, row in risk_up.iterrows():
                st.write(f"- {row['Feature']}: {row['Impact']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

        with insight_col_2:
            st.markdown(
                """
                <div class="section-card">
                    <div class="card-label">Risk Reduction</div>
                    <h3>Signals Reducing Fraud Risk</h3>
                """,
                unsafe_allow_html=True,
            )
            for _, row in risk_down.iterrows():
                st.write(f"- {row['Feature']}: {row['Impact']:.4f}")
            st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown(
        """
        <div class="section-card empty-state-card">
            <h3>No active assessment selected</h3>
            <p>
                Load a claim from history or complete a new fraud assessment to view model explanations
                and directional risk insights.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if hasattr(st, "page_link"):
        st.page_link("pages/1_Fraud_Risk_Assessment.py", label="Open Fraud Risk Assessment", icon="🧾")