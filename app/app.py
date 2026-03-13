import streamlit as st

from common import apply_theme, compute_model_performance, load_metadata, validate_artifacts

st.set_page_config(page_title="Insurance Fraud Triage Workspace", layout="wide")

apply_theme()
validate_artifacts()

metadata = load_metadata()
metrics, summary = compute_model_performance()

st.markdown(
    """
    <div class="hero-panel">
        <h1>Insurance Fraud Triage Workspace</h1>
        <p>Decision-support dashboard for insurance analysts. Use the page navigation to assess claims,
        inspect feature-level model drivers, review claim history, and monitor model health.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

latest_assessment = st.session_state.get("latest_assessment")

metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)
metric_col_1.metric("Model ROC AUC", f"{metrics['roc_auc']:.3f}")
metric_col_2.metric("Fraud Base Rate", f"{summary['fraud_rate']:.2%}")
metric_col_3.metric("Input Features", int(summary["feature_count"]))
metric_col_4.metric("Recent Assessment", "Available" if latest_assessment else "None")

st.markdown("## Workflow Overview")

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown(
        """
        <div class="section-card">
            <h3>How Analysts Use This App</h3>
            <ol>
                <li>Complete the multi-step assessment workflow with claim, policy, and incident details.</li>
                <li>Review risk score, risk band, and recommended action for triage.</li>
                <li>Inspect SHAP feature impact to understand drivers of the fraud score.</li>
                <li>Track assessed claims in the history log for follow-up analysis.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if latest_assessment:
        st.markdown(
            f"""
            <div class="section-card">
                <h3>Latest Assessed Claim</h3>
                <p><strong>Claim ID:</strong> {latest_assessment['claim_id']}</p>
                <p><strong>Risk Band:</strong> {latest_assessment['risk_band']}</p>
                <p><strong>Fraud Probability:</strong> {latest_assessment['fraud_probability']:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown(
        """
        <div class="section-card">
            <h3>Page Map</h3>
            <p><strong>Fraud Risk Assessment:</strong> Multi-step input and scoring workflow.</p>
            <p><strong>Explanation And Insights:</strong> SHAP impact and model reasoning view.</p>
            <p><strong>Claim History Logs:</strong> Historical triage records and quick filtering.</p>
            <p><strong>Model Information:</strong> Model metadata and validation metrics.</p>
            <p class="small-note">Tip: Use the Streamlit sidebar navigation to move between pages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

if hasattr(st, "page_link"):
    st.markdown("## Quick Actions")
    action_col_1, action_col_2 = st.columns(2)
    with action_col_1:
        st.page_link("pages/1_Fraud_Risk_Assessment.py", label="Open Fraud Risk Assessment")
        st.page_link("pages/2_Explanation_Insights.py", label="Open Explanation And Insights")
    with action_col_2:
        st.page_link("pages/3_Claim_History_Logs.py", label="Open Claim History Logs")
        st.page_link("pages/4_Model_Information.py", label="Open Model Information")

st.markdown(
    f"""
    <div class="small-note">
        Data records available for monitoring: {int(summary['record_count'])}.<br>
        Categorical fields: {len(metadata['categorical_columns'])}. Numeric fields: {len(metadata['numeric_columns'])}.
    </div>
    """,
    unsafe_allow_html=True,
)