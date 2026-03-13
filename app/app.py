import streamlit as st

from common import apply_theme, compute_model_performance, load_metadata, validate_artifacts

st.set_page_config(page_title="Insurance Fraud Triage Workspace", layout="wide")

apply_theme()
validate_artifacts()

metadata = load_metadata()
metrics, summary = compute_model_performance()
latest_assessment = st.session_state.get("latest_assessment")

st.markdown(
    """
    <div class="hero-panel">
        <div class="hero-eyebrow">Insurance Operations · Fraud Decision Support</div>
        <h1>Insurance Fraud Triage Workspace</h1>
        <p>
            A structured analyst workspace for claim screening, fraud risk triage, model interpretation,
            and review monitoring. Use the actions below to begin an assessment or inspect prior outputs.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Portfolio Snapshot")
metric_col_1, metric_col_2, metric_col_3, metric_col_4 = st.columns(4)

with metric_col_1:
    st.metric("Model ROC AUC", f"{metrics['roc_auc']:.3f}")

with metric_col_2:
    st.metric("Fraud Base Rate", f"{summary['fraud_rate']:.2%}")

with metric_col_3:
    st.metric("Input Features", int(summary["feature_count"]))

with metric_col_4:
    st.metric("Recent Assessment", "Available" if latest_assessment else "None")

if hasattr(st, "page_link"):
    st.markdown("### Quick Actions")
    action_col_1, action_col_2, action_col_3, action_col_4 = st.columns(4)

    with action_col_1:
        st.page_link(
            "pages/1_Fraud_Risk_Assessment.py",
            label="Start New Assessment",
            icon="🧾",
        )

    with action_col_2:
        st.page_link(
            "pages/2_Explanation_Insights.py",
            label="View Explanations",
            icon="📊",
        )

    with action_col_3:
        st.page_link(
            "pages/3_Claim_History_Logs.py",
            label="Open Claim History",
            icon="🗂️",
        )

    with action_col_4:
        st.page_link(
            "pages/4_Model_Information.py",
            label="Check Model Info",
            icon="🧠",
        )

st.markdown("### Analyst Workflow")
left_col, right_col = st.columns([1.25, 1], gap="large")

with left_col:
    st.markdown(
        """
        <div class="section-card">
            <h3>How Analysts Use This Workspace</h3>
            <ol>
                <li>Complete the multi-step assessment workflow with claim, policy, and incident details.</li>
                <li>Review the fraud score, risk band, and triage recommendation.</li>
                <li>Inspect SHAP-based feature drivers to understand model reasoning.</li>
                <li>Track prior assessments in the claim history log for follow-up review.</li>
            </ol>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if latest_assessment:
        st.markdown(
            f"""
            <div class="highlight-card">
                <div class="card-label">Latest Assessed Claim</div>
                <h3>{latest_assessment['claim_id']}</h3>
                <p><strong>Risk Band:</strong> {latest_assessment['risk_band']}</p>
                <p><strong>Fraud Probability:</strong> {latest_assessment['fraud_probability']:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="highlight-card muted-card">
                <div class="card-label">Latest Assessed Claim</div>
                <h3>No recent assessment</h3>
                <p>Run a fraud risk assessment to generate the latest triage result for review.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right_col:
    st.markdown(
        """
        <div class="section-card">
            <h3>Workspace Map</h3>
            <p><strong>Fraud Risk Assessment</strong><br>Complete the guided claim scoring workflow.</p>
            <p><strong>Explanation And Insights</strong><br>Inspect SHAP drivers and feature-level impact.</p>
            <p><strong>Claim History Logs</strong><br>Review prior claims and assessment outcomes.</p>
            <p><strong>Model Information</strong><br>Inspect metadata, validation metrics, and monitoring context.</p>
            <p class="small-note">Use the Streamlit sidebar or quick action links above to move between pages.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="section-card">
            <h3>System Summary</h3>
            <p><strong>Records Available:</strong> {int(summary['record_count'])}</p>
            <p><strong>Categorical Fields:</strong> {len(metadata['categorical_columns'])}</p>
            <p><strong>Numeric Fields:</strong> {len(metadata['numeric_columns'])}</p>
            <p class="small-note">This summary provides context for model monitoring and current input coverage.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="status-strip">
        Workspace ready. Use the assessment page to begin a new fraud triage workflow.
    </div>
    """,
    unsafe_allow_html=True,
)