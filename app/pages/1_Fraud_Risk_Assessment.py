from pathlib import Path
import sys

import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from common import (
    append_claim_log,
    apply_theme,
    assess_claim,
    collect_form_data,
    get_step_fields,
    humanize_field_name,
    initialize_form_state,
    load_metadata,
    load_pipeline,
    render_input_fields,
    render_risk_summary,
    render_shap_contributions,
    trigger_rerun,
    validate_artifacts,
    validate_claim_data,
)

st.set_page_config(page_title="Fraud Risk Assessment", layout="wide")

apply_theme()
validate_artifacts()

pipeline = load_pipeline()
metadata = load_metadata()
initialize_form_state(metadata)

steps = get_step_fields(metadata)
step_count = len(steps)
current_step = int(st.session_state.get("claim_form_step", 1))
current_step = max(1, min(current_step, step_count))
st.session_state["claim_form_step"] = current_step

st.markdown(
    """
    <div class="hero-panel">
        <h1>Fraud Risk Assessment</h1>
        <p>Complete the guided workflow to submit a structured insurance claim for model triage.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

step_columns = st.columns(step_count)
for index, (step_name, _) in enumerate(steps, start=1):
    if index < current_step:
        chip_status = "done"
        state_text = "Completed"
    elif index == current_step:
        chip_status = "active"
        state_text = "Current"
    else:
        chip_status = "pending"
        state_text = "Pending"

    step_columns[index - 1].markdown(
        f"<div class='step-chip {chip_status}'><strong>{index}. {step_name}</strong><br>{state_text}</div>",
        unsafe_allow_html=True,
    )

st.progress(current_step / step_count, text=f"Step {current_step} of {step_count}")

step_name, step_fields = steps[current_step - 1]
st.markdown(f"## {step_name}")

if step_fields:
    render_input_fields(step_fields, metadata, columns=2)
else:
    review_input = collect_form_data(metadata)
    review_df = pd.DataFrame(
        {
            "Field": [humanize_field_name(key) for key in review_input.keys()],
            "Value": list(review_input.values()),
        }
    )

    st.markdown("Review all values before running model inference.")
    st.dataframe(review_df, use_container_width=True, hide_index=True)

    validation_errors = validate_claim_data(review_input)
    if validation_errors:
        st.warning("Please resolve validation issues before submission.")
        for item in validation_errors:
            st.error(item)

navigation_col_1, navigation_col_2, _ = st.columns([1, 1, 2])

if current_step > 1:
    if navigation_col_1.button("Previous", use_container_width=True):
        st.session_state["claim_form_step"] = current_step - 1
        trigger_rerun()

if current_step < step_count:
    if navigation_col_2.button("Next", type="primary", use_container_width=True):
        st.session_state["claim_form_step"] = current_step + 1
        trigger_rerun()
else:
    if navigation_col_2.button("Submit Assessment", type="primary", use_container_width=True):
        input_data = collect_form_data(metadata)
        validation_errors = validate_claim_data(input_data)

        if validation_errors:
            for item in validation_errors:
                st.error(item)
        else:
            with st.spinner("Running fraud triage model..."):
                assessment = assess_claim(pipeline, metadata, input_data)

            st.session_state["latest_assessment"] = assessment
            append_claim_log(assessment, metadata)
            st.success(
                "Assessment completed and logged. You can continue in Explanation And Insights for deeper analysis."
            )

latest_assessment = st.session_state.get("latest_assessment")
if latest_assessment:
    st.markdown("---")
    st.markdown("## Latest Assessment Result")
    render_risk_summary(latest_assessment)

    st.markdown("### Top Feature Signals")
    render_shap_contributions(latest_assessment, limit=8)

    if hasattr(st, "page_link"):
        st.page_link("pages/2_Explanation_Insights.py", label="Open Explanation And Insights")
