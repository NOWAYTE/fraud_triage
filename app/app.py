import json
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(page_title="AI-Assisted Insurance Fraud Triage", layout="wide")

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "fraud_triage_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "fraud_triage_metadata.json"


def validate_artifacts():
    missing = [
        path.relative_to(BASE_DIR)
        for path in [PIPELINE_PATH, METADATA_PATH]
        if not path.exists()
    ]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        st.error(
            f"Missing model artifacts: {missing_list}. "
            "Run training first from the project root using: "
            "python src/train_and_save_model.py"
        )
        st.stop()


def apply_sklearn_pickle_compat_shims():
    try:
        import sklearn.compose._column_transformer as column_transformer
    except Exception:
        return

    if not hasattr(column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        column_transformer._RemainderColsList = _RemainderColsList

@st.cache_resource
def load_pipeline():
    try:
        return joblib.load(PIPELINE_PATH)
    except AttributeError as exc:
        if "_RemainderColsList" in str(exc) or "Can't get attribute" in str(exc):
            apply_sklearn_pickle_compat_shims()
            try:
                return joblib.load(PIPELINE_PATH)
            except Exception:
                pass
            st.error(
                "Model artifact is incompatible with the installed scikit-learn version. "
                "Use the same scikit-learn version as the environment where this model was trained "
                "(or re-export the model in a cross-version format)."
            )
            st.stop()
        raise

@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

validate_artifacts()

pipeline = load_pipeline()
metadata = load_metadata()

cat_cols = metadata["categorical_columns"]
num_cols = metadata["numeric_columns"]
cat_options = metadata["categorical_options"]
num_defaults = metadata["numeric_defaults"]

st.title("AI-Assisted Insurance Fraud Triage Dashboard")
st.caption("Prototype decision-support tool for insurance claim fraud screening")

st.markdown("### Claim Input Form")

with st.form("claim_form"):
    col1, col2 = st.columns(2)
    input_data = {}

    with col1:
        for col in num_cols[: len(num_cols)//2 + 1]:
            input_data[col] = st.number_input(
                label=col,
                value=float(num_defaults[col]),
                step=1.0
            )

        for col in cat_cols[: len(cat_cols)//2]:
            input_data[col] = st.selectbox(
                label=col,
                options=cat_options[col]
            )

    with col2:
        for col in num_cols[len(num_cols)//2 + 1:]:
            input_data[col] = st.number_input(
                label=col,
                value=float(num_defaults[col]),
                step=1.0
            )

        for col in cat_cols[len(cat_cols)//2:]:
            input_data[col] = st.selectbox(
                label=col,
                options=cat_options[col]
            )

    submitted = st.form_submit_button("Assess Fraud Risk")

def get_risk_band(prob):
    if prob < 0.30:
        return "Low"
    elif prob < 0.70:
        return "Medium"
    return "High"

def get_recommendation(prob):
    if prob < 0.30:
        return "Proceed with standard processing."
    elif prob < 0.70:
        return "Send for manual review."
    return "Escalate for priority fraud investigation."

if submitted:
    input_df = pd.DataFrame([input_data])

    fraud_prob = pipeline.predict_proba(input_df)[0, 1]
    pred_class = pipeline.predict(input_df)[0]
    risk_band = get_risk_band(fraud_prob)
    recommendation = get_recommendation(fraud_prob)

    st.markdown("## Assessment Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Fraud Probability", f"{fraud_prob:.2%}")
    c2.metric("Predicted Class", "Fraud" if pred_class == 1 else "Not Fraud")
    c3.metric("Risk Band", risk_band)

    if risk_band == "High":
        st.error(f"Recommended Action: {recommendation}")
    elif risk_band == "Medium":
        st.warning(f"Recommended Action: {recommendation}")
    else:
        st.success(f"Recommended Action: {recommendation}")

    st.markdown("## SHAP-Based Explanation")

    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    transformed = preprocess.transform(input_df)

    ohe = preprocess.named_transformers_["cat"].named_steps["onehot"]
    feature_names = np.concatenate([
        num_cols,
        ohe.get_feature_names_out(cat_cols)
    ])

    transformed_df = pd.DataFrame(transformed, columns=feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_df)

    
    shap_series = pd.Series(shap_values[0], index=feature_names)
    top_contrib = shap_series.abs().sort_values(ascending=False).head(10)

    st.markdown("### Top Contributing Factors")
    contrib_df = pd.DataFrame({
        "Feature": top_contrib.index,
        "Impact": shap_series[top_contrib.index].values
    })
    st.dataframe(contrib_df, use_container_width=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    top_contrib.sort_values().plot(kind="barh", ax=ax)
    ax.set_title("Top 10 SHAP Contributions")
    ax.set_xlabel("Absolute SHAP Impact")
    st.pyplot(fig)