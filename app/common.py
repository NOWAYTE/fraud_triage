import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fraud_oracle.csv"
LOGS_PATH = BASE_DIR / "data" / "claim_history.csv"
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "fraud_triage_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "fraud_triage_metadata.json"

FIELD_HELP_TEXT = {
    "Month": "Month the incident occurred.",
    "WeekOfMonth": "Week number of the incident month.",
    "DayOfWeek": "Day of week when incident occurred.",
    "MonthClaimed": "Month the claim was reported.",
    "WeekOfMonthClaimed": "Week number when claim was filed.",
    "DayOfWeekClaimed": "Day of week when claim was filed.",
    "AccidentArea": "Location context of the accident.",
    "PolicyType": "Policy package selected by the insured party.",
    "BasePolicy": "Core insurance policy type.",
    "Deductible": "Deductible amount configured on the policy.",
    "VehicleCategory": "Vehicle category for the insured asset.",
    "VehiclePrice": "Declared value range of the vehicle.",
    "NumberOfCars": "Number of insured vehicles in the household.",
    "AddressChange_Claim": "Recent address change before the claim.",
    "Days_Policy_Accident": "Time from policy start to accident.",
    "Days_Policy_Claim": "Time from policy start to claim filing.",
    "Age": "Claimant age.",
    "Sex": "Claimant sex.",
    "MaritalStatus": "Claimant marital status.",
    "Fault": "Who is identified at fault.",
    "DriverRating": "Internal driver risk rating.",
    "Make": "Vehicle manufacturer.",
    "PoliceReportFiled": "Whether a police report was filed.",
    "WitnessPresent": "Whether witness information is available.",
    "AgentType": "Claim handling agent type.",
    "NumberOfSuppliments": "Number of supporting supplements submitted.",
    "PastNumberOfClaims": "Claim history volume.",
    "AgeOfVehicle": "Vehicle age band.",
    "AgeOfPolicyHolder": "Policy holder age band.",
    "Year": "Claim record year.",
}


NUMERIC_BOUNDS = {
    "WeekOfMonth": (1.0, 5.0, 1.0),
    "WeekOfMonthClaimed": (1.0, 5.0, 1.0),
    "Age": (16.0, 100.0, 1.0),
    "Deductible": (100.0, 2000.0, 50.0),
    "DriverRating": (1.0, 4.0, 1.0),
    "Year": (1990.0, 2030.0, 1.0),
}


def trigger_rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()


def validate_artifacts() -> None:
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


def apply_sklearn_pickle_compat_shims() -> None:
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
                "Use the same scikit-learn version as the training environment "
                "or re-export the model artifact."
            )
            st.stop()
        raise


@st.cache_data
def load_metadata() -> Dict[str, object]:
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_claim_logs() -> pd.DataFrame:
    if not LOGS_PATH.exists():
        return pd.DataFrame()

    logs_df = pd.read_csv(LOGS_PATH)
    if "timestamp" in logs_df.columns:
        logs_df = logs_df.sort_values(by="timestamp", ascending=False, ignore_index=True)
    return logs_df


def append_claim_log(assessment: Dict[str, object], metadata: Dict[str, object]) -> None:
    LOGS_PATH.parent.mkdir(parents=True, exist_ok=True)

    row = {
        "timestamp": assessment["timestamp"],
        "claim_id": assessment["claim_id"],
        "fraud_probability": assessment["fraud_probability"],
        "predicted_class": assessment["predicted_class"],
        "risk_band": assessment["risk_band"],
        "recommended_action": assessment["recommended_action"],
    }

    for column in metadata["numeric_columns"]:
        row[column] = assessment["input_data"].get(column)

    for column in metadata["categorical_columns"]:
        row[column] = assessment["input_data"].get(column)

    write_header = not LOGS_PATH.exists()
    pd.DataFrame([row]).to_csv(
        LOGS_PATH,
        mode="a",
        index=False,
        header=write_header,
    )

    load_claim_logs.clear()


def humanize_field_name(field_name: str) -> str:
    field_name = field_name.replace("_", " ")
    field_name = re.sub(r"(?<!^)(?=[A-Z])", " ", field_name)
    return re.sub(r"\s+", " ", field_name).strip().title()


def format_feature_name(feature_name: str, categorical_columns: List[str]) -> str:
    for column in categorical_columns:
        prefix = f"{column}_"
        if feature_name.startswith(prefix):
            category_value = feature_name[len(prefix):]
            return f"{humanize_field_name(column)}: {category_value}"
    return humanize_field_name(feature_name)


def get_step_fields(metadata: Dict[str, object]) -> List[Tuple[str, List[str]]]:
    all_fields = metadata["numeric_columns"] + metadata["categorical_columns"]

    claim_details = [
        "Month",
        "WeekOfMonth",
        "DayOfWeek",
        "MonthClaimed",
        "WeekOfMonthClaimed",
        "DayOfWeekClaimed",
        "AccidentArea",
    ]

    policy_information = [
        "PolicyType",
        "BasePolicy",
        "Deductible",
        "VehicleCategory",
        "VehiclePrice",
        "NumberOfCars",
        "AddressChange_Claim",
        "Days_Policy_Accident",
        "Days_Policy_Claim",
    ]

    incident_details = [
        "Age",
        "Sex",
        "MaritalStatus",
        "Fault",
        "DriverRating",
        "Make",
        "PoliceReportFiled",
        "WitnessPresent",
        "AgentType",
        "NumberOfSuppliments",
        "PastNumberOfClaims",
        "AgeOfVehicle",
        "AgeOfPolicyHolder",
        "Year",
    ]

    used = set()
    steps = []

    for step_name, fields in [
        ("Claim Details", claim_details),
        ("Policy Information", policy_information),
        ("Incident Details", incident_details),
    ]:
        present_fields = [field for field in fields if field in all_fields]
        used.update(present_fields)
        steps.append((step_name, present_fields))

    remaining_fields = [field for field in all_fields if field not in used]
    steps[2][1].extend(remaining_fields)
    steps.append(("Review And Submit", []))
    return steps


def initialize_form_state(metadata: Dict[str, object]) -> None:
    if "claim_form_step" not in st.session_state:
        st.session_state["claim_form_step"] = 1

    if "latest_assessment" not in st.session_state:
        st.session_state["latest_assessment"] = None

    for column in metadata["numeric_columns"]:
        key = f"field_{column}"
        default_value = float(metadata["numeric_defaults"][column])
        if key not in st.session_state:
            st.session_state[key] = default_value

    for column in metadata["categorical_columns"]:
        key = f"field_{column}"
        options = metadata["categorical_options"][column]
        default_value = options[0] if options else ""
        if key not in st.session_state:
            st.session_state[key] = default_value


def collect_form_data(metadata: Dict[str, object]) -> Dict[str, object]:
    form_data: Dict[str, object] = {}

    for column in metadata["numeric_columns"]:
        form_data[column] = float(st.session_state.get(f"field_{column}", 0.0))

    for column in metadata["categorical_columns"]:
        form_data[column] = str(st.session_state.get(f"field_{column}", ""))

    return form_data


def restore_input_from_log_row(log_row: pd.Series, metadata: Dict[str, object]) -> Dict[str, object]:
    restored_input: Dict[str, object] = {}

    for column in metadata["numeric_columns"]:
        restored_input[column] = float(log_row.get(column, metadata["numeric_defaults"][column]))

    for column in metadata["categorical_columns"]:
        options = metadata["categorical_options"].get(column, [])
        value = str(log_row.get(column, options[0] if options else ""))
        if value not in options and options:
            value = options[0]
        restored_input[column] = value

    return restored_input


def populate_form_state(input_data: Dict[str, object], metadata: Dict[str, object]) -> None:
    for column in metadata["numeric_columns"]:
        st.session_state[f"field_{column}"] = float(input_data[column])

    for column in metadata["categorical_columns"]:
        st.session_state[f"field_{column}"] = str(input_data[column])


def render_input_fields(fields: List[str], metadata: Dict[str, object], columns: int = 2) -> None:
    ui_columns = st.columns(columns)

    for idx, field in enumerate(fields):
        target_col = ui_columns[idx % columns]
        label = humanize_field_name(field)
        help_text = FIELD_HELP_TEXT.get(field)

        with target_col:
            if field in metadata["numeric_columns"]:
                bounds = NUMERIC_BOUNDS.get(field)
                widget_params = {
                    "label": label,
                    "key": f"field_{field}",
                    "help": help_text,
                }
                if bounds:
                    min_value, max_value, step_value = bounds
                    widget_params["min_value"] = min_value
                    widget_params["max_value"] = max_value
                    widget_params["step"] = step_value
                else:
                    widget_params["step"] = 1.0
                st.number_input(**widget_params)
            else:
                options = metadata["categorical_options"][field]
                key = f"field_{field}"
                if st.session_state.get(key) not in options and options:
                    st.session_state[key] = options[0]
                st.selectbox(
                    label,
                    options=options,
                    key=key,
                    help=help_text,
                )


def validate_claim_data(input_data: Dict[str, object]) -> List[str]:
    errors: List[str] = []

    age = input_data.get("Age", 0)
    if age < 16 or age > 100:
        errors.append("Age must be between 16 and 100.")

    for week_field in ["WeekOfMonth", "WeekOfMonthClaimed"]:
        week_value = input_data.get(week_field, 0)
        if week_value < 1 or week_value > 5:
            errors.append(f"{humanize_field_name(week_field)} must be between 1 and 5.")

    deductible = input_data.get("Deductible", 0)
    if deductible < 100:
        errors.append("Deductible must be at least 100.")

    claim_month = str(input_data.get("MonthClaimed", ""))
    claim_day = str(input_data.get("DayOfWeekClaimed", ""))
    if claim_month == "0" or claim_day == "0":
        errors.append("Claim date fields should be set to a real month/day value.")

    return errors


def get_risk_band(probability: float) -> str:
    if probability < 0.30:
        return "Low"
    if probability < 0.70:
        return "Medium"
    return "High"


def get_recommendation(probability: float) -> str:
    if probability < 0.30:
        return "Proceed with standard processing."
    if probability < 0.70:
        return "Send for manual review."
    return "Escalate for priority fraud investigation."


def _normalize_shap_values(shap_values: object) -> np.ndarray:
    if isinstance(shap_values, list):
        if len(shap_values) > 1:
            return np.asarray(shap_values[1])
        return np.asarray(shap_values[0])

    shap_array = np.asarray(shap_values)
    if shap_array.ndim == 3 and shap_array.shape[-1] == 2:
        return shap_array[:, :, 1]
    if shap_array.ndim == 3 and shap_array.shape[0] == 2:
        return shap_array[1]
    return shap_array


def calculate_shap_table(
    pipeline,
    metadata: Dict[str, object],
    input_df: pd.DataFrame,
    top_n: int = 12,
) -> pd.DataFrame:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]

    transformed = preprocess.transform(input_df)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    categorical_columns = metadata["categorical_columns"]
    numeric_columns = metadata["numeric_columns"]

    onehot = preprocess.named_transformers_["cat"].named_steps["onehot"]
    transformed_feature_names = list(numeric_columns) + list(onehot.get_feature_names_out(categorical_columns))

    transformed_df = pd.DataFrame(transformed, columns=transformed_feature_names)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(transformed_df)
    normalized_shap = _normalize_shap_values(shap_values)

    if normalized_shap.ndim == 2:
        shap_vector = normalized_shap[0]
    elif normalized_shap.ndim == 1:
        shap_vector = normalized_shap
    else:
        shap_vector = normalized_shap.reshape(-1)

    if len(shap_vector) != len(transformed_feature_names):
        return pd.DataFrame(columns=["Feature", "Impact", "AbsoluteImpact"])

    shap_series = pd.Series(shap_vector, index=transformed_feature_names)
    top_features = shap_series.reindex(shap_series.abs().sort_values(ascending=False).index).head(top_n)

    shap_table = pd.DataFrame(
        {
            "Feature": [format_feature_name(name, categorical_columns) for name in top_features.index],
            "Impact": top_features.values,
            "AbsoluteImpact": np.abs(top_features.values),
        }
    )
    return shap_table


def assess_claim(pipeline, metadata: Dict[str, object], input_data: Dict[str, object]) -> Dict[str, object]:
    input_df = pd.DataFrame([input_data])

    fraud_probability = float(pipeline.predict_proba(input_df)[0, 1])
    predicted_class = int(pipeline.predict(input_df)[0])
    risk_band = get_risk_band(fraud_probability)
    recommended_action = get_recommendation(fraud_probability)

    try:
        shap_table = calculate_shap_table(pipeline, metadata, input_df, top_n=12)
        shap_records = shap_table.to_dict(orient="records")
    except Exception:
        shap_records = []

    current_time = datetime.now(timezone.utc)
    claim_id = f"CLM-{current_time.strftime('%Y%m%d-%H%M%S-%f')[:-3]}"

    return {
        "timestamp": current_time.replace(microsecond=0).isoformat(),
        "claim_id": claim_id,
        "fraud_probability": fraud_probability,
        "predicted_class": predicted_class,
        "risk_band": risk_band,
        "recommended_action": recommended_action,
        "input_data": input_data,
        "shap_contributions": shap_records,
    }


@st.cache_data(show_spinner=False)
def compute_model_performance() -> Tuple[Dict[str, float], Dict[str, float]]:
    pipeline = load_pipeline()
    dataset = load_dataset()

    features = dataset.drop(columns=["FraudFound_P", "PolicyNumber", "RepNumber"], errors="ignore")
    target = dataset["FraudFound_P"]

    _, x_test, _, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=42,
        stratify=target,
    )

    y_prob = pipeline.predict_proba(x_test)[:, 1]
    y_pred = pipeline.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
    }

    summary = {
        "record_count": float(len(dataset)),
        "fraud_rate": float(target.mean()),
        "feature_count": float(features.shape[1]),
    }

    return metrics, summary


def apply_theme() -> None:
    st.markdown(
        """
        <style>
            :root {
                --brand-navy: #10314b;
                --brand-slate: #2f4e68;
                --brand-blue: #2a6f97;
                --brand-mist: #eef4f8;
                --brand-ink: #1f2933;
                --risk-low: #1f7a5c;
                --risk-medium: #b7791f;
                --risk-high: #b0353a;
            }

            .stApp {
                background: linear-gradient(180deg, #f7fafc 0%, #edf2f7 100%);
                color: var(--brand-ink);
            }

            .block-container {
                padding-top: 1.2rem;
                padding-bottom: 2rem;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0f2c44 0%, #1e3e58 100%);
            }

            [data-testid="stSidebar"] * {
                color: #f8fbff;
            }

            .hero-panel {
                background: linear-gradient(135deg, #11314d 0%, #2b5c84 100%);
                border-radius: 14px;
                padding: 1.4rem;
                color: #f9fbfe;
                border: 1px solid rgba(255, 255, 255, 0.12);
                margin-bottom: 1rem;
                box-shadow: 0 10px 26px rgba(16, 49, 75, 0.14);
            }

            .section-card {
                background: #ffffff;
                border-radius: 12px;
                padding: 1rem;
                border: 1px solid #dbe5ee;
                margin-bottom: 0.8rem;
                box-shadow: 0 6px 18px rgba(15, 35, 56, 0.06);
            }

            .step-chip {
                border-radius: 10px;
                padding: 0.7rem 0.85rem;
                border: 1px solid #d2dee8;
                text-align: center;
                font-size: 0.9rem;
                background: #ffffff;
                min-height: 68px;
            }

            .step-chip.done {
                border-color: #9ac4db;
                background: #ecf5fa;
            }

            .step-chip.active {
                border-color: #2a6f97;
                background: #dfeef7;
                font-weight: 600;
            }

            .step-chip.pending {
                opacity: 0.86;
            }

            .risk-badge {
                display: inline-block;
                padding: 0.4rem 0.75rem;
                border-radius: 999px;
                font-weight: 700;
                margin: 0.4rem 0 0.75rem;
                border: 1px solid transparent;
            }

            .risk-low {
                background: #e6f7f1;
                color: var(--risk-low);
                border-color: #b6e3d4;
            }

            .risk-medium {
                background: #fff4dd;
                color: var(--risk-medium);
                border-color: #f0daa8;
            }

            .risk-high {
                background: #fde9ea;
                color: var(--risk-high);
                border-color: #efb5b8;
            }

            .action-panel {
                border-radius: 10px;
                padding: 0.9rem 1rem;
                border: 1px solid #dbe5ee;
                background: #f8fbff;
                margin-top: 0.5rem;
            }

            .action-panel.low {
                border-left: 6px solid var(--risk-low);
            }

            .action-panel.medium {
                border-left: 6px solid var(--risk-medium);
            }

            .action-panel.high {
                border-left: 6px solid var(--risk-high);
            }

            .small-note {
                font-size: 0.92rem;
                color: #52606d;
            }

            .stButton > button {
                border-radius: 8px;
                border: 1px solid #2a6f97;
            }

            .stButton > button[kind="primary"] {
                background: linear-gradient(135deg, #1f5f88 0%, #2f7fab 100%);
                color: #ffffff;
                border: 1px solid #1f5f88;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_risk_summary(assessment: Dict[str, object]) -> None:
    fraud_probability = float(assessment["fraud_probability"])
    predicted_class = int(assessment["predicted_class"])
    risk_band = str(assessment["risk_band"])
    recommendation = str(assessment["recommended_action"])

    col1, col2, col3 = st.columns(3)
    col1.metric("Fraud Probability", f"{fraud_probability:.2%}")
    col2.metric("Predicted Class", "Fraud" if predicted_class == 1 else "Not Fraud")
    col3.metric("Risk Band", risk_band)

    st.progress(fraud_probability, text=f"Fraud risk score: {fraud_probability:.2%}")

    css_class = "risk-low"
    if risk_band == "Medium":
        css_class = "risk-medium"
    if risk_band == "High":
        css_class = "risk-high"

    st.markdown(f"<div class='risk-badge {css_class}'>{risk_band} risk band</div>", unsafe_allow_html=True)
    st.markdown(
        f"<div class='action-panel {risk_band.lower()}'>"
        f"<strong>Recommended action:</strong><br>{recommendation}</div>",
        unsafe_allow_html=True,
    )


def render_shap_contributions(assessment: Dict[str, object], limit: int = 12) -> None:
    shap_df = pd.DataFrame(assessment.get("shap_contributions", []))
    if shap_df.empty:
        st.info("No SHAP contribution data is available for this record.")
        return

    shap_df = shap_df.head(limit).copy()
    st.dataframe(shap_df[["Feature", "Impact", "AbsoluteImpact"]], use_container_width=True, hide_index=True)

    plot_df = shap_df.iloc[::-1].copy()
    colors = ["#b0353a" if value > 0 else "#2a6f97" for value in plot_df["Impact"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(plot_df["Feature"], plot_df["Impact"], color=colors)
    ax.axvline(0, color="#1f2933", linewidth=1)
    ax.set_title("Top SHAP Feature Contributions")
    ax.set_xlabel("Signed impact toward fraud prediction")
    ax.set_ylabel("Feature")
    fig.tight_layout()

    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
