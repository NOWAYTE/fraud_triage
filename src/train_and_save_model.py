import json
import joblib
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "fraud_oracle.csv"
MODELS_DIR = BASE_DIR / "models"
PIPELINE_PATH = MODELS_DIR / "fraud_triage_pipeline.joblib"
METADATA_PATH = MODELS_DIR / "fraud_triage_metadata.json"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Load dataset
df = pd.read_csv(DATA_PATH)

# Prepare data
X = df.drop(columns=["FraudFound_P", "PolicyNumber", "RepNumber"], errors="ignore")
y = df["FraudFound_P"]

cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

# Preprocessing
numeric_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = SkPipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols)
    ]
)

# Best model pipeline
best_pipeline = ImbPipeline(steps=[
    ("preprocess", preprocess),
    ("sampler", RandomOverSampler(random_state=42)),
    ("model", GradientBoostingClassifier(random_state=42))
])

# Train
best_pipeline.fit(X_train, y_train)

# Save full pipeline
joblib.dump(best_pipeline, PIPELINE_PATH)

# Save form metadata
metadata = {
    "categorical_columns": cat_cols,
    "numeric_columns": num_cols,
    "categorical_options": {
        col: sorted([str(v) for v in X[col].dropna().unique().tolist()])
        for col in cat_cols
    },
    "numeric_defaults": {
        col: float(X[col].median()) for col in num_cols
    }
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print("Saved:")
print(f"- {PIPELINE_PATH.relative_to(BASE_DIR)}")
print(f"- {METADATA_PATH.relative_to(BASE_DIR)}")