import json
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from matplotlib import pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

load_dotenv()

st.set_page_config(page_title="NexaAgent - AutoML EDA", layout="wide")

st.title("NexaAgent: LangChain + Streamlit AutoML Assistant")
st.caption("Upload a CSV, describe the task, and view EDA + model results.")


def get_llm() -> ChatGroq:
    model_name = st.session_state.get("llm_model", "llama-3.3-70b-versatile")
    return ChatGroq(model=model_name, temperature=0.2)


def infer_task_settings(task: str, df: pd.DataFrame) -> Dict[str, str]:
    if not os.getenv("GROQ_API_KEY"):
        return {}

    prompt = (
        "You are a data science assistant.\n"
        "Given the task and dataset columns, decide: target_col and task_type.\n"
        "task_type must be one of: classification, regression.\n"
        "Return a JSON object with keys: target_col, task_type.\n\n"
        f"Task: {task}\n"
        f"Columns: {list(df.columns)}\n"
        "Respond with JSON only."
    )
    llm = get_llm()
    response = llm.invoke(prompt)
    content = response.content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {}


def split_features_target(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )
    return preprocessor


def get_models(task_type: str) -> Dict[str, object]:
    if task_type == "regression":
        return {
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=300, random_state=42),
            "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
            "KNN Regressor": KNeighborsRegressor(n_neighbors=7),
            "Extra Trees Regressor": ExtraTreesRegressor(n_estimators=400, random_state=42),
        }
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM (RBF)": SVC(kernel="rbf"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "Naive Bayes": GaussianNB(),
        "Extra Trees": ExtraTreesClassifier(n_estimators=400, random_state=42),
    }


def train_model(
    df: pd.DataFrame,
    target_col: str,
    model_name: str,
    test_size: float,
    task_type: str,
) -> Tuple[float, object, np.ndarray, np.ndarray]:
    X, y = split_features_target(df, target_col)
    stratify_target = None
    if task_type == "classification":
        class_counts = y.value_counts(dropna=False)
        min_count = class_counts.min()
        if min_count >= 2:
            min_test = min_count * test_size
            min_train = min_count * (1 - test_size)
            if min_test >= 1 and min_train >= 1:
                stratify_target = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_target
    )

    preprocessor = build_preprocessor(X_train)
    model = get_models(task_type)[model_name]

    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    if task_type == "regression":
        r2 = float(r2_score(y_test, preds))
        return r2, clf, y_test.to_numpy(), preds
    acc = accuracy_score(y_test, preds)
    return acc, clf, y_test.to_numpy(), preds


def render_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, ax=ax, colorbar=False)
    st.pyplot(fig)


def render_score_bar(scores: Dict[str, float], task_type: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    names = list(scores.keys())
    values = list(scores.values())
    sns.barplot(x=values, y=names, hue=names, ax=ax, palette="viridis", legend=False)
    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 1)
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v * 100:.1f}%")
    st.pyplot(fig)


def render_task_summary(task_type: str, target_col: str) -> None:
    st.markdown("### Task Interpretation")
    st.write(f"Task type: **{task_type}**")
    st.write(f"Target column: **{target_col}**")


def render_progress_steps(steps: Dict[str, bool], placeholder: st.delta_generator.DeltaGenerator) -> None:
    lines = [f"- [{'x' if done else ' '}] {label}" for label, done in steps.items()]
    placeholder.markdown("\n".join(lines))


def fallback_task_settings(task: str, df: pd.DataFrame) -> Dict[str, str]:
    target_col = df.columns[-1]
    numeric_target = pd.api.types.is_numeric_dtype(df[target_col])
    keywords = ("price", "forecast", "prediction", "revenue", "sales", "demand", "cost")
    task_lower = task.lower()
    is_regression = numeric_target and any(k in task_lower for k in keywords)
    if numeric_target and df[target_col].nunique() > 15:
        is_regression = True
    return {
        "target_col": target_col,
        "task_type": "regression" if is_regression else "classification",
    }


uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

task_input = st.text_input("Task / Problem Statement", value="", placeholder="e.g., stock price prediction")

run_pipeline = st.button("Run Analysis")

if not run_pipeline:
    st.info("Upload a CSV and enter a task, then click Run Analysis.")
    st.stop()

if uploaded_file is None:
    st.warning("Upload a CSV to begin.")
    st.stop()

if not task_input.strip():
    st.warning("Enter a task to continue.")
    st.stop()

try:
    df = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Failed to read CSV: {exc}")
    st.stop()

st.subheader("Data Preview")
st.dataframe(df.head(20), use_container_width=True)

st.subheader("Column Summary")
col1, col2 = st.columns(2)
with col1:
    st.write("Shape:", df.shape)
    st.write("Missing values:")
    st.dataframe(df.isna().sum().to_frame("missing"))
    st.write("Duplicates:", df.duplicated().sum())
with col2:
    st.write("Dtypes:")
    st.dataframe(df.dtypes.astype(str).to_frame("dtype"))

st.divider()

if df.columns.size < 2:
    st.warning("Dataset needs at least 2 columns.")
    st.stop()

task_settings = infer_task_settings(task_input, df)
if not task_settings.get("target_col") or not task_settings.get("task_type"):
    task_settings = fallback_task_settings(task_input, df)
    if not os.getenv("GROQ_API_KEY"):
        st.info("GROQ_API_KEY not set. Using automatic heuristics for task and target selection.")

target_col = task_settings["target_col"]
task_type = task_settings["task_type"]
if target_col not in df.columns:
    target_col = df.columns[-1]

if task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_col]):
    task_type = "classification"

render_task_summary(task_type, target_col)

st.divider()

st.subheader("EDA Outputs")

numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

if numeric_cols:
    st.markdown("#### Numeric Summary")
    st.dataframe(df[numeric_cols].describe().T)

if task_type == "classification":
    st.markdown("#### Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=df[target_col], ax=ax)
    ax.set_xlabel(target_col)
    ax.set_ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    st.pyplot(fig)
else:
    st.markdown("#### Target Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[target_col], kde=True, ax=ax)
    ax.set_xlabel(target_col)
    st.pyplot(fig)

if len(numeric_cols) >= 2:
    st.markdown("#### Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(7, 5))
    corr = df[numeric_cols].corr(numeric_only=True)
    sns.heatmap(corr, cmap="viridis", ax=ax)
    st.pyplot(fig)

if numeric_cols:
    st.markdown("#### Numeric Feature Distributions")
    for col in numeric_cols[:6]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.histplot(df[col], kde=True, ax=ax)
        ax.set_title(col)
        st.pyplot(fig)

if categorical_cols:
    st.markdown("#### Categorical Feature Distributions")
    for col in categorical_cols[:6]:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        top_vals = df[col].value_counts().head(10)
        sns.barplot(x=top_vals.values, y=top_vals.index, hue=top_vals.index, ax=ax, palette="viridis", legend=False)
        ax.set_title(col)
        st.pyplot(fig)

st.divider()

st.subheader("Model Training Results")

progress_placeholder = st.empty()
progress_steps = {
    "Preparing data": False,
    "Training models": False,
    "Computing scores": False,
    "Rendering charts": False,
}
render_progress_steps(progress_steps, progress_placeholder)

progress_steps["Preparing data"] = True
render_progress_steps(progress_steps, progress_placeholder)

test_size = 0.2
scores: Dict[str, float] = {}
display_scores: Dict[str, float] = {}
best_model_name = None
best_score = None
best_preds = None
best_y_true = None

for model_name in get_models(task_type).keys():
    try:
        score, model, y_true, y_pred = train_model(df, target_col, model_name, test_size, task_type)
    except Exception as exc:
        st.warning(f"{model_name} failed: {exc}")
        continue
    scores[model_name] = score
    if task_type == "regression":
        display_scores[model_name] = max(0.0, min(1.0, score))
    else:
        display_scores[model_name] = max(0.0, min(1.0, score))
    if best_score is None or score > best_score:
        best_score = score
        best_model_name = model_name
        best_preds = y_pred
        best_y_true = y_true

progress_steps["Training models"] = True
render_progress_steps(progress_steps, progress_placeholder)

if task_type == "classification":
    st.write("Accuracy Scores (test split):")
    scores_df = pd.DataFrame.from_dict(display_scores, orient="index", columns=["accuracy"])
    scores_df["accuracy"] = scores_df["accuracy"].map(lambda v: f"{v * 100:.1f}%")
    st.dataframe(scores_df)
else:
    st.write("Accuracy Scores (R² on test split):")
    st.caption("R² is converted to a percent-style score; higher is better.")
    scores_df = pd.DataFrame.from_dict(display_scores, orient="index", columns=["r2_score"])
    scores_df["r2_score"] = scores_df["r2_score"].map(lambda v: f"{v * 100:.1f}%")
    st.dataframe(scores_df)

progress_steps["Computing scores"] = True
render_progress_steps(progress_steps, progress_placeholder)

render_score_bar(display_scores, task_type)

progress_steps["Rendering charts"] = True
render_progress_steps(progress_steps, progress_placeholder)

if best_model_name and best_preds is not None and best_y_true is not None:
    st.markdown(f"#### Best Model: {best_model_name}")
    if task_type == "classification":
        render_confusion_matrix(best_y_true, best_preds)
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=best_y_true, y=best_preds, ax=ax)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)
