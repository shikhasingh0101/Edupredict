"""
EduPredict — Streamlit Product UI

Presentation layer for the EduPredict ML system.

Architecture:
    Streamlit UI
        ↓
    FastAPI backend
        ↓
    Production ML pipeline
"""

from pathlib import Path
import json
import os

import pandas as pd
import requests
import streamlit as st


# ============================================================
# CONFIG
# ============================================================

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="EduPredict",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# THEME / CSS
# ============================================================

st.html("""
<style>

:root {
    --navy: #0f172a;
    --navy-2: #172554;
    --blue: #2563eb;
    --blue-light: #eff6ff;
    --green: #16a34a;
    --green-light: #f0fdf4;
    --orange: #ea580c;
    --purple: #7c3aed;
    --text: #172033;
    --muted: #64748b;
    --border: #e2e8f0;
    --background: #f8fafc;
}

/* Main background */

.stApp {
    background:
        radial-gradient(
            circle at 10% 0%,
            rgba(37, 99, 235, 0.07),
            transparent 25%
        ),
        radial-gradient(
            circle at 90% 10%,
            rgba(124, 58, 237, 0.05),
            transparent 25%
        ),
        var(--background);
}

/* Content width */

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* Sidebar */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #0f172a 0%,
        #172554 100%
    );
}

section[data-testid="stSidebar"] * {
    color: #f8fafc;
}

/* Sidebar radio */

section[data-testid="stSidebar"] [data-testid="stRadio"] label {
    border-radius: 10px;
}

/* Buttons */

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 10px;
    border: 0;
    font-weight: 700;
    min-height: 44px;
}

/* Inputs */

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    border-radius: 9px;
}

/* Dataframe */

[data-testid="stDataFrame"] {
    border-radius: 12px;
    overflow: hidden;
}

/* Metric */

[data-testid="stMetric"] {
    background: white;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 4px 16px rgba(15, 23, 42, 0.04);
}

/* Tabs */

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
}

/* Hide excessive top padding */

div[data-testid="stToolbar"] {
    visibility: hidden;
}

</style>
""")


# ============================================================
# DATA HELPERS
# ============================================================

def api_get(endpoint: str, timeout: int = 5):
    response = requests.get(
        f"{API_URL}{endpoint}",
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def api_post(endpoint: str, payload: dict, timeout: int = 15):
    response = requests.post(
        f"{API_URL}{endpoint}",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def load_json(path: str):
    file_path = Path(path)

    if not file_path.exists():
        return None

    try:
        with file_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return None


def load_csv(path: str):
    file_path = Path(path)

    if not file_path.exists():
        return None

    try:
        return pd.read_csv(file_path)
    except Exception:
        return None


# ============================================================
# API STATUS
# ============================================================

def get_api_status():
    try:
        health = api_get("/health", timeout=3)

        if health.get("status") == "healthy":
            return True

        return False

    except Exception:
        return False


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("# 🎓 EduPredict")

    st.caption(
        "AI Student Performance Intelligence"
    )

    st.divider()

    st.markdown("### Navigation")

    page = st.radio(
        "Go to",
        [
            "🏠 Dashboard",
            "🎯 Predict",
            "📊 Model Performance",
            "⚙️ MLOps",
            "ℹ️ About",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### System")

    if get_api_status():
        st.success("● API Online")
    else:
        st.error("● API Offline")

    st.caption(
        f"Backend\n`{API_URL}`"
    )

    st.divider()

    st.caption(
        "EduPredict v1.0\n"
        "Feature Engineering + MLOps"
    )


# ============================================================
# COMMON HEADER
# ============================================================

def render_header(
    title="EduPredict",
    subtitle="AI-powered student exam performance prediction",
):

    st.title(f"🎓 {title}")

    st.caption(subtitle)

    st.divider()


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    render_header(
        "EduPredict",
        "AI Student Exam Performance Prediction & Academic Performance Intelligence",
    )

    # --------------------------------------------------------
    # Hero
    # --------------------------------------------------------

    with st.container(border=True):

        left, right = st.columns(
            [2.4, 1],
            gap="large",
        )

        with left:

            st.markdown("### 🚀 End-to-End ML Decision Support")

            st.write(
                "EduPredict transforms student academic, behavioral "
                "and learning-environment factors into an estimated "
                "exam score using a production-ready machine learning pipeline."
            )

            st.info(
                "Streamlit → FastAPI → Production ML Pipeline"
            )

        with right:

            st.metric(
                "Model Status",
                "Production",
            )

            st.metric(
                "API",
                "Online" if get_api_status() else "Offline",
            )

    st.write("")

    # --------------------------------------------------------
    # KPIs
    # --------------------------------------------------------

    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Dataset Records",
        "6,607",
    )

    c2.metric(
        "Input Features",
        "19",
    )

    c3.metric(
        "Test RMSE",
        "1.569",
    )

    c4.metric(
        "Test R²",
        "0.814",
    )

    st.write("")

    # --------------------------------------------------------
    # Architecture
    # --------------------------------------------------------

    st.subheader("How EduPredict Works")

    architecture = st.columns(5)

    architecture_steps = [
        ("📊", "Student Data", "Raw factors"),
        ("🧹", "Data Pipeline", "Clean + validate"),
        ("⚙️", "Features", "Engineer + select"),
        ("🧠", "ML Model", "Predict score"),
        ("🚀", "API + UI", "Serve prediction"),
    ]

    for col, (icon, title, description) in zip(
        architecture,
        architecture_steps,
    ):

        with col:

            with st.container(border=True):

                st.markdown(f"## {icon}")

                st.markdown(
                    f"**{title}**"
                )

                st.caption(description)

    st.write("")

    # --------------------------------------------------------
    # Why architecture
    # --------------------------------------------------------

    st.subheader("Why This Architecture?")

    a, b, c = st.columns(3)

    with a:

        with st.container(border=True):

            st.markdown("### 🔒 Consistent Predictions")

            st.write(
                "The preprocessing pipeline is stored with the model, "
                "so training and serving apply the same transformations."
            )

    with b:

        with st.container(border=True):

            st.markdown("### ⚡ API-First Serving")

            st.write(
                "Streamlit handles presentation while FastAPI owns "
                "prediction serving."
            )

    with c:

        with st.container(border=True):

            st.markdown("### 🔁 Reproducible Delivery")

            st.write(
                "Git, DVC, MLflow, Docker and GitHub Actions connect "
                "experimentation with production delivery."
            )

    st.write("")

    # --------------------------------------------------------
    # Project facts
    # --------------------------------------------------------

    st.subheader("Project Snapshot")

    snapshot_left, snapshot_right = st.columns(2)

    with snapshot_left:

        with st.container(border=True):

            st.markdown("### Dataset")

            st.write(
                "Student Performance Factors"
            )

            st.write(
                "**Rows:** 6,607  \n"
                "**Original input features:** 19  \n"
                "**Target:** `Exam_Score`  \n"
                "**Task:** Regression"
            )

    with snapshot_right:

        with st.container(border=True):

            st.markdown("### Feature Engineering")

            st.write(
                "**Study × Attendance**  \n"
                "Hours Studied × Attendance"
            )

            st.write(
                "**Study × Sleep**  \n"
                "Hours Studied × Sleep Hours"
            )


# ============================================================
# PREDICT
# ============================================================

elif page == "🎯 Predict":

    render_header(
        "Predict Student Exam Performance",
        "Enter student factors and request a live prediction from the FastAPI backend.",
    )

    # --------------------------------------------------------
    # API check
    # --------------------------------------------------------

    if not get_api_status():

        st.error(
            "FastAPI is offline. Start the backend before running predictions."
        )

        st.code(
            "uvicorn api.main:app --host 0.0.0.0 --port 8000",
            language="bash",
        )

        st.stop()

    # --------------------------------------------------------
    # Form
    # --------------------------------------------------------

    with st.form(
        "prediction_form",
        border=True,
    ):

        st.subheader("📚 Academic Factors")

        c1, c2, c3 = st.columns(3)

        with c1:

            hours_studied = st.number_input(
                "Hours Studied",
                min_value=0,
                max_value=24,
                value=5,
            )

        with c2:

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0,
                max_value=100,
                value=80,
            )

        with c3:

            previous_scores = st.number_input(
                "Previous Scores",
                min_value=0,
                max_value=100,
                value=70,
            )

        c1, c2, c3 = st.columns(3)

        with c1:

            tutoring_sessions = st.number_input(
                "Tutoring Sessions",
                min_value=0,
                max_value=20,
                value=2,
            )

        with c2:

            physical_activity = st.number_input(
                "Physical Activity",
                min_value=0,
                max_value=10,
                value=3,
            )

        with c3:

            sleep_hours = st.number_input(
                "Sleep Hours",
                min_value=0,
                max_value=24,
                value=7,
            )

        st.divider()

        st.subheader("🏫 Learning Environment")

        c1, c2, c3 = st.columns(3)

        with c1:

            parental_involvement = st.selectbox(
                "Parental Involvement",
                ["Low", "Medium", "High"],
            )

            access_resources = st.selectbox(
                "Access to Resources",
                ["Low", "Medium", "High"],
            )

            motivation = st.selectbox(
                "Motivation Level",
                ["Low", "Medium", "High"],
            )

        with c2:

            extracurricular = st.selectbox(
                "Extracurricular Activities",
                ["No", "Yes"],
            )

            internet_access = st.selectbox(
                "Internet Access",
                ["No", "Yes"],
            )

            teacher_quality = st.selectbox(
                "Teacher Quality",
                ["Low", "Medium", "High"],
            )

        with c3:

            school_type = st.selectbox(
                "School Type",
                ["Public", "Private"],
            )

            peer_influence = st.selectbox(
                "Peer Influence",
                ["Negative", "Neutral", "Positive"],
            )

            family_income = st.selectbox(
                "Family Income",
                ["Low", "Medium", "High"],
            )

        st.divider()

        st.subheader("👤 Student Profile")

        c1, c2, c3 = st.columns(3)

        with c1:

            learning_disabilities = st.selectbox(
                "Learning Disabilities",
                ["No", "Yes"],
            )

        with c2:

            parental_education = st.selectbox(
                "Parental Education Level",
                [
                    "High School",
                    "College",
                    "Postgraduate",
                ],
            )

        with c3:

            distance = st.selectbox(
                "Distance from Home",
                ["Near", "Moderate", "Far"],
            )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"],
        )

        st.write("")

        submitted = st.form_submit_button(
            "🚀 Generate Exam Score Prediction",
            use_container_width=True,
            type="primary",
        )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    if submitted:

        payload = {
            "Hours_Studied": int(hours_studied),
            "Attendance": int(attendance),
            "Parental_Involvement": parental_involvement,
            "Access_to_Resources": access_resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": int(sleep_hours),
            "Previous_Scores": int(previous_scores),
            "Motivation_Level": motivation,
            "Internet_Access": internet_access,
            "Tutoring_Sessions": int(tutoring_sessions),
            "Family_Income": family_income,
            "Teacher_Quality": teacher_quality,
            "School_Type": school_type,
            "Peer_Influence": peer_influence,
            "Physical_Activity": int(physical_activity),
            "Learning_Disabilities": learning_disabilities,
            "Parental_Education_Level": parental_education,
            "Distance_from_Home": distance,
            "Gender": gender,
        }

        try:

            with st.spinner(
                "Sending prediction request to FastAPI..."
            ):

                result = api_post(
                    "/predict",
                    payload,
                )

            score = float(
                result["predicted_score"]
            )

            band = result[
                "performance_band"
            ]

            version = result[
                "model_version"
            ]

            st.write("")

            # ------------------------------------------------
            # Result
            # ------------------------------------------------

            st.success(
                "Prediction generated successfully."
            )

            r1, r2, r3 = st.columns(3)

            with r1:

                st.metric(
                    "Predicted Exam Score",
                    f"{score:.2f} / 100",
                )

            with r2:

                st.metric(
                    "Performance Band",
                    band,
                )

            with r3:

                st.metric(
                    "Model",
                    "LinearRegression",
                )

            st.progress(
                min(
                    max(score / 100, 0.0),
                    1.0,
                ),
                text=f"Predicted score: {score:.2f} / 100",
            )

            with st.container(border=True):

                st.markdown(
                    "### 📌 Prediction Details"
                )

                st.write(
                    f"**Predicted Score:** {score:.2f}"
                )

                st.write(
                    f"**Performance Band:** {band}"
                )

                st.write(
                    f"**Model Version:** `{version}`"
                )

                st.caption(
                    "This is a model-generated estimate based on "
                    "the supplied student factors. It is intended "
                    "for academic decision-support and is not a guaranteed outcome."
                )

        except requests.exceptions.RequestException as exc:

            st.error(
                f"Prediction API request failed: {exc}"
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "📊 Model Performance":

    render_header(
        "Model Performance",
        "Evaluation on the untouched test set and comparison of candidate models.",
    )

    metrics = load_json(
        "outputs/metrics/final_model_evaluation.json"
    )

    if metrics is None:

        metrics = {
            "rmse": 1.5686764381239588,
            "mae": 0.5071414143775117,
            "r2": 0.8139212785920882,
            "test_rows": 1322,
        }

    rmse = metrics.get(
        "rmse",
        metrics.get("RMSE", 1.5687),
    )

    mae = metrics.get(
        "mae",
        metrics.get("MAE", 0.5071),
    )

    r2 = metrics.get(
        "r2",
        metrics.get(
            "R2",
            metrics.get(
                "r_squared",
                0.8139,
            ),
        ),
    )

    test_rows = metrics.get(
        "test_rows",
        metrics.get(
            "n_test",
            1322,
        ),
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    st.subheader("Final Test Set Results")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "RMSE",
        f"{float(rmse):.3f}",
        help="Root Mean Squared Error",
    )

    c2.metric(
        "MAE",
        f"{float(mae):.3f}",
        help="Mean Absolute Error",
    )

    c3.metric(
        "R²",
        f"{float(r2):.3f}",
        help="Coefficient of determination",
    )

    c4.metric(
        "Test Rows",
        f"{int(test_rows):,}",
    )

    st.write("")

    # --------------------------------------------------------
    # Champion
    # --------------------------------------------------------

    with st.container(border=True):

        st.markdown(
            "### 🏆 Champion Model"
        )

        st.success(
            "LinearRegression"
        )

        st.write(
            "Selected using validation RMSE and then refit "
            "on the complete development set."
        )

    st.write("")

    # --------------------------------------------------------
    # Model comparison
    # --------------------------------------------------------

    experiment_results = load_csv(
        "outputs/metrics/model_experiment_results.csv"
    )

    if experiment_results is not None:

        st.subheader("Model Experiment Comparison")

        display_df = experiment_results.copy()

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )

        if (
            "model" in experiment_results.columns
            and "rmse" in experiment_results.columns
        ):

            st.subheader("Validation RMSE")

            chart = (
                experiment_results[
                    ["model", "rmse"]
                ]
                .copy()
                .sort_values(
                    "rmse",
                    ascending=True,
                )
                .set_index("model")
            )

            st.bar_chart(
                chart,
                y="rmse",
            )

    else:

        st.warning(
            "Model experiment results file was not found."
        )

    # --------------------------------------------------------
    # Metric explanation
    # --------------------------------------------------------

    st.subheader("Understanding the Metrics")

    c1, c2, c3 = st.columns(3)

    with c1:

        with st.container(border=True):

            st.markdown("### RMSE")

            st.write(
                "Penalizes larger prediction errors more heavily."
            )

    with c2:

        with st.container(border=True):

            st.markdown("### MAE")

            st.write(
                "Represents the average absolute prediction error."
            )

    with c3:

        with st.container(border=True):

            st.markdown("### R²")

            st.write(
                "Measures the proportion of target variance "
                "explained by the model on the test set."
            )

    # --------------------------------------------------------
    # Feature engineering evidence
    # --------------------------------------------------------

    st.subheader("Feature Engineering Evidence")

    feature_data = pd.DataFrame(
        {
            "Feature": [
                "Study × Attendance",
                "Attendance",
                "Hours Studied",
                "Study × Sleep",
            ],
            "Correlation": [
                0.653,
                0.582,
                0.447,
                0.351,
            ],
        }
    )

    st.bar_chart(
        feature_data.set_index("Feature")
    )

    st.caption(
        "Correlation values shown here come from the project's "
        "cleaned dataset analysis."
    )


# ============================================================
# MLOPS
# ============================================================

elif page == "⚙️ MLOps":

    render_header(
        "MLOps Architecture",
        "The engineering layer that turns the ML experiment into a reproducible application.",
    )

    # --------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------

    st.subheader("End-to-End Delivery Pipeline")

    pipeline = st.columns(7)

    pipeline_items = [
        ("01", "Git", "Code"),
        ("02", "DVC", "Data"),
        ("03", "MLflow", "Experiments"),
        ("04", "FastAPI", "Serving"),
        ("05", "Streamlit", "UI"),
        ("06", "Docker", "Runtime"),
        ("07", "Actions", "CI/CD"),
    ]

    for col, (number, name, purpose) in zip(
        pipeline,
        pipeline_items,
    ):

        with col:

            with st.container(border=True):

                st.caption(number)

                st.markdown(
                    f"**{name}**"
                )

                st.caption(purpose)

    st.write("")

    # --------------------------------------------------------
    # Tool justification
    # --------------------------------------------------------

    st.subheader("Why Each Tool Exists")

    tools = [
        (
            "Git",
            "Code Versioning",
            "Maintains the real development history of the project.",
        ),
        (
            "DVC",
            "Data & Pipeline Versioning",
            "Tracks the dataset and reproducible ML pipeline stages.",
        ),
        (
            "MLflow",
            "Experiment Tracking",
            "Records model experiments and registers the selected model.",
        ),
        (
            "FastAPI",
            "Model Serving",
            "Provides a reusable HTTP prediction service.",
        ),
        (
            "Streamlit",
            "Product Interface",
            "Provides the interactive user-facing application.",
        ),
        (
            "Docker",
            "Reproducible Runtime",
            "Packages the application and dependencies consistently.",
        ),
        (
            "GitHub Actions",
            "CI/CD",
            "Runs tests, builds the image and publishes it automatically.",
        ),
    ]

    for name, purpose, reason in tools:

        with st.container(border=True):

            left, right = st.columns(
                [1, 3],
                gap="large",
            )

            with left:

                st.markdown(
                    f"### {name}"
                )

                st.caption(
                    purpose
                )

            with right:

                st.write(reason)

    st.write("")

    # --------------------------------------------------------
    # Actual deployment flow
    # --------------------------------------------------------

    st.subheader("CI/CD Flow")

    st.code(
        """Developer
   │
   ▼
Git Push
   │
   ▼
GitHub Actions
   │
   ├── Install Dependencies
   │
   ├── Run Pytest
   │
   ├── Build Docker Image
   │
   ├── Login to Docker Hub
   │
   └── Push Image
           │
           ▼
     Docker Hub
           │
           ▼
   Production Container
           │
      ┌────┴────┐
      ▼         ▼
 Streamlit   FastAPI
  :8501       :8000
                │
                ▼
          ML Pipeline
                │
                ▼
           Prediction""",
        language="text",
    )

    # --------------------------------------------------------
    # Live monitoring
    # --------------------------------------------------------

    st.subheader("Live API Monitoring")

    try:

        live_metrics = api_get(
            "/metrics",
            timeout=3,
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Requests",
            live_metrics.get(
                "total_requests",
                0,
            ),
        )

        c2.metric(
            "Successful Predictions",
            live_metrics.get(
                "successful_predictions",
                0,
            ),
        )

        c3.metric(
            "Errors",
            live_metrics.get(
                "errors",
                0,
            ),
        )

        latency = live_metrics.get(
            "average_latency_seconds",
            0,
        )

        c4.metric(
            "Avg Latency",
            f"{float(latency) * 1000:.1f} ms",
        )

    except Exception:

        st.warning(
            "Live monitoring metrics are currently unavailable."
        )


# ============================================================
# ABOUT
# ============================================================

elif page == "ℹ️ About":

    render_header(
        "About EduPredict",
        "End-to-end Feature Engineering and MLOps project.",
    )

    with st.container(border=True):

        st.subheader(
            "🎓 What is EduPredict?"
        )

        st.write(
            "EduPredict is an end-to-end machine learning system "
            "that predicts student exam scores from academic, behavioral "
            "and learning-environment factors."
        )

        st.write(
            "The project demonstrates the complete journey from "
            "raw data and feature engineering to model experimentation, "
            "API serving, interactive UI and automated container delivery."
        )

    st.write("")

    st.subheader("🎯 Project Objective")

    c1, c2 = st.columns(2)

    with c1:

        with st.container(border=True):

            st.markdown("### Prediction")

            st.write(
                "Predict the expected `Exam_Score` using "
                "19 student input features."
            )

    with c2:

        with st.container(border=True):

            st.markdown("### Productionization")

            st.write(
                "Demonstrate how a trained model can be transformed "
                "into a reproducible MLOps application."
            )

    st.write("")

    st.subheader("🧰 Technology Stack")

    tech1, tech2, tech3 = st.columns(3)

    with tech1:

        with st.container(border=True):

            st.markdown("### Machine Learning")

            st.write(
                "Python\n\n"
                "Pandas\n\n"
                "Scikit-learn\n\n"
                "Joblib"
            )

    with tech2:

        with st.container(border=True):

            st.markdown("### MLOps")

            st.write(
                "Git\n\n"
                "DVC\n\n"
                "MLflow\n\n"
                "Pytest"
            )

    with tech3:

        with st.container(border=True):

            st.markdown("### Application")

            st.write(
                "FastAPI\n\n"
                "Streamlit\n\n"
                "Docker\n\n"
                "GitHub Actions"
            )

    st.write("")

    st.subheader("📌 Important Note")

    st.info(
        "EduPredict provides model-generated estimates for academic "
        "decision-support. Predictions are not guaranteed outcomes "
        "and should be interpreted in the context of the supplied inputs."
    )

    st.divider()

    st.caption(
        "EduPredict • Feature Engineering & MLOps • 2026"
    )