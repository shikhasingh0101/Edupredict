"""EduPredict Streamlit frontend.

The frontend communicates with FastAPI and never loads model.joblib directly.
"""

import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="EduPredict", page_icon="🎓", layout="wide")

st.title("🎓 EduPredict")
st.caption("AI Student Exam Performance Prediction & Academic Risk Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Dashboard", "Predict", "Model Performance", "About"]
)

if page == "Dashboard":
    st.header("Dashboard")
    st.write("Use the Predict page to send student inputs to the FastAPI model service.")
    try:
        health = requests.get(f"{API_URL}/health", timeout=3).json()
        st.success(f"API status: {health.get('status')}")
    except requests.RequestException:
        st.error("FastAPI is not reachable. Start the API on port 8000.")

elif page == "Predict":
    st.header("Predict Exam Score")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        hours = c1.number_input("Hours Studied", min_value=0, value=20)
        attendance = c2.number_input("Attendance (%)", min_value=0, max_value=100, value=80)
        sleep = c3.number_input("Sleep Hours", min_value=0, value=7)

        previous = c1.number_input("Previous Scores", min_value=0, value=70)
        tutoring = c2.number_input("Tutoring Sessions", min_value=0, value=2)
        physical = c3.number_input("Physical Activity", min_value=0, value=3)

        parental = c1.selectbox("Parental Involvement", ["Low", "Medium", "High"])
        resources = c2.selectbox("Access to Resources", ["Low", "Medium", "High"])
        extracurricular = c3.selectbox("Extracurricular Activities", ["No", "Yes"])
        motivation = c1.selectbox("Motivation Level", ["Low", "Medium", "High"])
        internet = c2.selectbox("Internet Access", ["No", "Yes"])
        income = c3.selectbox("Family Income", ["Low", "Medium", "High"])
        teacher = c1.selectbox("Teacher Quality", ["Low", "Medium", "High"])
        school = c2.selectbox("School Type", ["Public", "Private"])
        peer = c3.selectbox("Peer Influence", ["Negative", "Neutral", "Positive"])
        disability = c1.selectbox("Learning Disabilities", ["No", "Yes"])
        education = c2.selectbox("Parental Education Level", ["High School", "College", "Postgraduate"])
        distance = c3.selectbox("Distance from Home", ["Near", "Moderate", "Far"])
        gender = st.selectbox("Gender", ["Male", "Female"])

        submitted = st.form_submit_button("Predict")

    if submitted:
        payload = {
            "Hours_Studied": hours,
            "Attendance": attendance,
            "Parental_Involvement": parental,
            "Access_to_Resources": resources,
            "Extracurricular_Activities": extracurricular,
            "Sleep_Hours": sleep,
            "Previous_Scores": previous,
            "Motivation_Level": motivation,
            "Internet_Access": internet,
            "Tutoring_Sessions": tutoring,
            "Family_Income": income,
            "Teacher_Quality": teacher,
            "School_Type": school,
            "Peer_Influence": peer,
            "Physical_Activity": physical,
            "Learning_Disabilities": disability,
            "Parental_Education_Level": education,
            "Distance_from_Home": distance,
            "Gender": gender,
        }
        try:
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            st.metric("Predicted Exam Score", result["predicted_score"])
            st.info(f"Performance band: {result['performance_band']}")
        except requests.RequestException as exc:
            st.error(f"Prediction request failed: {exc}")

elif page == "Model Performance":
    st.header("Model Performance")
    st.info("This page reads actual generated metrics when outputs/metrics/final_metrics.json exists.")

    import json
    from pathlib import Path
    path = Path("outputs/metrics/final_metrics.json")
    if path.exists():
        st.json(json.loads(path.read_text()))
    else:
        st.warning("Final metrics have not been generated yet.")

else:
    st.header("About EduPredict")
    st.write(
        "EduPredict is an end-to-end Feature Engineering and MLOps project "
        "for predicting student exam scores. Predictions are decision-support "
        "outputs and are not educational or medical diagnoses."
    )
