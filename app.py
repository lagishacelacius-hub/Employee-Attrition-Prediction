import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# PAGE SETTINGS
# -----------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="👨‍💼",
    layout="wide"
)

# -----------------------------
# LOAD MODEL
# -----------------------------
pipeline = joblib.load("model/employee_attrition_model.pkl")

# -----------------------------
# LOAD DATASET
# -----------------------------
df = pd.read_csv("dataset/employee_attrition.csv")

# -----------------------------
# TITLE
# -----------------------------
st.title("👨‍💼 Employee Attrition Prediction System")

st.write(
    """
This application predicts whether an employee is likely to leave the company
using a Machine Learning model.
"""
)

st.markdown("---")

# -----------------------------
# INPUT SECTION
# -----------------------------

st.header("Employee Details")

col1, col2 = st.columns(2)

with col1:

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=60,
        value=30
    )

    income = st.number_input(
        "Monthly Income",
        min_value=1000,
        value=30000
    )

    distance = st.number_input(
        "Distance From Home",
        min_value=1,
        value=5
    )

    years = st.number_input(
        "Years At Company",
        min_value=0,
        value=5
    )

    overtime = st.selectbox(
        "OverTime",
        [
            "No",
            "Yes"
        ]
    )

with col2:

    job_satisfaction = st.selectbox(
        "Job Satisfaction",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    environment = st.selectbox(
        "Environment Satisfaction",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

    worklife = st.selectbox(
        "Work Life Balance",
        [
            "Bad",
            "Better",
            "Good",
            "Best"
        ]
    )

    job_level = st.selectbox(
        "Job Level",
        [
            "Entry Level",
            "Junior Level",
            "Mid Level",
            "Senior Level",
            "Executive Level"
        ]
    )

    involvement = st.selectbox(
        "Job Involvement",
        [
            "Low",
            "Medium",
            "High",
            "Very High"
        ]
    )

predict = st.button("Predict Employee Attrition")
# -----------------------------
# PREDICTION
# -----------------------------

if predict:

    input_df = pd.DataFrame(
        {
            "Age": [age],
            "MonthlyIncome": [income],
            "DistanceFromHome": [distance],
            "YearsAtCompany": [years],
            "JobSatisfaction": [job_satisfaction],
            "EnvironmentSatisfaction": [environment],
            "WorkLifeBalance": [worklife],
            "OverTime": [overtime],
            "JobLevel": [job_level],
            "JobInvolvement": [involvement]
        }
    )

    prediction = pipeline.predict(input_df)[0]

    probability = pipeline.predict_proba(input_df)[0]

    confidence = max(probability) * 100

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("⚠️ Employee is likely to leave the company.")

    else:

        st.success("✅ Employee is likely to stay in the company.")

    st.info(f"Prediction Confidence: {confidence:.2f}%")

    st.markdown("---")

    st.subheader("HR Recommendations")

    if prediction == 1:

        if overtime == "Yes":
            st.write("✅ Reduce overtime to improve employee well-being.")

        if job_satisfaction in ["Low", "Medium"]:
            st.write("✅ Improve job satisfaction through recognition and rewards.")

        if environment in ["Low", "Medium"]:
            st.write("✅ Improve workplace environment.")

        if worklife == "Bad":
            st.write("✅ Improve work-life balance policies.")

        if income < 30000:
            st.write("✅ Consider salary revision and employee benefits.")

        if years < 2:
            st.write("✅ Provide mentoring and career development opportunities.")

    else:

        st.write("✅ Continue maintaining employee engagement.")
        st.write("✅ Encourage career growth.")
        st.write("✅ Maintain healthy work-life balance.")
    # -----------------------------
# DASHBOARD
# -----------------------------

st.markdown("---")

st.header("📊 Employee Attrition Dashboard")

# Dataset summary
col1, col2, col3 = st.columns(3)

col1.metric("Total Employees", len(df))
col2.metric("Employees Stayed", len(df[df["Attrition"] == "No"]))
col3.metric("Employees Left", len(df[df["Attrition"] == "Yes"]))

st.markdown("---")

# Attrition Distribution
st.subheader("Employee Attrition Distribution")

fig, ax = plt.subplots(figsize=(6,4))

df["Attrition"].value_counts().plot(
    kind="bar",
    ax=ax
)

ax.set_xlabel("Attrition")
ax.set_ylabel("Count")

st.pyplot(fig)

# Age Distribution
st.subheader("Employee Age Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(df["Age"], bins=15)

ax.set_xlabel("Age")
ax.set_ylabel("Employees")

st.pyplot(fig)

# Monthly Income Distribution
st.subheader("Monthly Income Distribution")

fig, ax = plt.subplots(figsize=(8,4))

ax.hist(df["MonthlyIncome"], bins=20)

ax.set_xlabel("Monthly Income")
ax.set_ylabel("Employees")

st.pyplot(fig)

# Feature Importance
st.subheader("⭐ Feature Importance")

feature_names = [
    "Job Satisfaction",
    "Environment Satisfaction",
    "Work Life Balance",
    "OverTime",
    "Job Level",
    "Job Involvement",
    "Age",
    "Monthly Income",
    "Distance From Home",
    "Years At Company"
]

importance = pipeline.named_steps["model"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

fig, ax = plt.subplots(figsize=(8,5))

ax.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

ax.set_xlabel("Importance")

st.pyplot(fig)

st.markdown("---")

st.success("Project Developed using Machine Learning (Random Forest) and Streamlit")    