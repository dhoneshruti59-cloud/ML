import streamlit as st
import pandas as pd
import joblib

model = joblib.load("student_LR.pkl")
scaler = joblib.load("student_scaler.pkl")
columns = joblib.load("student_columns.pkl")

st.set_page_config(page_title="Student Performance Prediction", layout="centered")

st.title("Student Performance Prediction System")

gender=st.selectbox("Select Gender", ["Male", "Female"])

study_time_hours=st.number_input("Enter Study Time (in hours)", min_value=0, max_value=24, value=1)

attendance_percent=st.number_input("Enter Attendance Percentage", min_value=0, max_value=100, value=75)

sleep_hours=st.number_input("Enter Sleep Hours (per day)", min_value=0, max_value=24, value=8)

parental_education=st.selectbox("Select Parental Education Level", ["High School", "Bachelor's", "Master's", "PhD"])

internet_access=st.selectbox("Do you have Internet Access?", ["Yes", "No"])

extracurricular_activities=st.selectbox("Are you involved in Extracurricular Activities?", ["Yes", "No"])

part_time_job=st.selectbox("Do you have a Part-time Job?", ["Yes", "No"])

previous_grade=st.number_input("Enter Previous Grade (if any)", min_value=0, max_value=100, value=0)

final_exam_score=st.number_input("Enter Final Exam Score (if any)", min_value=0, max_value=100, value=0)

if st.button("Predict"):
    user_data = pd.DataFrame({
        "gender": [gender],
        "study_time_hours": [study_time_hours],
        "attendance_percent": [attendance_percent],
        "sleep_hours": [sleep_hours],
        "parental_education": [parental_education],
        "internet_access": [internet_access],
        "extracurricular_activities": [extracurricular_activities],
        "part_time_job": [part_time_job],
        "previous_grade": [previous_grade],
        "final_exam_score": [final_exam_score]
    })

    user_data_encoded = pd.get_dummies(user_data, drop_first=True)

    user_data_encoded = user_data_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(user_data_encoded)

    st.success(f"Predicted Student Performance: {prediction[0]}")