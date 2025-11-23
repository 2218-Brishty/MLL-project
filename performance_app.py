import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================================================
# 1) Load Required Files
# ============================================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")     # dictionary of LabelEncoders
columns = joblib.load("columns.pkl")       # list of column names used during training

# Columns used during training
num_cols = ['Study_Hours_per_Week', 'Attendance_Rate',
            'Past_Exam_Scores', 'Final_Exam_Score']

cat_cols = ['Gender', 'Parental_Education_Level',
            'Internet_Access_at_Home', 'Extracurricular_Activities']


# ============================================================
# 2) Prediction Function (PERFECT PIPELINE MATCH)
# ============================================================
def predict_result(values):

    # Convert dictionary → DataFrame
    df = pd.DataFrame([values])

    # Apply LabelEncoders exactly as training
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Scale numerical features
    df[num_cols] = scaler.transform(df[num_cols])

    # Create final dataframe EXACT as model was trained on
    final_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Fill existing columns
    for col in df.columns:
        final_df[col] = df[col].iloc[0]

    # Predict
    pred = model.predict(final_df)[0]

    return "Pass" if pred == 1 else "Fail"


# ============================================================
# 3) Streamlit UI
# ============================================================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🎓 Student Performance Predictor</h1>", 
            unsafe_allow_html=True)
st.write("Enter student details to predict Pass/Fail.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    study = st.slider("Weekly Study Hours", 1, 30, 10)
    attendance = st.slider("Attendance (%)", 50, 100, 85)

with col2:
    past = st.number_input("Past Exam Score", 0, 100, 60)
    final = st.number_input("Final Exam Score", 0, 100, 75)
    parent = st.selectbox("Parental Education Level",
                          ["High School", "College", "University", "Masters", "PhD"])
    internet = st.selectbox("Internet Access at Home", ["Yes", "No"])
    extra = st.selectbox("Extracurricular Activities", ["Yes", "No"])

# Prepare input dictionary
data = {
    "Gender": gender,
    "Study_Hours_per_Week": study,
    "Attendance_Rate": attendance,
    "Past_Exam_Scores": past,
    "Final_Exam_Score": final,
    "Parental_Education_Level": parent,
    "Internet_Access_at_Home": internet,
    "Extracurricular_Activities": extra
}

if st.button("Predict Result"):
    result = predict_result(data)

    if result == "Pass":
        st.success("🎉 Prediction: PASS ✓ Congratulations!")
        st.balloons()
    else:
        st.error("❌ Prediction: FAIL")
        st.warning("The model predicts this student may fail. Support needed.")

st.markdown("---")
st.caption("© ML Student Evaluation — Streamlit App")
