import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===================================================
# Load Model Assets
# ===================================================

model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

num_cols = ['Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores','Final_Exam_Score']
cat_cols = ['Gender','Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']


# ===================================================
# Prediction Function
# ===================================================

def predict_result(values):

    df = pd.DataFrame([values])

    # Apply saved LabelEncoders on categorical columns
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Scale numerical values
    df[num_cols] = scaler.transform(df[num_cols])

    # Align columns exactly like training time
    final_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    for col in df.columns:
        final_df[col] = df[col].values[0]

    prediction = model.predict(final_df)[0]
    return "Pass" if prediction == 1 else "Fail"


# ===================================================
# Streamlit UI
# ===================================================

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E8B57;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.write("Use the fields below to predict whether a student will Pass or Fail based on academic features.")
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    study = st.slider("Weekly Study Hours", 1, 30, 10)
    attendance = st.slider("Attendance (%)", 50, 100, 85)

with col2:
    past = st.number_input("Past Exam Score", 0, 100, 60)
    final = st.number_input("Final Exam Score", 0, 100, 75)
    parent = st.selectbox("Parental Education Level", 
                          ["High School","College","University","Masters","PhD"])
    internet = st.selectbox("Internet Access at Home", ["Yes","No"])
    extra = st.selectbox("Extracurricular Activities", ["Yes","No"])

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
        st.success("🎉 Prediction: PASS ✓")
        st.balloons()
    else:
        st.error("❌ Prediction: FAIL")
        st.warning("The model predicts this student may fail. Consider giving additional support.")

st.markdown("---")
st.caption("ML Student Performance Prediction — Streamlit App")


