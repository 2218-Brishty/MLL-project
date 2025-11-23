import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Performance Predictor", layout="centered")

# Load assets
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

# Numerical + Categorical Columns
num_cols = ['Study_Hours_per_Week','Attendance_Rate','Past_Exam_Scores','Final_Exam_Score']
cat_cols = ['Gender','Parental_Education_Level','Internet_Access_at_Home','Extracurricular_Activities']


def predict_result(values):

    df = pd.DataFrame([values])

    # Apply saved label encoders
    for col in cat_cols:
        df[col] = encoders[col].transform(df[col])

    # Scale numerical values
    df[num_cols] = scaler.transform(df[num_cols])

    # Align columns
    final_df = pd.DataFrame(columns=columns)
    final_df.loc[0] = 0
    for col in df.columns:
        final_df[col] = df[col]

    pred = model.predict(final_df)[0]

    return "উত্তীর্ণ (Pass)" if pred == 1 else "অনুত্তীর্ণ (Fail)"


st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male","Female"])
    study = st.slider("Study Hours Per Week",1,30,10)
    attendance = st.slider("Attendance (%)",50,100,85)

with col2:
    past = st.number_input("Past Exam Score",0,100,75)
    final = st.number_input("Final Exam Score",0,100,70)
    parent = st.selectbox("Parental Education",["High School","College","University","Masters","PhD"])
    internet = st.selectbox("Internet Access",["Yes","No"])
    extra = st.selectbox("Extracurricular Activities",["Yes","No"])

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

if st.button("Predict"):
    result = predict_result(data)
    st.success(f"Result: {result}")

