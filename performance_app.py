import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =====================================
# Load model files
# =====================================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
columns = joblib.load("columns.pkl")

cat_cols = ['Gender', 'Parental_Education_Level',
            'Internet_Access_at_Home', 'Extracurricular_Activities']

num_cols = [
    'Study_Hours_per_Week',
    'Attendance_Rate',
    'Past_Exam_Scores',
    'Final_Exam_Score'
]

# =====================================
# Prediction Function (Safe LabelEncoder)
# =====================================
def predict(values):
    df = pd.DataFrame([values])

    # Encode categorical columns safely
    for col in cat_cols:
        le = encoders[col]
        safe_vals = []
        for v in df[col]:
            if v in le.classes_:
                safe_vals.append(le.transform([v])[0])
            else:
                safe_vals.append(-1)  # fallback for unseen label
        df[col] = safe_vals

    # Scale numerical columns
    df[num_cols] = scaler.transform(df[num_cols])

    # Create final input matching exact columns
    final_df = pd.DataFrame(np.zeros((1, len(columns))), columns=columns)

    # Insert values into correct columns
    for col in columns:
        if col in df:
            final_df[col] = df[col].iloc[0]

    # Predict
    pred = model.predict(final_df)[0]
    return "Pass" if pred == "Pass" or pred == 1 else "Fail"

# =====================================
# Streamlit UI
# =====================================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.markdown(
    "<h2 style='text-align:center; color:#2E8B57;'>🎓 Student Performance Predictor</h2>",
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

# Use encoder classes to limit dropdown options to training values
with col1:
    gender = st.selectbox("Gender", encoders['Gender'].classes_.tolist())
    study = st.slider("Weekly Study Hours", 1, 30, 10)
    attendance = st.slider("Attendance (%)", 50, 100, 85)

with col2:
    past = st.number_input("Past Exam Score", 0, 100, 60)
    final = st.number_input("Final Exam Score", 0, 100, 75)
    parent = st.selectbox("Parental Education", encoders['Parental_Education_Level'].classes_.tolist())
    internet = st.selectbox("Internet Access", encoders['Internet_Access_at_Home'].classes_.tolist())
    extra = st.selectbox("Extracurricular Activities", encoders['Extracurricular_Activities'].classes_.tolist())

# Build input data
input_data = {
    "Gender": gender,
    "Study_Hours_per_Week": study,
    "Attendance_Rate": attendance,
    "Past_Exam_Scores": past,
    "Final_Exam_Score": final,
    "Parental_Education_Level": parent,
    "Internet_Access_at_Home": internet,
    "Extracurricular_Activities": extra
}

# Predict on button click
if st.button("Predict Result"):
    result = predict(input_data)
    if result == "Pass":
        st.success("🎉 Result: PASS")
        st.balloons()
    else:
        st.error("❌ Result: FAIL")
