"""
Heart Disease Prediction - Streamlit App
Hamza Tahir (FA24-BSE-142, Section A) - Machine Learning Fundamentals, Assignment 2

Serves the SVC saved by Heart_Disease_Prediction.ipynb. The LabelEncoders are
re-fitted on the same value lists as notebook Step 5.1, which gives the same codes.
"""

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Heart Disease Prediction", page_icon="🫀")

MODEL_FILE = Path(__file__).resolve().parent.parent / "heart_disease_model.pkl"

ATTRIBUTE_VALUES = {
    "Sex": ["Male", "Female"],
    "ExerciseAngina": ["Yes", "No"],
    "MajorVessels": ["Zero", "One", "Two", "Three"],
    "STSlope": ["Upsloping", "Flat", "Downsloping"],
}


@st.cache_resource
def load_resources():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    encoders = {name: LabelEncoder().fit(values) for name, values in ATTRIBUTE_VALUES.items()}
    return model, encoders


model, encoders = load_resources()

st.title("🫀 Heart Disease Prediction")
st.write(
    "Enter four results from a patient's cardiac examination. A Support Vector Classifier trained on the "
    "UCI Cleveland Heart Disease dataset predicts whether heart disease is present."
)
st.caption("Test accuracy 0.90 · 10-fold cross validation 0.83")

with st.form("patient_form"):
    sex = st.radio("Sex", ATTRIBUTE_VALUES["Sex"], horizontal=True)
    angina = st.radio("Chest pain during exercise (exercise-induced angina)?", ["No", "Yes"], horizontal=True)
    vessels = st.select_slider("Major vessels coloured by fluoroscopy", ATTRIBUTE_VALUES["MajorVessels"])
    slope = st.selectbox("Slope of the peak exercise ST segment", ATTRIBUTE_VALUES["STSlope"])
    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    patient = pd.DataFrame([{"Sex": sex, "ExerciseAngina": angina, "MajorVessels": vessels, "STSlope": slope}])
    encoded = patient.copy()
    for column in patient.columns:
        encoded[column] = encoders[column].transform(patient[column])

    if model.predict(encoded)[0] == 1:
        st.error("**Prediction: Heart disease is likely.**")
    else:
        st.success("**Prediction: No heart disease.**")

    st.write("Encoded input sent to the model:")
    st.dataframe(encoded, hide_index=True)

st.divider()
st.caption("Hamza Tahir · FA24-BSE-142 · Section A · COMSATS University Islamabad, Lahore Campus. "
           "Educational project, not medical advice.")
