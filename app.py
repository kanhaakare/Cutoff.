import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Set up title and description
st.title("🎓 Cutoff Prediction App")
st.write("Enter the details below to predict the admission cutoff figure using the optimized Random Forest model.")

# Load the quantized model pipeline
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')

try:
    pipeline = load_model()
    st.success("Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

# Input fields for user
university = st.text_input("University", "University of Delhi")
university_id = st.text_input("University ID", "university-of-delhi")
year = st.number_input("Year", min_value=2020, max_value=2030, value=2025)
programme = st.text_input("Programme", "B.A. (Honours) — Economics")
programme_id = st.text_input("Programme ID", "university-of-delhi-b-a-honours")
college = st.text_input("College", "St. Stephen's College")
variant = st.text_input("Variant", "Economics + Mathematics")
round_num = st.text_input("Round", "1")
category = st.text_input("Category", "UR")
seat_type = st.text_input("Seat Type", "general")
figure_kind = st.selectbox("Figure Kind", ["score", "rank"])
basis = st.selectbox("Basis", ["official", "unofficial"])
source_id = st.text_input("Source ID", "university-of-delhi-2025-r1")
source_page = st.number_input("Source Page", min_value=1.0, value=21.0)

# Predict button
if st.button("Predict Cutoff"):
    # Map inputs to model structure (DataFrame)
    input_data = pd.DataFrame({
        'university_id': [university_id],
        'university': [university],
        'year': [int(year)],
        'programme_id': [programme_id],
        'programme': [programme],
        'college': [college],
        'variant': [variant],
        'round': [str(round_num)],
        'category': [category],
        'seat_type': [seat_type],
        'figure_kind': [figure_kind],
        'basis': [basis],
        'source_id': [source_id],
        'source_page': [float(source_page)]
    })

    try:
        prediction = pipeline.predict(input_data)[0]
        st.metric(label="Predicted Cutoff Figure", value=f"{prediction:.4f}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
