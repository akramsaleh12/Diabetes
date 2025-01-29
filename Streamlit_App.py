import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the saved model and median values
model_filename = 'diabetes_model.pkl'
saved_data = joblib.load(model_filename)
model = saved_data['model']
medians = saved_data['medians']

st.title("Diabetes Prediction Model")
st.write("Enter the patient's data to predict the diabetes outcome.")

# Input fields for user data
pregnancies = st.number_input("Pregnancies", min_value=0.000, step=1.000)
glucose = st.number_input("Glucose", min_value=0.000, format="%.3f")
blood_pressure = st.number_input("Blood Pressure", min_value=0.000, format="%.3f")
skin_thickness = st.number_input("Skin Thickness", min_value=0.000, format="%.3f")
insulin = st.number_input("Insulin", min_value=0.000, format="%.3f")
bmi = st.number_input("BMI", min_value=0.000, format="%.3f")
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.000, format="%.3f")
age = st.number_input("Age", min_value=0.000, step=1.000)

if st.button("Predict"):
    try:
        # Prepare the input data
        input_data = [
            pregnancies,
            glucose if glucose != 0 else medians['Glucose'],
            blood_pressure if blood_pressure != 0 else medians['BloodPressure'],
            skin_thickness if skin_thickness != 0 else medians['SkinThickness'],
            insulin if insulin != 0 else medians['Insulin'],
            bmi if bmi != 0 else medians['BMI'],
            diabetes_pedigree_function,
            age
        ]

        # Make prediction
        prediction = model.predict([input_data])[0]
        result = "Diabetic" if prediction == 1 else "Non-Diabetic"
        st.success(f"Prediction: {result}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
