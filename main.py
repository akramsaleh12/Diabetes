import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import subprocess


# Load the datasets
data_dict_file = 'Diabetes Data Dictionary.csv'
data_file = 'Diabetes DataSet.csv'

data_dict = pd.read_csv(data_dict_file)
data = pd.read_csv(data_file)

# Streamlit App Title
st.title("Diabetes Dataset Descriptive Analytics")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = ["Dataset Overview", "Data Dictionary", "Descriptive Statistics", "Correlation Analysis", "Outlier Analysis", "Predictive Model"]
choice = st.sidebar.radio("Choose an option:", options)

# Dataset Overview
if choice == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Below is a sample of the dataset:")
    st.dataframe(data.head())

    st.write("Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    st.text(buffer.getvalue())

    st.write("Shape of the dataset:")
    st.text(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

# Data Dictionary
elif choice == "Data Dictionary":
    st.header("Data Dictionary")
    st.write("Understanding the dataset columns based on the dictionary:")
    st.dataframe(data_dict)

    st.write("Search for specific information in the data dictionary:")
    search_column = st.text_input("Enter column name or keyword:")

    if search_column:
        filtered_dict = data_dict[data_dict.apply(lambda row: row.astype(str).str.contains(search_column, case=False).any(), axis=1)]
        st.dataframe(filtered_dict)

# Descriptive Statistics
elif choice == "Descriptive Statistics":
    st.header("Descriptive Statistics")
    
    st.write("Summary Statistics for Numeric Columns:")
    st.dataframe(data.describe().T)

    st.write("Distribution of Columns:")
    column = st.selectbox("Select a column to visualize:", data.select_dtypes(include=['int64', 'float64']).columns)
    st.bar_chart(data[column])

# Correlation Analysis
elif choice == "Correlation Analysis":
    st.header("Correlation Analysis")
    
    # Add explanation about correlation analysis
    st.write("""
    **What is Correlation Analysis?**
    
    Correlation analysis measures the strength and direction of the relationship between two variables. 
    In this dataset, it helps us understand:
    - How each feature relates to the target variable (`Outcome`).
    - Relationships between features to detect multicollinearity.
    
    **Correlation Values:**
    - **1**: Perfect positive relationship (as one variable increases, the other increases).
    - **-1**: Perfect negative relationship (as one variable increases, the other decreases).
    - **0**: No relationship.
    """)
    
    st.write("### Correlation Table Between Features and Target (Outcome):")
    correlation = data.corr()[["Outcome"]].sort_values(by="Outcome", ascending=False)
    st.dataframe(correlation)

    st.write("### Correlation Heatmap:")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True, ax=ax)
    st.pyplot(fig)

# Outlier Analysis
elif choice == "Outlier Analysis":
    st.header("Outlier Analysis")

    st.write("Outliers are detected using the Z-score method. Z-scores above 3 or below -3 indicate potential outliers.")

    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns

    for column in numeric_columns:
        if column != 'Outcome':
            st.write(f"### Outlier Analysis for {column}")
    
            # Calculate Z-scores
            data['z_score'] = (data[column] - data[column].mean()) / data[column].std()
            outliers = data[np.abs(data['z_score']) > 3]
    
            st.write(f"Number of outliers in '{column}': {outliers.shape[0]}")
            st.dataframe(outliers[[column, 'z_score']])
    
            # Scatter plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(data.index, data[column], label="Data", alpha=0.7)
            ax.scatter(outliers.index, outliers[column], color='red', label="Outliers", alpha=0.7)
            ax.axhline(data[column].mean() + 3 * data[column].std(), color='orange', linestyle='--', label="+3 Z-score")
            ax.axhline(data[column].mean() - 3 * data[column].std(), color='orange', linestyle='--', label="-3 Z-score")
            ax.set_title(f"Outlier Detection for {column}")
            ax.set_xlabel("Index")
            ax.set_ylabel(column)
            ax.legend()
            st.pyplot(fig)

elif choice == "Predictive Model":
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

