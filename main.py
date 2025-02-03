import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
import numpy as np
import os
import joblib

# Load user credentials
def load_users():
    users_file = '/mnt/data/users.csv'
    users_df = pd.read_csv(users_file)
    return {row['username']: row['password'] for _, row in users_df.iterrows()}

# Authentication function
def authenticate(username, password, users):
    return username in users and users[username] == password

# Streamlit App Title
st.title("Diabetes Dataset Descriptive Analytics")

# User Authentication
users = load_users()
st.sidebar.header("User Login")
username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")
login_button = st.sidebar.button("Login")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if login_button:
    if authenticate(username, password, users):
        st.sidebar.success("Login successful!")
        st.session_state.authenticated = True
    else:
        st.sidebar.error("Invalid username or password. Please try again.")
        st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.stop()
# Custom CSS for Styling with Background Color
st.markdown("""
    <style>
        body {
            background-color: #e6f7ff;
            color: #333333;
         }
        .stApp {
            background-color: #F0F8FF;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
                }
        .stDataFrame {
            background-color: #f0f8ff;
            border-radius: 5px;
                }
        .stSidebar {
                    background-color: #007acc;
                    color: white;
                    padding: 15px;
                    border-radius: 10px;
                }
        h1, h2, h3 {
                    color: #007acc;
                }
        </style>
        """, unsafe_allow_html=True)

# Load the datasets
data_dict_file = 'Diabetes Data Dictionary.csv'
data_file = 'Diabetes DataSet.csv'
        
data_dict = pd.read_csv(data_dict_file)
data = pd.read_csv(data_file)
        
# Streamlit App Title
st.image("image2.jpg")
# st.title("Diabetes Dataset Descriptive Analytics")
        
# Sidebar for navigation
st.sidebar.title("Navigation")
st.sidebar.markdown("<hr style='border:1px solid white'>", unsafe_allow_html=True)
options = ["Dataset Overview", "Data Dictionary", "Descriptive Statistics", "Correlation Analysis", "Outlier Analysis", "Predictive Model"]
choice = st.sidebar.radio("Select an option:", options)

# Dataset Overview
if choice == "Dataset Overview":
    st.header("Dataset Overview")
    st.write("Below is a sample of the dataset:")
    st.dataframe(data.head())
        
    st.write("Dataset Information:")
    buffer = io.StringIO()
    data.info(buf=buffer)
    info_str = buffer.getvalue()
    st.text(info_str)
        
    st.write("Shape of the dataset:")
    st.text(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")
    
    # Null values
    st.write("Number of Null Values per Feature:")
    st.dataframe(data.isnull().sum().reset_index().rename(columns={"index": "Feature", 0: "Null Values"}))
            
    # Zero values
    st.write("Number of Zero Values per Feature:")
    zero_values = data.apply(lambda x: (x == 0).sum()).reset_index()
    zero_values.columns = ["Feature", "Zero Values"]
    st.dataframe(zero_values)
        
# Descriptive Statistics
elif choice == "Descriptive Statistics":
    st.header("Descriptive Statistics")
    
    st.write("Summary Statistics for Numeric Columns:")
    st.dataframe(data.describe().T)
            
    # Interactive histograms with explanation
    st.header("Interactive Histogram")
    numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns
    if len(numerical_columns) > 0:
        column = st.selectbox("Select a column for the histogram", numerical_columns)
        bins = st.slider("Number of bins", min_value=5, max_value=50, value=20)
                
        # Plotting the histogram
        fig, ax = plt.subplots()
        sns.histplot(data[column], bins=bins, kde=True, color="blue", ax=ax)
        plt.title(f"Histogram of {column}")
        st.pyplot(fig)
        
        # Explanation of the histogram
        st.subheader("Explanation of Histogram:")
        st.write(f"""
                    - The histogram for **{column}** represents the frequency distribution of its values.
                    - **Bins**: The x-axis is divided into intervals (bins) that group the data points.
                        - The number of bins is adjustable using the slider.
                    - **Height of Bars**: The y-axis represents the count of data points falling into each bin.
                    - **Distribution Shape**:
                        - A symmetric, bell-shaped histogram may indicate a normal distribution.
                        - Skewed histograms (left or right) may highlight potential trends or biases.
                    - **KDE (Kernel Density Estimate)**:
                        - The smooth curve overlays the histogram and shows the estimated data distribution.
                """)
        
        # Highlighting additional insights
        st.write("**Key Insights:**")
        mean_val = data[column].mean()
        median_val = data[column].median()
        st.write(f"- **Mean:** {mean_val:.2f}")
        st.write(f"- **Median:** {median_val:.2f}")
        st.write(f"- The histogram helps identify trends, clusters, or potential outliers in the data.")
    else:
        st.warning("No numerical columns available for the histogram.")

# Correlation Analysis
elif choice == "Correlation Analysis":
    st.header(" Correlation Analysis")
        
    # Add explanation about correlation analysis
    st.write("""
    **What is Correlation Analysis?**
            
    Correlation analysis measures the strength and direction of the relationship between two variables. 
    in this dataset, it helps us understand:
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
        
    st.write("### Feature Importance Based on Correlation with Outcome:")
    feature_importance = correlation.drop("Outcome").sort_values(by="Outcome", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=feature_importance["Outcome"], y=feature_importance.index, palette="Blues_r")
    ax.set_xlabel("Correlation with Outcome")
    ax.set_title("Feature Importance")
    st.pyplot(fig)

    st.write("### Explanation of Feature Importance")
    st.markdown("""
            Feature importance is determined by measuring the correlation of each feature with the target variable (Outcome). 
            Features with higher absolute correlation values have a stronger relationship with diabetes prediction. 
            However, correlation does not imply causation. Further analysis and feature engineering may be required.
            """)
        
# Outlier Analysis
elif choice == "Outlier Analysis":
            st.header("Outlier Analysis")
        
            st.write("Outliers are detected using the Z-score method. Z-scores above 3 or below -3 indicate potential outliers.")
        
            numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
        
            for column in numeric_columns:
                st.write(f"### Outlier Analysis for {column}")
        
                # Calculate Z-scores
                z_scores = (data[column] - data[column].mean()) / data[column].std()
                outliers = data[np.abs(z_scores) > 3]
        
                st.write(f"Number of outliers in '{column}': {outliers.shape[0]}")
                st.dataframe(outliers[[column]])
        
                # Scatter plot
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(data.index, data[column], label="Data", alpha=0.7, color='blue')
                ax.scatter(outliers.index, outliers[column], color='red', label="Outliers", alpha=0.7)
                ax.axhline(data[column].mean() + 3 * data[column].std(), color='orange', linestyle='--', label="+3 Z-score")
                ax.axhline(data[column].mean() - 3 * data[column].std(), color='orange', linestyle='--', label="-3 Z-score")
                ax.set_title(f"Outlier Detection for {column}")
                ax.set_xlabel("Index")
                ax.set_ylabel(column)
                ax.legend()
                st.pyplot(fig)
        
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

elif choice == "Predictive Model":
              
            # Load the saved model and median values
            model_filename = 'diabetes_model.pkl'
            saved_data = joblib.load(model_filename)
            model = saved_data['model']
            medians = saved_data['medians']
            
            st.markdown(
                """
                <style>
                .stApp {
                    background-color: #F0F8FF;  # Light blue background
                }
                </style>
                """,
                unsafe_allow_html=True
            )
    
            # st.image("image1.jpg")
            st.header("Diabetes Prediction Model")
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
        

    


    
