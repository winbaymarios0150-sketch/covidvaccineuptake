import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Define the clean_age function as used in preprocessing
def clean_age(age):
    if isinstance(age, (int, float)):
        return age
    age = str(age).lower().replace('years', '').replace('yrs', '').replace('+', '').replace('.', '').strip()
    if age == 'nineteen':
        return 19
    elif age == 'fifty two':
        return 52
    elif age == 'twenty-six':
        return 26
    elif age == 'eighteen':
        return 18
    elif age == 'twenty two':
        return 22
    try:
        return float(age)
    except ValueError:
        return np.nan

# Load the trained model
try:
    model = joblib.load("randomforestwb.joblib")
except FileNotFoundError:
    st.error("Model file 'linearwb.joblib' not found. Please ensure the model is trained and saved.")
    st.stop() # Stop the app if the model file is not found

# Define the mapping for categorical features based on the training data
# These mappings should correspond to the LabelEncoder transformations done during training
# You would typically save these mappings during training, but for this example,
# we'll manually define them based on the exploration of filtered_data
gender_mapping = {'Female': 0, 'Male': 1, 'Other': 2}
education_mapping = {
    'School - Standard 10+': 0,
    'Other': 1,
    'School - Standard 12+': 2,
    'Graduation (B.A/B.Sc/B.Com/B. Tech etc.)': 3,
    'Post Graduation (M.A/M.Sc/M.Tech/M. Phil/PhD etc.)': 4,
    'Professional Degree (MBA/Law/Medical etc.)': 5
}
occupation_mapping = {
    'Employed in private organization': 1,
    'Employed in government organization': 0,
    'Housewife': 2,
    'IT Professional': 3,
    'Not Employed': 4,
    'Other': 5,
    'Retired': 6,
    'Student': 7,
    'Self Employed':8 # Added Self Employed based on data
}
income_mapping = {
    '10L': 0,
    '5 Lakhs': 2,
    'Between 20-30 lacs': 8,
    'NaN': 12, # Handle NaN if needed, though we dropped NaNs in training
    'Between 40-50 lacs': 10,
    'Between 10-20 lacs': 7,
    '5 lacs or less': 3,
    'Above 50 lacs': 1,
    'Between 30-40 lacs': 9,
    'Between 5-10 lacs': 11,
    '6 lacs or less': 4, # Added based on data
    '7 lacs or less': 5, # Added based on data
    '4 lacs or less': 6 # Added based on data
}
contracted_covid_19_mapping = {'No': 0, 'Yes': 1}
willing_mapping = {0: 'No', 1: 'Yes', 2: 'Maybe'} # Mapping back for output


# App Title and Description
st.title("COVID-19 Vaccine Willingness Prediction")
st.write("Predicting an individual's willingness to take the COVID-19 vaccine based on various factors.")

# Input Fields
st.header("Enter Your Information")

perception = st.slider("Perception (How likely you think you are to contract COVID-19?)", 0.0, 10.0, 5.0)
certainty = st.slider("Certainty (How certain are you about your perception?)", 0.0, 10.0, 5.0)
age = st.number_input("Age", min_value=0, max_value=120, value=30)
gender = st.selectbox("Gender", list(gender_mapping.keys()))
education = st.selectbox("Education", list(education_mapping.keys()))
occupation = st.selectbox("Occupation", list(occupation_mapping.keys()))
income = st.selectbox("Approximately, what is your yearly family income? (In Rupees)", list(income_mapping.keys()))
contracted_covid_19 = st.selectbox("Have you ever contracted COVID-19?", list(contracted_covid_19_mapping.keys()))

# Prediction Button
if st.button("Predict Willingness"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame({
        'perception': [perception],
        'Certainity': [certainty],
        'Age': [clean_age(age)], # Apply clean_age function
        'gender': [gender_mapping[gender]], # Map categorical to numerical
        'education': [education_mapping[education]], # Map categorical to numerical
        'occupation': [occupation_mapping[occupation]], # Map categorical to numerical
        'income': [income_mapping[income]], # Map categorical to numerical
        'contracted_covid_19': [contracted_covid_19_mapping[contracted_covid_19]] # Map categorical to numerical
    })

    # Ensure column order matches training data
    # This is crucial for consistent predictions
    input_data = input_data[model.feature_names_in_]


    # Make prediction
    prediction_numeric = model.predict(input_data)[0]
    prediction_text = willing_mapping[prediction_numeric]

    # Display prediction
    st.subheader("Prediction:")
    st.write(f"Based on the information provided, the predicted willingness to take the COVID-19 vaccine is: **{prediction_text}**")
