import streamlit as st
import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the trained model
model = joblib.load('crime_model.pkl')

# App Title
st.title("Crime Rate Predictor")
st.write("Enter details to predict if an area is crime-prone and forecast crime rates.")

# User Inputs (expanded to more fields — add more as needed)
city = st.selectbox("City", ["City1", "City2", "City3"])  # Replace with actual from dataset
victim_age = st.number_input("Victim Age", 0, 100, 30)
victim_gender = st.selectbox("Victim Gender", ["Male", "Female", "Unknown"])
crime_domain = st.selectbox("Crime Domain", ["Theft", "Assault", "Murder"])

# Preprocess Input (create full DataFrame with all 15 features, defaults for missing)
input_data = pd.DataFrame({
    'Report Number': [0],  # Default; add input if needed
    'Date Reported': [0],
    'Date of Occurrence': [0],
    'Time of Occurrence': [0],
    'City': [city],
    'Crime Code': [0],
    'Crime Description': [0],
    'Victim Age': [victim_age],
    'Victim Gender': [victim_gender],
    'Weapon Used': [0],
    'Crime Domain': [crime_domain],
    'Police Deployed': [0],
    'Case Closed': [0],
    'Date Case Closed': [0],
    'Crime Prone': [0]  # Or derived if needed
})

# Encode categorical (match notebook)
le = LabelEncoder()
for col in ['City', 'Victim Gender', 'Crime Domain']:
    input_data[col] = le.fit_transform(input_data[col])

# Normalize (simplified — use if your model expects it)
scaler = StandardScaler()
input_data = scaler.fit_transform(input_data)

# FIX: Reindex to match the model's exact 15 features (replace this list with your notebook's printout)
training_columns = [
    'Report Number', 'Date Reported', 'Date of Occurrence', 'Time of Occurrence',
    'City', 'Crime Code', 'Crime Description', 'Victim Age', 'Victim Gender',
    'Weapon Used', 'Crime Domain', 'Police Deployed', 'Case Closed', 'Date Case Closed',
    'Crime Prone'  # Adjust to your exact 15 columns from notebook print
]
input_data = pd.DataFrame(input_data, columns=training_columns)  # Reindex to match

# Prediction Button
if st.button("Predict"):
    prediction = model.predict(input_data)
    st.subheader("Predicted Crime Rate:")
    st.success(f"{prediction[0]:.2f}")

    # Simple classification (based on threshold)
    if prediction[0] > 50:  # Example threshold
        st.warning("High Crime-Prone Area")
    else:
        st.info("Low Crime-Prone Area")