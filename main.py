import streamlit as st
import pandas as pd
import joblib

# Load the preprocessed data and trained model
data = pd.read_csv("medical data.csv")

# Drop columns that were not used for training
data = data.drop(columns=["DateOfBirth", "Medicine"])

# Load the model and encoders
rf_classifier = joblib.load("rf_classifier.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Streamlit interface
st.title("Medicine Recommendation System")

# User inputs
gender = st.selectbox("Gender", label_encoders["Gender"].classes_)
symptoms = st.selectbox("Symptoms", label_encoders["Symptoms"].classes_)
causes = st.selectbox("Causes", label_encoders["Causes"].classes_)
disease = st.selectbox("Disease", label_encoders["Disease"].classes_)

# Encode user inputs
user_input = {
    "Gender": label_encoders["Gender"].transform([gender])[0],
    "Symptoms": label_encoders["Symptoms"].transform([symptoms])[0],
    "Causes": label_encoders["Causes"].transform([causes])[0],
    "Disease": label_encoders["Disease"].transform([disease])[0],
}

# Create a DataFrame with the same columns as the training data
user_df = pd.DataFrame([user_input])

# Add any missing columns from the training data, set them to 0
missing_cols = set(data.columns) - set(user_df.columns)
for col in missing_cols:
    user_df[col] = 0

# Ensure columns are in the same order as during training
user_df = user_df[data.columns]

# Add a button to recommend medicine
if st.button("Recommend"):
    # Predict the medicine
    predicted_medicine = rf_classifier.predict(user_df)[0]
    predicted_medicine = label_encoders["Medicine"].inverse_transform(
        [predicted_medicine]
    )[0]

    st.write(f"Recommended Medicine: {predicted_medicine}")

