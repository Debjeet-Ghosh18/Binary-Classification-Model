import streamlit as st
import pickle
import numpy as np

# Load your trained binary classification model
with open("Binary_Classification_Model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("🔮 Binary Classification Prediction App")
st.write("Upload features below to get predictions using your trained model.")

# Example feature inputs (you can change these based on your dataset)
feature1 = st.number_input("Feature 1", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)

# Collect all features into an array
features = np.array([[feature1, feature2, feature3]])

if st.button("Predict"):
    prediction = model.predict(features)[0]
    proba = model.predict_proba(features)[0][1]

    if prediction == 1:
        st.success(f"✅ Prediction: Positive (Class 1) with probability {proba:.2f}")
    else:
        st.error(f"❌ Prediction: Negative (Class 0) with probability {proba:.2f}")
