import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# Load the model, label encoder, and unique values
output_dir = 'output'
model = joblib.load(os.path.join(output_dir, 'model.joblib'))
le = joblib.load(os.path.join(output_dir, 'label_encoder.joblib'))
unique_values = joblib.load(os.path.join(output_dir, 'unique_values.joblib'))

st.title('SLE Prediction App')

# Create input fields
input_data = {}
for column, values in unique_values.items():
    if column != 'AGE' and column != 'age onset':
        input_data[column] = st.selectbox(f"Select {column}", values)
    else:
        input_data[column] = st.number_input(f"Enter {column}", min_value=0, max_value=120)

# Make prediction
if st.button('Predict'):
    input_df = pd.DataFrame([input_data])
    
    # Make prediction
    probability = model.predict_proba(input_df)[0][1]
    
    st.write(f"Likelihood of SLE: {probability * 100:.2f}%")

    # SHAP explanation
    explainer = shap.TreeExplainer(model['classifier'])
    shap_values = explainer.shap_values(model['preprocessor'].transform(input_df))
    
    st.write("SHAP Feature Importance:")
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values[1], input_df, plot_type="bar", show=False)
    st.pyplot(fig)
