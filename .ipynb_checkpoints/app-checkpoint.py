import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict price
def predict_price(user_base, cost_to_develop):
    # Create DataFrame for the input data
    input_data = pd.DataFrame({
        'User_Base': [user_base],
        'Cost_to_Develop': [cost_to_develop]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the price (log-transformed)
    predicted_price_log = model.predict(input_data_scaled)
    
    # Handle negative log predictions by forcing them to a minimum
    predicted_price_log = np.maximum(predicted_price_log, 0)

    # Inverse log transformation to get the original price
    predicted_price_original = np.power(10, predicted_price_log) - 1

    return predicted_price_original[0]

# Streamlit app layout
st.title('SaaS Product Pricing Prediction')

st.write('Enter the details of the SaaS product below to predict its price:')

user_base = st.number_input('User Base', min_value=0, value=4000)
cost_to_develop = st.number_input('Cost to Develop', min_value=0.0, value=150000.0)

if st.button('Predict Price'):
    predicted_price = predict_price(user_base, cost_to_develop)
    st.write(f"Predicted SaaS Price: ${predicted_price:.2f}")
