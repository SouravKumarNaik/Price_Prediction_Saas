import streamlit as st
import numpy as np
import joblib

# Load the pre-trained model and scaler
model = joblib.load('Model_Saved')
scaler = joblib.load('StandardScaler_Saved')

# Streamlit UI
st.title('SaaS Pricing Prediction with Ridge Regression')

# Input fields
feature_set = st.number_input('Enter Feature Set', min_value=0)
user_base = st.number_input('Enter User Base', min_value=0)
market_position = st.selectbox('Select Market Position', ['Niche', 'Mainstream', 'Premium'])
customer_satisfaction = st.slider('Enter Customer Satisfaction', min_value=0.0, max_value=5.0, format="%.2f")
competitor_price = st.number_input('Enter Competitor Price', min_value=0.0, format="%.2f")
cost_to_develop = st.number_input('Enter Cost to Develop', min_value=0.0, format="%.2f")
desired_profit_margin = st.slider('Desired Profit Margin', min_value=0, max_value=100)
buyer_budget = st.number_input('Enter Buyer Budget', min_value=0.0, format="%.2f")

# Encode Market Position
market_position_encoded = {'Niche': 1, 'Mainstream': 2, 'Premium': 3}[market_position]

# Collect input data
features = np.array([feature_set, user_base, market_position_encoded, customer_satisfaction, 
                     competitor_price, cost_to_develop, desired_profit_margin, buyer_budget]).reshape(1, -1)

# Scale input data
features_scaled = scaler.transform(features)

# Predict the price
if st.button('Predict Price'):
    predicted_price = model.predict(features_scaled)
    st.write(f"Predicted Price: ${predicted_price[0]:.2f}")
