import streamlit as st
import joblib
import pandas as pd

# Load the trained model and scaler
model = joblib.load('best_reg_model.pkl')
scaler = joblib.load('scaler.pkl')

# Function to predict price
def predict_price(user_base, competitor_price, cost_to_develop, buyer_budget):
    # Create DataFrame for the input data
    input_data = pd.DataFrame({
        'User_Base': [user_base],
        'Competitor_Price': [competitor_price],
        'Cost_to_Develop': [cost_to_develop],
        'Buyer_Budget': [buyer_budget]
    })

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict the price
    predicted_price = model.predict(input_data_scaled)

    return predicted_price[0]

# Streamlit app layout
st.title('SaaS Product Pricing Prediction')

st.write('Enter the details of the SaaS product below to predict its price:')

user_base = st.number_input('User Base', min_value=0, value=75000)
competitor_price = st.number_input('Competitor Price', min_value=0.0, value=20.0)
cost_to_develop = st.number_input('Cost to Develop', min_value=0.0, value=175000.0)
buyer_budget = st.number_input('Buyer Budget', min_value=0.0, value=17.0)

if st.button('Predict Price'):
    predicted_price = predict_price(
        user_base,
        competitor_price,
        cost_to_develop,
        buyer_budget
    )
    st.write(f"Predicted Price: ${predicted_price:.2f}")
