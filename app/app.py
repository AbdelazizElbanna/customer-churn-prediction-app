import sys
import os
import streamlit as st
import pandas as pd

# --- CRITICAL PATH FIX ---
# This tells Python to look in the parent directory so it can find the 'src' folder.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Now we can safely import from src
try:
    from src.preprocessing import preprocess_data
    from src.prediction import predict_churn
except ModuleNotFoundError as e:
    st.error(f"Module Import Error: {e}")
    st.error("Please ensure the 'src' folder exists and contains preprocessing.py and prediction.py")
    st.stop()


# --- App Configuration & Styling ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📊", layout="centered")

st.title("📊 Customer Churn Predictor")
st.markdown("""
Enter the customer's details below to predict whether they are likely to churn (cancel their subscription). 
The prediction is powered by a fine-tuned LightGBM model.
""")

# --- Input Form ---
with st.form("customer_data_form"):
    st.subheader("Customer Information")
    
    # Use columns to make the layout look professional and compact
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Demographics & Account**")
        age = st.number_input("Age", min_value=18, max_value=100, value=30, step=1)
        gender = st.selectbox("Gender", options=['Male', 'Female'])
        tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12, step=1)
        subscription_type = st.selectbox("Subscription Type", options=['Basic', 'Standard', 'Premium'])
        contract_length = st.selectbox("Contract Length", options=['Monthly', 'Quarterly', 'Annual'])
        
    with col2:
        st.markdown("**Behavior & Financials**")
        total_spend = st.number_input("Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
        payment_delay = st.number_input("Payment Delay (Days)", min_value=0, max_value=60, value=0, step=1)
        support_calls = st.number_input("Support Calls Made", min_value=0, max_value=20, value=1, step=1)
        usage_frequency = st.number_input("Usage Frequency (Days/Month)", min_value=0, max_value=30, value=15, step=1)
        last_interaction = st.number_input("Last Interaction (Days Ago)", min_value=0, max_value=100, value=5, step=1)

    # Submit button
    submit_button = st.form_submit_button(label="Predict Churn Risk")

# --- Prediction Logic ---
if submit_button:
    # 1. Gather all inputs into a dictionary matching the exact column names expected
    user_data = {
        'Age': age,
        'Gender': gender,
        'Tenure': tenure,
        'Usage Frequency': usage_frequency,
        'Support Calls': support_calls,
        'Payment Delay': payment_delay,
        'Subscription Type': subscription_type,
        'Contract Length': contract_length,
        'Total Spend': total_spend,
        'Last Interaction': last_interaction
    }
    
    with st.spinner('Analyzing customer data...'):
        try:
            # 2. Pass data through the preprocessing pipeline
            processed_data = preprocess_data(user_data)
            
            # 3. Pass cleaned data to the model
            result = predict_churn(processed_data)
            
            # 4. Display Results beautifully
            st.markdown("---")
            st.subheader("Prediction Results")
            
            prob = result['churn_probability']
            
            # Show visual alerts based on the prediction
            if result['prediction'] == 1:
                st.error(f"⚠️ **High Risk:** This customer is likely to CHURN.")
            else:
                st.success(f"✅ **Low Risk:** This customer is likely to STAY.")
            
            # Show Probability metrics
            st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
            st.progress(float(prob))
            
            # Add some dynamic business advice based on EDA insights
            if result['prediction'] == 1:
                st.markdown("### Suggested Actions:")
                if payment_delay > 20:
                    st.info("- **Billing:** The customer has high payment delays. Consider a grace period or a gentle billing reminder.")
                if support_calls > 3:
                    st.info("- **Support:** The customer has contacted support multiple times. Reach out directly to resolve lingering issues.")
                if contract_length == 'Monthly':
                    st.info("- **Retention:** Offer a discount to upgrade them to a Quarterly or Annual plan.")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")