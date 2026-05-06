import sys
import os
import streamlit as st

# --- CRITICAL PATH FIX ---
# This allows app.py to see the src folder outside the app directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.preprocessing import preprocess_data
    from src.prediction import predict_churn
except ModuleNotFoundError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()


# --- App Configuration & Styling ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📈", layout="centered")

st.title("📈 Customer Churn Predictor")
st.markdown("""
Enter the customer's details below to predict whether they are likely to churn. 
*Powered by an Optimized LightGBM Model with Feature Selection.*
""")

# --- Input Form ---
with st.form("customer_data_form"):
    st.subheader("Customer Information")
    
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

    submit_button = st.form_submit_button(label="Predict Churn Risk")

# --- Prediction Logic ---
if submit_button:
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
    
    with st.spinner('Analyzing optimal features...'):
        try:
            # 1. Preprocess the data (outputs 15 original features)
            processed_data = preprocess_data(user_data)
            
            # 2. Predict (Filters down to RFECV features automatically and predicts)
            result = predict_churn(processed_data)
            
            # 3. Display Results
            st.markdown("---")
            st.subheader("Prediction Results")
            
            prob = result['churn_probability']
            
            if result['prediction'] == 1:
                st.error(f"⚠️ **High Risk:** This customer is likely to CHURN.")
            else:
                st.success(f"✅ **Low Risk:** This customer is likely to STAY.")
            
            st.metric(label="Churn Probability", value=f"{prob * 100:.1f}%")
            st.progress(float(prob))
            
            if result['prediction'] == 1:
                st.markdown("### Suggested Actions:")
                if payment_delay > 20:
                    st.info("- **Billing:** The customer has high payment delays. Consider a gentle billing reminder.")
                if support_calls > 3:
                    st.info("- **Support:** The customer has contacted support multiple times. Reach out directly.")
                if contract_length == 'Monthly':
                    st.info("- **Retention:** Offer a discount to upgrade to a Quarterly or Annual plan.")
                    
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")