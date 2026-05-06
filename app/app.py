import sys
import os
import streamlit as st
import pandas as pd

# --- CRITICAL PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

try:
    from src.preprocessing import preprocess_data
    from src.prediction import predict_churn
except ModuleNotFoundError as e:
    st.error(f"Module Import Error: {e}")
    st.stop()

# --- App Configuration (Wide Layout) ---
st.set_page_config(page_title="Customer Churn Predictor", page_icon="⚡", layout="wide")

# --- Custom CSS for Dark UI & Glowing Effects ---
st.markdown("""
    <style>
    /* Main Background adjustments */
    .stApp {
        background-color: #0E1117;
    }
    
    /* Custom Styling for the Predict Button */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        background-color: #00ADB5;
        color: white;
        font-weight: bold;
        border: None;
        padding: 0.6rem 0rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 173, 181, 0.4);
    }
    .stButton > button:hover {
        background-color: #00FFF5;
        color: #121212;
        box-shadow: 0 6px 20px rgba(0, 255, 245, 0.6);
        transform: translateY(-2px);
    }

    /* Styling for the output cards */
    .css-1r6slb0, .css-12oz5g7 {
        background-color: #1A1D24;
        border-radius: 10px;
        padding: 20px;
        border: 1px solid #2A2E39;
    }
    </style>
""", unsafe_allow_html=True)

# --- Header Section ---
st.markdown("<h1 style='text-align: center; color: #00FFF5;'>⚡ Customer Churn Prediction Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #A0AAB2; font-size: 1.1rem;'>AI-Powered Risk Analysis using Optimized LightGBM</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Main Layout: Two Main Columns ---
# Left column for Inputs (User stays here), Right column for Results
left_panel, right_panel = st.columns([1.2, 1], gap="large")

with left_panel:
    st.markdown("### 📋 Customer Profile Inputs")
    st.caption("Adjust the metrics below to simulate different customer scenarios.")
    
    with st.form("customer_data_form"):
        # Split inputs nicely inside the form
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("👤 Age", min_value=18, max_value=100, value=30, step=1)
            gender = st.selectbox("🚻 Gender", options=['Male', 'Female'])
            tenure = st.number_input("📅 Tenure (Months)", min_value=0, max_value=100, value=12, step=1)
            subscription_type = st.selectbox("🎟️ Subscription Type", options=['Basic', 'Standard', 'Premium'])
            contract_length = st.selectbox("📜 Contract Length", options=['Monthly', 'Quarterly', 'Annual'])
            
        with col2:
            total_spend = st.number_input("💰 Total Spend ($)", min_value=0.0, max_value=10000.0, value=500.0, step=10.0)
            payment_delay = st.number_input("⏳ Payment Delay (Days)", min_value=0, max_value=60, value=0, step=1)
            support_calls = st.number_input("📞 Support Calls Made", min_value=0, max_value=20, value=1, step=1)
            usage_frequency = st.number_input("⏱️ Usage (Days/Month)", min_value=0, max_value=30, value=15, step=1)
            last_interaction = st.number_input("💬 Last Interaction (Days Ago)", min_value=0, max_value=100, value=5, step=1)

        # The stylish submit button
        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button(label="🚀 RUN PREDICTION ANALYSIS")

with right_panel:
    st.markdown("### 📊 Live Analysis Result")
    
    if submit_button:
        # Prepare data
        user_data = {
            'Age': age, 'Gender': gender, 'Tenure': tenure,
            'Usage Frequency': usage_frequency, 'Support Calls': support_calls,
            'Payment Delay': payment_delay, 'Subscription Type': subscription_type,
            'Contract Length': contract_length, 'Total Spend': total_spend,
            'Last Interaction': last_interaction
        }
        
        with st.spinner('Running LightGBM Inference Engine...'):
            try:
                processed_data = preprocess_data(user_data)
                result = predict_churn(processed_data)
                
                prob = result['churn_probability']
                
                # --- Result Dashboard UI ---
                st.markdown("<br>", unsafe_allow_html=True)
                
                if result['prediction'] == 1:
                    st.error("#### ⚠️ CRITICAL RISK DETECTED\nThis customer exhibits behavior strongly correlated with churning.")
                    progress_color = "red"
                else:
                    st.success("#### ✅ LOW RISK\nThis customer is currently stable and likely to remain.")
                    progress_color = "green"
                
                st.markdown("---")
                
                # Metrics Row
                m_col1, m_col2 = st.columns(2)
                with m_col1:
                    st.metric(label="Calculated Churn Probability", value=f"{prob * 100:.1f}%")
                with m_col2:
                    st.metric(label="Primary Contract", value=contract_length)
                
                # Visual Progress Bar
                st.progress(float(prob))
                
                # Smart Actionable Insights Engine
                if result['prediction'] == 1:
                    st.markdown("<br><b>💡 Recommended Retention Strategies:</b>", unsafe_allow_html=True)
                    if payment_delay > 15:
                        st.warning("💸 **High Payment Delay:** Proactively offer a temporary payment plan or discount to ease financial friction.")
                    if support_calls > 3:
                        st.warning("🛠️ **Support Frustration:** Trigger a priority call from a senior customer success manager immediately.")
                    if contract_length == 'Monthly':
                        st.info("🔄 **Contract Vulnerability:** Send a targeted email offering 20% off if they lock into an Annual plan today.")
                        
            except Exception as e:
                st.error(f"System Error: {e}")
    else:
        # Default state before user clicks predict
        st.info("👈 Please configure the customer profile on the left and click **Run Prediction Analysis** to view results here.")
        
        # Placeholder design
        st.markdown("""
        <div style="opacity: 0.3; text-align: center; padding-top: 50px;">
            <h1 style="font-size: 80px;">🎯</h1>
            <p>Waiting for data input...</p>
        </div>
        """, unsafe_allow_html=True)