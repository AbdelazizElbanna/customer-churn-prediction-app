import pandas as pd
import joblib
import os

# Define absolute paths to load the saved artifacts
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '../models/encoder.pkl')

# Load the saved Scaler and Encoder
try:
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise FileNotFoundError(f"Could not load preprocessing artifacts. Error: {e}")

def preprocess_data(raw_data: dict) -> pd.DataFrame:
    """
    Preprocesses raw customer data for the LightGBM model.
    """
    df = pd.DataFrame([raw_data])
    
    cols_categorical = ['Gender', 'Subscription Type', 'Contract Length']
    cols_numerical = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                      'Payment Delay', 'Total Spend', 'Last Interaction']
    
    # --- 1. Categorical Encoding ---
    encoded_array = encoder.transform(df[cols_categorical])
    cols_names = encoder.get_feature_names_out(cols_categorical)
    df_encoded = pd.DataFrame(encoded_array, columns=cols_names, index=df.index)
    
    # --- 2. Numerical Scaling ---
    df[cols_numerical] = scaler.transform(df[cols_numerical])
    
    # --- 3. Drop original categorical and concatenate ---
    df = df.drop(columns=cols_categorical)
    df_processed = pd.concat([df, df_encoded], axis=1)
    
    # --- 4. Downcast to int8 ---
    int_cols = [
        'Gender_Female', 'Gender_Male',
        'Subscription Type_Basic', 'Subscription Type_Premium',
        'Subscription Type_Standard', 'Contract Length_Annual',
        'Contract Length_Monthly', 'Contract Length_Quarterly'
    ]
    
    for col in int_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('int8')
            
    # --- 5. Order columns exactly as original X_train ---
    final_feature_order = cols_numerical + int_cols
    df_final = df_processed[final_feature_order]
    
    return df_final