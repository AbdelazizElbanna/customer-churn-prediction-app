import pandas as pd
import joblib
import os

# Define absolute paths to load the saved artifacts
# Assuming the structure is:
# project_folder/
# ├── models/
# │   ├── scaler.pkl
# │   └── encoder.pkl
# └── src/
#     └── preprocessing.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, '../models/scaler.pkl')
ENCODER_PATH = os.path.join(BASE_DIR, '../models/encoder.pkl')

# Load the saved Scaler and Encoder
try:
    scaler = joblib.load(SCALER_PATH)
    encoder = joblib.load(ENCODER_PATH)
except Exception as e:
    raise FileNotFoundError(f"Could not load preprocessing artifacts. Ensure models exist. Error: {e}")

def preprocess_data(raw_data: dict) -> pd.DataFrame:
    """
    Preprocesses raw customer data from the Streamlit UI to be ready for the Random Forest model.
    
    Args:
        raw_data (dict): A dictionary representing a single customer's data.
                         Keys must match the original feature names.
                         
    Returns:
        pd.DataFrame: A processed DataFrame with exact columns expected by the model.
    """
    # 1. Convert the input dictionary into a pandas DataFrame (1 row)
    df = pd.DataFrame([raw_data])
    
    # 2. Define the exact column groupings as done in the notebook
    cols_categorical = ['Gender', 'Subscription Type', 'Contract Length']
    cols_numerical = ['Age', 'Tenure', 'Usage Frequency', 'Support Calls', 
                      'Payment Delay', 'Total Spend', 'Last Interaction']
    
    # 3. Categorical Encoding (OneHotEncoder)
    encoded_array = encoder.transform(df[cols_categorical])
    cols_names = encoder.get_feature_names_out(cols_categorical)
    df_encoded = pd.DataFrame(encoded_array, columns=cols_names, index=df.index)
    
    # 4. Numerical Scaling (StandardScaler)
    df[cols_numerical] = scaler.transform(df[cols_numerical])
    
    # 5. Drop original categorical columns and concatenate processed data
    df = df.drop(columns=cols_categorical)
    df_processed = pd.concat([df, df_encoded], axis=1)
    
    # 6. Downcast binary columns to int8 to match notebook training state exactly
    int_cols = [
        'Gender_Female', 'Gender_Male',
        'Subscription Type_Basic', 'Subscription Type_Premium',
        'Subscription Type_Standard', 'Contract Length_Annual',
        'Contract Length_Monthly', 'Contract Length_Quarterly'
    ]
    
    for col in int_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('int8')
            
    # 7. Enforce strict column order (Model expects columns in the exact same order as X_train)
    final_feature_order = cols_numerical + int_cols
    df_final = df_processed[final_feature_order]
    
    return df_final