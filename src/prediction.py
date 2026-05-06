import joblib
import os
import pandas as pd

# Define absolute path to load the saved LightGBM model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/LGBMClassifier.pkl')

# Load the model
try:
    lgbm_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Could not load the model. Ensure LGBMClassifier.pkl exists. Error: {e}")

def predict_churn(processed_df: pd.DataFrame) -> dict:
    """
    Takes the preprocessed DataFrame, dynamically selects the RFECV features, 
    and returns the churn prediction.
    """
    # Get the exact features the LightGBM model was trained on
    expected_features = lgbm_model.feature_name_
    
    # --- THE FIX: Replace spaces with underscores in the columns ---
    processed_df.columns = processed_df.columns.str.replace(' ', '_')
    
    # Filter the processed dataframe to keep ONLY those features
    final_df = processed_df[expected_features]
    
    # Get the prediction and probability
    prediction = lgbm_model.predict(final_df)[0]
    probability = lgbm_model.predict_proba(final_df)[0][1]
    
    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "status": "Will Churn" if prediction == 1 else "Will Stay"
    }