import joblib
import os
import pandas as pd

# Define absolute path to load the saved Random Forest model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '../models/RandomForest.pkl')

# Load the model once when the script runs (Fast for Streamlit)
try:
    rf_model = joblib.load(MODEL_PATH)
except Exception as e:
    raise FileNotFoundError(f"Could not load the model. Ensure RandomForest.pkl exists. Error: {e}")

def predict_churn(processed_df: pd.DataFrame) -> dict:
    """
    Takes the preprocessed DataFrame and returns the churn prediction.
    
    Args:
        processed_df (pd.DataFrame): The single-row dataframe output from preprocess_data().
        
    Returns:
        dict: A dictionary containing the prediction (0 or 1) and the probability.
    """
    # Get the prediction (0 for Not Churn, 1 for Churn)
    prediction = rf_model.predict(processed_df)[0]
    
    # Get the probability of churn (Class 1 probability)
    probability = rf_model.predict_proba(processed_df)[0][1]
    
    # Return as a clean dictionary so Streamlit can easily display it
    return {
        "prediction": int(prediction),
        "churn_probability": float(probability),
        "status": "Will Churn" if prediction == 1 else "Will Stay"
    }