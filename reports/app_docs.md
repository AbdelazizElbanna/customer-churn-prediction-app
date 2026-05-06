# 📱 Application Documentation: Customer Churn Predictor

## 1. Overview
This document provides a technical overview of the Streamlit web application built for predicting customer churn. The application leverages a LightGBM machine learning model with dynamic feature selection, providing real-time predictions based on user input.

## 2. System Architecture & Data Flow
The application follows a strictly decoupled architecture to separate the user interface from the machine learning logic.

**The Data Flow:**
1. **Input:** The user enters customer demographic and behavioral data via the Streamlit UI (`app.py`).
2. **Collection:** The UI compiles this data into a Python dictionary (`raw_data`).
3. **Preprocessing:** The dictionary is passed to `src/preprocessing.py`, where it is scaled, encoded, and transformed into a pandas DataFrame matching the exact structure of the original training data.
4. **Prediction:** The preprocessed DataFrame is sent to `src/prediction.py`. The LightGBM model dynamically filters the required features (based on prior RFECV selection) and calculates the churn probability.
5. **Output:** The final prediction and business logic recommendations are sent back to the UI for rendering.

## 3. Functions Reference

### `src/preprocessing.py`
**Function:** `preprocess_data(raw_data: dict) -> pd.DataFrame`
* **Purpose:** Cleans and formats the raw user input.
* **Dependencies:** Requires `scaler.pkl` and `encoder.pkl` located in the `models/` directory.
* **Operations:**
  * Applies `OneHotEncoder` to categorical features (Gender, Subscription Type, Contract Length).
  * Applies `StandardScaler` to numerical features (Age, Tenure, Total Spend, etc.).
  * Downcasts binary columns to `int8` for memory optimization.
  * Restores the exact column order expected by the original `X_train`.

### `src/prediction.py`
**Function:** `predict_churn(processed_df: pd.DataFrame) -> dict`
* **Purpose:** Executes the model inference.
* **Dependencies:** Requires `LGBMClassifier.pkl` located in the `models/` directory.
* **Operations:**
  * Extracts required features dynamically using `lgbm_model.feature_name_`.
  * Sanitizes column names (replacing spaces with underscores) to match LightGBM internal formatting.
  * Filters the DataFrame to include only the strictly necessary features.
  * Returns a dictionary containing the binary prediction (0 or 1) and the exact churn probability.

## 4. User Guide (UI Interaction)
The Streamlit interface is divided into two logical sections:
* **Demographics & Account:** Captures stable customer information (Age, Gender, Tenure, etc.).
* **Behavior & Financials:** Captures dynamic interactions (Spend, Delays, Calls).

**Alerts & Recommendations:**
The UI includes dynamic business logic. If a customer is flagged as "High Risk", the app dynamically generates suggested actions based on the specific input parameters (e.g., suggesting a billing reminder if payment delays exceed 20 days).