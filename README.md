<h1 align="center">📊 Customer Churn Prediction App</h1>

<p align="center">
  An end-to-end Machine Learning pipeline to predict customer churn using an optimized <b>LightGBM</b> model with dynamic feature selection, deployed via a highly interactive <b>Streamlit</b> web application.
</p>

<div align="center">
  <a href="https://customer-churn-prediction-app-abdelazizelbanna.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🔴_Live_Demo-Access_App_Here-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Live Demo" />
  </a>
</div>
<br>

---
---

## 🛠️ Tech Stack & Tools

Click on any icon below to visit the official documentation:

<p align="center">
  <a href="https://www.python.org/" target="_blank">
    <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python" />
  </a>
  <a href="https://pandas.pydata.org/" target="_blank">
    <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas" />
  </a>
  <a href="https://scikit-learn.org/" target="_blank">
    <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn" />
  </a>
  <a href="https://lightgbm.readthedocs.io/" target="_blank">
    <img src="https://img.shields.io/badge/LightGBM-000000?style=for-the-badge&logo=lightgbm&logoColor=white" alt="LightGBM" />
  </a>
  <a href="https://streamlit.io/" target="_blank">
    <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit" />
  </a>
  <a href="https://jupyter.org/" target="_blank">
    <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter" />
  </a>
</p>

---

## 🏗️ System Architecture

The application follows a highly modular, decoupled architecture, separating the User Interface, Data Processing, and Model Inference to ensure scalability and maintainability:

1. **User Interface (`app/app.py`):** Built with Streamlit, it captures customer data and sends it as a raw dictionary to the backend.
2. **Data Preprocessing (`src/preprocessing.py`):** 
   - Loads the pre-trained `scaler.pkl` and `encoder.pkl`.
   - Cleans the raw input, applies One-Hot Encoding, Standard Scaling, and ensures exact column mapping and data-type casting (`int8` memory optimization).
3. **Model Prediction (`src/prediction.py`):**
   - Implements **Dynamic Feature Selection**: Automatically reads the `feature_name_` attribute from the loaded LightGBM model to filter out discarded features.
   - Outputs the final Churn Probability and binary prediction back to the UI.

---

## 🔍 Exploratory Data Analysis (EDA) & Insights

During the EDA phase, we discovered that customer churn in this dataset isn't linear but rather driven by **sharp, synthetic-like thresholds**. 

* **Support Calls & Payment Delays:** Customers who made more than 4 support calls or had payment delays exceeding 20 days showed a massive spike in churn rate.
* **Contract Length:** Monthly contracts exhibited significantly higher churn vulnerability compared to Annual contracts.

### Key Visualizations:

<p align="center">
  <img src="reports/figures/eda/correlation%20heatmap.png" alt="Correlation Heatmap" width="45%" />
  <img src="reports/figures/eda/churn%20rate%20num%20features.png" alt="Churn Rate Numerical Features" width="45%" />
</p>
<p align="center">
  <em>(Left) Feature Correlation Heatmap | (Right) Churn Rate vs Numerical Features</em>
</p>

---

## 🤖 Modeling & Evaluation

Initially, a baseline `RandomForestClassifier` was trained, yielding excellent results. However, to optimize for **production speed** and reduce noise, we migrated to a **Light Gradient Boosting Machine (LightGBM)**.

### Model Optimization Pipeline:
1. **Handling Imbalance:** Applied `scale_pos_weight` dynamically based on class distribution.
2. **Feature Selection (RFECV):** Utilized Recursive Feature Elimination with Cross-Validation to strip away noisy features, retaining only the most critical predictors.
3. **Threshold Tuning:** Adjusted the decision threshold (e.g., `> 0.3`) to maximize **Recall**—prioritizing the identification of *all* potential churners over strict precision to match business objectives.

### Performance & Feature Importance:

<p align="center">
  <img src="reports/figures/evaluation/feature%20importance%20for%20LGBM%20model.png" alt="LGBM Feature Importance" width="45%" />
  <img src="reports/figures/evaluation/LGBM%20reslut%20after%20feature%20selection.png" alt="LGBM Confusion Matrix" width="45%" />
</p>

---

## 📁 Folder Architecture

The project adheres to a clean, production-ready structure:
```text
customer-churn-prediction-app/
├── app/
│   └── app.py                      # Main Streamlit UI script
├── data/
│   ├── cleaned/                    # Processed datasets
│   └── raw/                        # Original raw datasets
├── models/
│   ├── encoder.pkl                 # Fitted One-Hot Encoder
│   ├── LGBMClassifier.pkl          # Final tuned LightGBM Model
│   └── scaler.pkl                  # Fitted Standard Scaler
├── notebooks/
│   ├── Cleaning_eda_insights.ipynb # Data cleaning & visualization
│   └── preprocessing_modeling.ipynb# Feature selection & training
├── reports/
│   └── figures/                    # Saved EDA & Evaluation plots
│       ├── eda/
│       └── evaluation/
├── src/
│   ├── prediction.py               # Inference logic & dynamic feature mapping
│   └── preprocessing.py            # Data transformation pipeline
├── .gitignore
├── README.md
└── requirements.txt                # Project dependencies
```
## 🚀 Setup & Installation Guide

Follow these steps to run the project locally on your machine:

---

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/AbdelazizElbanna/customer-churn-prediction-app.git
cd customer-churn-prediction-app
```

---

### 2️⃣ Create & Activate Virtual Environment (Recommended)

```bash
python -m venv venv
```

**Windows:**

```bash
venv\Scripts\activate
```

**Mac/Linux:**

```bash
source venv/bin/activate
```

---

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4️⃣ Run the Application

```bash
streamlit run app/app.py
```

---

### 🌐 Access the App

After running the app, it will be available at:

```text
http://localhost:8501
```

---

## 📌 Notes

* Make sure you have Python installed (>= 3.8 recommended)
* Update pip if needed:

```bash
pip install --upgrade pip
```
