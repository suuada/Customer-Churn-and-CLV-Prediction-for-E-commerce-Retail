
##  Project Overview
This repository contains the complete source code, machine learning pipelines, and multi-tier application architecture developed for my Bachelor of Science Thesis in Computer Science at the University of New York Tirana.

Unlike subscription-based services, non-contractual retail environments suffer from invisible churn—meaning businesses must mathematically infer when a customer has departed. This system addresses this challenge by converting raw transaction records into strategic business indicators using a modular, three-tier client-server framework.

---

##  Key Architectural Components

### 1. Data Processing & Modeling Layer
* **Data Cleaning Pipeline:** Ingests over 1 million transactions from the UCI Online Retail II dataset, handling deduplication, invalid prices, and missing records.
* **RFM Feature Engineering:** Quantifies customer profiles across Recency, Frequency, and Monetary dimensions.
* **Churn Classifier:** Trains and contrasts predictive models (Logistic Regression, Random Forest, XGBoost), selecting a high-recall baseline ($AUC = 0.778$) to catch at-risk customers early.
* **Probabilistic CLV Forecasting:** Fits BG/NBD and Gamma-Gamma sub-models to estimate customer buying behavior and project revenue shares over 12-month periods.

### 2. Middleware (FastAPI REST Engine)
* Serves analytical outputs through six custom endpoints hosted on port `8000`.
* Includes automated OpenAPI interactive documentation (`/docs`) for programmatic, asynchronous system execution.
* Robust background design with local CSV failovers to ensure operational resilience.

### 3. Presentation Layer (Streamlit Dashboard)
* An interactive multi-page enterprise web platform built completely in Python.
* **Primary Dashboard:** Renders business-wide health metrics, segmented user tiers (Champions, Loyal, At Risk, Promising, Lost), and cohort distributions.
* **Self-Upload Suite:** Allows external companies to drop a transactional CSV file directly into memory for real-time model forecasting without data persistence.


##  Repository Structure

```text
├── data/
│   ├── online_retail_I.xlsx        # Raw transactional records for 2010-2011
│   └── online_retail_II.xlsx       # Raw transactional records for 2009-2010
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data quality assessment & exploration
│   └── 02_full_analysis.ipynb      # RFM engineering, ML model training, & CLV pipeline
├── models/
│   ├── churn_model.pkl             # Serialized Logistic Regression prediction model
│   └── scaler.pkl                  # Serialized StandardScaler object
├── reports/
│   ├── churn_distribution.png      # Active vs Churned distribution bar chart
│   ├── customer_segments.png       # 5-Tier RFM customer segmentation distribution
│   ├── feature_importance.png      # XGBoost feature attribution chart
│   ├── model_comparison.png        # AUC-ROC curve comparing LR, RF, and XGBoost
│   ├── model_evaluation.png        # Best model's confusion matrix & ROC curve
│   └── year_comparison.csv         # Exported Year-on-Year performance summary
├── app/
│   ├── main.py                     # Streamlit multi-page entry point (Welcome Page)
│   └── pages/
│       ├── 1_Dashboard.py          # Primary interactive analytics dashboard page
│       └── 2_Company_Analysis.py   # External business data CSV upload engine
├── api.py                          # FastAPI REST API application script
├── requirements.txt                # System package dependency registry (pandas, fastapi, etc.)
└── README.md                       # Comprehensive setup guide and repository abstracttapi, etc.)
└── README.md                       # Comprehensive setup guide and repository abstract

## Step 1: at the anaconda prompt inside of bash:
               cd C:\Users\suada\Desktop\graduation_project
               python -m uvicorn api:app --reload
## Step 2: cd C:\Users\suada\Desktop\graduation_project\app
             python -m streamlit run main.py
