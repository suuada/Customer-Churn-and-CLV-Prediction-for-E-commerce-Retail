# Customer-Churn-and-CLV-Prediction-for-E-commerce-Retail
An end-to-end customer analytics platform for non-contractual retail settings. Features an RFM feature engineering pipeline, an XGBoost &amp; Logistic Regression churn predictor, probabilistic CLV modeling, and a FastAPI middleware layer serving an interactive multi-page Streamlit dashboard. Deployed for BS Thesis in Computer Science at UNYT.
# Customer Churn & Lifetime Value Prediction for E-Commerce & Retail

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


Structure of the roject:
├── data/
│   ├── online_retail_I.xlsx        # Raw transactional records for 2010-2011 [cite: 443]
│   └── online_retail_II.xlsx       # Raw transactional records for 2009-2010 [cite: 443]
├── notebooks/
│   ├── 01_data_exploration.ipynb   # Data quality assessment & exploration [cite: 443]
│   └── 02_full_analysis.ipynb      # RFM engineering, ML model training, & CLV pipeline [cite: 443]
├── models/
│   ├── churn_model.pkl             # Serialized Logistic Regression prediction model [cite: 443]
│   └── scaler.pkl                  # Serialized StandardScaler object [cite: 443]
├── reports/
│   ├── churn_distribution.png      # Active vs Churned distribution bar chart [cite: 443]
│   ├── customer_segments.png       # 5-Tier RFM customer segmentation distribution [cite: 443]
│   ├── feature_importance.png      # XGBoost feature attribution chart [cite: 443]
│   ├── model_comparison.png        # AUC-ROC curve comparing LR, RF, and XGBoost [cite: 443]
│   ├── model_evaluation.png        # Best model's confusion matrix & ROC curve [cite: 443]
│   └── year_comparison.csv         # Exported Year-on-Year performance summary 
├── app/
│   ├── main.py                     # Streamlit multi-page entry point (Welcome Page) [cite: 444]
│   └── pages/
│       ├── 1_Dashboard.py          # Primary interactive analytics dashboard page [cite: 444]
│       └── 2_Company_Analysis.py   # External business data CSV upload engine [cite: 444]
├── api.py                          # FastAPI REST API application script [cite: 444]
├── requirements.txt                # System package dependency registry (pandas, fastapi, etc.)
└── README.md                       # Comprehensive setup guide and repository abstract
