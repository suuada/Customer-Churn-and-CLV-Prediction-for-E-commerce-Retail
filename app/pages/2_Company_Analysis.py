import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np
import os
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data
import sys
sys.path.append('..')

# ── FIXED MODEL PATH (WORKS EVERYWHERE) ─────────────────────────
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

model_path = os.path.join(BASE_DIR, 'models', 'churn_model.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# ── UI ─────────────────────────────────────────────────────────
st.title("Analyse Your Own Customer Data")
st.write("Upload your transaction data and get churn prediction, CLV estimation, segmentation, and insights.")

# Upload type
upload_type = st.radio(
    "What type of file are you uploading?",
    ["Raw transaction data (Invoice, Customer ID, Quantity, Price, Date)",
     "Pre-processed RFM data (Customer ID, frequency, monetary, recency)"]
)

st.markdown("---")

# File upload
if "Raw transaction" in upload_type:
    st.info("CSV must include: Customer ID, Invoice/Order ID, Date, Quantity, Price")
else:
    st.info("CSV must include: Customer ID, frequency, monetary, recency")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

# ── FUNCTIONS ─────────────────────────────────────────────────
def segment_customer(row):
    if row['recency'] <= 30 and row['frequency'] >= 5:
        return 'Champion'
    elif row['recency'] <= 60 and row['frequency'] >= 3:
        return 'Loyal'
    elif row['recency'] <= 90:
        return 'Promising'
    elif row['recency'] <= 180:
        return 'At Risk'
    else:
        return 'Lost'

def build_rfm(data, date_col, id_col, inv_col, price_col, qty_col):
    data[date_col] = pd.to_datetime(data[date_col])
    data = data.dropna(subset=[id_col])
    data = data[data[qty_col] > 0]
    data = data[data[price_col] > 0]

    data['TotalPrice'] = data[qty_col] * data[price_col]
    ref = data[date_col].max() + pd.Timedelta(days=1)

    rfm = data.groupby(id_col).agg(
        last_purchase=(date_col, 'max'),
        frequency=(inv_col, 'nunique'),
        monetary=('TotalPrice', 'sum')
    ).reset_index()

    rfm['recency'] = (ref - rfm['last_purchase']).dt.days
    rfm = rfm.drop(columns=['last_purchase'])
    rfm['churn'] = (rfm['recency'] > 90).astype(int)

    return rfm, data

# ── MAIN LOGIC ────────────────────────────────────────────────
if uploaded_file is not None:
    uploaded_df = pd.read_csv(uploaded_file)
    uploaded_df.columns = [c.strip() for c in uploaded_df.columns]

    st.write("Preview of uploaded data:")
    st.dataframe(uploaded_df.head())

    try:
        if "Raw transaction" in upload_type:
            # Auto-detect columns
            date_col = [c for c in uploaded_df.columns if 'date' in c.lower()][0]
            qty_col = [c for c in uploaded_df.columns if 'qty' in c.lower() or 'quantity' in c.lower()][0]
            price_col = [c for c in uploaded_df.columns if 'price' in c.lower() or 'unit' in c.lower()][0]
            id_col = [c for c in uploaded_df.columns if 'customer' in c.lower()][0]
            inv_col = [c for c in uploaded_df.columns if 'invoice' in c.lower() or 'order' in c.lower()][0]

            rfm_up, clean_data = build_rfm(uploaded_df, date_col, id_col, inv_col, price_col, qty_col)

            # CLV
            try:
                summary = summary_data_from_transaction_data(
                    clean_data,
                    customer_id_col=id_col,
                    datetime_col=date_col,
                    monetary_value_col='TotalPrice',
                    observation_period_end=clean_data[date_col].max()
                )

                returning = summary[summary['frequency'] > 0]

                if len(returning) > 10:
                    bgf = BetaGeoFitter(penalizer_coef=0.5)
                    bgf.fit(summary['frequency'], summary['recency'], summary['T'])

                    ggf = GammaGammaFitter(penalizer_coef=0.5)
                    ggf.fit(returning['frequency'], returning['monetary_value'])

                    summary['clv'] = ggf.customer_lifetime_value(
                        bgf,
                        summary['frequency'],
                        summary['recency'],
                        summary['T'],
                        summary['monetary_value'],
                        time=12,
                        freq='D'
                    )

                    summary['clv'] = summary['clv'].fillna(0).clip(lower=0)

                    rfm_up = rfm_up.set_index(id_col)
                    rfm_up = rfm_up.join(summary[['clv']], how='left')
                    rfm_up['clv'] = rfm_up['clv'].fillna(0)
                    rfm_up = rfm_up.reset_index()

                    id_col_name = rfm_up.columns[0]
                else:
                    rfm_up['clv'] = 0
                    id_col_name = id_col
            except:
                rfm_up['clv'] = 0
                id_col_name = id_col

        else:
            rfm_up = uploaded_df.copy()
            id_col_name = [c for c in rfm_up.columns if 'customer' in c.lower() or 'id' in c.lower()][0]
            rfm_up['churn'] = (rfm_up['recency'] > 90).astype(int)
            rfm_up['clv'] = 0

        # ── MODEL PREDICTION ─────────────────────────────
        features = scaler.transform(rfm_up[['frequency', 'monetary']])
        rfm_up['churn_probability'] = model.predict_proba(features)[:, 1]
        rfm_up['segment'] = rfm_up.apply(segment_customer, axis=1)

        # ── RESULTS ──────────────────────────────────────
        st.success(f"Analysis complete! {len(rfm_up)} customers processed.")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Customers", len(rfm_up))
        col2.metric("Churn Rate", f"{rfm_up['churn'].mean()*100:.1f}%")
        col3.metric("Avg CLV", f"£{rfm_up['clv'].mean():,.0f}")
        col4.metric("High Risk", len(rfm_up[rfm_up['churn_probability'] > 0.5]))

        st.markdown("---")

        # Segmentation chart
        st.subheader("Customer Segmentation")
        seg_counts = rfm_up['segment'].value_counts()

        fig, ax = plt.subplots()
        seg_counts.plot(kind='bar', ax=ax)
        st.pyplot(fig)

        # Download
        csv = rfm_up.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results",
            data=csv,
            file_name='customer_analysis.csv',
            mime='text/csv'
        )

    except Exception as e:
        st.error(f"Error: {e}")