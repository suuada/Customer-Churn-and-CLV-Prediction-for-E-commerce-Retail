import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import joblib
import numpy as np
import sys
sys.path.append('..')

# ── Load main data ─────────────────────────────────────
df = pd.read_csv('../reports/master_customer_table_full.csv', index_col='Customer ID')
comparison_df = pd.read_csv('../reports/year_comparison.csv')

# ── Load model and scaler for company upload ───────────
model = joblib.load('../models/churn_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

# ── API connection ─────────────────────────────────────
try:
    response = requests.get("http://127.0.0.1:8000/stats", timeout=3)
    stats = response.json()
    api_connected = True
except:
    api_connected = False

# ── Page config ────────────────────────────────────────
st.set_page_config(page_title="Customer Analytics Dashboard", layout="wide")
st.title("Customer Churn & Lifetime Value Dashboard")

if api_connected:
    st.success("Connected to Customer Analytics API")
else:
    st.warning("API offline - showing local data")

# ── KPI Row ────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
if api_connected:
    col1.metric("Total Customers", stats['total_customers'])
    col2.metric("Churn Rate", f"{stats['churn_rate']}%")
    col3.metric("Avg CLV", f"£{stats['average_clv']:,.0f}")
    col4.metric("High Risk Customers", stats['high_risk_customers'])
else:
    col1.metric("Total Customers", len(df))
    col2.metric("Churn Rate", f"{df['churn'].mean()*100:.1f}%")
    col3.metric("Avg CLV", f"£{df['clv'].mean():,.0f}")
    col4.metric("High Risk Customers", len(df[df['churn_probability'] > 0.5]))

st.divider()

# ── Customer Segmentation ──────────────────────────────
st.subheader("Customer Segmentation")
seg_counts = df['segment'].value_counts()
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(df.groupby('segment')[['frequency', 'monetary', 'churn_probability']].mean().round(2))
with col2:
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {'Champion': 'gold', 'Loyal': 'steelblue', 'Promising': 'seagreen',
              'At Risk': 'orange', 'Lost': 'tomato'}
    seg_counts.plot(kind='bar', ax=ax,
                   color=[colors[s] for s in seg_counts.index], edgecolor='white')
    ax.set_title('Customers per Segment')
    ax.set_ylabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Customer Lookup ────────────────────────────────────
st.subheader("Customer Lookup")
customer_id = st.selectbox("Select a Customer ID", df.index.tolist())
if api_connected:
    try:
        r = requests.get(f"http://127.0.0.1:8000/customer/{customer_id}", timeout=3)
        c = r.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segment", c['segment'])
        col2.metric("Churn Risk", f"{c['churn_probability']}%")
        col3.metric("Predicted CLV (12mo)", f"£{c['predicted_clv']:,.0f}")
        col4.metric("Total Spent", f"£{c['total_spent']:,.0f}")
    except:
        customer = df.loc[customer_id]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Segment", customer['segment'])
        col2.metric("Churn Risk", f"{customer['churn_probability']*100:.1f}%")
        col3.metric("Predicted CLV (12mo)", f"£{customer['clv']:,.0f}")
        col4.metric("Total Spent", f"£{customer['monetary']:,.0f}")
else:
    customer = df.loc[customer_id]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Segment", customer['segment'])
    col2.metric("Churn Risk", f"{customer['churn_probability']*100:.1f}%")
    col3.metric("Predicted CLV (12mo)", f"£{customer['clv']:,.0f}")
    col4.metric("Total Spent", f"£{customer['monetary']:,.0f}")

st.divider()

# ── Top valuable customers ─────────────────────────────
st.subheader("Top 10 Most Valuable Customers")
top10 = df.nlargest(10, 'clv')[['segment', 'clv', 'churn_probability', 'frequency', 'monetary']]
st.dataframe(top10.style.format({'clv': '£{:,.0f}', 'churn_probability': '{:.1%}', 'monetary': '£{:,.0f}'}))

st.divider()

# ── High risk customers ────────────────────────────────
st.subheader("High Churn Risk Customers (>50% probability)")
high_risk = df[df['churn_probability'] > 0.5][['segment', 'churn_probability', 'clv', 'monetary']].sort_values('churn_probability', ascending=False)
st.dataframe(high_risk.style.format({'clv': '£{:,.0f}', 'churn_probability': '{:.1%}', 'monetary': '£{:,.0f}'}))

st.divider()

# ── Year Comparison ────────────────────────────────────
st.subheader("Year-on-Year Comparison: 2009-2010 vs 2010-2011")
col1, col2 = st.columns([1, 2])
with col1:
    st.dataframe(comparison_df.set_index('Metric'))
with col2:
    metrics_to_plot = ['Churn Rate (%)', 'Avg Order Frequency', 'Avg Monetary Value (£)']
    plot_df = comparison_df[comparison_df['Metric'].isin(metrics_to_plot)].set_index('Metric')
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(plot_df))
    width = 0.35
    bars1 = ax.bar(x - width/2, plot_df['Year 2009-2010'], width, label='2009-2010', color='steelblue')
    bars2 = ax.bar(x + width/2, plot_df['Year 2010-2011'], width, label='2010-2011', color='tomato')
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=15, ha='right')
    ax.legend()
    ax.set_title('Key Metrics: Year on Year')
    plt.tight_layout()
    st.pyplot(fig)

st.divider()

# ── Distribution Charts ────────────────────────────────
st.subheader("Distribution Analysis")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(df['churn_probability'], bins=30, color='steelblue', edgecolor='white')
axes[0].set_title('Churn Probability Distribution')
axes[0].set_xlabel('Churn Probability')
axes[0].set_ylabel('Number of Customers')
axes[1].hist(df[df['clv'] > 0]['clv'], bins=30, color='seagreen', edgecolor='white')
axes[1].set_title('CLV Distribution (Customers with CLV > 0)')
axes[1].set_xlabel('Predicted CLV (£)')
axes[1].set_ylabel('Number of Customers')
plt.tight_layout()
st.pyplot(fig)