from fastapi import FastAPI, HTTPException
import pandas as pd

df = pd.read_csv('reports/master_customer_table_full.csv', index_col='Customer ID')

app = FastAPI(
    title="Customer Analytics API",
    description="API for Customer Churn Prediction and Lifetime Value Estimation",
    version="1.0.0"
)

@app.get("/")
def root():
    return {"message": "Customer Analytics API is running"}

@app.get("/stats")
def get_stats():
    return {
        "total_customers": len(df),
        "churn_rate": round(df['churn'].mean() * 100, 2),
        "average_clv": round(df['clv'].mean(), 2),
        "high_risk_customers": int((df['churn_probability'] > 0.5).sum()),
        "segments": df['segment'].value_counts().to_dict()
    }

@app.get("/customer/{customer_id}")
def get_customer(customer_id: int):
    if customer_id not in df.index:
        raise HTTPException(status_code=404, detail="Customer not found")
    customer = df.loc[customer_id]
    return {
        "customer_id": customer_id,
        "segment": customer['segment'],
        "churn_probability": round(customer['churn_probability'] * 100, 2),
        "predicted_clv": round(customer['clv'], 2),
        "total_spent": round(customer['monetary'], 2),
        "frequency": int(customer['frequency']),
        "recency_days": int(customer['recency'])
    }

@app.get("/customers/top")
def get_top_customers(limit: int = 10):
    top = df.nlargest(limit, 'clv')[['segment', 'clv', 'churn_probability', 'monetary', 'frequency']]
    return top.reset_index().to_dict(orient='records')

@app.get("/customers/highrisk")
def get_high_risk(threshold: float = 0.5):
    risk = df[df['churn_probability'] > threshold][['segment', 'churn_probability', 'clv', 'monetary']]
    risk = risk.sort_values('churn_probability', ascending=False)
    return risk.reset_index().to_dict(orient='records')

@app.get("/customers/segment/{segment_name}")
def get_by_segment(segment_name: str):
    valid = ['Champion', 'Loyal', 'Promising', 'At Risk', 'Lost']
    if segment_name not in valid:
        raise HTTPException(status_code=400, detail=f"Segment must be one of {valid}")
    result = df[df['segment'] == segment_name][['churn_probability', 'clv', 'monetary', 'frequency']]
    return {
        "segment": segment_name,
        "count": len(result),
        "customers": result.reset_index().to_dict(orient='records')
    }