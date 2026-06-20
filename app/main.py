import streamlit as st

st.set_page_config(
    page_title="Customer Analytics Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Customer Analytics Platform")
st.write("Welcome! Use the sidebar to navigate between pages.")

st.markdown("""
### What this platform does:
- **Dashboard** — Explore the full retail customer analysis with churn predictions, CLV, and segmentation
- **Company Analysis** — Upload your own transaction data and get instant predictions for your business

""")