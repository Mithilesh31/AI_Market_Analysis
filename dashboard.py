import streamlit as st
import pandas as pd
import plotly.express as px
from scripts.sales_forecast import forecast_sales
from scripts.customer_segmentation import segment_customers
from scripts.pricing_analysis import analyze_pricing

# Page config
st.set_page_config(
    page_title="AI Market Analysis",
    layout="wide",
    page_icon="📊"
)

# Custom CSS
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stButton>button {border-radius: 5px;}
    .stSelectbox div {border-radius: 5px;}
    .css-1aumxhk {background-color: #ffffff;}
</style>
""", unsafe_allow_html=True)

# Title
st.title("📈 AI-Powered Market Trend Analysis")

# Sidebar
st.sidebar.header("Navigation")
page = st.sidebar.radio("Select Analysis", 
    ["Sales Forecast", "Customer Segments", "Pricing Analysis"])

# Data loading with cache
@st.cache_data
def load_data():
    return pd.read_csv('data/retail_dataset.csv')

df = load_data()

if page == "Sales Forecast":
    st.header("🔮 Sales Forecasting")
    st.write("30-day sales forecast using Prophet time series modeling")
    
    with st.spinner("Generating forecast..."):
        forecast = forecast_sales()
    
    col1, col2 = st.columns(2)
    with col1:
        st.image('outputs/sales_forecast.png', use_column_width=True)
    with col2:
        st.dataframe(forecast[['ds', 'yhat']].tail(30).rename(columns={
            'ds': 'Date', 'yhat': 'Predicted Sales'
        }).style.background_gradient(cmap='Blues'))

elif page == "Customer Segments":
    st.header("👥 Customer Segmentation")
    st.write("Behavioral clusters based on purchasing patterns")
    
    with st.spinner("Analyzing customer segments..."):
        segments = segment_customers(df)
    
    st.plotly_chart(segments['plot'], use_container_width=True)
    
    st.subheader("Segment Characteristics")
    st.dataframe(segments['segment_stats'].style.format({
        'Avg Units Purchased': '{:.1f}',
        'Avg Spending': '₹{:.2f}',
        'Avg Purchase Frequency': '{:.1f}'
    }))

elif page == "Pricing Analysis":
    st.header("💰 Pricing Sensitivity")
    st.write("Impact of pricing on product demand")
    
    with st.spinner("Calculating price elasticity..."):
        pricing = analyze_pricing(df)
    
    st.plotly_chart(pricing['price_demand_curve'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Price Elasticity R² Score", f"{pricing['r2_score']:.2f}")
    with col2:
        st.write("Optimal Pricing Recommendations:")
        st.dataframe(pricing['price_recommendations'][['product', 'price', 'elasticity']]
                     .rename(columns={
                         'product': 'Product',
                         'price': 'Current Price',
                         'elasticity': 'Elasticity'
                     }).style.format({
                         'Current Price': '₹{:.2f}',
                         'Elasticity': '{:.3f}'
                     }))

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "AI Market Analysis Dashboard\n\n"
    "Using synthetic retail data to demonstrate:\n"
    "- Time Series Forecasting\n"
    "- Customer Segmentation\n"
    "- Price Elasticity Modeling"
)