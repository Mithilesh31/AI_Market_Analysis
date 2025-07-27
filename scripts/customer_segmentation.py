import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import os

def segment_customers(df):
    # Calculate customer metrics
    customer_data = df.groupby('customer').agg({
        'quantity': 'sum',
        'price': 'mean',
        'date': 'count'
    }).rename(columns={
        'quantity': 'total_quantity',
        'price': 'avg_spend',
        'date': 'purchase_frequency'
    }).reset_index()

    # Scale features
    scaler = StandardScaler()
    features = scaler.fit_transform(customer_data[['total_quantity', 'avg_spend', 'purchase_frequency']])

    # Cluster customers
    kmeans = KMeans(n_clusters=3, random_state=42)
    customer_data['segment'] = kmeans.fit_predict(features)
    
    # Create visualization
    fig = px.scatter_3d(customer_data, 
                       x='total_quantity', 
                       y='avg_spend', 
                       z='purchase_frequency',
                       color='segment',
                       title='Customer Segmentation')
    
    # Calculate segment statistics
    segment_stats = customer_data.groupby('segment').agg({
        'total_quantity': 'mean',
        'avg_spend': 'mean',
        'purchase_frequency': 'mean'
    }).rename(columns={
        'total_quantity': 'Avg Units Purchased',
        'avg_spend': 'Avg Spending',
        'purchase_frequency': 'Avg Purchase Frequency'
    })
    
    # Save outputs
    os.makedirs('../outputs', exist_ok=True)
    fig.write_html('../outputs/customer_segments.html')
    segment_stats.to_csv('../outputs/segment_stats.csv')
    
    return {
        'plot': fig,
        'segment_stats': segment_stats
    }

if __name__ == "__main__":
    df = pd.read_csv('../data/retail_dataset.csv')
    print("Running customer segmentation...")
    results = segment_customers(df)
    print("Segmentation completed. Results saved to ../outputs/")