import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import plotly.express as px
import os

def analyze_pricing(df):
    # Calculate price-demand relationship
    price_demand = df.groupby('product').agg({
        'price': 'mean',
        'quantity': 'sum'
    }).reset_index()
    
    # Fit regression model
    X = price_demand[['price']]
    y = price_demand['quantity']
    model = LinearRegression()
    model.fit(X, y)
    
    # Generate predictions for visualization
    price_range = np.linspace(price_demand['price'].min(), price_demand['price'].max(), 100)
    predicted_demand = model.predict(price_range.reshape(-1, 1))
    
    # Create visualization
    fig = px.scatter(price_demand, x='price', y='quantity', 
                     trendline="lowess", 
                     title='Price-Demand Relationship')
    fig.add_scatter(x=price_range, y=predicted_demand, 
                    mode='lines', name='Linear Regression')
    
    # Calculate optimal price points
    price_demand['elasticity'] = (price_demand['quantity'] - model.predict(X)) / price_demand['quantity']
    recommendations = price_demand.sort_values('elasticity').head(5)
    
    # Save outputs
    os.makedirs('../outputs', exist_ok=True)
    fig.write_html('../outputs/price_demand_curve.html')
    recommendations.to_csv('../outputs/price_recommendations.csv', index=False)
    
    return {
        'price_demand_curve': fig,
        'r2_score': model.score(X, y),
        'price_recommendations': recommendations
    }

if __name__ == "__main__":
    df = pd.read_csv('../data/retail_dataset.csv')
    print("Running pricing analysis...")
    results = analyze_pricing(df)
    print("Analysis completed. Results saved to ../outputs/")