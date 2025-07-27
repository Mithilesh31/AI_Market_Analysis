import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os

def forecast_sales():
    # Load data
    df = pd.read_csv('../data/retail_dataset.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    # Aggregate daily sales
    daily_sales = df.groupby('date')['quantity'].sum().reset_index()
    daily_sales.columns = ['ds', 'y']
    
    # Train model
    model = Prophet(weekly_seasonality=True)
    model.fit(daily_sales)
    
    # Make future dataframe (30 days)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    
    # Plot
    fig = model.plot(forecast)
    plt.title('30-Day Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    
    # Save results
    os.makedirs('../outputs', exist_ok=True)
    plt.savefig('../outputs/sales_forecast.png')
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30).to_csv('../outputs/forecast_results.csv', index=False)
    
    return forecast

if __name__ == "__main__":
    print("Running sales forecasting...")
    forecast = forecast_sales()
    print("Forecast completed. Results saved to ../outputs/")