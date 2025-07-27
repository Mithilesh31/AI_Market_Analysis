import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
from pathlib import Path

def forecast_sales():
    """Generate sales forecast with Prophet"""
    # 1. Set up paths (works from any directory)
    project_root = Path(__file__).parent.parent  # Goes up from scripts/
    data_path = project_root / 'data' / 'retail_dataset.csv'
    output_dir = project_root / 'outputs'

    # 2. Validate paths
    if not data_path.exists():
        raise FileNotFoundError(
            f"❌ Data file missing at: {data_path}\n"
            f"Current working directory: {os.getcwd()}\n"
            f"Files in data/: {[f.name for f in (project_root/'data').iterdir()]}"
        )

    # 3. Load and prepare data
    try:
        df = pd.read_csv(data_path, parse_dates=['date'])
        daily_sales = df.groupby('date')['quantity'].sum().reset_index()
        daily_sales.columns = ['ds', 'y']
    except Exception as e:
        raise ValueError(f"Data loading failed: {str(e)}")

    # 4. Model training
    model = Prophet(weekly_seasonality=True)
    model.fit(daily_sales)

    # 5. Generate forecast
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    # 6. Save results
    output_dir.mkdir(exist_ok=True)
    fig = model.plot(forecast)
    plt.title('30-Day Sales Forecast')
    plt.xlabel('Date')
    plt.ylabel('Units Sold')
    plt.savefig(output_dir / 'sales_forecast.png', bbox_inches='tight')
    plt.close()

    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)\
        .to_csv(output_dir / 'forecast_results.csv', index=False)

    print(f"✅ Forecast saved to: {output_dir}/")
    return forecast

if __name__ == "__main__":
    print("Running sales forecasting...")
    try:
        forecast = forecast_sales()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        exit(1)
