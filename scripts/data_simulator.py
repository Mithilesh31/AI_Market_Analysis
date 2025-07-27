import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_retail_data():
    # ABSOLUTE output path
    output_path = '/Users/mithileshr.desale/Desktop/AI_Market_Analysis/data/retail_dataset.csv'
    
    # Generate data (same as before)
    np.random.seed(42)
    dates = [datetime.now() - timedelta(days=np.random.randint(1, 730)) for _ in range(7000)]
    
    df = pd.DataFrame({
        'date': [d.strftime('%Y-%m-%d') for d in dates],
        'product': [f"Product_{i}" for i in np.random.randint(1, 51, 7000)],
        'customer': [f"Cust_{i}" for i in np.random.randint(1, 501, 7000)],
        'price': np.round(np.abs(np.random.normal(300, 100, 7000)), 2),
        'quantity': np.random.randint(1, 10, 7000)
    })

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CORRECT location
    df.to_csv(output_path, index=False)
    print(f"✅ Data saved to:\n{output_path}")

if __name__ == "__main__":
    generate_retail_data()