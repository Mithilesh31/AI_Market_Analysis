import pandas as pd
from config import FIGS_DIR
import matplotlib.pyplot as plt

def save_fig(name, dpi=300, fig=None):
    """Save matplotlib figure to outputs/figures"""
    if fig is None:
        fig = plt.gcf()
    fig.savefig(FIGS_DIR / f'{name}.png', dpi=dpi, bbox_inches='tight')
    plt.close()

def resample_sales(df, freq='W'):
    """Resample sales data with flexible frequency"""
    return df.resample(freq, on='date')['quantity'].sum()

def format_currency(series):
    """Format pandas series as currency"""
    return series.apply(lambda x: f'₹{x:,.2f}')
    # Example: Add this to notebooks/utils.py
def new_analysis_function():
    """Your new analysis logic"""
    pass