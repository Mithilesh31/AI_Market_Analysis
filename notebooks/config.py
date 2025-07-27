from pathlib import Path

# Data paths
DATA_DIR = Path('../data')
RAW_DATA = DATA_DIR / 'retail_dataset.csv'
CLEAN_DATA = DATA_DIR / 'cleaned_data.csv'

# Output paths
OUTPUT_DIR = Path('../outputs')
FIGS_DIR = OUTPUT_DIR / 'figures'

# Model constants
DEFAULT_SEED = 42
DATE_FORMAT = '%Y-%m-%d'

# Create directories if missing
for dir in [DATA_DIR, OUTPUT_DIR, FIGS_DIR]:
    dir.mkdir(exist_ok=True)