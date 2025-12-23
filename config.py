"""
Configuration settings for the LHI system.

These are the key parameters that control how the system works.
Adjust these based on your market observations and trading needs.
"""

# Liquidity stress thresholds
# These were chosen based on observing BTC/ETH markets during various conditions
SPREAD_THRESHOLD_BPS = 20      # Spreads above this are concerning
DEPTH_DROP_THRESHOLD = 0.5     # 50% depth drop indicates stress
VOLUME_SPIKE_THRESHOLD = 3.0   # 3x normal volume is unusual

# LHI scoring weights
# These reflect what matters most for trade execution quality
LHI_WEIGHTS = {
    'spread': 0.40,      # 40% - directly affects execution cost
    'depth': 0.30,       # 30% - affects market impact
    'volume': 0.15,      # 15% - unusual activity indicator
    'volatility': 0.15   # 15% - affects slippage
}

# Rolling window sizes (in minutes)
METRICS_WINDOW = 60        # 1 hour for basic metrics
PERCENTILE_WINDOW = 1440   # 24 hours for percentile calculations

# LHI interpretation thresholds
LHI_HEALTHY_THRESHOLD = 80    # Above this = healthy
LHI_STRESS_THRESHOLD = 50     # Below this = stress

# Chart settings
CHART_STYLE = 'default'
CHART_DPI = 300
CHART_SIZE = (15, 12)

# File paths
DATA_FOLDER = 'data'
OUTPUT_FOLDER = 'output'

# Symbols to analyze
DEFAULT_SYMBOLS = [
    ('BTC-USDT', 'data/btc_usdt_data.csv'),
    ('ETH-USDT', 'data/eth_usdt_data.csv')
]

# Report settings
MAX_STRESS_PERIODS_IN_REPORT = 10  # Show worst N periods
RECENT_STRESS_DAYS = 1             # Consider last N days as "recent"