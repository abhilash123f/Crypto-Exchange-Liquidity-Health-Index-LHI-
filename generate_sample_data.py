"""
Quick script to generate some realistic-looking crypto data for testing.
Not meant to be perfect, just good enough to test our LHI system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Set random seed for reproducible "market data"
np.random.seed(42)

def generate_crypto_data(symbol, days=30, base_price=50000):
    """Generate fake but realistic crypto market data"""
    
    # Create timestamps - every minute for the last N days
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days)
    timestamps = pd.date_range(start_time, end_time, freq='1min')
    
    n_points = len(timestamps)
    
    # Generate price walk - more volatile than stocks
    price_changes = np.random.normal(0, 0.002, n_points)  # 0.2% std per minute
    # Add some trending periods
    trend_periods = np.random.choice([0, 1, -1], n_points, p=[0.7, 0.15, 0.15])
    price_changes += trend_periods * 0.001
    
    mid_prices = base_price * np.exp(np.cumsum(price_changes))
    
    # Generate spreads - wider during volatile periods
    volatility = pd.Series(price_changes).rolling(60).std().fillna(0.001)
    base_spread_bps = 5  # 5 basis points base spread
    spread_multiplier = 1 + 10 * volatility  # spreads widen with volatility
    spreads = base_spread_bps * spread_multiplier * mid_prices / 10000
    
    bid_prices = mid_prices - spreads/2
    ask_prices = mid_prices + spreads/2
    
    # Generate volumes - higher during volatile periods
    base_volume = 100
    volume_multiplier = 1 + 5 * volatility
    trade_volumes = np.random.exponential(base_volume * volume_multiplier)
    
    # Order book depths - thinner during stress
    stress_factor = np.maximum(0.1, 1 - 3 * volatility)  # depth drops with volatility
    bid_volumes = np.random.exponential(50) * stress_factor
    ask_volumes = np.random.exponential(50) * stress_factor
    
    # OHLC data (simplified)
    opens = mid_prices
    highs = mid_prices * (1 + np.abs(np.random.normal(0, 0.005, n_points)))
    lows = mid_prices * (1 - np.abs(np.random.normal(0, 0.005, n_points)))
    closes = mid_prices
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'symbol': symbol,
        'bid_price': bid_prices,
        'ask_price': ask_prices,
        'bid_volume': bid_volumes,
        'ask_volume': ask_volumes,
        'trade_volume': trade_volumes,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes
    })
    
    return df

if __name__ == "__main__":
    print("Generating sample crypto data...")
    
    # Generate data for BTC and ETH
    btc_data = generate_crypto_data('BTC-USDT', days=30, base_price=45000)
    eth_data = generate_crypto_data('ETH-USDT', days=30, base_price=2500)
    
    # Save to CSV
    btc_data.to_csv('data/btc_usdt_data.csv', index=False)
    eth_data.to_csv('data/eth_usdt_data.csv', index=False)
    
    print(f"Generated {len(btc_data)} BTC data points")
    print(f"Generated {len(eth_data)} ETH data points")
    print("Data saved to data/ folder")