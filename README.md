# Crypto Exchange Liquidity Health Index (LHI)

Internal tool for monitoring market liquidity quality on BTC-USDT and ETH-USDT pairs.

## What This Does

Computes a simple liquidity health score (0-100) to help our trade ops team spot when markets get sketchy. Built for practical use, not academic research.

**Key Features:**
- Single 0-100 liquidity health score combining spread, depth, volume, and volatility
- Automatic identification of liquidity stress periods
- Visual charts highlighting problematic periods
- Written reports with trading recommendations
- Quick-check script for current conditions

## Quick Start

1. **Install requirements:**
   ```bash
   pip install pandas numpy matplotlib
   ```

2. **Generate sample data (or use your own CSV files):**
   ```bash
   python generate_sample_data.py
   ```

3. **Run full analysis:**
   ```bash
   python lhi_analyzer.py
   ```

4. **Quick health check:**
   ```bash
   python quick_analysis.py
   ```

5. **Interactive analysis:**
   ```bash
   jupyter notebook liquidity_analysis_demo.ipynb
   ```

## Data Format Expected

CSV files with columns:
- `timestamp`: DateTime of the observation
- `bid_price`, `ask_price`: Best bid/ask prices
- `bid_volume`, `ask_volume`: Volume at best bid/ask
- `trade_volume`: Trading volume in the period
- `open`, `high`, `low`, `close`: OHLC data (optional)

## LHI Score Interpretation

- **80-100**: Healthy liquidity (normal trading conditions)
- **50-80**: Mild stress (increased caution, consider splitting large orders)
- **0-50**: Significant stress (reduce position sizes, use limit orders)

## What Gets Flagged as Stress

- Spreads widening beyond 20 basis points
- Order book depth dropping by 50%+ from recent average
- Volume spikes 3x+ normal levels
- LHI score dropping below 50

## Files Generated

- `output/[symbol]_liquidity_analysis.png`: Visual analysis charts
- `output/[symbol]_liquidity_report.txt`: Written summary with trading recommendations

## Methodology Notes

**Weights used in LHI calculation:**
- Spread: 40% (directly affects execution cost)
- Depth: 30% (affects market impact)
- Volume: 15% (unusual activity indicator)
- Volatility: 15% (affects slippage)

**Limitations (we're honest about these):**
- Based on top-of-book data only
- Manually set thresholds (not dynamically optimized)
- Doesn't account for cross-exchange conditions
- Volume spikes might be legitimate market activity

## Built for Trade Ops

This system prioritizes practical utility over mathematical perfection. The goal is to help traders make better execution decisions, not to publish research papers.

Built by trade ops for trade ops.