"""
Quick Analysis Script
====================

For when you just want to run a fast check on current market conditions.
No fancy charts, just the key numbers.

Usage: python quick_analysis.py
"""

from lhi_analyzer import LiquidityAnalyzer
import sys

def quick_check(symbol, data_file):
    """Run a quick liquidity health check"""
    
    try:
        analyzer = LiquidityAnalyzer(symbol)
        analyzer.load_data(data_file)
        analyzer.compute_basic_metrics()
        analyzer.compute_lhi_score()
        analyzer.identify_stress_periods()
        
        # Get latest data point (most recent)
        latest = analyzer.lhi_scores.iloc[-1]
        
        # Summary stats
        avg_lhi = analyzer.lhi_scores['lhi_score'].mean()
        current_lhi = latest['lhi_score']
        stress_pct = (analyzer.lhi_scores['is_stress'].sum() / len(analyzer.lhi_scores)) * 100
        
        print(f"\n{symbol} QUICK CHECK")
        print("-" * 20)
        print(f"Current LHI: {current_lhi:.1f}/100")
        print(f"30-day Avg:  {avg_lhi:.1f}/100")
        print(f"Stress Time: {stress_pct:.1f}%")
        
        # Current condition
        if current_lhi >= 80:
            print("Status: ✓ HEALTHY")
        elif current_lhi >= 50:
            print("Status: ! CAUTION")
        else:
            print("Status: X STRESS")
            
        # Recent stress periods (last 24 hours)
        recent_stress = [p for p in analyzer.stress_periods 
                        if (analyzer.lhi_scores['timestamp'].max() - p['start']).days < 1]
        
        if recent_stress:
            print(f"Recent stress periods: {len(recent_stress)}")
        else:
            print("No recent stress periods")
            
        return analyzer
        
    except Exception as e:
        print(f"Error analyzing {symbol}: {e}")
        return None

if __name__ == "__main__":
    print("CRYPTO LIQUIDITY QUICK CHECK")
    print("=" * 30)
    
    # Check both symbols
    symbols = [
        ('BTC-USDT', 'data/btc_usdt_data.csv'),
        ('ETH-USDT', 'data/eth_usdt_data.csv')
    ]
    
    results = {}
    for symbol, data_file in symbols:
        analyzer = quick_check(symbol, data_file)
        if analyzer:
            results[symbol] = analyzer
    
    # Quick comparison
    if len(results) == 2:
        btc_lhi = results['BTC-USDT'].lhi_scores['lhi_score'].iloc[-1]
        eth_lhi = results['ETH-USDT'].lhi_scores['lhi_score'].iloc[-1]
        
        print(f"\nCOMPARISON")
        print("-" * 10)
        if btc_lhi > eth_lhi:
            print(f"BTC liquidity is better ({btc_lhi:.1f} vs {eth_lhi:.1f})")
        elif eth_lhi > btc_lhi:
            print(f"ETH liquidity is better ({eth_lhi:.1f} vs {btc_lhi:.1f})")
        else:
            print("Both pairs have similar liquidity")
    
    print("\nFor detailed analysis, run: python lhi_analyzer.py")