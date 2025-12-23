"""
Liquidity Health Index (LHI) Analyzer
=====================================

Internal tool for monitoring crypto market liquidity quality.
Built for practical trade ops use - not academic perfection.

Author: Trade Ops Team
Last Updated: Dec 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')  # suppress pandas warnings for cleaner output

class LiquidityAnalyzer:
    """
    Main class for computing liquidity metrics and health index.
    
    Design philosophy: Keep it simple and interpretable.
    We'd rather have a metric that makes sense to traders than 
    something mathematically perfect but opaque.
    """
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.data = None
        self.metrics = None
        self.lhi_scores = None
        
        # These thresholds were chosen based on observing BTC/ETH markets
        # during various stress periods. Adjust as needed.
        self.spread_threshold_bps = 20  # 20 bps = concerning spread
        self.depth_drop_threshold = 0.5  # 50% depth drop = stress
        self.volume_spike_threshold = 3.0  # 3x normal volume       
 
    def load_data(self, filepath):
        """Load market data from CSV. Assumes standard format."""
        print(f"Loading data for {self.symbol} from {filepath}")
        
        self.data = pd.read_csv(filepath)
        self.data['timestamp'] = pd.to_datetime(self.data['timestamp'])
        self.data = self.data.sort_values('timestamp').reset_index(drop=True)
        
        # Basic data validation
        required_cols = ['bid_price', 'ask_price', 'bid_volume', 'ask_volume', 'trade_volume']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        print(f"Loaded {len(self.data)} data points")
        print(f"Date range: {self.data['timestamp'].min()} to {self.data['timestamp'].max()}")
        
    def compute_basic_metrics(self):
        """
        Compute core liquidity metrics.
        
        These are the bread and butter metrics that any trader understands.
        Nothing fancy, just the stuff that matters for execution quality.
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        df = self.data.copy()
        
        # 1. Bid-Ask Spread Metrics
        df['mid_price'] = (df['bid_price'] + df['ask_price']) / 2
        df['spread_abs'] = df['ask_price'] - df['bid_price']
        df['spread_bps'] = (df['spread_abs'] / df['mid_price']) * 10000  # basis points
        
        # 2. Order Book Depth
        df['total_depth'] = df['bid_volume'] + df['ask_volume']
        df['depth_imbalance'] = abs(df['bid_volume'] - df['ask_volume']) / df['total_depth']
        
        # Rolling averages for comparison (using 1 hour = 60 minutes)
        window = 60
        df['spread_bps_ma'] = df['spread_bps'].rolling(window).mean()
        df['total_depth_ma'] = df['total_depth'].rolling(window).mean()
        
        # 3. Volume Intensity 
        # Compare current volume to recent average
        df['volume_ma'] = df['trade_volume'].rolling(window).mean()
        df['volume_ratio'] = df['trade_volume'] / df['volume_ma']
        
        # 4. Price Impact Proxy
        # Rough estimate: how much would a $10k trade move the price?
        trade_size_usd = 10000
        df['trade_size_coins'] = trade_size_usd / df['mid_price']
        # Simplified: assume linear price impact within the top level
        df['estimated_slippage_bps'] = np.minimum(
            (df['trade_size_coins'] / df['total_depth']) * df['spread_bps'],
            df['spread_bps'] * 2  # cap at 2x spread
        )
        
        # 5. Short-term Volatility
        # Rolling realized volatility (annualized)
        df['price_return'] = df['mid_price'].pct_change()
        df['volatility'] = df['price_return'].rolling(window).std() * np.sqrt(365 * 24 * 60)  # annualized
        
        self.metrics = df
        print("Computed basic liquidity metrics")  
      
    def compute_lhi_score(self):
        """
        Compute the Liquidity Health Index (LHI).
        
        This is where we combine everything into a single 0-100 score.
        The weights are based on what we've seen matter most for trade execution.
        
        Higher score = better liquidity
        """
        if self.metrics is None:
            raise ValueError("No metrics computed. Call compute_basic_metrics() first.")
            
        df = self.metrics.copy()
        
        # Normalize each metric to 0-100 scale
        # We use percentile-based normalization to handle outliers
        
        # 1. Spread Score (lower spread = higher score)
        # Use simpler percentile calculation to avoid the multi-index issue
        df['spread_p10'] = df['spread_bps'].rolling(1440, min_periods=100).quantile(0.1)
        df['spread_p90'] = df['spread_bps'].rolling(1440, min_periods=100).quantile(0.9)
        df['spread_score'] = 100 * (1 - np.clip(
            (df['spread_bps'] - df['spread_p10']) / (df['spread_p90'] - df['spread_p10']), 0, 1
        ))
        
        # 2. Depth Score (higher depth = higher score)
        df['depth_p10'] = df['total_depth'].rolling(1440, min_periods=100).quantile(0.1)
        df['depth_p90'] = df['total_depth'].rolling(1440, min_periods=100).quantile(0.9)
        df['depth_score'] = 100 * np.clip(
            (df['total_depth'] - df['depth_p10']) / (df['depth_p90'] - df['depth_p10']), 0, 1
        )
        
        # 3. Volume Score (moderate volume = good, extreme volume = concerning)
        # Sweet spot is around 1-2x normal volume
        df['volume_score'] = 100 * np.exp(-0.5 * (np.log(np.maximum(df['volume_ratio'], 0.1)))**2)
        
        # 4. Volatility Score (lower vol = higher score)
        df['vol_p10'] = df['volatility'].rolling(1440, min_periods=100).quantile(0.1)
        df['vol_p90'] = df['volatility'].rolling(1440, min_periods=100).quantile(0.9)
        df['volatility_score'] = 100 * (1 - np.clip(
            (df['volatility'] - df['vol_p10']) / (df['vol_p90'] - df['vol_p10']), 0, 1
        ))
        
        # Combine scores with weights
        # These weights reflect what matters most for our trading:
        # - Spread is most important (40%) - directly affects execution cost
        # - Depth is second (30%) - affects market impact
        # - Volume and volatility are supporting indicators (15% each)
        weights = {
            'spread': 0.40,
            'depth': 0.30, 
            'volume': 0.15,
            'volatility': 0.15
        }
        
        df['lhi_score'] = (
            weights['spread'] * df['spread_score'].fillna(50) +
            weights['depth'] * df['depth_score'].fillna(50) +
            weights['volume'] * df['volume_score'].fillna(50) +
            weights['volatility'] * df['volatility_score'].fillna(50)
        )
        
        # Add interpretation categories
        df['lhi_category'] = pd.cut(df['lhi_score'], 
                                   bins=[0, 50, 80, 100],
                                   labels=['Stress', 'Mild Stress', 'Healthy'])
        
        self.lhi_scores = df
        print("Computed LHI scores") 
       
    def identify_stress_periods(self):
        """
        Flag periods of liquidity stress.
        
        This is the practical part - when should traders be extra careful?
        """
        if self.lhi_scores is None:
            raise ValueError("No LHI scores computed. Call compute_lhi_score() first.")
            
        df = self.lhi_scores.copy()
        
        # Define stress conditions
        stress_conditions = {
            'wide_spreads': df['spread_bps'] > self.spread_threshold_bps,
            'thin_depth': df['total_depth'] < (df['total_depth_ma'] * self.depth_drop_threshold),
            'volume_spike': df['volume_ratio'] > self.volume_spike_threshold,
            'low_lhi': df['lhi_score'] < 50
        }
        
        # Combine conditions
        df['is_stress'] = (
            stress_conditions['wide_spreads'] |
            stress_conditions['thin_depth'] |
            stress_conditions['volume_spike'] |
            stress_conditions['low_lhi']
        )
        
        # Find stress periods (consecutive stress points)
        df['stress_group'] = (df['is_stress'] != df['is_stress'].shift()).cumsum()
        stress_periods = []
        
        for group_id, group in df.groupby('stress_group'):
            if group['is_stress'].iloc[0]:  # this is a stress period
                start_time = group['timestamp'].iloc[0]
                end_time = group['timestamp'].iloc[-1]
                duration_minutes = len(group)
                
                # Summarize what went wrong
                issues = []
                if group['spread_bps'].max() > self.spread_threshold_bps:
                    issues.append(f"Wide spreads (max {group['spread_bps'].max():.1f} bps)")
                if (group['total_depth'] < group['total_depth_ma'] * self.depth_drop_threshold).any():
                    issues.append("Thin order book")
                if group['volume_ratio'].max() > self.volume_spike_threshold:
                    issues.append(f"Volume spike ({group['volume_ratio'].max():.1f}x normal)")
                
                stress_periods.append({
                    'start': start_time,
                    'end': end_time,
                    'duration_minutes': duration_minutes,
                    'min_lhi': group['lhi_score'].min(),
                    'issues': '; '.join(issues)
                })
        
        self.stress_periods = stress_periods
        self.lhi_scores = df
        
        print(f"Identified {len(stress_periods)} stress periods")
        return stress_periods
        
    def create_charts(self, save_path=None):
        """
        Create exploratory charts for liquidity analysis.
        
        These aren't meant to be perfect dashboards - just useful plots
        for understanding what's happening in the market.
        """
        if self.lhi_scores is None:
            raise ValueError("No LHI scores computed. Call compute_lhi_score() first.")
            
        df = self.lhi_scores.copy()
        
        # Create a figure with multiple subplots
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        fig.suptitle(f'{self.symbol} Liquidity Analysis', fontsize=16, fontweight='bold')
        
        # 1. LHI Score vs Price
        ax1 = axes[0]
        ax1_twin = ax1.twinx()
        
        # Plot LHI score
        ax1.plot(df['timestamp'], df['lhi_score'], color='blue', alpha=0.7, linewidth=1)
        ax1.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Stress Threshold')
        ax1.axhline(y=80, color='orange', linestyle='--', alpha=0.5, label='Mild Stress Threshold')
        ax1.set_ylabel('LHI Score', color='blue')
        ax1.set_ylim(0, 100)
        ax1.legend(loc='upper left')
        
        # Plot price on secondary axis
        ax1_twin.plot(df['timestamp'], df['mid_price'], color='black', alpha=0.5, linewidth=0.8)
        ax1_twin.set_ylabel('Price (USDT)', color='black')
        
        # Shade stress periods
        if hasattr(self, 'stress_periods'):
            for period in self.stress_periods:
                ax1.axvspan(period['start'], period['end'], alpha=0.2, color='red')
        
        ax1.set_title('Liquidity Health Index vs Price')
        ax1.grid(True, alpha=0.3)
        
        # 2. Spread Analysis
        ax2 = axes[1]
        ax2.plot(df['timestamp'], df['spread_bps'], color='red', alpha=0.7, linewidth=1)
        ax2.axhline(y=self.spread_threshold_bps, color='red', linestyle='--', alpha=0.5)
        ax2.set_ylabel('Spread (bps)')
        ax2.set_title('Bid-Ask Spread Over Time')
        ax2.grid(True, alpha=0.3)
        
        # 3. Order Book Depth
        ax3 = axes[2]
        ax3.plot(df['timestamp'], df['total_depth'], color='green', alpha=0.7, linewidth=1, label='Total Depth')
        ax3.plot(df['timestamp'], df['total_depth_ma'], color='darkgreen', alpha=0.5, linewidth=1, label='1h Average')
        ax3.set_ylabel('Order Book Depth')
        ax3.set_title('Order Book Depth')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Volume Intensity
        ax4 = axes[3]
        ax4.plot(df['timestamp'], df['volume_ratio'], color='purple', alpha=0.7, linewidth=1)
        ax4.axhline(y=self.volume_spike_threshold, color='red', linestyle='--', alpha=0.5)
        ax4.set_ylabel('Volume Ratio (vs 1h avg)')
        ax4.set_xlabel('Time')
        ax4.set_title('Volume Intensity')
        ax4.grid(True, alpha=0.3)
        
        # Format x-axis for all subplots
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        else:
            plt.show()
            
        return fig
    
    def generate_report(self, save_path=None):
        """
        Generate a summary report of liquidity conditions.
        
        This is meant to be a quick read for trade ops - highlight the key issues
        and periods to watch out for.
        """
        if self.lhi_scores is None:
            raise ValueError("No LHI scores computed. Call compute_lhi_score() first.")
            
        df = self.lhi_scores.copy()
        
        # Calculate summary stats
        avg_lhi = df['lhi_score'].mean()
        min_lhi = df['lhi_score'].min()
        stress_time_pct = (df['is_stress'].sum() / len(df)) * 100
        
        avg_spread = df['spread_bps'].mean()
        max_spread = df['spread_bps'].max()
        
        report = f"""
LIQUIDITY HEALTH REPORT - {self.symbol}
{'='*50}

EXECUTIVE SUMMARY
-----------------
Average LHI Score: {avg_lhi:.1f}/100
Worst LHI Score: {min_lhi:.1f}/100
Time in Stress: {stress_time_pct:.1f}% of period

Average Spread: {avg_spread:.1f} bps
Worst Spread: {max_spread:.1f} bps

INTERPRETATION
--------------
"""
        
        if avg_lhi >= 80:
            report += "✓ Overall liquidity conditions are HEALTHY\n"
        elif avg_lhi >= 50:
            report += "! Overall liquidity shows MILD STRESS\n"
        else:
            report += "X Overall liquidity is under SIGNIFICANT STRESS\n"
            
        if stress_time_pct > 20:
            report += f"! High stress frequency ({stress_time_pct:.1f}% of time)\n"
        
        report += f"\nSTRESS PERIODS IDENTIFIED\n{'-'*25}\n"
        
        if hasattr(self, 'stress_periods') and self.stress_periods:
            # Sort by severity (lowest LHI first)
            sorted_periods = sorted(self.stress_periods, key=lambda x: x['min_lhi'])
            
            for i, period in enumerate(sorted_periods[:10]):  # Show worst 10
                report += f"{i+1}. {period['start'].strftime('%m-%d %H:%M')} - {period['end'].strftime('%H:%M')}\n"
                report += f"   Duration: {period['duration_minutes']} minutes\n"
                report += f"   Min LHI: {period['min_lhi']:.1f}\n"
                report += f"   Issues: {period['issues']}\n\n"
        else:
            report += "No significant stress periods identified.\n"
            
        report += f"\nTRADING RECOMMENDATIONS\n{'-'*23}\n"
        
        if avg_lhi < 50:
            report += "- Consider reducing position sizes during stress periods\n"
            report += "- Use limit orders instead of market orders\n"
            report += "- Monitor spreads closely before executing large trades\n"
        elif avg_lhi < 80:
            report += "- Normal trading with increased caution during flagged periods\n"
            report += "- Consider splitting large orders\n"
        else:
            report += "- Normal trading conditions\n"
            report += "- Standard execution strategies should work well\n"
            
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        else:
            print(report)
            
        return report

def analyze_symbol(symbol, data_file):
    """
    Run complete liquidity analysis for a symbol.
    
    This is the main workflow - load data, compute metrics, identify stress,
    create charts, and generate reports.
    """
    print(f"\n{'='*60}")
    print(f"ANALYZING {symbol}")
    print(f"{'='*60}")
    
    # Initialize analyzer
    analyzer = LiquidityAnalyzer(symbol)
    
    try:
        # Load and process data
        analyzer.load_data(data_file)
        analyzer.compute_basic_metrics()
        analyzer.compute_lhi_score()
        stress_periods = analyzer.identify_stress_periods()
        
        # Create outputs
        chart_path = f"output/{symbol.lower().replace('-', '_')}_liquidity_analysis.png"
        report_path = f"output/{symbol.lower().replace('-', '_')}_liquidity_report.txt"
        
        analyzer.create_charts(chart_path)
        analyzer.generate_report(report_path)
        
        print(f"\n✅ Analysis complete for {symbol}")
        print(f"📊 Chart: {chart_path}")
        print(f"📄 Report: {report_path}")
        
        return analyzer
        
    except Exception as e:
        print(f"❌ Error analyzing {symbol}: {str(e)}")
        return None

if __name__ == "__main__":
    """
    Main execution - analyze both BTC and ETH data.
    
    This is set up to run the full analysis pipeline.
    Just run: python lhi_analyzer.py
    """
    
    print("Crypto Liquidity Health Index (LHI) Analyzer")
    print("=" * 50)
    print("Internal tool for trade ops team")
    print("Analyzing BTC-USDT and ETH-USDT liquidity conditions...")
    
    # Analyze both symbols
    symbols_to_analyze = [
        ('BTC-USDT', 'data/btc_usdt_data.csv'),
        ('ETH-USDT', 'data/eth_usdt_data.csv')
    ]
    
    results = {}
    
    for symbol, data_file in symbols_to_analyze:
        analyzer = analyze_symbol(symbol, data_file)
        if analyzer:
            results[symbol] = analyzer
    
    # Quick comparison summary
    if len(results) > 1:
        print(f"\n{'='*60}")
        print("CROSS-SYMBOL COMPARISON")
        print(f"{'='*60}")
        
        for symbol, analyzer in results.items():
            if analyzer.lhi_scores is not None:
                avg_lhi = analyzer.lhi_scores['lhi_score'].mean()
                stress_pct = (analyzer.lhi_scores['is_stress'].sum() / len(analyzer.lhi_scores)) * 100
                print(f"{symbol:10} | Avg LHI: {avg_lhi:5.1f} | Stress Time: {stress_pct:5.1f}%")
    
    print(f"\n🎯 Analysis complete! Check the output/ folder for detailed results.")