
"""
Enhanced Trading Bot with Real TradingView Data Integration
Combines the existing trading logic with real market data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingview_data_service import TradingViewDataService
from database_manager import DatabaseManager
from trendline_detector import TrendlineDetector
from signal_generator import SignalGenerator
from data_models import Candle, TrendLine, TradingSignal

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTradingBot:
    """Enhanced trading bot that uses real TradingView data"""
    
    def __init__(self, symbol: str = "BTC-USD", db_path: str = "enhanced_trading_bot.db"):
        self.symbol = symbol
        self.db_path = db_path
        self.data_service = TradingViewDataService()
        
        # Initialize components
        self.db_manager = DatabaseManager(db_path)
        self.trendline_detector = TrendlineDetector()
        self.signal_generator = SignalGenerator()
        
        # Data storage
        self.candles_df = None
        self.candles_list = []
        self.current_signals = []
        self.active_trades = []
        
        logger.info(f"Enhanced trading bot initialized for {symbol}")
    
    def fetch_and_store_data(self, timeframe: str = "1m", days: int = 7):
        """Fetch real market data and store in database"""
        logger.info(f"Fetching {days} days of {timeframe} data for {self.symbol}")
        
        # Fetch data from TradingView
        df = self.data_service.fetch_data_from_tradingview(
            symbol=self.symbol,
            timeframe=timeframe,
            days=days
        )
        
        if df is None or df.empty:
            logger.error("Failed to fetch market data")
            return False
        
        self.candles_df = df
        self.convert_df_to_candles()
        
        # Store in database
        self.db_manager.store_market_data(self.symbol, df)
        
        logger.info(f"Successfully fetched and stored {len(df)} candles")
        return True
    
    def convert_df_to_candles(self):
        """Convert DataFrame to Candle objects"""
        self.candles_list = []
        
        for timestamp, row in self.candles_df.iterrows():
            candle = Candle(
                timestamp=timestamp,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume'])
            )
            self.candles_list.append(candle)
    
    def detect_trendlines(self) -> List[TrendLine]:
        """Detect trendlines from real market data"""
        trendlines = self.trendline_detector.detect_trendlines(self.candles_list)
        
        # Store trendlines in database
        self.db_manager.store_trendlines(self.symbol, trendlines)
        
        return trendlines
    
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete technical analysis on real data"""
        logger.info("Starting complete technical analysis")
        
        results = {
            'symbol': self.symbol,
            'analysis_time': datetime.now().isoformat(),
            'data_points': len(self.candles_list) if self.candles_list else 0,
            'trendlines': [],
            'support_resistance': [],
            'signals': [],
            'current_price': None
        }
        
        if not self.candles_list:
            logger.warning("No data available for analysis")
            return results
        
        # Get current price
        results['current_price'] = self.candles_list[-1].close
        
        # Detect trendlines
        trendlines = self.detect_trendlines()
        results['trendlines'] = [
            {
                'type': tl.type,
                'direction': tl.direction,
                'strength': tl.strength,
                'touches': tl.touches,
                'start_price': tl.start_point[1],
                'end_price': tl.end_point[1]
            }
            for tl in trendlines
        ]
        
        # Generate trading signals
        signals = self.generate_trading_signals(trendlines)
        results['signals'] = [
            {
                'type': signal.signal_type,
                'price': signal.price,
                'confidence': signal.confidence,
                'timestamp': signal.timestamp.isoformat()
            }
            for signal in signals
        ]
        
        logger.info(f"Analysis complete: {len(trendlines)} trendlines, {len(signals)} signals")
        return results
    
    def generate_trading_signals(self, trendlines: List[TrendLine]) -> List[TradingSignal]:
        """Generate trading signals based on analysis"""
        signals = self.signal_generator.generate_trading_signals(self.candles_list, trendlines)
        
        # Store signals
        self.db_manager.store_trading_signals(self.symbol, signals)
        
        return signals
    
    def create_analysis_chart(self, save_path: str = None):
        """Create comprehensive analysis chart with trendlines"""
        if self.candles_df is None or self.candles_df.empty:
            logger.warning("No data available for charting")
            return None
        
        # Create candlestick chart
        fig = self.data_service.plot_candlestick_chart(
            self.candles_df, 
            symbol=self.symbol,
            save_path=save_path
        )
        
        if fig and hasattr(self, 'current_trendlines'):
            # Add trendlines to chart (if available)
            for trendline in getattr(self, 'current_trendlines', []):
                # Add trendline as a line trace
                fig.add_trace(
                    go.Scatter(
                        x=[trendline.start_point[0], trendline.end_point[0]],
                        y=[trendline.start_point[1], trendline.end_point[1]],
                        mode='lines',
                        name=f'{trendline.type} {trendline.direction}',
                        line=dict(
                            color='yellow' if trendline.type == 'major' else 'orange',
                            width=2 if trendline.type == 'major' else 1,
                            dash='solid'
                        )
                    ),
                    row=1, col=1
                )
        
        return fig
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        status = self.db_manager.get_status_summary(self.symbol)
        
        # Add current price if available
        if self.candles_list:
            status['current_price'] = self.candles_list[-1].close
        
        return status

def main():
    """Example usage of the enhanced trading bot"""
    
    # Test symbols
    test_symbols = ["BTC-USD", "ETH-USD", "AAPL"]
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"Testing Enhanced Trading Bot with {symbol}")
        print(f"{'='*60}")
        
        # Initialize bot
        bot = EnhancedTradingBot(symbol=symbol)
        
        # Fetch real data
        success = bot.fetch_and_store_data(timeframe="1m", days=2)
        
        if success:
            # Run analysis
            analysis_results = bot.run_analysis()
            print(f"\nAnalysis Results:")
            print(f"Data points: {analysis_results['data_points']}")
            print(f"Current price: ${analysis_results['current_price']:.2f}")
            print(f"Trendlines: {len(analysis_results['trendlines'])}")
            print(f"Signals: {len(analysis_results['signals'])}")
            
            # Create chart
            chart_path = f"{symbol.replace('-', '_')}_analysis_chart.html"
            bot.create_analysis_chart(save_path=chart_path)
            print(f"Analysis chart saved to: {chart_path}")
            
            # Get status
            status = bot.get_status_summary()
            print(f"\nBot Status: {status}")
            
        else:
            print(f"Failed to fetch data for {symbol}")
        
        print(f"\nDatabase file: {bot.db_path}")

if __name__ == "__main__":
    main()
