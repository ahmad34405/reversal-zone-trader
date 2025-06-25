
"""
Enhanced Trading Bot with Real TradingView Data Integration
Combines the existing trading logic with real market data
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tradingview_data_service import TradingViewDataService
import pandas as pd
import numpy as np
import sqlite3
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import json
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass
class TrendLine:
    start_point: tuple
    end_point: tuple
    type: str
    direction: str
    strength: float
    touches: int

@dataclass
class SupportResistanceLevel:
    price: float
    type: str
    touches: int
    strength: float
    first_touch: datetime
    last_touch: datetime

@dataclass
class TradingSignal:
    timestamp: datetime
    signal_type: str
    price: float
    confidence: float
    components: List[str]

class EnhancedTradingBot:
    """Enhanced trading bot that uses real TradingView data"""
    
    def __init__(self, symbol: str = "BTC-USD", db_path: str = "enhanced_trading_bot.db"):
        self.symbol = symbol
        self.db_path = db_path
        self.data_service = TradingViewDataService()
        
        # Trading parameters
        self.major_lookback = 200
        self.minor_lookback = 30
        self.min_touches = 3
        self.price_tolerance = 0.05
        
        # Initialize database
        self.init_database()
        
        # Data storage
        self.candles_df = None
        self.candles_list = []
        self.current_signals = []
        self.active_trades = []
        
        logger.info(f"Enhanced trading bot initialized for {symbol}")
    
    def init_database(self):
        """Initialize enhanced database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Enhanced trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                direction TEXT NOT NULL,
                quantity REAL DEFAULT 1.0,
                pnl REAL,
                status TEXT DEFAULT 'OPEN',
                signal_type TEXT,
                confidence REAL,
                notes TEXT
            )
        ''')
        
        # Market data table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                open_price REAL NOT NULL,
                high_price REAL NOT NULL,
                low_price REAL NOT NULL,
                close_price REAL NOT NULL,
                volume INTEGER NOT NULL,
                UNIQUE(symbol, timestamp)
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trading_signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                signal_type TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT DEFAULT 'ACTIVE',
                components TEXT,
                notes TEXT
            )
        ''')
        
        # Trendlines table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trendlines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                start_time TEXT NOT NULL,
                start_price REAL NOT NULL,
                end_time TEXT NOT NULL,
                end_price REAL NOT NULL,
                type TEXT NOT NULL,
                direction TEXT NOT NULL,
                strength REAL NOT NULL,
                touches INTEGER NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Enhanced database schema initialized")
    
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
        self.store_market_data(df)
        
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
    
    def store_market_data(self, df: pd.DataFrame):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            for timestamp, row in df.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.symbol,
                    timestamp.isoformat(),
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    int(row['Volume'])
                ))
            
            conn.commit()
            logger.info(f"Stored {len(df)} candles in database")
            
        except Exception as e:
            logger.error(f"Error storing market data: {e}")
        finally:
            conn.close()
    
    def detect_trendlines(self) -> List[TrendLine]:
        """Detect trendlines from real market data"""
        if not self.candles_list:
            logger.warning("No candle data available for trendline detection")
            return []
        
        trendlines = []
        
        # Major trendlines (using more data points)
        if len(self.candles_list) >= self.major_lookback:
            major_trendlines = self._find_trendlines(
                self.candles_list[-self.major_lookback:], 
                "major"
            )
            trendlines.extend(major_trendlines)
        
        # Minor trendlines (using recent data)
        if len(self.candles_list) >= self.minor_lookback:
            minor_trendlines = self._find_trendlines(
                self.candles_list[-self.minor_lookback:], 
                "minor"
            )
            trendlines.extend(minor_trendlines)
        
        # Store trendlines in database
        self.store_trendlines(trendlines)
        
        logger.info(f"Detected {len(trendlines)} trendlines")
        return trendlines
    
    def _find_trendlines(self, candles: List[Candle], trendline_type: str) -> List[TrendLine]:
        """Find trendlines from swing points"""
        if len(candles) < 10:
            return []
        
        # Find swing highs and lows
        swing_highs = self._find_swing_points(candles, "high")
        swing_lows = self._find_swing_points(candles, "low")
        
        trendlines = []
        
        # Create trendlines from swing points
        if len(swing_highs) >= 2:
            high_trendline = self._create_trendline_from_points(
                swing_highs, candles, trendline_type, "resistance"
            )
            if high_trendline:
                trendlines.append(high_trendline)
        
        if len(swing_lows) >= 2:
            low_trendline = self._create_trendline_from_points(
                swing_lows, candles, trendline_type, "support"
            )
            if low_trendline:
                trendlines.append(low_trendline)
        
        return trendlines
    
    def _find_swing_points(self, candles: List[Candle], point_type: str) -> List[tuple]:
        """Find swing high or low points"""
        points = []
        lookback = 5  # Look 5 candles back and forward
        
        for i in range(lookback, len(candles) - lookback):
            current_candle = candles[i]
            is_swing = True
            
            # Check if current point is a swing high/low
            for j in range(i - lookback, i + lookback + 1):
                if j == i:
                    continue
                
                compare_candle = candles[j]
                
                if point_type == "high":
                    if compare_candle.high >= current_candle.high:
                        is_swing = False
                        break
                else:  # low
                    if compare_candle.low <= current_candle.low:
                        is_swing = False
                        break
            
            if is_swing:
                price = current_candle.high if point_type == "high" else current_candle.low
                points.append((current_candle.timestamp, price, i))
        
        return points
    
    def _create_trendline_from_points(self, points: List[tuple], candles: List[Candle], 
                                    trendline_type: str, direction_hint: str) -> Optional[TrendLine]:
        """Create trendline from swing points"""
        if len(points) < 2:
            return None
        
        # Use first and last points
        start_point = points[0]
        end_point = points[-1]
        
        # Determine direction
        direction = "ascending" if end_point[1] > start_point[1] else "descending"
        
        # Count touches (points near the trendline)
        touches = self._count_trendline_touches(start_point, end_point, points)
        
        if touches < 2:  # Need at least 2 touches
            return None
        
        # Calculate strength
        strength = min(touches / len(points), 1.0)
        
        return TrendLine(
            start_point=(start_point[0], start_point[1]),
            end_point=(end_point[0], end_point[1]),
            type=trendline_type,
            direction=direction,
            strength=strength,
            touches=touches
        )
    
    def _count_trendline_touches(self, start_point: tuple, end_point: tuple, 
                                points: List[tuple]) -> int:
        """Count how many points touch the trendline"""
        if start_point[0] == end_point[0]:  # Vertical line
            return 0
        
        touches = 0
        tolerance = self.price_tolerance
        
        # Calculate line parameters
        time_diff = (end_point[0] - start_point[0]).total_seconds()
        price_diff = end_point[1] - start_point[1]
        slope = price_diff / time_diff if time_diff != 0 else 0
        
        for point in points:
            # Calculate expected price at this time
            point_time_diff = (point[0] - start_point[0]).total_seconds()
            expected_price = start_point[1] + slope * point_time_diff
            
            # Check if point is close to the line
            if abs(point[1] - expected_price) <= tolerance:
                touches += 1
        
        return touches
    
    def store_trendlines(self, trendlines: List[TrendLine]):
        """Store detected trendlines in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Clear old trendlines for this symbol
            conn.execute('DELETE FROM trendlines WHERE symbol = ?', (self.symbol,))
            
            for trendline in trendlines:
                conn.execute('''
                    INSERT INTO trendlines 
                    (symbol, start_time, start_price, end_time, end_price, 
                     type, direction, strength, touches, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    self.symbol,
                    trendline.start_point[0].isoformat(),
                    trendline.start_point[1],
                    trendline.end_point[0].isoformat(),
                    trendline.end_point[1],
                    trendline.type,
                    trendline.direction,
                    trendline.strength,
                    trendline.touches,
                    datetime.now().isoformat()
                ))
            
            conn.commit()
            logger.info(f"Stored {len(trendlines)} trendlines in database")
            
        except Exception as e:
            logger.error(f"Error storing trendlines: {e}")
        finally:
            conn.close()
    
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
        signals = []
        current_price = self.candles_list[-1].close if self.candles_list else 0
        current_time = datetime.now()
        
        # Signal generation logic based on trendlines
        for trendline in trendlines:
            # Calculate current trendline price
            trendline_price = self._calculate_trendline_price_at_time(trendline, current_time)
            
            if trendline_price is None:
                continue
            
            # Check if price is near trendline
            price_diff = abs(current_price - trendline_price)
            
            if price_diff <= self.price_tolerance:
                signal_type = f"{trendline.type.upper()}_{trendline.direction.upper()}_TOUCH"
                confidence = trendline.strength * 0.8  # Base confidence on trendline strength
                
                signal = TradingSignal(
                    timestamp=current_time,
                    signal_type=signal_type,
                    price=trendline_price,
                    confidence=confidence,
                    components=[f"{trendline.type}_trendline"]
                )
                
                signals.append(signal)
        
        # Store signals
        self.store_trading_signals(signals)
        
        return signals
    
    def _calculate_trendline_price_at_time(self, trendline: TrendLine, target_time: datetime) -> Optional[float]:
        """Calculate trendline price at a specific time"""
        start_time, start_price = trendline.start_point
        end_time, end_price = trendline.end_point
        
        if start_time == end_time:
            return None
        
        # Linear interpolation/extrapolation
        time_total = (end_time - start_time).total_seconds()
        time_to_target = (target_time - start_time).total_seconds()
        
        if time_total == 0:
            return start_price
        
        price_diff = end_price - start_price
        slope = price_diff / time_total
        
        calculated_price = start_price + slope * time_to_target
        return calculated_price
    
    def store_trading_signals(self, signals: List[TradingSignal]):
        """Store trading signals in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            for signal in signals:
                conn.execute('''
                    INSERT INTO trading_signals 
                    (symbol, timestamp, signal_type, price, confidence, components)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    self.symbol,
                    signal.timestamp.isoformat(),
                    signal.signal_type,
                    signal.price,
                    signal.confidence,
                    json.dumps(signal.components)
                ))
            
            conn.commit()
            logger.info(f"Stored {len(signals)} trading signals")
            
        except Exception as e:
            logger.error(f"Error storing trading signals: {e}")
        finally:
            conn.close()
    
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
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get data statistics
            data_count = conn.execute(
                'SELECT COUNT(*) FROM market_data WHERE symbol = ?', 
                (self.symbol,)
            ).fetchone()[0]
            
            # Get recent signals
            recent_signals = conn.execute('''
                SELECT COUNT(*) FROM trading_signals 
                WHERE symbol = ? AND datetime(timestamp) > datetime('now', '-1 hour')
            ''', (self.symbol,)).fetchone()[0]
            
            # Get trendline count
            trendline_count = conn.execute(
                'SELECT COUNT(*) FROM trendlines WHERE symbol = ?',
                (self.symbol,)
            ).fetchone()[0]
            
            current_price = None
            if self.candles_list:
                current_price = self.candles_list[-1].close
            
            return {
                'symbol': self.symbol,
                'current_price': current_price,
                'data_points': data_count,
                'trendlines_detected': trendline_count,
                'recent_signals': recent_signals,
                'last_analysis': datetime.now().isoformat(),
                'status': 'active' if self.candles_list else 'no_data'
            }
            
        except Exception as e:
            logger.error(f"Error getting status summary: {e}")
            return {'error': str(e)}
        finally:
            conn.close()

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
