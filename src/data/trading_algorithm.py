
"""
Complete Modular Trading Bot Implementation
Price Action-Based Reversal Strategy

This file contains the complete Python implementation that would run
alongside the React dashboard. The React app simulates the real-time
data that this Python backend would generate.
"""

import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===== DATA STRUCTURES =====

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
    start_point: Tuple[datetime, float]
    end_point: Tuple[datetime, float]
    type: str  # 'major' or 'minor'
    direction: str  # 'ascending' or 'descending'
    strength: float  # confidence score
    touches: int

@dataclass
class SupportResistanceLevel:
    price: float
    type: str  # 'support' or 'resistance'
    touches: int
    strength: float
    first_touch: datetime
    last_touch: datetime

@dataclass
class IntersectionPoint:
    price: float
    timestamp: datetime
    type: str  # type of intersection
    confidence: float
    components: List[str]  # what creates this intersection

@dataclass
class Trade:
    id: int
    entry_price: float
    exit_price: float
    entry_time: datetime
    exit_time: datetime
    direction: str  # 'LONG' or 'SHORT'
    pnl: float
    duration_minutes: int
    intersection_type: str

# ===== DATA LOADER =====

class DataLoader:
    """Handles loading and generating OHLC candlestick data"""
    
    def __init__(self):
        self.data = []
    
    def generate_synthetic_data(self, periods: int = 500, start_price: float = 100.0) -> List[Candle]:
        """Generate synthetic OHLC data for testing"""
        candles = []
        current_time = datetime.now() - timedelta(minutes=periods)
        price = start_price
        
        for i in range(periods):
            # Create realistic price movement
            price_change = np.random.normal(0, 0.2)
            open_price = price
            
            # Generate high and low with some volatility
            volatility = np.random.uniform(0.1, 0.5)
            high = open_price + np.random.uniform(0, volatility)
            low = open_price - np.random.uniform(0, volatility)
            
            # Close price influenced by trend and noise
            trend = np.sin(i / 50) * 0.1  # Long-term trend
            close = open_price + price_change + trend
            close = max(low, min(high, close))  # Ensure close is between high and low
            
            volume = np.random.randint(100, 1000)
            
            candle = Candle(
                timestamp=current_time + timedelta(minutes=i),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(close, 2),
                volume=volume
            )
            
            candles.append(candle)
            price = close
        
        logger.info(f"Generated {len(candles)} synthetic candles")
        return candles
    
    def load_from_csv(self, filepath: str) -> List[Candle]:
        """Load data from CSV file"""
        try:
            df = pd.read_csv(filepath)
            candles = []
            
            for _, row in df.iterrows():
                candle = Candle(
                    timestamp=pd.to_datetime(row['timestamp']),
                    open=float(row['open']),
                    high=float(row['high']),
                    low=float(row['low']),
                    close=float(row['close']),
                    volume=int(row['volume'])
                )
                candles.append(candle)
            
            logger.info(f"Loaded {len(candles)} candles from {filepath}")
            return candles
            
        except Exception as e:
            logger.error(f"Error loading data from {filepath}: {e}")
            return []

# ===== TRENDLINE DETECTION =====

class TrendLineDetector:
    """Detects major and minor trendlines from candlestick data"""
    
    def __init__(self):
        self.major_lookback = 200
        self.minor_lookback = 30
        self.min_touches = 2
    
    def find_swing_points(self, candles: List[Candle], lookback: int) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
        """Find swing highs and lows"""
        if len(candles) < lookback:
            return [], []
        
        highs = []
        lows = []
        
        for i in range(lookback//2, len(candles) - lookback//2):
            window = candles[i - lookback//2:i + lookback//2 + 1]
            current_high = candles[i].high
            current_low = candles[i].low
            
            # Check if it's a swing high
            if all(current_high >= c.high for c in window):
                highs.append((i, current_high))
            
            # Check if it's a swing low
            if all(current_low <= c.low for c in window):
                lows.append((i, current_low))
        
        return highs, lows
    
    def calculate_trendline(self, points: List[Tuple[int, float]], candles: List[Candle]) -> Optional[TrendLine]:
        """Calculate trendline from swing points"""
        if len(points) < 2:
            return None
        
        # Use first and last points for trendline
        start_idx, start_price = points[0]
        end_idx, end_price = points[-1]
        
        # Calculate how many points are near this trendline
        touches = 0
        tolerance = 0.1  # Price tolerance for considering a "touch"
        
        for idx, price in points:
            # Calculate expected price at this index based on trendline
            progress = (idx - start_idx) / (end_idx - start_idx)
            expected_price = start_price + (end_price - start_price) * progress
            
            if abs(price - expected_price) <= tolerance:
                touches += 1
        
        if touches < self.min_touches:
            return None
        
        direction = 'ascending' if end_price > start_price else 'descending'
        strength = min(touches / len(points), 1.0)
        
        return TrendLine(
            start_point=(candles[start_idx].timestamp, start_price),
            end_point=(candles[end_idx].timestamp, end_price),
            type='major' if len(candles) >= self.major_lookback else 'minor',
            direction=direction,
            strength=strength,
            touches=touches
        )
    
    def detect_trendlines(self, candles: List[Candle]) -> List[TrendLine]:
        """Detect both major and minor trendlines"""
        trendlines = []
        
        # Major trendlines (200 candles)
        if len(candles) >= self.major_lookback:
            major_data = candles[-self.major_lookback:]
            major_highs, major_lows = self.find_swing_points(major_data, 20)
            
            # Create trendlines from highs and lows
            high_trendline = self.calculate_trendline(major_highs, major_data)
            low_trendline = self.calculate_trendline(major_lows, major_data)
            
            if high_trendline:
                high_trendline.type = 'major'
                trendlines.append(high_trendline)
            
            if low_trendline:
                low_trendline.type = 'major'
                trendlines.append(low_trendline)
        
        # Minor trendlines (30 candles)
        if len(candles) >= self.minor_lookback:
            minor_data = candles[-self.minor_lookback:]
            minor_highs, minor_lows = self.find_swing_points(minor_data, 5)
            
            high_trendline = self.calculate_trendline(minor_highs, minor_data)
            low_trendline = self.calculate_trendline(minor_lows, minor_data)
            
            if high_trendline:
                high_trendline.type = 'minor'
                trendlines.append(high_trendline)
            
            if low_trendline:
                low_trendline.type = 'minor'
                trendlines.append(low_trendline)
        
        logger.info(f"Detected {len(trendlines)} trendlines")
        return trendlines

# ===== SUPPORT/RESISTANCE LEVELS =====

class SupportResistanceDetector:
    """Detects support and resistance levels based on body rejections"""
    
    def __init__(self):
        self.min_touches = 3
        self.price_tolerance = 0.05  # 5 cents tolerance
    
    def identify_rejection_zones(self, candles: List[Candle]) -> List[float]:
        """Identify price levels where rejections occurred"""
        rejection_levels = []
        
        for i in range(1, len(candles) - 1):
            current = candles[i]
            prev_candle = candles[i-1]
            next_candle = candles[i+1]
            
            # Look for rejection patterns
            # Bullish rejection at support (hammer-like)
            body_size = abs(current.close - current.open)
            lower_wick = current.open - current.low if current.close > current.open else current.close - current.low
            upper_wick = current.high - current.close if current.close > current.open else current.high - current.open
            
            # Strong lower wick suggests support
            if lower_wick > body_size * 2 and lower_wick > upper_wick:
                rejection_levels.append(current.low)
            
            # Strong upper wick suggests resistance
            if upper_wick > body_size * 2 and upper_wick > lower_wick:
                rejection_levels.append(current.high)
        
        return rejection_levels
    
    def cluster_levels(self, levels: List[float]) -> List[SupportResistanceLevel]:
        """Cluster nearby price levels and count touches"""
        if not levels:
            return []
        
        clustered_levels = []
        levels.sort()
        
        i = 0
        while i < len(levels):
            level_group = [levels[i]]
            base_level = levels[i]
            
            # Find all levels within tolerance
            j = i + 1
            while j < len(levels) and abs(levels[j] - base_level) <= self.price_tolerance:
                level_group.append(levels[j])
                j += 1
            
            # Only consider levels with minimum touches
            if len(level_group) >= self.min_touches:
                avg_price = sum(level_group) / len(level_group)
                
                # Determine if it's support or resistance based on recent price action
                level_type = 'support'  # Default to support
                
                clustered_levels.append(SupportResistanceLevel(
                    price=round(avg_price, 2),
                    type=level_type,
                    touches=len(level_group),
                    strength=min(len(level_group) / 5.0, 1.0),
                    first_touch=datetime.now() - timedelta(hours=1),
                    last_touch=datetime.now()
                ))
            
            i = j
        
        logger.info(f"Identified {len(clustered_levels)} support/resistance levels")
        return clustered_levels
    
    def detect_levels(self, candles: List[Candle]) -> List[SupportResistanceLevel]:
        """Main method to detect support and resistance levels"""
        rejection_levels = self.identify_rejection_zones(candles)
        return self.cluster_levels(rejection_levels)

# ===== INTERSECTION CALCULATOR =====

class IntersectionCalculator:
    """Calculate intersection points between trendlines and levels"""
    
    def __init__(self):
        self.future_periods = 50  # Look ahead 50 minutes
    
    def line_intersection(self, line1: TrendLine, line2: TrendLine) -> Optional[Tuple[datetime, float]]:
        """Calculate intersection between two trendlines"""
        # Convert to slope-intercept form
        x1_1 = line1.start_point[0].timestamp()
        y1_1 = line1.start_point[1]
        x1_2 = line1.end_point[0].timestamp()
        y1_2 = line1.end_point[1]
        
        x2_1 = line2.start_point[0].timestamp()
        y2_1 = line2.start_point[1]
        x2_2 = line2.end_point[0].timestamp()
        y2_2 = line2.end_point[1]
        
        # Calculate slopes
        if x1_2 == x1_1 or x2_2 == x2_1:
            return None  # Vertical line
        
        m1 = (y1_2 - y1_1) / (x1_2 - x1_1)
        m2 = (y2_2 - y2_1) / (x2_2 - x2_1)
        
        if abs(m1 - m2) < 1e-10:
            return None  # Parallel lines
        
        # Calculate intersection
        b1 = y1_1 - m1 * x1_1
        b2 = y2_1 - m2 * x2_1
        
        x_intersect = (b2 - b1) / (m1 - m2)
        y_intersect = m1 * x_intersect + b1
        
        intersection_time = datetime.fromtimestamp(x_intersect)
        
        # Only return future intersections
        if intersection_time > datetime.now():
            return (intersection_time, y_intersect)
        
        return None
    
    def trendline_level_intersection(self, trendline: TrendLine, level: SupportResistanceLevel) -> Optional[Tuple[datetime, float]]:
        """Calculate when a trendline will intersect a horizontal level"""
        # Extend trendline into the future
        x1 = trendline.start_point[0].timestamp()
        y1 = trendline.start_point[1]
        x2 = trendline.end_point[0].timestamp()
        y2 = trendline.end_point[1]
        
        if x2 == x1:
            return None  # Vertical line
        
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
        
        # Solve for when line equals the level price
        # level.price = slope * x + intercept
        # x = (level.price - intercept) / slope
        
        if abs(slope) < 1e-10:
            return None  # Horizontal line
        
        x_intersect = (level.price - intercept) / slope
        intersection_time = datetime.fromtimestamp(x_intersect)
        
        # Only return future intersections within reasonable time frame
        if intersection_time > datetime.now() and intersection_time < datetime.now() + timedelta(hours=24):
            return (intersection_time, level.price)
        
        return None
    
    def find_all_intersections(self, trendlines: List[TrendLine], levels: List[SupportResistanceLevel]) -> List[IntersectionPoint]:
        """Find all intersection points"""
        intersections = []
        
        # Trendline-Trendline intersections
        for i, line1 in enumerate(trendlines):
            for j, line2 in enumerate(trendlines[i+1:], i+1):
                intersection = self.line_intersection(line1, line2)
                if intersection:
                    time_point, price = intersection
                    
                    # Calculate confidence based on trendline strengths
                    confidence = (line1.strength + line2.strength) / 2
                    
                    intersections.append(IntersectionPoint(
                        price=round(price, 2),
                        timestamp=time_point,
                        type=f"{line1.type.upper()}_TREND_INTERSECTION",
                        confidence=confidence,
                        components=[f"{line1.type}_trendline", f"{line2.type}_trendline"]
                    ))
        
        # Trendline-Level intersections
        for trendline in trendlines:
            for level in levels:
                intersection = self.trendline_level_intersection(trendline, level)
                if intersection:
                    time_point, price = intersection
                    
                    confidence = (trendline.strength + level.strength) / 2
                    
                    intersections.append(IntersectionPoint(
                        price=round(price, 2),
                        timestamp=time_point,
                        type=f"{trendline.type.upper()}_TREND_{level.type.upper()}",
                        confidence=confidence,
                        components=[f"{trendline.type}_trendline", f"{level.type}_level"]
                    ))
        
        # Sort by timestamp
        intersections.sort(key=lambda x: x.timestamp)
        
        logger.info(f"Found {len(intersections)} intersection points")
        return intersections

# ===== TRADING LOGIC =====

class TradingEngine:
    """Handles trading logic and execution"""
    
    def __init__(self, db_manager):
        self.db_manager = db_manager
        self.active_intersections = []
        self.current_trade = None
        self.trade_duration_minutes = [3, 5]  # Random choice between 3 and 5 minutes
    
    def update_intersections(self, intersections: List[IntersectionPoint]):
        """Update the list of active intersection points"""
        # Filter out past intersections and low confidence ones
        current_time = datetime.now()
        self.active_intersections = [
            intersection for intersection in intersections
            if intersection.timestamp > current_time and intersection.confidence > 0.6
        ]
        
        logger.info(f"Updated active intersections: {len(self.active_intersections)}")
    
    def check_intersection_trigger(self, current_price: float) -> Optional[IntersectionPoint]:
        """Check if current price has reached any intersection point"""
        price_tolerance = 0.05  # 5 cents tolerance
        
        for intersection in self.active_intersections:
            if abs(current_price - intersection.price) <= price_tolerance:
                logger.info(f"Triggered intersection at {intersection.price} (current: {current_price})")
                return intersection
        
        return None
    
    def execute_trade(self, intersection: IntersectionPoint, current_price: float):
        """Execute a reversal trade at intersection point"""
        if self.current_trade is not None:
            logger.warning("Already in a trade, skipping new entry")
            return
        
        # Determine trade direction based on intersection type and price action
        # For simplicity, assume we're looking for reversals
        direction = 'LONG' if 'SUPPORT' in intersection.type else 'SHORT'
        
        # Random duration between 3 and 5 minutes
        duration = np.random.choice(self.trade_duration_minutes)
        
        trade_id = int(datetime.now().timestamp())
        
        self.current_trade = {
            'id': trade_id,
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'direction': direction,
            'duration_minutes': duration,
            'intersection_type': intersection.type,
            'target_exit_time': datetime.now() + timedelta(minutes=duration)
        }
        
        logger.info(f"Entered {direction} trade at {current_price}, duration: {duration}min")
        
        # Remove this intersection from active list
        self.active_intersections.remove(intersection)
    
    def check_trade_exit(self, current_price: float):
        """Check if current trade should be exited"""
        if self.current_trade is None:
            return
        
        current_time = datetime.now()
        
        # Exit based on time
        if current_time >= self.current_trade['target_exit_time']:
            self.exit_trade(current_price)
    
    def exit_trade(self, exit_price: float):
        """Exit the current trade"""
        if self.current_trade is None:
            return
        
        exit_time = datetime.now()
        entry_price = self.current_trade['entry_price']
        direction = self.current_trade['direction']
        
        # Calculate P&L
        if direction == 'LONG':
            pnl = exit_price - entry_price
        else:  # SHORT
            pnl = entry_price - exit_price
        
        duration = int((exit_time - self.current_trade['entry_time']).total_seconds() / 60)
        
        # Create trade record
        trade = Trade(
            id=self.current_trade['id'],
            entry_price=entry_price,
            exit_price=exit_price,
            entry_time=self.current_trade['entry_time'],
            exit_time=exit_time,
            direction=direction,
            pnl=round(pnl, 2),
            duration_minutes=duration,
            intersection_type=self.current_trade['intersection_type']
        )
        
        # Save to database
        self.db_manager.save_trade(trade)
        
        logger.info(f"Exited {direction} trade: Entry={entry_price}, Exit={exit_price}, P&L={pnl}")
        
        # Clear current trade
        self.current_trade = None
    
    def get_trade_status(self) -> dict:
        """Get current trading status"""
        return {
            'active_trade': self.current_trade,
            'active_intersections': len(self.active_intersections),
            'next_intersection': self.active_intersections[0] if self.active_intersections else None
        }

# ===== DATABASE MANAGER =====

class DatabaseManager:
    """Handles all database operations"""
    
    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT NOT NULL,
                direction TEXT NOT NULL,
                pnl REAL NOT NULL,
                duration_minutes INTEGER NOT NULL,
                intersection_type TEXT NOT NULL
            )
        ''')
        
        # Signals table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                type TEXT NOT NULL,
                price REAL NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL
            )
        ''')
        
        # Performance metrics table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                total_trades INTEGER NOT NULL,
                winning_trades INTEGER NOT NULL,
                total_pnl REAL NOT NULL,
                win_rate REAL NOT NULL
            )
        ''')
        
        conn.commit()
        conn.close()
        
        logger.info("Database initialized successfully")
    
    def save_trade(self, trade: Trade):
        """Save a completed trade to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (id, entry_price, exit_price, entry_time, exit_time, 
                              direction, pnl, duration_minutes, intersection_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id,
            trade.entry_price,
            trade.exit_price,
            trade.entry_time.isoformat(),
            trade.exit_time.isoformat(),
            trade.direction,
            trade.pnl,
            trade.duration_minutes,
            trade.intersection_type
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved trade {trade.id} to database")
    
    def save_signal(self, intersection: IntersectionPoint, status: str = 'ACTIVE'):
        """Save a trading signal to database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO signals (timestamp, type, price, confidence, status)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            intersection.timestamp.isoformat(),
            intersection.type,
            intersection.price,
            intersection.confidence,
            status
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, limit: int = 50) -> List[Trade]:
        """Get recent trades from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM trades ORDER BY exit_time DESC LIMIT ?
        ''', (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        trades = []
        for row in rows:
            trade = Trade(
                id=row[0],
                entry_price=row[1],
                exit_price=row[2],
                entry_time=datetime.fromisoformat(row[3]),
                exit_time=datetime.fromisoformat(row[4]),
                direction=row[5],
                pnl=row[6],
                duration_minutes=row[7],
                intersection_type=row[8]
            )
            trades.append(trade)
        
        return trades
    
    def get_performance_metrics(self) -> dict:
        """Calculate and return performance metrics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all trades
        cursor.execute('SELECT pnl, direction FROM trades')
        rows = cursor.fetchall()
        
        if not rows:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'total_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0
            }
        
        total_trades = len(rows)
        winning_trades = len([r for r in rows if r[0] > 0])
        losing_trades = len([r for r in rows if r[0] < 0])
        total_pnl = sum([r[0] for r in rows])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        winning_pnls = [r[0] for r in rows if r[0] > 0]
        losing_pnls = [r[0] for r in rows if r[0] < 0]
        
        avg_win = sum(winning_pnls) / len(winning_pnls) if winning_pnls else 0
        avg_loss = sum(losing_pnls) / len(losing_pnls) if losing_pnls else 0
        
        conn.close()
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': round(win_rate, 2),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2)
        }

# ===== MAIN TRADING BOT CLASS =====

class TradingBot:
    """Main trading bot orchestrator"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.trendline_detector = TrendLineDetector()
        self.sr_detector = SupportResistanceDetector()
        self.intersection_calculator = IntersectionCalculator()
        self.db_manager = DatabaseManager()
        self.trading_engine = TradingEngine(self.db_manager)
        
        self.candles = []
        self.is_running = False
        
        logger.info("Trading bot initialized")
    
    def load_data(self, use_synthetic: bool = True):
        """Load historical data"""
        if use_synthetic:
            self.candles = self.data_loader.generate_synthetic_data(periods=300)
        else:
            self.candles = self.data_loader.load_from_csv("market_data.csv")
        
        logger.info(f"Loaded {len(self.candles)} candles")
    
    def analyze_market(self):
        """Perform complete market analysis"""
        if len(self.candles) < 50:
            logger.warning("Not enough data for analysis")
            return
        
        # Detect trendlines
        trendlines = self.trendline_detector.detect_trendlines(self.candles)
        
        # Detect support/resistance levels
        sr_levels = self.sr_detector.detect_levels(self.candles)
        
        # Calculate intersections
        intersections = self.intersection_calculator.find_all_intersections(trendlines, sr_levels)
        
        # Update trading engine
        self.trading_engine.update_intersections(intersections)
        
        # Save signals to database
        for intersection in intersections:
            self.db_manager.save_signal(intersection)
        
        return {
            'trendlines': trendlines,
            'sr_levels': sr_levels,
            'intersections': intersections
        }
    
    def process_new_candle(self, price: float):
        """Process new price data (simulating real-time)"""
        # Create new candle (simplified)
        new_candle = Candle(
            timestamp=datetime.now(),
            open=price,
            high=price + np.random.uniform(0, 0.1),
            low=price - np.random.uniform(0, 0.1),
            close=price,
            volume=np.random.randint(100, 500)
        )
        
        self.candles.append(new_candle)
        
        # Keep only recent candles for performance
        if len(self.candles) > 500:
            self.candles = self.candles[-400:]
        
        # Check for trade triggers
        triggered_intersection = self.trading_engine.check_intersection_trigger(price)
        if triggered_intersection:
            self.trading_engine.execute_trade(triggered_intersection, price)
        
        # Check for trade exits
        self.trading_engine.check_trade_exit(price)
    
    def run_backtest(self):
        """Run a backtest on historical data"""
        logger.info("Starting backtest...")
        
        analysis = self.analyze_market()
        
        # Simulate trading on remaining data
        test_data = self.candles[-50:]  # Test on last 50 candles
        
        for candle in test_data:
            self.process_new_candle(candle.close)
            time.sleep(0.1)  # Simulate real-time processing
        
        # Get performance metrics
        metrics = self.db_manager.get_performance_metrics()
        logger.info(f"Backtest completed. Metrics: {metrics}")
        
        return metrics
    
    def start_live_trading(self):
        """Start live trading simulation"""
        self.is_running = True
        logger.info("Starting live trading...")
        
        # Initial analysis
        self.analyze_market()
        
        # Simulate live price feed
        current_price = self.candles[-1].close if self.candles else 100.0
        
        while self.is_running:
            # Simulate price movement
            price_change = np.random.normal(0, 0.05)
            current_price += price_change
            current_price = round(current_price, 2)
            
            # Process new price
            self.process_new_candle(current_price)
            
            # Re-analyze market every 10 minutes
            if len(self.candles) % 10 == 0:
                self.analyze_market()
            
            # Status update
            status = self.trading_engine.get_trade_status()
            logger.info(f"Price: {current_price}, Active trade: {status['active_trade'] is not None}")
            
            time.sleep(60)  # Wait 1 minute (simulated)
    
    def stop_trading(self):
        """Stop live trading"""
        self.is_running = False
        logger.info("Trading stopped")
    
    def get_status(self) -> dict:
        """Get current bot status"""
        metrics = self.db_manager.get_performance_metrics()
        trading_status = self.trading_engine.get_trade_status()
        
        return {
            'is_running': self.is_running,
            'total_candles': len(self.candles),
            'current_price': self.candles[-1].close if self.candles else 0,
            'performance': metrics,
            'trading_status': trading_status
        }

# ===== MAIN EXECUTION =====

def main():
    """Main execution function"""
    # Initialize trading bot
    bot = TradingBot()
    
    # Load data
    bot.load_data(use_synthetic=True)
    
    # Run backtest
    print("Running backtest...")
    backtest_results = bot.run_backtest()
    print(f"Backtest results: {backtest_results}")
    
    # Start live trading simulation
    print("\nStarting live trading simulation...")
    try:
        bot.start_live_trading()
    except KeyboardInterrupt:
        bot.stop_trading()
        print("\nTrading stopped by user")
    
    # Final status
    final_status = bot.get_status()
    print(f"\nFinal status: {final_status}")

if __name__ == "__main__":
    main()

"""
This complete implementation provides:

1. **Modular Architecture**: Separate classes for each component
2. **Data Handling**: Synthetic data generation and CSV loading
3. **Technical Analysis**: 
   - Major/minor trendline detection from swing points
   - Support/resistance levels based on rejection patterns
   - Intersection point calculations
4. **Trading Logic**: 
   - Real-time monitoring and trade execution
   - 3-5 minute trade duration
   - P&L calculation and tracking
5. **Database Management**: SQLite for storing trades and signals
6. **Performance Tracking**: Win rate, P&L, and other metrics
7. **Logging**: Comprehensive logging for debugging and monitoring

To use this in production:
1. Replace synthetic data with real market data API
2. Add proper risk management and position sizing
3. Implement real broker integration
4. Add more sophisticated entry/exit logic
5. Enhance the web dashboard integration

The React dashboard you see simulates the real-time data that this Python
backend would generate in a production environment.
"""
