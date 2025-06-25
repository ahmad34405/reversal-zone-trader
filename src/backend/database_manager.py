
"""
Database operations for the Enhanced Trading Bot
"""

import sqlite3
import logging
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any
from data_models import TrendLine, TradingSignal

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handles all database operations for the trading bot"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
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
    
    def store_market_data(self, symbol: str, df: pd.DataFrame):
        """Store market data in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            for timestamp, row in df.iterrows():
                conn.execute('''
                    INSERT OR REPLACE INTO market_data 
                    (symbol, timestamp, open_price, high_price, low_price, close_price, volume)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
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
    
    def store_trendlines(self, symbol: str, trendlines: List[TrendLine]):
        """Store detected trendlines in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Clear old trendlines for this symbol
            conn.execute('DELETE FROM trendlines WHERE symbol = ?', (symbol,))
            
            for trendline in trendlines:
                conn.execute('''
                    INSERT INTO trendlines 
                    (symbol, start_time, start_price, end_time, end_price, 
                     type, direction, strength, touches, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
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
    
    def store_trading_signals(self, symbol: str, signals: List[TradingSignal]):
        """Store trading signals in database"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            for signal in signals:
                conn.execute('''
                    INSERT INTO trading_signals 
                    (symbol, timestamp, signal_type, price, confidence, components)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    symbol,
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
    
    def get_status_summary(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive status summary"""
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Get data statistics
            data_count = conn.execute(
                'SELECT COUNT(*) FROM market_data WHERE symbol = ?', 
                (symbol,)
            ).fetchone()[0]
            
            # Get recent signals
            recent_signals = conn.execute('''
                SELECT COUNT(*) FROM trading_signals 
                WHERE symbol = ? AND datetime(timestamp) > datetime('now', '-1 hour')
            ''', (symbol,)).fetchone()[0]
            
            # Get trendline count
            trendline_count = conn.execute(
                'SELECT COUNT(*) FROM trendlines WHERE symbol = ?',
                (symbol,)
            ).fetchone()[0]
            
            return {
                'symbol': symbol,
                'data_points': data_count,
                'trendlines_detected': trendline_count,
                'recent_signals': recent_signals,
                'last_analysis': datetime.now().isoformat(),
                'status': 'active' if data_count > 0 else 'no_data'
            }
            
        except Exception as e:
            logger.error(f"Error getting status summary: {e}")
            return {'error': str(e)}
        finally:
            conn.close()
