
# Price Action Trading Bot

A sophisticated modular trading bot that implements a price action-based reversal strategy using technical analysis, trendlines, and support/resistance levels.

## 🎯 Features

### Core Trading Strategy
- **1-minute candlestick analysis** with OHLC data processing
- **Major trendlines** detection from last 200 candles using swing highs/lows
- **Minor trendlines** detection from last 30 candles
- **Support & resistance zones** based on body rejections (minimum 3 touches)
- **Intersection point analysis** for high-probability reversal zones
- **Real-time trade simulation** with 3-5 minute duration trades

### Technical Components
- **Modular Python architecture** with clean separation of concerns
- **SQLite database** for storing trades, signals, and performance metrics
- **Interactive React dashboard** with real-time visualizations
- **Comprehensive logging** and error handling
- **Performance tracking** with win rate, P&L, and trade analytics

### Dashboard Features
- 📊 **Live price charts** with trendline overlays
- 🎯 **Active signal monitoring** with confidence scoring
- 📈 **Trade history tracking** with detailed P&L analysis
- 🔍 **Technical analysis display** showing support/resistance levels
- ⚡ **Real-time status updates** and bot control

## 🚀 Quick Start

### React Dashboard
The dashboard is already running and displays simulated trading data. Click **"Start Bot"** to see real-time price updates and trading signals.

### Python Backend (Production Setup)

```bash
# Install required dependencies
pip install pandas numpy matplotlib plotly sqlalchemy flask streamlit

# The complete Python implementation is in src/data/trading_algorithm.py
# This includes all modules: data_loader, trendlines, levels, intersections, trading_logic, db, and main orchestrator
```

## 📁 Project Structure

```
src/
├── components/
│   └── TradingBot.tsx          # Main React dashboard component
├── data/
│   └── trading_algorithm.py    # Complete Python trading bot implementation
└── pages/
    └── Index.tsx              # Main page

Python Modules (in trading_algorithm.py):
├── DataLoader                 # OHLC data loading and generation
├── TrendLineDetector         # Major/minor trendline detection  
├── SupportResistanceDetector # S/R level identification
├── IntersectionCalculator    # Intersection point analysis
├── TradingEngine            # Trade execution logic
├── DatabaseManager          # SQLite database operations
└── TradingBot               # Main orchestrator class
```

## 🔧 Technical Implementation

### Trendline Detection
- **Swing Point Analysis**: Identifies local highs/lows using configurable lookback periods
- **Multi-timeframe Approach**: Major (200 candles) and minor (30 candles) trendlines
- **Strength Scoring**: Based on number of touches and point alignment
- **Direction Classification**: Ascending/descending trend identification

### Support/Resistance Levels
- **Rejection Pattern Recognition**: Analyzes candlestick body vs. wick ratios
- **Touch Validation**: Minimum 3 touches required for level confirmation
- **Price Clustering**: Groups nearby levels within tolerance range
- **Strength Calculation**: Based on touch frequency and rejection intensity

### Intersection Analysis
- **Trendline Intersections**: Major-minor and major-major trendline crosses
- **Level Intersections**: Trendline-support/resistance crossovers
- **Future Projection**: Calculates intersection points ahead of current price
- **Confidence Scoring**: Combines component strengths for signal quality

### Trading Logic
- **Real-time Monitoring**: Checks price against active intersection points
- **Reversal Strategy**: Enters trades at high-probability reversal zones
- **Risk Management**: Fixed duration trades (3-5 minutes) with automatic exits
- **Performance Tracking**: Comprehensive P&L and win rate analysis

## 📊 Database Schema

### Trades Table
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    direction TEXT NOT NULL,
    pnl REAL NOT NULL,
    duration_minutes INTEGER NOT NULL,
    intersection_type TEXT NOT NULL
);
```

### Signals Table
```sql
CREATE TABLE signals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    type TEXT NOT NULL,
    price REAL NOT NULL,
    confidence REAL NOT NULL,
    status TEXT NOT NULL
);
```

## 🎮 Dashboard Usage

1. **Start/Stop Bot**: Control trading bot execution
2. **Price Chart**: View live 1-minute candlestick data with trendlines
3. **Live Signals**: Monitor active intersection points and their confidence
4. **Trade History**: Review past trades with entry/exit details and P&L
5. **Technical Analysis**: View detected support/resistance levels and trendlines

## ⚙️ Configuration

### Trading Parameters
```python
# Trendline Detection
major_lookback = 200        # Candles for major trendlines
minor_lookback = 30         # Candles for minor trendlines
min_touches = 2            # Minimum touches for trendline validation

# Support/Resistance
min_touches = 3            # Minimum touches for S/R levels
price_tolerance = 0.05     # Price tolerance for level clustering

# Trading
trade_duration = [3, 5]    # Trade duration options (minutes)
confidence_threshold = 0.6  # Minimum confidence for trade entry
```

## 🔮 Future Enhancements

- **Real Market Data Integration**: Connect to live data feeds (Binance, Alpha Vantage, etc.)
- **Advanced Risk Management**: Stop-loss, take-profit, position sizing
- **Multiple Timeframe Analysis**: Combine different timeframe signals
- **Machine Learning**: Pattern recognition and signal optimization
- **Real Broker Integration**: Execute trades through broker APIs
- **Advanced Visualization**: 3D charts, heatmaps, and correlation analysis

## 🛠️ Development

The codebase is designed for easy extension and modification:

- **Modular Architecture**: Each component is independent and testable
- **Clean Interfaces**: Well-defined data structures and method signatures  
- **Comprehensive Logging**: Debug and monitor all operations
- **Database Abstraction**: Easy to switch from SQLite to PostgreSQL
- **Configurable Parameters**: Adjust strategy parameters without code changes

## 📈 Performance Metrics

The bot tracks comprehensive performance metrics:
- **Total Trades**: Number of completed trades
- **Win Rate**: Percentage of profitable trades
- **Total P&L**: Cumulative profit/loss
- **Average Win/Loss**: Average profit per winning/losing trade
- **Trade Duration**: Average time in trade
- **Signal Accuracy**: Intersection point prediction accuracy

---

**Note**: This is a demonstration/educational project using simulated data. For live trading, implement proper risk management, backtesting, and regulatory compliance.
