
# Enhanced Trading Bot Backend

This enhanced trading bot backend integrates real TradingView data with advanced technical analysis and candlestick chart generation.

## 🚀 Features

### Real Market Data
- **TradingView Integration**: Fetches real OHLC data using yfinance
- **Multiple Timeframes**: 1m, 5m, 15m, 30m, 1h, 1d support
- **Real-time Price Updates**: Live price monitoring
- **Fallback Data**: Generates realistic synthetic data when real data unavailable

### Advanced Charting
- **Interactive Candlestick Charts**: Using Plotly with zoom, hover, and pan
- **Volume Analysis**: Volume bars with color-coded bull/bear indicators
- **Trendline Overlays**: Major and minor trendlines on charts
- **Multiple Export Formats**: HTML, JSON, and image export

### Technical Analysis
- **Swing Point Detection**: Automated swing high/low identification
- **Trendline Detection**: Major (200 candles) and minor (30 candles) trendlines
- **Support/Resistance**: Price level identification with touch counting
- **Signal Generation**: Trading signals based on technical analysis

### API Server
- **FastAPI Backend**: RESTful API for React frontend integration
- **Real-time Endpoints**: Live data and analysis endpoints
- **Background Processing**: Async data fetching and analysis
- **CORS Support**: Cross-origin requests for web frontend

## 📦 Installation

```bash
# Install Python dependencies
pip install -r requirements.txt

# Or install individually:
pip install pandas numpy yfinance plotly mplfinance fastapi uvicorn websocket-client python-dotenv
```

## 🏃‍♂️ Quick Start

### 1. Run Standalone Bot
```bash
# Run with default symbol (BTC-USD)
python run_bot.py

# Run with specific symbol
python run_bot.py ETH-USD
python run_bot.py AAPL
```

### 2. Start API Server
```bash
# Start FastAPI server
python api_server.py

# Server runs on http://localhost:8000
# API docs available at http://localhost:8000/docs
```

### 3. Test Data Service
```bash
# Test TradingView data fetching
python -c "from tradingview_data_service import TradingViewDataService; TradingViewDataService().main()"
```

## 🔧 Usage Examples

### Fetch Real Market Data
```python
from tradingview_data_service import TradingViewDataService

service = TradingViewDataService()

# Fetch 1-minute data for Bitcoin
df = service.fetch_data_from_tradingview("BTC-USD", "1m", days=7)
print(f"Fetched {len(df)} candles")
print(df.head())
```

### Create Candlestick Chart
```python
# Create interactive chart
fig = service.plot_candlestick_chart(df, "BTC-USD")
fig.write_html("btc_chart.html")

# Alternative with mplfinance
service.plot_mplfinance_chart(df, "BTC-USD", "btc_chart.png")
```

### Run Complete Analysis
```python
from enhanced_trading_bot import EnhancedTradingBot

bot = EnhancedTradingBot("ETH-USD")
bot.fetch_and_store_data("1m", days=2)
results = bot.run_analysis()

print(f"Current Price: ${results['current_price']}")
print(f"Signals: {len(results['signals'])}")
```

## 🌐 API Endpoints

### Market Data
```bash
# Get historical data
GET /api/market-data/BTC-USD?timeframe=1m&days=1

# Get real-time price
GET /api/real-time/BTC-USD
```

### Analysis
```bash
# Run technical analysis
GET /api/analysis/BTC-USD

# Get chart data
GET /api/chart/BTC-USD

# Get bot status
GET /api/status/BTC-USD
```

### Management
```bash
# List available symbols
GET /api/symbols

# Initialize new symbol
POST /api/symbols/AAPL/initialize

# Test data service
GET /api/test/data-service
```

## 📊 Supported Symbols

### Cryptocurrencies
- BTC-USD, ETH-USD, ADA-USD, DOT-USD, LINK-USD

### Stocks
- AAPL, TSLA, GOOGL, MSFT, AMZN, NVDA

### Adding Custom Symbols
The bot automatically converts common trading symbols:
- BTCUSDT → BTC-USD
- ETHUSDT → ETH-USD
- Any symbol supported by yfinance

## 🗄️ Database Schema

### Market Data Table
```sql
CREATE TABLE market_data (
    symbol TEXT,
    timestamp TEXT,
    open_price REAL,
    high_price REAL,
    low_price REAL,
    close_price REAL,
    volume INTEGER
);
```

### Trading Signals Table
```sql
CREATE TABLE trading_signals (
    symbol TEXT,
    timestamp TEXT,
    signal_type TEXT,
    price REAL,
    confidence REAL,
    status TEXT,
    components TEXT
);
```

### Trendlines Table
```sql
CREATE TABLE trendlines (
    symbol TEXT,
    start_time TEXT,
    start_price REAL,
    end_time TEXT,
    end_price REAL,
    type TEXT,
    direction TEXT,
    strength REAL,
    touches INTEGER
);
```

## ⚙️ Configuration

### Trading Parameters
```python
# Trendline detection
major_lookback = 200     # Candles for major trendlines
minor_lookback = 30      # Candles for minor trendlines
min_touches = 3          # Minimum touches for validation

# Price analysis
price_tolerance = 0.05   # Price tolerance for level detection

# Data fetching
default_timeframe = "1m" # Default timeframe
default_days = 7         # Default data history
```

### Environment Variables
```bash
# Optional: Set custom API keys (if using premium data sources)
export TRADING_API_KEY="your_api_key"
export DATABASE_PATH="custom_trading.db"
```

## 🔍 Technical Analysis Features

### Trendline Detection
- **Swing Point Algorithm**: Identifies local highs and lows
- **Linear Regression**: Fits trendlines through swing points
- **Touch Validation**: Counts price touches to validate trendlines
- **Strength Scoring**: Confidence based on touches and alignment

### Signal Generation
- **Trendline Touches**: Signals when price approaches trendlines
- **Breakout Detection**: Identifies trendline breaks
- **Confluence Areas**: Multiple indicator convergence
- **Confidence Scoring**: 0-1 scale based on technical strength

### Chart Features
- **Interactive Navigation**: Zoom, pan, hover tooltips
- **Multi-timeframe**: Switch between different timeframes
- **Indicator Overlays**: Trendlines, support/resistance levels
- **Volume Analysis**: Volume bars with bull/bear coloring

## 🚀 Integration with React Frontend

The backend is designed to work seamlessly with the React dashboard:

```javascript
// Fetch market data
const response = await fetch('/api/market-data/BTC-USD');
const data = await response.json();

// Get analysis results
const analysis = await fetch('/api/analysis/BTC-USD');
const results = await analysis.json();

// Real-time price updates
const price = await fetch('/api/real-time/BTC-USD');
const priceData = await price.json();
```

## 🛠️ Development

### File Structure
```
src/backend/
├── tradingview_data_service.py    # Core data fetching and charting
├── enhanced_trading_bot.py        # Trading bot with real data
├── api_server.py                  # FastAPI server
├── run_bot.py                     # Standalone bot runner
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Testing
```bash
# Test data service
python -m pytest test_data_service.py

# Test API endpoints
python -m pytest test_api.py

# Manual testing
python run_bot.py BTC-USD
```

## 📝 Notes

- **Data Sources**: Uses yfinance for reliable, free market data
- **Rate Limits**: Respects yfinance rate limits and includes fallback data
- **Error Handling**: Comprehensive error handling with logging
- **Performance**: Optimized for real-time analysis with caching
- **Scalability**: Designed to handle multiple symbols simultaneously

## 🔮 Future Enhancements

- **Live WebSocket Feeds**: Real-time streaming data
- **Advanced Indicators**: RSI, MACD, Bollinger Bands
- **Machine Learning**: Pattern recognition and prediction
- **Multi-Exchange Support**: Binance, Coinbase, etc.
- **Paper Trading**: Simulated trading with real data
- **Alert System**: Email/SMS notifications for signals

---

This enhanced backend provides a robust foundation for professional-grade trading bot development with real market data integration.
