
"""
FastAPI server to serve trading data and charts to the React frontend
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
import uvicorn
import json
import os
from datetime import datetime, timedelta
from enhanced_trading_bot import EnhancedTradingBot
from tradingview_data_service import TradingViewDataService
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Enhanced Trading Bot API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global bot instances
trading_bots = {}
data_service = TradingViewDataService()

@app.on_event("startup")
async def startup_event():
    """Initialize trading bots for common symbols"""
    default_symbols = ["BTC-USD", "ETH-USD", "AAPL"]
    
    for symbol in default_symbols:
        try:
            bot = EnhancedTradingBot(symbol=symbol)
            # Fetch initial data
            await asyncio.create_task(
                asyncio.to_thread(bot.fetch_and_store_data, "1m", 1)
            )
            trading_bots[symbol] = bot
            logger.info(f"Initialized trading bot for {symbol}")
        except Exception as e:
            logger.error(f"Failed to initialize bot for {symbol}: {e}")

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Enhanced Trading Bot API",
        "version": "1.0.0",
        "endpoints": {
            "market_data": "/api/market-data/{symbol}",
            "analysis": "/api/analysis/{symbol}",
            "chart": "/api/chart/{symbol}",
            "real_time": "/api/real-time/{symbol}",
            "status": "/api/status/{symbol}"
        }
    }

@app.get("/api/market-data/{symbol}")
async def get_market_data(symbol: str, timeframe: str = "1m", days: int = 1):
    """Get historical market data for a symbol"""
    try:
        # Get or create bot for symbol
        if symbol not in trading_bots:
            trading_bots[symbol] = EnhancedTradingBot(symbol=symbol)
        
        bot = trading_bots[symbol]
        
        # Fetch fresh data if needed
        if not bot.candles_list or len(bot.candles_list) < 100:
            success = await asyncio.to_thread(
                bot.fetch_and_store_data, timeframe, days
            )
            if not success:
                raise HTTPException(status_code=500, detail="Failed to fetch market data")
        
        # Convert candles to JSON format
        market_data = []
        for candle in bot.candles_list[-500:]:  # Return last 500 candles
            market_data.append({
                "timestamp": candle.timestamp.isoformat(),
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "volume": candle.volume
            })
        
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": market_data,
            "count": len(market_data)
        }
        
    except Exception as e:
        logger.error(f"Error getting market data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{symbol}")
async def get_analysis(symbol: str, background_tasks: BackgroundTasks):
    """Get technical analysis for a symbol"""
    try:
        # Get or create bot for symbol
        if symbol not in trading_bots:
            trading_bots[symbol] = EnhancedTradingBot(symbol=symbol)
            # Fetch data in background
            background_tasks.add_task(
                trading_bots[symbol].fetch_and_store_data, "1m", 2
            )
        
        bot = trading_bots[symbol]
        
        # Run analysis
        analysis_results = await asyncio.to_thread(bot.run_analysis)
        
        return analysis_results
        
    except Exception as e:
        logger.error(f"Error running analysis for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/real-time/{symbol}")
async def get_real_time_data(symbol: str):
    """Get real-time price data for a symbol"""
    try:
        price_data = await asyncio.to_thread(
            data_service.get_real_time_price, symbol
        )
        
        if price_data is None:
            raise HTTPException(status_code=404, detail="Real-time data not available")
        
        return price_data
        
    except Exception as e:
        logger.error(f"Error getting real-time data for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/chart/{symbol}")
async def get_chart_data(symbol: str, save_file: bool = False):
    """Generate and return chart data for a symbol"""
    try:
        # Get or create bot for symbol
        if symbol not in trading_bots:
            trading_bots[symbol] = EnhancedTradingBot(symbol=symbol)
            await asyncio.to_thread(
                trading_bots[symbol].fetch_and_store_data, "1m", 1
            )
        
        bot = trading_bots[symbol]
        
        if bot.candles_df is None or bot.candles_df.empty:
            raise HTTPException(status_code=404, detail="No chart data available")
        
        # Create chart
        chart_filename = f"{symbol.replace('-', '_')}_chart.html"
        chart_path = os.path.join("charts", chart_filename)
        
        # Ensure charts directory exists
        os.makedirs("charts", exist_ok=True)
        
        fig = await asyncio.to_thread(
            bot.create_analysis_chart, chart_path if save_file else None
        )
        
        if fig is None:
            raise HTTPException(status_code=500, detail="Failed to create chart")
        
        # Return chart data as JSON
        chart_data = fig.to_dict()
        
        return {
            "symbol": symbol,
            "chart_data": chart_data,
            "chart_file": chart_path if save_file else None
        }
        
    except Exception as e:
        logger.error(f"Error creating chart for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{symbol}")
async def get_bot_status(symbol: str):
    """Get trading bot status for a symbol"""
    try:
        if symbol not in trading_bots:
            return {
                "symbol": symbol,
                "status": "not_initialized",
                "message": "Bot not yet initialized for this symbol"
            }
        
        bot = trading_bots[symbol]
        status = await asyncio.to_thread(bot.get_status_summary)
        
        return status
        
    except Exception as e:
        logger.error(f"Error getting status for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/symbols")
async def get_available_symbols():
    """Get list of available trading symbols"""
    return {
        "symbols": list(trading_bots.keys()),
        "supported_symbols": [
            "BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD", "LINK-USD",
            "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "NVDA"
        ]
    }

@app.post("/api/symbols/{symbol}/initialize")
async def initialize_symbol(symbol: str, background_tasks: BackgroundTasks):
    """Initialize a new trading bot for a symbol"""
    try:
        if symbol in trading_bots:
            return {
                "symbol": symbol,
                "status": "already_initialized",
                "message": "Bot already exists for this symbol"
            }
        
        # Create new bot
        bot = EnhancedTradingBot(symbol=symbol)
        trading_bots[symbol] = bot
        
        # Fetch initial data in background
        background_tasks.add_task(bot.fetch_and_store_data, "1m", 2)
        
        return {
            "symbol": symbol,
            "status": "initialized",
            "message": "Bot initialized and data fetching started"
        }
        
    except Exception as e:
        logger.error(f"Error initializing bot for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/test/data-service")
async def test_data_service():
    """Test the data service functionality"""
    try:
        # Test data fetching
        test_symbol = "BTC-USD"
        df = await asyncio.to_thread(
            data_service.fetch_data_from_tradingview, test_symbol, "1m", 1
        )
        
        if df is None or df.empty:
            return {"status": "error", "message": "No data fetched"}
        
        # Test real-time data
        real_time = await asyncio.to_thread(
            data_service.get_real_time_price, test_symbol
        )
        
        return {
            "status": "success",
            "data_service": "working",
            "test_symbol": test_symbol,
            "historical_data_points": len(df),
            "real_time_price": real_time['current_price'] if real_time else None,
            "sample_data": df.head(3).to_dict() if not df.empty else None
        }
        
    except Exception as e:
        logger.error(f"Data service test failed: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
