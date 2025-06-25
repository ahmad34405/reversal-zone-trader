
"""
Simple script to run the enhanced trading bot with real TradingView data
"""

import asyncio
import sys
import os
from enhanced_trading_bot import EnhancedTradingBot
from tradingview_data_service import TradingViewDataService

async def main():
    """Main function to demonstrate the enhanced trading bot"""
    
    print("🤖 Enhanced Trading Bot with Real TradingView Data")
    print("=" * 60)
    
    # Get symbol from command line or use default
    symbol = sys.argv[1] if len(sys.argv) > 1 else "BTC-USD"
    
    print(f"📊 Symbol: {symbol}")
    print(f"⏰ Starting analysis...")
    
    # Initialize the enhanced trading bot
    bot = EnhancedTradingBot(symbol=symbol)
    
    # Fetch real market data
    print("📥 Fetching real market data from TradingView...")
    success = bot.fetch_and_store_data(timeframe="1m", days=2)
    
    if not success:
        print("❌ Failed to fetch market data")
        return
    
    print(f"✅ Successfully fetched {len(bot.candles_list)} candles")
    
    # Run technical analysis
    print("🔍 Running technical analysis...")
    analysis_results = bot.run_analysis()
    
    # Display results
    print("\n📈 Analysis Results:")
    print(f"Current Price: ${analysis_results['current_price']:.2f}")
    print(f"Data Points: {analysis_results['data_points']}")
    print(f"Trendlines Detected: {len(analysis_results['trendlines'])}")
    print(f"Trading Signals: {len(analysis_results['signals'])}")
    
    # Show trendlines
    if analysis_results['trendlines']:
        print("\n📊 Detected Trendlines:")
        for i, tl in enumerate(analysis_results['trendlines'], 1):
            print(f"  {i}. {tl['type'].title()} {tl['direction']} "
                  f"(Strength: {tl['strength']:.2f}, Touches: {tl['touches']})")
    
    # Show signals
    if analysis_results['signals']:
        print("\n🎯 Trading Signals:")
        for i, signal in enumerate(analysis_results['signals'], 1):
            print(f"  {i}. {signal['type']} at ${signal['price']:.2f} "
                  f"(Confidence: {signal['confidence']:.2f})")
    
    # Create analysis chart
    print("\n📊 Creating candlestick chart...")
    chart_filename = f"{symbol.replace('-', '_')}_analysis.html"
    fig = bot.create_analysis_chart(save_path=chart_filename)
    
    if fig:
        print(f"✅ Chart saved as: {chart_filename}")
        print(f"🌐 Open {chart_filename} in your browser to view the interactive chart")
    
    # Get bot status
    status = bot.get_status_summary()
    print(f"\n🤖 Bot Status: {status['status']}")
    print(f"📂 Database: {bot.db_path}")
    
    print("\n✨ Analysis complete!")

if __name__ == "__main__":
    asyncio.run(main())
