
"""
Trading signal generation logic for the Enhanced Trading Bot
"""

import logging
from typing import List
from datetime import datetime
from data_models import Candle, TrendLine, TradingSignal

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Handles trading signal generation based on analysis"""
    
    def __init__(self, price_tolerance: float = 0.05):
        self.price_tolerance = price_tolerance
    
    def generate_trading_signals(self, candles_list: List[Candle], 
                                trendlines: List[TrendLine]) -> List[TradingSignal]:
        """Generate trading signals based on analysis"""
        signals = []
        
        if not candles_list:
            logger.warning("No candle data available for signal generation")
            return signals
        
        current_price = candles_list[-1].close
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
        
        return signals
    
    def _calculate_trendline_price_at_time(self, trendline: TrendLine, target_time: datetime) -> float:
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
