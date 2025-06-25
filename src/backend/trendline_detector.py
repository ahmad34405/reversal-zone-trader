
"""
Trendline detection logic for the Enhanced Trading Bot
"""

import logging
from typing import List, Optional
from datetime import datetime
from data_models import Candle, TrendLine

logger = logging.getLogger(__name__)

class TrendlineDetector:
    """Handles trendline detection from market data"""
    
    def __init__(self, major_lookback: int = 200, minor_lookback: int = 30, 
                 min_touches: int = 3, price_tolerance: float = 0.05):
        self.major_lookback = major_lookback
        self.minor_lookback = minor_lookback
        self.min_touches = min_touches
        self.price_tolerance = price_tolerance
    
    def detect_trendlines(self, candles_list: List[Candle]) -> List[TrendLine]:
        """Detect trendlines from real market data"""
        if not candles_list:
            logger.warning("No candle data available for trendline detection")
            return []
        
        trendlines = []
        
        # Major trendlines (using more data points)
        if len(candles_list) >= self.major_lookback:
            major_trendlines = self._find_trendlines(
                candles_list[-self.major_lookback:], 
                "major"
            )
            trendlines.extend(major_trendlines)
        
        # Minor trendlines (using recent data)
        if len(candles_list) >= self.minor_lookback:
            minor_trendlines = self._find_trendlines(
                candles_list[-self.minor_lookback:], 
                "minor"
            )
            trendlines.extend(minor_trendlines)
        
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
    
    def calculate_trendline_price_at_time(self, trendline: TrendLine, target_time: datetime) -> Optional[float]:
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
